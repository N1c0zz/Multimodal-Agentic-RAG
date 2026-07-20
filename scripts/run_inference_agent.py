"""
ReAct agentic inference with two retrieval tools:
- retrieve_knowledge: image-only first pass
- refine_search: image + hypothesis fused second pass

Enforces (outside the agent, to avoid relying on a known-unreliable
smolagents FinalAnswerTool override, see huggingface/smolagents#1254) that
every episode retrieves at least once: if the agent answers without ever
calling retrieve_knowledge, we force one retrieval pass and re-answer using
it, matching what the non-agentic RAG baseline would always do.

Falls back to a plain (no-tool, no-context) Qwen call only if the agent
fails entirely (exception or empty answer).

Also logs evidence_in_context (whether the oracle Wikipedia page was among
the URLs retrieved by EITHER tool, or by the forced fallback retrieval, at
any point during the episode).
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from smolagents import ToolCallingAgent

from retriever_agent import RetrieverAgent
from agent_tools import KnowledgeRetrievalTool, RefineSearchTool, EpisodeState
from qwen_agent_model import QwenAgentModel

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

CUSTOM_INSTRUCTIONS = (
    "You are answering knowledge-intensive visual questions about an entity "
    "shown in an image (a plant, animal, building, etc.). "
    "\n\n"
    "You have access to EXACTLY THREE tools, and no others: "
    "'retrieve_knowledge', 'refine_search', and 'final_answer'. "
    "Never call any other tool name (for example: image_transformer, "
    "image_search, image_generator, document_qa, web_search do NOT exist "
    "here and calling them will always fail). "
    "\n\n"
    "You MUST always call retrieve_knowledge at least once before calling "
    "final_answer, even if you think you already know the answer -- your "
    "own knowledge is often wrong on this task, and the retrieved evidence "
    "is more reliable.\n"
    "\n"
    "Example of a correct first action:\n"
    '{"name": "retrieve_knowledge", "arguments": {"reasoning": '
    '"I need background information about the entity shown in the image."}}\n\n'
    "\n"
    "Call retrieve_knowledge AT MOST ONCE per question -- it is based only "
    "on the image, so calling it again returns the exact same evidence. "
    "If that evidence is insufficient or seems to be about the wrong entity, "
    "call refine_search ONCE with your best hypothesis about the entity's "
    "specific identity (e.g. a species name) -- this CAN return different, "
    "more targeted evidence. Do not call refine_search more than once either.\n"
    "\n"
    "Base your final answer primarily on the retrieved evidence and the image."
    "\n\n"
    "The 'answer' argument of final_answer must be EXTREMELY SHORT: a single "
    "word, a name, a number, or a short comma-separated list. NEVER restate "
    "the question, NEVER write a full explanatory sentence.\n"
    "WRONG: {\"name\": \"final_answer\", \"arguments\": {\"answer\": "
    "\"The size of an adult Argiope catenulata typically ranges from 15 to "
    "25 mm.\"}}\n"
    "RIGHT: {\"name\": \"final_answer\", \"arguments\": {\"answer\": "
    "\"15-25 mm\"}}"
)


def plain_vlm_fallback(question: str, image: Image.Image, model: QwenAgentModel) -> str:
    """Used only if the agent fails entirely (exception / empty answer)."""
    prompt_text = (
        f"{question}\n\n"
        "Answer with the shortest possible response: "
        "a single word, name, or brief phrase. "
        "Do not explain or use full sentences."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        gen_ids = model.model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return model.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def context_augmented_fallback(question: str, image: Image.Image, context: str, model: QwenAgentModel) -> str:
    """Used when the agent answered WITHOUT ever calling retrieve_knowledge."""
    prompt_text = (
        f"Context:\n{context}\n\n"
        f"Based ONLY on the context above, answer the following question. "
        f"If the answer is not in the context, use the image and your best judgment.\n\n"
        f"Question: {question}\n\n"
        "Answer with the shortest possible response: "
        "a single word, name, or brief phrase. "
        "Do not explain or use full sentences."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        gen_ids = model.model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    return model.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--text_weight", type=float, default=0.3)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--verbosity_level", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    retriever = RetrieverAgent(top_k=args.top_k, text_weight=args.text_weight)
    episode_state = EpisodeState()
    tool_retrieve = KnowledgeRetrievalTool(retriever, episode_state)
    tool_refine = RefineSearchTool(retriever, episode_state, text_weight=args.text_weight)
    model = QwenAgentModel(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    model.set_episode_state(episode_state)

    agent = ToolCallingAgent(
        tools=[tool_retrieve, tool_refine],
        model=model,
        instructions=CUSTOM_INSTRUCTIONS,
        max_steps=args.max_steps,
        verbosity_level=args.verbosity_level,
    )

    results = []
    fallback_count = 0
    forced_retrieval_count = 0

    for sample in tqdm(samples, desc="Agentic Inference"):
        image_path = str(IMAGE_ROOT / sample["related_images"])
        image = Image.open(image_path).convert("RGB")

        tool_retrieve.set_image(image)
        tool_refine.set_image(image)
        model.set_image(image)
        episode_state.reset()

        prediction = ""
        n_steps = None
        used_fallback = False
        forced_retrieval = False

        try:
            answer = agent.run(sample["question"], images=[image], reset=True)
            prediction = str(answer).strip()
            try:
                n_steps = len(agent.memory.steps)
            except Exception:
                n_steps = None
        except Exception as e:
            print(f"Agent error on {sample['unique_id']}: {e}")

        if prediction and not episode_state.has_retrieved:
            print(f"Agent skipped retrieval on {sample['unique_id']}, forcing one pass")
            forced_retrieval = True
            forced_retrieval_count += 1
            try:
                context, urls = retriever.retrieve(image, query_text="", text_weight=0.0)
                tool_retrieve.retrieved_urls_log.extend(urls)
                if context:
                    prediction = context_augmented_fallback(sample["question"], image, context, model)
            except Exception as e:
                print(f"Forced retrieval failed on {sample['unique_id']}: {e}")

        if not prediction:
            print(f"Empty/failed agent answer on {sample['unique_id']}, using fallback")
            try:
                prediction = plain_vlm_fallback(sample["question"], image, model)
                used_fallback = True
                fallback_count += 1
            except Exception as e:
                print(f"Fallback also failed on {sample['unique_id']}: {e}")
                prediction = ""

        reference = sample.get("answer", "")

        retrieved_urls = list(set(tool_retrieve.retrieved_urls_log + tool_refine.retrieved_urls_log))
        oracle_urls = [u.strip() for u in sample.get('wikipedia_url', '').split('|') if u.strip()]
        evidence_in_context = any(url in retrieved_urls for url in oracle_urls)

        results.append({
            "data_id": sample["unique_id"],
            "question": sample["question"],
            "reference": reference,
            "answers": prediction,
            "question_type": sample.get("question_type", "automatic"),
            "n_steps": n_steps,
            "used_fallback": used_fallback,
            "forced_retrieval": forced_retrieval,
            "evidence_in_context": evidence_in_context,
        })

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")
    print(f"Fallback used on {fallback_count}/{len(samples)} samples")
    print(f"Forced retrieval (agent skipped it) on {forced_retrieval_count}/{len(samples)} samples")


if __name__ == "__main__":
    main()