"""
ReAct agentic inference with a 3-tool pipeline (+ built-in final_answer):
- assess_retrieval_need: decides RETRIEVE vs ANSWER_DIRECTLY (always first)
- retrieve_knowledge: image + auto-generated guesses + question, with
  automatic ReAG-Critic relevance filtering
- refine_search: optional, agent-triggered, grounded dedicated re-guess,
  also with automatic filtering

Relevance filtering (ReAG-Critic) is applied automatically inside the
retrieval tools rather than as a separate optional tool -- see
agent_tools.py's module docstring for why.

Safety net: forces one retrieval pass ONLY if retrieval was recommended (or
never assessed) and the agent skipped it -- an ANSWER_DIRECTLY decision that
the agent respects is NOT overridden, since honoring it is the entire point
of adding this decision point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import json
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from smolagents import ToolCallingAgent

from qwen_utils import generate_greedy
from eval_utils import load_dataset, build_result_record
from retriever_agent import RetrieverAgent
from reag_critic import ReAGCritic
from agent_tools import (
    AssessRetrievalNeedTool, KnowledgeRetrievalTool, RefineSearchTool,
    EpisodeState, _generate_dedicated, GUESS_PROMPT,
)
from qwen_agent_model import QwenAgentModel

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

CUSTOM_INSTRUCTIONS = (
    "You are answering knowledge-intensive visual questions about an entity "
    "shown in an image (a plant, animal, building, etc.). "
    "\n\n"
    "You have access to EXACTLY THREE actions, and no others: "
    "'assess_retrieval_need', 'retrieve_knowledge', 'refine_search', and "
    "'final_answer'. Never call any other tool name (image_search, "
    "web_search, etc. do NOT exist here and will always fail). Retrieved "
    "evidence is automatically filtered for relevance -- you do not need to "
    "do this yourself."
    "\n\n"
    "Your workflow:\n"
    "1. ALWAYS call assess_retrieval_need FIRST. It tells you whether to "
    "retrieve external knowledge or answer directly.\n"
    "2. If it recommends RETRIEVE, call retrieve_knowledge (at most once).\n"
    "3. If the retrieved evidence seems to be about the wrong entity, or you "
    "are told all retrieved evidence was filtered out as irrelevant, you may "
    "call refine_search ONCE to try again.\n"
    "4. Call final_answer when you are ready to respond.\n"
    "\n"
    "If assess_retrieval_need recommends ANSWER_DIRECTLY, you may call "
    "final_answer right away using the image and your own knowledge, without "
    "retrieving -- this is a valid and often correct choice, not a shortcut "
    "to avoid.\n"
    "\n"
    "Example of a correct first action:\n"
    '{"name": "assess_retrieval_need", "arguments": {"reasoning": '
    '"Checking whether I need external knowledge for this question."}}\n\n'
    "\n"
    "The 'answer' argument of final_answer must be EXTREMELY SHORT: a single "
    "word, a name, a number, or a short comma-separated list. NEVER restate "
    "the question, NEVER write a full sentence.\n"
    "WRONG: {\"name\": \"final_answer\", \"arguments\": {\"answer\": "
    "\"The size of an adult Argiope catenulata ranges from 15 to 25 mm.\"}}\n"
    "RIGHT: {\"name\": \"final_answer\", \"arguments\": {\"answer\": \"15-25 mm\"}}"
    "Even if no relevant evidence was found, ALWAYS give your best guess based "
    "on the image and your own knowledge -- never answer 'Unknown', 'I don't "
    "know', or similar. A specific guess is always better than admitting "
    "uncertainty for this task.\n"
)


def plain_vlm_fallback(question, image, model):
    """Used only if the agent fails entirely (exception / empty answer)."""
    prompt_text = (
        f"{question}\n\nAnswer with the shortest possible response: "
        "a single word, name, or brief phrase. Do not explain."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text}]}]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.model.device)
    return generate_greedy(model.model, model.processor, inputs, max_new_tokens=64)


def context_augmented_fallback(question, image, context, model):
    """Used when the agent answered WITHOUT ever calling retrieve_knowledge."""
    prompt_text = (
        f"Context:\n{context}\n\nBased ONLY on the context above, answer the "
        f"following question. If the answer is not in the context, use the "
        f"image and your best judgment.\n\nQuestion: {question}\n\n"
        "Answer with the shortest possible response: a single word, name, or "
        "brief phrase. Do not explain."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text}]}]
    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.model.device)
    return generate_greedy(model.model, model.processor, inputs, max_new_tokens=64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--text_weight", type=float, default=0.3)
    parser.add_argument("--critic_threshold", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--verbosity_level", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    retriever = RetrieverAgent(top_k=args.top_k, text_weight=args.text_weight)
    critic = ReAGCritic(yes_prob_threshold=args.critic_threshold)
    episode_state = EpisodeState()

    model = QwenAgentModel(
        model_id=args.model_id, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, max_retries=args.max_retries,
    )
    model.set_episode_state(episode_state)

    tool_assess = AssessRetrievalNeedTool(episode_state, model.model, model.processor)
    tool_retrieve = KnowledgeRetrievalTool(retriever, episode_state, model.model, model.processor, critic)
    tool_refine = RefineSearchTool(retriever, episode_state, model.model, model.processor, critic, text_weight=args.text_weight)

    agent = ToolCallingAgent(
        tools=[tool_assess, tool_retrieve, tool_refine], model=model,
        instructions=CUSTOM_INSTRUCTIONS, max_steps=args.max_steps,
        verbosity_level=args.verbosity_level,
    )

    results = []
    fallback_count = 0
    forced_retrieval_count = 0
    answer_directly_count = 0
    filter_removed_all_count = 0

    for sample in tqdm(samples, desc="Agentic Inference"):
        image_path = str(IMAGE_ROOT / sample["related_images"])
        image = Image.open(image_path).convert("RGB")
        question = sample["question"]

        for t in (tool_assess, tool_retrieve, tool_refine):
            t.set_image(image)
            if hasattr(t, "set_question"):
                t.set_question(question)
        model.set_image(image)
        episode_state.reset()

        prediction = ""; n_steps = None; used_fallback = False; forced_retrieval = False

        try:
            answer = agent.run(question, images=[image], reset=True)
            prediction = str(answer).strip()
            try:
                n_steps = len(agent.memory.steps)
            except Exception:
                n_steps = None
        except Exception as e:
            print(f"Agent error on {sample['unique_id']}: {e}")

        skipped_retrieval = prediction and not episode_state.has_retrieved
        respected_answer_directly = (
            episode_state.has_assessed
            and episode_state.retrieval_recommended == "ANSWER_DIRECTLY"
        )

        if skipped_retrieval and not respected_answer_directly:
            print(f"Forcing retrieval on {sample['unique_id']} (recommended or unassessed)")
            forced_retrieval = True; forced_retrieval_count += 1
            try:
                guesses = _generate_dedicated(image, GUESS_PROMPT, model.model, model.processor)
                combined_query = f"Image tags: {guesses}. Question: {question}"
                _raw_context, urls, labeled_passages = retriever.retrieve(image, query_text=combined_query, text_weight=args.text_weight)
                tool_retrieve.retrieved_urls_log.extend(urls)
                filtered_passages = critic.filter_passages(image, question, labeled_passages)
                filtered_context = "\n\n".join(f"[{label}]\n{text}" for label, text in filtered_passages)
                if filtered_context:
                    prediction = context_augmented_fallback(question, image, filtered_context, model)
            except Exception as e:
                print(f"Forced retrieval failed on {sample['unique_id']}: {e}")
        elif skipped_retrieval and respected_answer_directly:
            answer_directly_count += 1

        if episode_state.filter_removed_all:
            filter_removed_all_count += 1

        if not prediction:
            print(f"Empty/failed agent answer on {sample['unique_id']}, using fallback")
            try:
                prediction = plain_vlm_fallback(question, image, model)
                used_fallback = True; fallback_count += 1
            except Exception as e:
                print(f"Fallback also failed on {sample['unique_id']}: {e}")
                prediction = ""

        retrieved_urls = list(set(tool_retrieve.retrieved_urls_log + tool_refine.retrieved_urls_log))

        results.append(build_result_record(
            sample, prediction, retrieved_urls,
            extra_fields={
                "n_steps": n_steps,
                "used_fallback": used_fallback,
                "forced_retrieval": forced_retrieval,
                "retrieval_recommended": episode_state.retrieval_recommended,
                "retrieved": episode_state.has_retrieved,
                "filter_removed_all": episode_state.filter_removed_all,
            },
        ))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")
    print(f"Fallback used on {fallback_count}/{len(samples)} samples")
    print(f"Forced retrieval on {forced_retrieval_count}/{len(samples)} samples")
    print(f"Answered directly (no retrieval, respected) on {answer_directly_count}/{len(samples)} samples")
    print(f"Filter removed ALL retrieved passages on {filter_removed_all_count}/{len(samples)} samples")


if __name__ == "__main__":
    main()