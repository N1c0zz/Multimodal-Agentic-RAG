"""
ReAct agentic inference con pipeline a 2 tool (+ final_answer nativo), per
indicazione dei tutor: recuperare documenti INTERI, lasciare che un filtro
fine per sezione selezioni il rilevante, e limitare i tool a 2-3.
refine_search è stato rimosso (vedi il docstring di agent_tools.py).

FIXED: le risposte finali "bail-out" (Unknown, I don't know, ecc.) vengono
ora intercettate a livello di codice e sostituite con plain_vlm_fallback,
esattamente come già succedeva per le risposte vuote. Un'istruzione nel
prompt che vietava questo comportamento era stata provata in precedenza e
non aveva funzionato -- coerente con ogni altro problema di affidabilità
di questo modello, risolto solo intercettando il comportamento nel codice.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import json
import argparse
from tqdm import tqdm
from PIL import Image
from smolagents import ToolCallingAgent

from eval_utils import load_dataset, build_result_record
from retriever_agent import RetrieverAgent
from reag_critic import ReAGCritic
from agent_tools import (
    AssessRetrievalNeedTool, KnowledgeRetrievalTool,
    EpisodeState, _generate_dedicated, GUESS_PROMPT_TEMPLATE,
)
from qwen_agent_model import QwenAgentModel

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

BAILOUT_PATTERNS = [
    "unknown", "i don't know", "i do not know", "n/a", "not sure",
    "cannot determine", "no information", "not available", "unclear",
]


def _is_bailout(prediction: str) -> bool:
    if not prediction or not prediction.strip():
        return True
    normalized = prediction.strip().lower()
    if len(normalized) > 60:
        # risposte lunghe difficilmente sono un puro bail-out; evita falsi
        # positivi su risposte legittime che contengono per caso una di
        # queste parole (es. "origine sconosciuta, regione...")
        return False
    return any(p in normalized for p in BAILOUT_PATTERNS)


CUSTOM_INSTRUCTIONS = (
    "You are answering knowledge-intensive visual questions about an entity "
    "shown in an image (a plant, animal, building, etc.). "
    "\n\n"
    "You have access to EXACTLY TWO actions, and no others: "
    "'assess_retrieval_need' and 'retrieve_knowledge'. Never call any other "
    "tool name (image_search, web_search, refine_search, etc. do NOT exist "
    "here and will always fail). Retrieved evidence is automatically "
    "filtered for relevance at the section level -- you do not need to do "
    "this yourself."
    "\n\n"
    "Your workflow:\n"
    "1. ALWAYS call assess_retrieval_need FIRST. It tells you whether to "
    "retrieve external knowledge or answer directly.\n"
    "2. If it recommends RETRIEVE, call retrieve_knowledge (at most once).\n"
    "3. Call final_answer when you are ready to respond.\n"
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
)


def plain_vlm_fallback(question, image, model):
    prompt_text = (
        f"{question}\n\nAnswer with the shortest possible response: "
        "a single word, name, or brief phrase. Do not explain."
    )
    return model.generate_plain(image, prompt_text, max_new_tokens=64)


def context_augmented_fallback(question, image, context, model):
    prompt_text = (
        f"Context:\n{context}\n\nBased ONLY on the context above, answer the "
        f"following question. If the answer is not in the context, use the "
        f"image and your best judgment.\n\nQuestion: {question}\n\n"
        "Answer with the shortest possible response: a single word, name, or "
        "brief phrase. Do not explain."
    )
    return model.generate_plain(image, prompt_text, max_new_tokens=64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--text_weight", type=float, default=0.3)
    parser.add_argument("--critic_threshold", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=5)
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

    tool_assess = AssessRetrievalNeedTool(episode_state, model)
    tool_retrieve = KnowledgeRetrievalTool(retriever, episode_state, model, critic)

    agent = ToolCallingAgent(
        tools=[tool_assess, tool_retrieve], model=model,
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

        for t in (tool_assess, tool_retrieve):
            t.set_image(image)
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
                prompt = GUESS_PROMPT_TEMPLATE.format(question=question)
                guesses = _generate_dedicated(image, prompt, model)
                combined_query = f"{guesses}"
                urls, labeled_sections = retriever.retrieve(image, query_text=combined_query, text_weight=args.text_weight)
                tool_retrieve.retrieved_urls_log.extend(urls)
                filtered = critic.filter_passages(image, question, labeled_sections)
                filtered_context = "\n\n".join(f"[{label}]\n{text}" for label, text in filtered)
                if filtered_context:
                    prediction = context_augmented_fallback(question, image, filtered_context, model)
            except Exception as e:
                print(f"Forced retrieval failed on {sample['unique_id']}: {e}")
        elif skipped_retrieval and respected_answer_directly:
            answer_directly_count += 1

        if episode_state.filter_removed_all:
            filter_removed_all_count += 1

        if _is_bailout(prediction):
            print(f"Bailout answer on {sample['unique_id']} ('{prediction}'), using fallback")
            try:
                prediction = plain_vlm_fallback(question, image, model)
                used_fallback = True; fallback_count += 1
            except Exception as e:
                print(f"Fallback also failed on {sample['unique_id']}: {e}")
                prediction = prediction or ""

        retrieved_urls = list(set(tool_retrieve.retrieved_urls_log))

        results.append(build_result_record(
            sample, prediction, retrieved_urls,
            extra_fields={
                "model_id": args.model_id,
                "n_steps": n_steps,
                "used_fallback": used_fallback,
                "forced_retrieval": forced_retrieval,
                "retrieval_recommended": episode_state.retrieval_recommended,
                "retrieved": episode_state.has_retrieved,
                "filter_removed_all": episode_state.filter_removed_all,
                "n_sections_before_filter": episode_state.n_sections_before_filter,
                "n_sections_after_filter": episode_state.n_sections_after_filter,
            },
        ))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")
    print(f"Fallback used on {fallback_count}/{len(samples)} samples")
    print(f"Forced retrieval on {forced_retrieval_count}/{len(samples)} samples")
    print(f"Answered directly (no retrieval, respected) on {answer_directly_count}/{len(samples)} samples")
    print(f"Filter removed ALL retrieved sections on {filter_removed_all_count}/{len(samples)} samples")


if __name__ == "__main__":
    main()