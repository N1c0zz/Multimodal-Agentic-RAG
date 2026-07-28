"""
RAG Oracle inference script for Qwen2.5-VL-3B-Instruct.
Uses the pre-retrieved context (section_texts) already present in the dataset
to augment the prompt with relevant Wikipedia passages before answering.
This is an oracle RAG baseline: retrieval is not performed at inference time,
but is taken directly from the dataset annotations -- it measures an upper
bound on how well Qwen can use CORRECT context, decoupled from retrieval
quality.

NOTE: this script has no FAISS retrieval and therefore no retrieved_urls to
compare against an oracle URL. evidence_in_context here is computed by
checking whether the reference answer text literally appears in the given
context (see eval_utils.compute_evidence_in_context_by_answer_match) -- a
different method than every other RAG script in this project, which check
oracle-URL membership instead. The two are not directly comparable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import json
import argparse
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from qwen_utils import load_qwen, generate_greedy
from eval_utils import load_dataset, build_result_record, compute_evidence_in_context_by_answer_match

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4


def build_context(sample: dict) -> str:
    """Extract and concatenate relevant section_texts from the dataset's own 'retrieval' field."""
    retrieval_list = sample.get('retrieval', [])
    if not retrieval_list:
        return ""

    # The dataset has one retrieval entry per sample.
    retrieval = retrieval_list[0]
    section_texts = retrieval.get('section_texts', [])
    section_titles = retrieval.get('section_titles', [])

    context_parts = []
    for title, text in zip(section_titles, section_texts):
        if title.lower() in EXCLUDE_SECTIONS:
            continue
        if text.strip():
            context_parts.append(text.strip())
        if len(context_parts) >= MAX_SECTIONS:
            break

    return "\n\n".join(context_parts)


def run_inference(sample: dict, model, processor) -> tuple[str, bool]:
    image_path = str(IMAGE_ROOT / sample['related_images'])
    context = build_context(sample)
    evidence_in_context = compute_evidence_in_context_by_answer_match(sample, context)

    if context:
        prompt_text = (
            f"Here is some relevant context:\n{context}\n\n"
            f"Question: {sample['question']}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )
    else:
        # Fallback to plain VLM if no context available for this sample.
        prompt_text = (
            f"{sample['question']}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prediction = generate_greedy(model, processor, inputs, max_new_tokens=64)
    return prediction, evidence_in_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        default='/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json')
    parser.add_argument('--output_dir', required=True,
                        help="Directory where split_0.json will be written")
    parser.add_argument('--model_id', default='Qwen/Qwen2.5-VL-3B-Instruct')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="RAG Oracle Inference"):
        try:
            prediction, has_evidence = run_inference(sample, model, processor)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            has_evidence = False

        # retrieved_urls=None (no real retrieval here): pass the
        # substring-based evidence flag via extra_fields instead.
        results.append(build_result_record(
            sample, prediction,
            extra_fields={"evidence_in_context": has_evidence},
        ))

    out_file = output_dir / "split_0.json"
    with open(out_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()