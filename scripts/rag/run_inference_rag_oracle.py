"""
Oracle RAG Inference Baseline.

Evaluates an upper-bound scenario for context utilization by bypassing standard 
vector-store retrieval. Instead, it extracts the pre-retrieved oracle Wikipedia 
passages directly annotated within the dataset and feeds them as context.

Note on metrics: Since no runtime URL retrieval occurs, evidence containment 
must be evaluated using a substring-matching proxy 
(`compute_evidence_in_context_by_answer_match`) rather than the strict URL-based 
check utilized in actual retrieval pipelines.
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
    """
    Extracts and concatenates relevant textual sections directly from the 
    dataset's annotated oracle 'retrieval' field.
    """
    retrieval_list = sample.get('retrieval', [])
    if not retrieval_list:
        return ""

    # The dataset schema provides a single optimal retrieval entry per sample
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
    """Generates an answer given the image, question, and oracle context."""
    image_path = str(IMAGE_ROOT / sample['related_images'])
    context = build_context(sample)
    
    # Employs string-matching heuristic to verify evidence presence
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
        # Fallback to plain zero-shot VLM capability if oracle context is empty
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

        # retrieved_urls=None since no runtime vector search is performed.
        # The boolean proxy flag is passed via extra_fields.
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