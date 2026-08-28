"""
Dynamic Re-ranking RAG Inference Pipeline.

Integrates a two-stage retrieval mechanism with confidence-adaptive context selection.
The pipeline retrieves a broad candidate set, generates a visual caption to perform
cross-modal re-ranking, and dynamically restricts the context to the top-1 document 
if the confidence margin exceeds a defined threshold. This approach aims to reduce 
contextual noise in unambiguous retrieval scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import json
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

from qwen_utils import load_qwen, generate_greedy
from eval_utils import load_dataset, build_result_record
from retriever_rerank_dynamic import RetrieverRerankDynamic

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def run_inference(
    sample: dict,
    model,
    processor,
    retriever: RetrieverRerankDynamic,
    image: Image.Image,
) -> tuple[str, bool, list[str]]:
    """Executes the dynamic re-ranking inference pipeline for a single sample."""
    image_path = str(IMAGE_ROOT / sample["related_images"])

    # The visual caption is generated internally by the retriever for the re-ranking 
    # process and is discarded here, as it is not injected into the final QA prompt.
    context, top_urls, _caption, high_confidence = retriever.retrieve_rerank(
        image, sample["question"], model, processor
    )

    if context:
        prompt_text = (
            f"Context:\n{context}\n\n"
            f"Based ONLY on the context above, answer the following question. "
            f"If the answer is not in the context, use the image and your best judgment.\n\n"
            f"Question: {sample['question']}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )
    else:
        # Fallback prompt structure when retrieval yields an empty context
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
    return prediction, high_confidence, top_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_k_retrieval", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.08)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    retriever = RetrieverRerankDynamic(
        top_k=args.top_k,
        top_k_retrieval=args.top_k_retrieval,
        alpha=args.alpha,
        confidence_threshold=args.confidence_threshold,
    )
    model, processor = load_qwen(args.model_id)

    results = []
    high_conf_count = 0
    for sample in tqdm(samples, desc="RAG Rerank Dynamic Inference"):
        try:
            image_path = str(IMAGE_ROOT / sample["related_images"])
            image = Image.open(image_path).convert("RGB")
            prediction, high_confidence, retrieved_urls = run_inference(sample, model, processor, retriever, image)
            if high_confidence:
                high_conf_count += 1
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            high_confidence = False
            retrieved_urls = []

        results.append(build_result_record(
            sample, prediction, retrieved_urls,
            extra_fields={"high_confidence": high_confidence},
        ))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")
    print(f"High confidence (top-1 only) samples: {high_conf_count}/{len(samples)}")


if __name__ == "__main__":
    main()