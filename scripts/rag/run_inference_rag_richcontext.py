"""
Rich Context RAG Inference Pipeline.

Executes a standard single-pass visual retrieval setup, but structures the resulting 
context with section titles and distinct document boundaries (e.g., source labels). 
It evaluates the impact of augmented context formatting on language model generation 
in isolation, omitting query fusion or re-ranking mechanisms.
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
from retriever_richcontext import RetrieverRichContext

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def run_inference(
    sample: dict,
    model,
    processor,
    retriever: RetrieverRichContext,
    image: Image.Image,
) -> tuple[str, list[str]]:
    """Executes the rich-context inference pipeline for a single sample."""
    image_path = str(IMAGE_ROOT / sample["related_images"])

    context, retrieved_urls = retriever.retrieve(image)

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
    return prediction, retrieved_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    retriever = RetrieverRichContext(top_k=args.top_k)
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="RAG RichContext Inference"):
        try:
            image_path = str(IMAGE_ROOT / sample["related_images"])
            image = Image.open(image_path).convert("RGB")
            prediction, retrieved_urls = run_inference(sample, model, processor, retriever, image)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            retrieved_urls = []

        results.append(build_result_record(sample, prediction, retrieved_urls))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()