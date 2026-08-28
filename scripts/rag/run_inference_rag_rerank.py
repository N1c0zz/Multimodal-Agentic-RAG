"""
Caption-Based Re-ranking RAG Inference Pipeline.

Executes a two-stage visual-question answering workflow per sample:
1. Retrieval: Embeds the query image using EVA-CLIP and fetches the top-N candidates.
2. Re-ranking: A VLM generates a descriptive visual caption of the query image, 
   which is subsequently embedded alongside the candidate texts to re-rank the 
   documents based on a combined visual-textual similarity score.
3. QA Synthesis: The VLM generates a short-form answer using the top-k re-ranked 
   documents as context.

Unlike basic RAG pipelines, this script passes the initialized VLM and processor 
into the retriever class to perform the internal caption generation step.
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
from retriever_rerank import RetrieverRerank

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def run_inference(
    sample: dict,
    model,
    processor,
    retriever: RetrieverRerank,
    image: Image.Image,
) -> tuple[str, list[str]]:
    """Executes the re-ranking inference pipeline for a single sample."""
    image_path = str(IMAGE_ROOT / sample["related_images"])

    # The visual caption is generated internally by the retriever for the re-ranking 
    # process and is discarded here, as it is not injected into the final QA prompt.
    context, top_urls, _caption = retriever.retrieve_rerank(
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
    return prediction, top_urls


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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    retriever = RetrieverRerank(
        top_k=args.top_k,
        top_k_retrieval=args.top_k_retrieval,
        alpha=args.alpha,
    )
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="RAG Rerank Inference"):
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