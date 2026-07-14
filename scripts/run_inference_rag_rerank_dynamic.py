"""
RAG inference with caption-based re-ranking and dynamic top-k selection.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from retriever_rerank_dynamic import RetrieverRerankDynamic

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def load_qwen(model_id: str):
    print(f"Loading Qwen model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
    )

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=16384,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
    )
    return model, processor


def run_inference(
    sample: dict,
    model,
    processor,
    retriever: RetrieverRerankDynamic,
    image: Image.Image,
) -> tuple[str, bool]:
    image_path = str(IMAGE_ROOT / sample["related_images"])

    context, _, caption, high_confidence = retriever.retrieve_rerank(
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

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output[0].strip(), high_confidence


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

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data

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
            prediction, high_confidence = run_inference(sample, model, processor, retriever, image)
            if high_confidence:
                high_conf_count += 1
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            high_confidence = False

        reference = sample.get('answer', "")

        results.append({
            "data_id": sample["unique_id"],
            "question": sample["question"],
            "reference": reference,
            "answers": prediction,
            "question_type": sample.get('question_type', 'automatic'),
            "high_confidence": high_confidence,
        })

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")
    print(f"High confidence (top-1 only) samples: {high_conf_count}/{len(samples)}")


if __name__ == "__main__":
    main()