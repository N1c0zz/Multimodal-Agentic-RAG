"""
RAG inference script for Qwen2.5-VL-3B-Instruct.
For each sample: embeds the query image with EVA-CLIP-8B, retrieves
the top-k most similar Wikipedia documents from the FAISS index,
and passes the retrieved context to Qwen to answer the question.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
from retriever import Retriever

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def load_qwen(model_id: str):
    print(f"Loading Qwen model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=16384,
        min_pixels=3136,
        max_pixels=301056,
        cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
    )
    return model, processor


def run_inference(
    sample: dict,
    model,
    processor,
    retriever: Retriever,
    image: Image.Image,
) -> str:
    image_path = str(IMAGE_ROOT / sample["related_images"])

    # Retrieve context using the query image
    context, _ = retriever.retrieve(image)

    if context:
        prompt_text = (
            f"Here is some relevant context:\n{context}\n\n"
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
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output[0].strip()


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

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data

    # Load retriever first
    retriever = Retriever(top_k=args.top_k)

    # Load Qwen
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples[:1], desc="RAG Inference"):
        try:
            image_path = str(IMAGE_ROOT / sample["related_images"])
            image = Image.open(image_path).convert("RGB")
            prediction = run_inference(sample, model, processor, retriever, image)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""

        results.append({
            "data_id": sample["unique_id"],
            "prediction": prediction,
        })

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()