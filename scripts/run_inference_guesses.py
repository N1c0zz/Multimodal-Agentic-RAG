"""
Two-Stage Multimodal RAG Inference Script (Double-Pass)

This script orchestrates a pipeline using Qwen2.5-VL-3B-Instruct and a custom Hybrid Retriever.
- Stage 1 (Hypothesis Generation): Prompts the Vision-Language Model to generate the top-3 taxonomic guesses for the input image.
- Stage 2 (Hybrid Retrieval): Embeds the original image, the generated guesses, and the user query to retrieve top-k context passages from a FAISS index.
- Stage 3 (Final QA): Feeds the original image, user question, and retrieved text context back to the VLM to generate a strict, short-form answer.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from retriever_guesses import Retriever

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

def load_qwen(model_id: str):
    print(f"Loading Qwen model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
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

def run_inference(sample: dict, model, processor, retriever: Retriever, image: Image.Image) -> str:
    image_path = str(IMAGE_ROOT / sample["related_images"])
    question = sample["question"]

    phase1_prompt = (
        "Analyze the main subject of this image. Provide your top 3 most probable guesses "
        "for its specific proper name, biological species, or exact identity. "
        "Output ONLY a comma-separated list of these 3 names (e.g., Fuchsia magellanica, Hibiscus rosa-sinensis, Mandevilla sanderi). "
        "Do not write full sentences, background descriptions, or explanations."
    )

    messages_p1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": phase1_prompt},
            ],
        }
    ]

    text_p1 = processor.apply_chat_template(messages_p1, tokenize=False, add_generation_prompt=True)
    img_in_1, vid_in_1 = process_vision_info(messages_p1)
    
    inputs_p1 = processor(
        text=[text_p1], images=img_in_1, videos=vid_in_1, padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        gen_ids_1 = model.generate(**inputs_p1, max_new_tokens=40)

    trimmed_1 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_p1.input_ids, gen_ids_1)]
    caption_keywords = processor.batch_decode(trimmed_1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    print(f"\n[DEBUG] File: {sample['related_images']} | Keyword Qwen: '{caption_keywords}'")

    combined_query = f"Image tags: {caption_keywords}. Question: {question}"
    context, _ = retriever.retrieve(image, query_text=combined_query)

    if context:
        prompt_text = (
            f"Here is some relevant context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )
    else:
        prompt_text = (
            f"{question}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )

    messages_p2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_p2 = processor.apply_chat_template(messages_p2, tokenize=False, add_generation_prompt=True)
    img_in_2, vid_in_2 = process_vision_info(messages_p2)
    
    inputs_p2 = processor(
        text=[text_p2], images=img_in_2, videos=vid_in_2, padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        gen_ids_2 = model.generate(**inputs_p2, max_new_tokens=64)

    trimmed_2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_p2.input_ids, gen_ids_2)]
    output = processor.batch_decode(trimmed_2, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data

    retriever = Retriever(top_k=args.top_k)
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="Double-Pass Inference"):
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
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} predictions to {out_file}")

if __name__ == "__main__":
    main()