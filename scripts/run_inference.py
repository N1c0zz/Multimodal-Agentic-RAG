"""
Runs Qwen2.5-VL-3B-Instruct on the encyclopedic test subset and writes
predictions in the format expected by evaluation_infoseek.py:
  output_dir/split_0.json  -> [{"data_id": ..., "prediction": ...}, ...]
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


def run_inference(sample: dict, model, processor) -> str:
    image_rel_path = sample['related_images']
    image_path = str(IMAGE_ROOT / image_rel_path)

    prompt_instruction = " Answer strictly with a single word or a short phrase. Do not use full sentences."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": sample['question'] + prompt_instruction},
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
    parser.add_argument('--dataset_path', default='/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json')
    parser.add_argument('--output_dir', required=True,
                        help="Directory where split_0.json will be written")
    parser.add_argument('--model_id', default='Qwen/Qwen2.5-VL-3B-Instruct')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data

    model, processor = load_model(args.model_id)

    results = []
    for sample in tqdm(samples, desc="Inference"):
        try:
            prediction = run_inference(sample, model, processor)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""

        results.append({
            "data_id": sample['unique_id'],
            "prediction": prediction,
        })

    out_file = output_dir / "split_0.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()