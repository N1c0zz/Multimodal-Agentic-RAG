"""
Runs Qwen2.5-VL-3B-Instruct on the encyclopedic test subset and writes
predictions directly in the format expected by the Encyclopedic-VQA
eval script (evqa_compute_metrics.py):
  output_dir/split_0.json -> [{"data_id": ..., "question": ...,
                                "reference": ..., "answers": ...,
                                "question_type": ...}, ...]

No retrieval: this is the plain-VLM baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import argparse
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from qwen_utils import load_qwen, generate_greedy
from eval_utils import load_dataset, build_result_record

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def run_inference(sample: dict, model, processor) -> str:
    image_path = str(IMAGE_ROOT / sample["related_images"])
    prompt_instruction = " Answer strictly with a single word or a short phrase. Do not use full sentences."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": sample["question"] + prompt_instruction},
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

    return generate_greedy(model, processor, inputs, max_new_tokens=64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where split_0.json will be written")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="Inference"):
        try:
            prediction = run_inference(sample, model, processor)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""

        # No retrieval in the baseline, so no evidence_in_context field.
        results.append(build_result_record(sample, prediction))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()