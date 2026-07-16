"""
RAG Oracle inference script for Qwen2.5-VL-3B-Instruct.
Uses the pre-retrieved context (section_texts) already present in the dataset
to augment the prompt with relevant Wikipedia passages before answering.
This is an oracle RAG baseline: retrieval is not performed at inference time,
but is taken directly from the dataset annotations.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

# Sections that are not informative for answering questions
EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
    )
    
    # Disable sampling to enforce deterministic output
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


def build_context(sample: dict) -> str:
    """Extract and concatenate relevant section_texts from the retrieval field."""
    retrieval_list = sample.get('retrieval', [])
    if not retrieval_list:
        return ""

    # The dataset has one retrieval entry per sample
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
    image_rel_path = sample['related_images']
    image_path = str(IMAGE_ROOT / image_rel_path)
    context = build_context(sample)

    # Check if the correct answer is present in the oracle context
    evidence_in_context = False
    reference = sample.get('answer', "")
    if reference and context:
        ref_lower = str(reference).lower()
        ctx_lower = context.lower()
        if '|' in ref_lower:
            evidence_in_context = any(ans.strip() in ctx_lower for ans in ref_lower.split('|'))
        else:
            evidence_in_context = ref_lower in ctx_lower

    if context:
        prompt_text = (
            f"Here is some relevant context:\n{context}\n\n"
            f"Question: {sample['question']}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )
    else:
        # Fallback to plain VLM if no context available
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
    
    return output[0].strip(), evidence_in_context


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

    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    samples = list(data.values()) if isinstance(data, dict) else data

    model, processor = load_model(args.model_id)

    results = []
    for sample in tqdm(samples, desc="RAG Oracle Inference"):
        try:
            prediction, has_evidence = run_inference(sample, model, processor)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            has_evidence = False
            
        reference = sample.get('answer', "")

        results.append({
            "data_id": sample["unique_id"],
            "question": sample["question"],
            "reference": reference,
            "answers": prediction,
            "question_type": sample.get('question_type', 'automatic'),
            "evidence_in_context": has_evidence,
        })

    out_file = output_dir / "split_0.json"
    with open(out_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()