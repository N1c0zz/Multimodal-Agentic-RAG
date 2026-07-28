"""
Centralized Qwen2.5-VL-3B-Instruct loading and generation for all
non-agentic RAG scripts (baseline/, rag/). Previously this exact load_qwen()
logic was duplicated across 6+ scripts; centralizing it means a future fix
(like the do_sample greedy-decoding bug found earlier in this project) only
needs to happen in one place.

Standard settings, validated across the whole project:
- torch_dtype=bfloat16
- greedy decoding: do_sample=False, enforced both in generation_config and
  again at every generate() call (belt-and-suspenders)
- max_pixels=1280*28*28, min_pixels=256*28*28 (higher than an earlier
  512*28*28 default, restoring visual detail)
- model_max_length=16384 (explicit, avoids silent prompt truncation)
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from paths import CACHE_DIR


def load_qwen(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
    model_max_length: int = 16384,
):
    print(f"Loading Qwen model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )

    # Disable sampling explicitly: Qwen's default generation_config often has
    # do_sample=True with a temperature/top_p set, which introduces
    # non-deterministic, occasionally worse answers for this factual QA task.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    processor = AutoProcessor.from_pretrained(
        model_id,
        model_max_length=model_max_length,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        cache_dir=CACHE_DIR,
    )
    return model, processor


def generate_greedy(model, processor, inputs, max_new_tokens: int = 64) -> str:
    """
    Runs a single greedy generate() call and returns the decoded, trimmed
    output text. Every non-agentic script had this exact
    generate -> trim -> batch_decode pattern inline; centralized here.
    Works identically for single-stage scripts (rag, richcontext) and
    multi-stage ones (combined, guesses) -- just call it once per stage.
    """
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()