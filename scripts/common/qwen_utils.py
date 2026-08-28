"""
Vision-Language Model (VLM) Loading and Inference Utilities.

Centralizes the initialization and generation logic for the Qwen2.5-VL architecture 
across all non-agentic RAG pipelines.

Architectural configurations enforced by default:
- torch_dtype: bfloat16 for efficient memory usage without precision degradation.
- Decoding strategy: Deterministic (greedy) decoding (do_sample=False) to ensure 
  reproducibility and factual consistency in knowledge-intensive QA.
- Image resolution bounds: min_pixels and max_pixels are set to maintain high 
  visual detail, preventing aggressive downscaling by the processor.
- Context window: Explicitly set to 16384 tokens to prevent silent prompt truncation.
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
    """
    Loads the Qwen2.5-VL model and its corresponding processor.
    
    Explicitly overrides the model's default generation configuration to disable 
    sampling, ensuring deterministic outputs for factual reasoning tasks.
    
    Args:
        model_id (str): The HuggingFace model identifier.
        min_pixels (int): Minimum resolution bound for the image processor.
        max_pixels (int): Maximum resolution bound for the image processor.
        model_max_length (int): Maximum token length for the text processor.
        
    Returns:
        tuple: (model, processor) initialized and loaded into GPU memory.
    """
    print(f"Loading Qwen model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )

    # Disable sampling explicitly to prevent non-deterministic or hallucinated 
    # answers induced by default temperature/top_p settings.
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
    Executes a single-shot greedy generation pass.
    
    Processes the raw model output by stripping the input prompt tokens and 
    decoding the newly generated tokens into a clean string.
    
    Args:
        model: The loaded Qwen2.5-VL model.
        processor: The associated model processor.
        inputs: The tokenized and processed multimodal inputs.
        max_new_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        str: The decoded and trimmed output string.
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
        
    # Trim the input tokens from the generated sequence
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()