"""
Basic inference script for Qwen2.5-VL-3B-Instruct.
Optimized for deployment on GPUs with limited VRAM (e.g., 11GB).
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("Loading model in bfloat16 to optimize VRAM usage...")
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# 1. LOAD MODEL
# Forcing torch_dtype to bfloat16 instead of "auto" to explicitly half the memory footprint.
# This ensures the 3B model weights comfortably fit within an 11GB GPU.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. LOAD PROCESSOR AND LIMIT IMAGE RESOLUTION
# Qwen-VL processes images into tokens. High-resolution images can result in thousands of tokens,
# causing quadratic OOM (Out Of Memory) issues in the attention layer.
# We constrain the pixel range to balance performance and memory cost.
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28  # Capped explicitly for 11GB GPU environments

processor = AutoProcessor.from_pretrained(
    model_id, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

# 3. PREPARE THE INPUT MESSAGES
# Construct the conversational payload with an image URL and a text prompt.
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

print("Processing image and text inputs...")
# Apply the chat template to format the prompt as expected by the instruct model
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Parse vision inputs (handles downloading/loading the image behind the scenes)
image_inputs, video_inputs = process_vision_info(messages)

# Tokenize and push inputs to the GPU
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# 4. INFERENCE
print("Generating response...")
# Generate tokens with a strict limit to prevent unbounded memory scaling during generation
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Trim the prompt tokens from the output so we only decode the newly generated text
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode the generated token IDs back to human-readable text
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 5. DISPLAY RESULTS
print("\n" + "=" * 50)
print("MODEL RESPONSE:")
print("=" * 50)
print(output_text[0])
print("=" * 50 + "\n")