"""
Custom smolagents Model wrapping our already-validated Qwen2.5-VL-3B-Instruct
inference setup (high max_pixels, explicit model_max_length).

Design choice: rather than routing the query image through smolagents'
generic multimodal message-content pipeline, we keep the image as an
attribute of this Model instance, set once per episode via set_image(),
and always attach it to the *first* user turn we reconstruct for Qwen.

Uses light sampling (temperature) rather than pure greedy decoding: with
byte-identical retries after a tool-call error, greedy decoding reproduces
the exact same (wrong) output every time. A small temperature lets the
agent escape these loops.
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from smolagents import Model, ChatMessage, MessageRole

CACHE_DIR = "/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf"

ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool-call": "assistant",
    "tool-response": "user",
}


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


class QwenAgentModel(Model):
    """smolagents-compatible Model backed by a local Qwen2.5-VL-3B-Instruct."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ):
        super().__init__(model_id=model_id, flatten_messages_as_text=False, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.current_image = None

        print(f"Loading Qwen model for agent: {model_id}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            model_max_length=16384,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            cache_dir=CACHE_DIR,
        )
        print("Qwen agent model loaded.")

    def set_image(self, image: Image.Image):
        """Must be called once per episode/sample, before agent.run()."""
        self.current_image = image

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        qwen_messages = []
        image_attached = False

        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            role = getattr(role, "value", role)
            role = ROLE_MAP.get(str(role), "user")

            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            text = _flatten_content(content)

            if role == "user" and not image_attached and self.current_image is not None:
                qwen_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": self.current_image},
                            {"type": "text", "text": text},
                        ],
                    }
                )
                image_attached = True
            else:
                qwen_messages.append({"role": role, "content": text})

        chat_text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                top_k=None,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if stop_sequences:
            for stop in stop_sequences:
                idx = output_text.find(stop)
                if idx != -1:
                    output_text = output_text[:idx]
                    break

        return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)