"""
Custom smolagents Model wrapping Qwen2.5-VL-3B-Instruct, with tool-name
validation inside generate() (regenerate up to max_retries at rising
temperature, then coerce), which eliminated the ~34% hallucinated-tool
episodes seen at 1000-sample scale. Uses light sampling (temperature=0.3)
so byte-identical retries after an error don't reproduce the same output.

generate_plain() is a single-shot, no-tools, greedy generation method used
by run_inference_agent.py's fallback functions and by agent_tools.py's
dedicated calls (guesses, retrieval-need assessment, re-guess) -- kept as a
wrapper method so those callers don't need to know which backbone
(Qwen2.5-VL) is actually loaded.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from smolagents import Model, ChatMessage, MessageRole

from paths import CACHE_DIR

ROLE_MAP = {
    "system": "system", "user": "user", "assistant": "assistant",
    "tool-call": "assistant", "tool-response": "user",
}

VALID_TOOL_NAMES = {"assess_retrieval_need", "retrieve_knowledge", "final_answer"}
TOOL_ARG_KEY = {
    "assess_retrieval_need": "reasoning",
    "retrieve_knowledge": "reasoning",
    "final_answer": "answer",
}
NAME_PATTERN = re.compile(r'"name"\s*:\s*"([a-zA-Z_]+)"')
ARG_KEY_PATTERN = re.compile(r'"arguments"\s*:\s*\{\s*"([a-zA-Z_]+)"')


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


def _extract_tool_name(text: str):
    m = NAME_PATTERN.search(text)
    return m.group(1) if m else None


def _coerce_to_valid_tool(text: str, target_name: str) -> str:
    fixed = NAME_PATTERN.sub(f'"name": "{target_name}"', text, count=1)
    arg_match = ARG_KEY_PATTERN.search(fixed)
    if arg_match:
        old_key = arg_match.group(1)
        new_key = TOOL_ARG_KEY[target_name]
        if old_key != new_key:
            fixed = fixed.replace(f'"{old_key}"', f'"{new_key}"', 1)
    return fixed


class QwenAgentModel(Model):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(model_id=model_id, flatten_messages_as_text=False, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.current_image = None
        self.episode_state = None

        print(f"Loading Qwen model for agent: {model_id}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR,
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.processor = AutoProcessor.from_pretrained(
            model_id, model_max_length=16384,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, cache_dir=CACHE_DIR,
        )
        print("Qwen agent model loaded.")

    def set_image(self, image: Image.Image):
        self.current_image = image

    def set_episode_state(self, episode_state):
        self.episode_state = episode_state

    def generate_plain(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 64) -> str:
        """
        Single-shot, no-tools, greedy generation -- backbone-agnostic entry
        point used by run_inference_agent.py's fallbacks and by
        agent_tools.py's dedicated (guess/assessment/re-guess) calls.
        """
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _generate_once(self, inputs, temperature: float) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.9, top_k=None,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        qwen_messages = []
        image_attached = False
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            role = getattr(role, "value", role)
            role = ROLE_MAP.get(str(role), "user")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            text = _flatten_content(content)
            if role == "user" and not image_attached and self.current_image is not None:
                qwen_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.current_image},
                        {"type": "text", "text": text},
                    ],
                })
                image_attached = True
            else:
                qwen_messages.append({"role": role, "content": text})

        chat_text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        output_text = self._generate_once(inputs, self.temperature)
        tool_name = _extract_tool_name(output_text)

        attempt = 0
        while (tool_name is not None and tool_name not in VALID_TOOL_NAMES
               and attempt < self.max_retries):
            attempt += 1
            retry_temp = min(self.temperature + 0.2 * attempt, 0.9)
            output_text = self._generate_once(inputs, retry_temp)
            tool_name = _extract_tool_name(output_text)

        if tool_name is not None and tool_name not in VALID_TOOL_NAMES:
            if self.episode_state is None or not self.episode_state.has_assessed:
                target = "assess_retrieval_need"
            elif not self.episode_state.has_retrieved:
                target = "retrieve_knowledge"
            else:
                target = "final_answer"
            output_text = _coerce_to_valid_tool(output_text, target)

        if stop_sequences:
            for stop in stop_sequences:
                idx = output_text.find(stop)
                if idx != -1:
                    output_text = output_text[:idx]
                    break

        return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)