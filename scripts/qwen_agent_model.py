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

NEW: tool-name validation with internal retry + coercion. At 1000-sample
scale, prompt-level mitigations (instructions, examples, temperature) were
NOT sufficient: ~34% of episodes hallucinated a non-existent tool (e.g.
image_search, web_search -- plausibly memorized from generic smolagents
tutorial examples) and never recovered within max_steps. We now validate
the tool name in our own generate() before returning control to the agent:
- if invalid, regenerate internally (up to max_retries times, at increasing
  temperature) -- these retries do NOT count against max_steps.
- if still invalid after retries, coerce the model's own JSON text by
  substituting only the tool name (and argument key) for a valid one,
  preserving whatever surrounding format the model produced (safer than
  inventing new formatting, since it reuses a pattern known to parse
  correctly in successful calls from the same pathway).
"""

import re
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

VALID_TOOL_NAMES = {"retrieve_knowledge", "refine_search", "final_answer"}
TOOL_ARG_KEY = {
    "retrieve_knowledge": "reasoning",
    "refine_search": "hypothesis",
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
    match = NAME_PATTERN.search(text)
    return match.group(1) if match else None


def _coerce_to_valid_tool(text: str, target_name: str) -> str:
    """
    Rewrites an invalid tool-name JSON blob into a valid one, preserving the
    model's own surrounding format/prefix -- safer than inventing new syntax
    from scratch, since it reuses a pattern already known to parse correctly.
    """
    fixed = NAME_PATTERN.sub(f'"name": "{target_name}"', text, count=1)
    arg_match = ARG_KEY_PATTERN.search(fixed)
    if arg_match:
        old_key = arg_match.group(1)
        new_key = TOOL_ARG_KEY[target_name]
        if old_key != new_key:
            fixed = fixed.replace(f'"{old_key}"', f'"{new_key}"', 1)
    return fixed


class QwenAgentModel(Model):
    """smolagents-compatible Model backed by a local Qwen2.5-VL-3B-Instruct."""

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
        self.episode_state = None  # set externally, see set_episode_state()

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

    def set_episode_state(self, episode_state):
        """
        Called once, before the sample loop. The SAME episode_state object
        is reused across all samples (reset per-sample elsewhere), so this
        only needs to be set once for the whole run.
        """
        self.episode_state = episode_state

    def _generate_once(self, inputs, temperature: float) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=None,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

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

        # --- First attempt ---
        output_text = self._generate_once(inputs, self.temperature)
        tool_name = _extract_tool_name(output_text)

        # --- Internal retries if the tool name is invalid (does NOT count
        #     against agent max_steps, since we never hand this back yet) ---
        attempt = 0
        while (
            tool_name is not None
            and tool_name not in VALID_TOOL_NAMES
            and attempt < self.max_retries
        ):
            attempt += 1
            retry_temperature = min(self.temperature + 0.2 * attempt, 0.9)
            output_text = self._generate_once(inputs, retry_temperature)
            tool_name = _extract_tool_name(output_text)

        # --- Last resort: coerce the model's own text into a valid call ---
        if tool_name is not None and tool_name not in VALID_TOOL_NAMES:
            has_retrieved = self.episode_state.has_retrieved if self.episode_state else False
            target_name = "retrieve_knowledge" if not has_retrieved else "final_answer"
            output_text = _coerce_to_valid_tool(output_text, target_name)

        if stop_sequences:
            for stop in stop_sequences:
                idx = output_text.find(stop)
                if idx != -1:
                    output_text = output_text[:idx]
                    break

        return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)