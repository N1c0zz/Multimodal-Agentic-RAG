"""
Wrapper for the ReAG-Critic passage relevance filter.

This module integrates a Qwen2.5-VL-based relevance classifier specifically 
fine-tuned for Knowledge-Based Visual Question Answering (KB-VQA). It acts as 
an impartial relevance judge for the agent's context, evaluating retrieved 
passages section by section to maximize precision before the context is fed 
to the reasoning backbone.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from paths import CACHE_DIR

CRITIC_MODEL_NAME = "aimagelab/ReAG-Critic"

RELEVANCY_EVAL_SYSTEM_PROMPT = """You are a multimodal reasoning assistant specialized in Knowledge-Based Visual Question Answering (KB-VQA).
Your task is to evaluate whether a given text passage provides useful and relevant information for answering a question about an image.

You will be given:
- Image: a visual scene containing entities, actions, and context.
- Question: a natural-language question that refers to the image.
- Text Passage: an external knowledge snippet retrieved from a database or the web.

You must analyze the semantic alignment between the text, the image, and the question.
Follow these steps carefully before giving your final decision:
1. Understand the visual scene: Identify the key objects, people, actions, and context visible in the image.
2. Interpret the question: Determine what information the question seeks.
3. Analyze the text passage: Extract the main claims, facts, and entities mentioned in the text.

Compare for relevance: Assess whether the information in the text:
- Contains at least one sentence that supports answering the question about the image, OR
- Provides background knowledge needed to interpret or reason about the image-question pair.

Important:
- If even a single sentence in the passage is relevant or useful, consider the entire passage as relevant and answer "Yes".
- If no part of the passage contributes meaningfully to answering the question, answer "No".

Output only one word:
"Yes" -> if the text provides relevant or useful information for answering the question.
"No" -> if the text is irrelevant or unhelpful."""

SECTION_EVAL_USER_TEMPLATE = """Here is the question on the image above:
{question}

Here is the text passage to analyze:
{passage}

Does the text passage contain at least one sentence that may have some information useful to answer the user question?
"Yes"/"No" answer:"""


class ReAGCritic:
    def __init__(self, yes_prob_threshold: float = 0.1):
        self.yes_prob_threshold = yes_prob_threshold
        print("Loading ReAG-Critic (aimagelab/ReAG-Critic)...")
        self.processor = AutoProcessor.from_pretrained(
            CRITIC_MODEL_NAME, padding_side="left", use_fast=True, 
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
            cache_dir=CACHE_DIR,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CRITIC_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )  # device_map="auto" is intentionally omitted to bypass Accelerate's dispatch
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Architectural fix for a known issue in transformers >= 4.50 with Qwen2.5-VL:
        # Automatic weight-tying fails to properly tie the lm_head to the embeddings.
        input_embeddings = self.model.get_input_embeddings()
        self.model.lm_head.weight = input_embeddings.weight

        # Permanent validation: ensures the weight-tying is effectively applied in memory.
        # This prevents silent failures where the critic's probabilities would be invalid.
        if not torch.equal(self.model.lm_head.weight.data, input_embeddings.weight.data):
            print("WARNING: lm_head re-tie is NOT effective! Critic scores will be invalid.")
        else:
            print("lm_head successfully tied to input embeddings.")

        self.model.eval()
        print("ReAG-Critic loaded.")

    @torch.inference_mode()
    def score_passage(self, image: Image.Image, question: str, passage: str) -> dict:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": RELEVANCY_EVAL_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SECTION_EVAL_USER_TEMPLATE.format(question=question, passage=passage)},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)

        yes_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
        no_id = self.processor.tokenizer.convert_tokens_to_ids("No")
        yes_prob = probs[0, yes_id].item()
        return {"relevant": yes_prob > self.yes_prob_threshold, "yes_probability": yes_prob}

    def filter_passages(self, image: Image.Image, question: str, labeled_passages: list) -> list:
        """
        Evaluates a list of candidate passages and returns the relevant subset.

        Args:
            image (Image.Image): The query image.
            question (str): The user's visual question.
            labeled_passages (list): A list of tuples containing (source_label, text).

        Returns:
            list: The filtered subset of passages scoring above the probability threshold.
        """
        kept = []
        for label, text in labeled_passages:
            if not text.strip():
                continue
            result = self.score_passage(image, question, text)
            if result["relevant"]:
                kept.append((label, text))
        return kept