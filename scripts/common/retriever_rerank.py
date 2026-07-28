"""
Retriever with caption-based re-ranking.
Pipeline:
1. EVA-CLIP image embed -> FAISS search -> top-10 URLs
2. Qwen generates a caption describing the query image
3. EVA-CLIP text encoder embeds the caption
4. For each retrieved doc: embed its first section with EVA-CLIP text encoder
5. Combine visual score + textual score -> re-rank -> top-3
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from retriever import Retriever

from paths import CACHE_DIR

CAPTION_PROMPT = (
    "What is the specific species, name, or identity of the main subject "
    "in this image? If it is a plant or animal, provide the scientific name "
    "if possible. Be as specific as possible about what entity this is."
)


class RetrieverRerank(Retriever):
    def __init__(self, top_k: int = 3, top_k_retrieval: int = 10, alpha: float = 0.5):
        self.top_k_retrieval = top_k_retrieval
        self.alpha = alpha
        # keep_text_encoder=True: this subclass needs encode_text(), so the
        # base class keeps EVA-CLIP's text components loaded instead of
        # dropping them -- _load_text_encoder() below then reuses self.model
        # rather than loading a second full copy.
        super().__init__(top_k=top_k_retrieval, keep_text_encoder=True)
        self.top_k_final = top_k
        self._load_text_encoder()

    def _load_text_encoder(self):
        """
        Loads only the tokenizer here. The text encoder itself is
        self.model (already loaded by the base class with
        keep_text_encoder=True) -- no second EVA-CLIP-8B copy is loaded.
        Previously this method loaded an entire second ~16GB model just to
        call encode_text() on it.
        """
        print("Loading EVA-CLIP tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        print("Tokenizer loaded (reusing the shared EVA-CLIP-8B model for text encoding).")

    def _embed_text(self, text: str) -> np.ndarray:
        """Compute L2-normalized text embedding with EVA-CLIP."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens.input_ids)
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        return embedding.cpu().numpy().astype("float32")

    def _generate_caption(self, image: Image.Image, qwen_model, qwen_processor) -> str:
        """Use Qwen to generate a descriptive caption of the query image (greedy)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ]

        text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return caption

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a.flatten(), b.flatten()))

    def retrieve_rerank(
        self,
        image: Image.Image,
        question: str,
        qwen_model,
        qwen_processor,
    ) -> tuple[str, list[str], str]:
        """Returns (joined_context, top_urls, generated_caption)."""
        query_image_embedding = self._embed_image(image)
        visual_scores, indices = self.index.search(
            query_image_embedding, k=self.top_k_retrieval
        )

        candidates = []
        for score, idx in zip(visual_scores[0], indices[0]):
            url = self.knn[idx][0]
            candidates.append({"url": url, "visual_score": float(score)})

        caption = self._generate_caption(image, qwen_model, qwen_processor)
        caption_embedding = self._embed_text(caption)

        for candidate in candidates:
            url = candidate["url"]
            if url not in self.kb:
                candidate["textual_score"] = 0.0
                continue
            section_texts = self.kb[url].get("section_texts", [])
            first_text = ""
            for text in section_texts:
                if text.strip():
                    first_text = text.strip()[:512]
                    break
            if first_text:
                doc_embedding = self._embed_text(first_text)
                candidate["textual_score"] = self._cosine_similarity(caption_embedding, doc_embedding)
            else:
                candidate["textual_score"] = 0.0

        for candidate in candidates:
            candidate["final_score"] = (
                self.alpha * candidate["visual_score"]
                + (1 - self.alpha) * candidate["textual_score"]
            )

        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        top_urls = [c["url"] for c in candidates[:self.top_k_final]]
        context_parts = []
        for url in top_urls:
            context = self._build_context(url)
            if context:
                context_parts.append(context)

        return "\n\n---\n\n".join(context_parts), top_urls, caption