"""
Retriever combining Multi-Hypothesis Multimodal Fusion + Rich Context.

Same fusion mechanism as retriever_guesses.py (see that file's docstring),
but _build_context here adds section titles and [Source N] labels --
the "rich context" treatment from retriever_richcontext.py, applied on top
of the fused retrieval instead of the plain image-only one.
"""

import torch
import faiss
import json
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

from paths import INDEX_PATH, KNN_PATH, KB_PATH, CACHE_DIR

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4


class RetrieverCombined:
    def __init__(self, top_k: int = 3, text_weight: float = 0.3):
        self.top_k = top_k
        self.text_weight = text_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_index()
        self._load_kb()
        self._load_embedding_model()

    def _load_index(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(KNN_PATH, "r") as f:
            self.knn = json.load(f)
        print(f"  Vectors in index: {self.index.ntotal}")
        print(f"  KNN entries:      {len(self.knn)}")

    def _load_kb(self):
        print("Loading knowledge base...")
        with open(KB_PATH, "r") as f:
            self.kb = json.load(f)
        print(f"  KB entries: {len(self.kb)}")

    def _load_embedding_model(self):
        print("Loading EVA-CLIP-8B (Vision + Text)...")
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=CACHE_DIR,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        self.model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).eval()
        print("EVA-CLIP-8B loaded successfully.")

    def _embed_multimodal(self, image: Image.Image, text_query: str) -> np.ndarray:
        # 1. Visual features
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed.pixel_values.to(dtype=torch.float16, device=self.device)

        # 2. Text features (hard truncation to prevent CLIP's 77-token limit crashes)
        text_inputs = self.tokenizer(
            [text_query],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(self.device)

        # 3. Encode and fuse
        with torch.no_grad():
            img_emb = self.model.encode_image(pixel_values)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

            txt_emb = self.model.encode_text(input_ids)
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

            combined_emb = ((1.0 - self.text_weight) * img_emb) + (self.text_weight * txt_emb)
            combined_emb = combined_emb / combined_emb.norm(p=2, dim=-1, keepdim=True)

        return combined_emb.cpu().numpy().astype("float32")

    def _build_context(self, url: str) -> str:
        """Rich context: section titles kept as inline labels."""
        if url not in self.kb:
            return ""

        entry = self.kb[url]
        section_texts = entry.get("section_texts", [])
        section_titles = entry.get("section_titles", [""] * len(section_texts))

        parts = []
        for title, text in zip(section_titles, section_texts):
            if title.lower() in EXCLUDE_SECTIONS:
                continue
            if text.strip():
                label = title.strip() if title.strip() else "Overview"
                parts.append(f"[{label}] {text.strip()}")
            if len(parts) >= MAX_SECTIONS:
                break

        return "\n".join(parts)

    def retrieve(self, image: Image.Image, query_text: str) -> tuple[str, list[str]]:
        query_embedding = self._embed_multimodal(image, query_text)
        scores, indices = self.index.search(query_embedding, k=self.top_k)

        retrieved_urls = []
        context_parts = []

        for idx in indices[0]:
            if idx == -1:
                continue

            str_idx = str(idx)
            try:
                entry = self.knn[str_idx] if isinstance(self.knn, dict) else self.knn[int(idx)]
                url = entry[0] if isinstance(entry, list) else entry

                retrieved_urls.append(url)
                context = self._build_context(url)
                if context:
                    context_parts.append(context)
            except Exception:
                continue

        # Label each source explicitly ([Source N]) so the model can
        # attribute information to the correct document.
        labeled_context = "\n\n".join(
            f"[Source {i+1}]\n{part}" for i, part in enumerate(context_parts)
        )
        return labeled_context, retrieved_urls