"""
Base retriever module for the RAG pipeline.
Uses EVA-CLIP-8B to embed the query image and searches the pre-built FAISS
index. This is the parent class for the re-ranking variants
(retriever_rerank.py, retriever_rerank_dynamic.py).
"""

import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

from paths import INDEX_PATH, KNN_PATH, KB_PATH, CACHE_DIR

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4


class Retriever:
    def __init__(self, top_k: int = 3, keep_text_encoder: bool = False):
        """
        keep_text_encoder: if False (default, used for plain image-only
        retrieval), the text encoder is dropped after loading to save VRAM.
        Subclasses that also need encode_text() (RetrieverRerank,
        RetrieverRerankDynamic) pass True here so they can reuse THIS single
        loaded model for text embedding, instead of loading a second, full
        (~16GB) copy of EVA-CLIP-8B themselves.
        """
        self.top_k = top_k
        self.keep_text_encoder = keep_text_encoder
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
        mode = "vision+text" if self.keep_text_encoder else "vision only"
        print(f"Loading EVA-CLIP-8B ({mode})...")
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=CACHE_DIR,
        )
        self.model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).to(self.device).eval()

        if not self.keep_text_encoder:
            # Vision-only use case: drop the text encoder to save VRAM, since
            # this configuration never calls encode_text().
            if hasattr(self.model, "text_model"):
                del self.model.text_model
            if hasattr(self.model, "text_projection"):
                del self.model.text_projection
            torch.cuda.empty_cache()
        print(f"EVA-CLIP-8B loaded ({mode}).")

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed.pixel_values.to(dtype=torch.float16, device=self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(pixel_values)
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        return embedding.cpu().numpy().astype("float32")

    def _build_context(self, url: str) -> str:
        """Extract and concatenate relevant section_texts for a given URL."""
        if url not in self.kb:
            return ""

        entry = self.kb[url]
        section_texts = entry.get("section_texts", [])
        section_titles = entry.get("section_titles", [""] * len(section_texts))

        context_parts = []
        for title, text in zip(section_titles, section_texts):
            if title.lower() in EXCLUDE_SECTIONS:
                continue
            if text.strip():
                context_parts.append(text.strip())
            if len(context_parts) >= MAX_SECTIONS:
                break

        return "\n\n".join(context_parts)

    def retrieve(self, image: Image.Image) -> tuple[str, list[str]]:
        query_embedding = self._embed_image(image)
        scores, indices = self.index.search(query_embedding, k=self.top_k)

        retrieved_urls = []
        context_parts = []
        for idx in indices[0]:
            url = self.knn[idx][0]
            retrieved_urls.append(url)
            context = self._build_context(url)
            if context:
                context_parts.append(context)

        final_context = "\n\n---\n\n".join(context_parts)
        return final_context, retrieved_urls