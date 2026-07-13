"""
Retriever module for RAG pipeline.
Uses EVA-CLIP-8B (vision only) to embed the query image and searches
the pre-built FAISS index.
"""

import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4

INDEX_PATH = "/work/cvcs2026/encyclopedic/knn.index"
KNN_PATH   = "/work/cvcs2026/encyclopedic/knn.json"
KB_PATH    = "/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.json"


class Retriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
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
        print("Loading EVA-CLIP-8B (vision only)...")
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
        )
        self.model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir="/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf",
        ).to(self.device).eval()

        if hasattr(self.model, "text_model"):
            del self.model.text_model
        if hasattr(self.model, "text_projection"):
            del self.model.text_projection
        torch.cuda.empty_cache()
        print("EVA-CLIP-8B loaded.")

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