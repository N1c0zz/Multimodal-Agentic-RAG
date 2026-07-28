"""
Retriever with richer context: more sections per document, with section
titles included and clearer document boundaries between sources.
No re-ranking, no query fusion -- isolates the effect of richer context
alone, on top of the plain image-only FAISS retrieval.
"""

import torch
import faiss
import json
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

from paths import INDEX_PATH, KNN_PATH, KB_PATH, CACHE_DIR

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
# Intentionally higher than the base Retriever's MAX_SECTIONS=4: this is the
# whole point of the "rich context" experiment.
MAX_SECTIONS = 8


class RetrieverRichContext:
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

    def _load_kb(self):
        print("Loading knowledge base...")
        with open(KB_PATH, "r") as f:
            self.kb = json.load(f)
        print(f"  KB entries: {len(self.kb)}")

    def _load_embedding_model(self):
        print("Loading EVA-CLIP-8B (vision only)...")
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

        # Vision-only: drop the text encoder to save VRAM (this retriever
        # never calls encode_text()).
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
        """Rich context: more sections than the base retriever, each labeled with its title."""
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

        # Label each source explicitly so the model can attribute
        # information to the correct document.
        labeled_context = "\n\n".join(
            f"[Source {i+1}]\n{part}" for i, part in enumerate(context_parts)
        )
        return labeled_context, retrieved_urls