"""
Shared retriever for the ReAct agent: a single EVA-CLIP-8B (vision+text)
instance, used by both retrieval tools (retrieve_knowledge and
refine_search) via a per-call `text_weight` override.

retrieve() returns the individual labeled passages alongside the joined
context string, so filter_context (see agent_tools.py) can score each
source separately with ReAG-Critic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import torch
import faiss
import json
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

from paths import INDEX_PATH, KNN_PATH, KB_PATH, CACHE_DIR

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTIONS = 4
MAX_CONTEXT_CHARS = 3000


class RetrieverAgent:
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

    def _load_kb(self):
        print("Loading knowledge base...")
        with open(KB_PATH, "r") as f:
            self.kb = json.load(f)
        print(f"  KB entries: {len(self.kb)}")

    def _load_embedding_model(self):
        print("Loading EVA-CLIP-8B (Vision + Text)...")
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/EVA-CLIP-8B", trust_remote_code=True, cache_dir=CACHE_DIR,
        )
        self.model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        ).eval()
        print("EVA-CLIP-8B loaded successfully.")

    def _embed_multimodal(self, image: Image.Image, text_query: str, text_weight: float) -> np.ndarray:
        """
        text_query: a SINGLE combined string (e.g. "Image tags: X. Question:
        Y"), embedded in one call -- matching the static hypothesis-guided
        fusion experiment. text_weight=0.0 forces pure image-only retrieval
        (used by retrieve_knowledge's default first pass).
        """
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed.pixel_values.to(dtype=torch.float16, device=self.device)

        with torch.no_grad():
            img_emb = self.model.encode_image(pixel_values)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)

            if text_weight > 0.0 and text_query.strip():
                text_inputs = self.tokenizer(
                    [text_query], padding=True, truncation=True,
                    max_length=77, return_tensors="pt",
                )
                input_ids = text_inputs["input_ids"].to(self.device)
                txt_emb = self.model.encode_text(input_ids)
                txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)

                combined_emb = ((1.0 - text_weight) * img_emb) + (text_weight * txt_emb)
                combined_emb = combined_emb / combined_emb.norm(p=2, dim=-1, keepdim=True)
            else:
                combined_emb = img_emb

        return combined_emb.cpu().numpy().astype("float32")

    def _build_context(self, url: str) -> str:
        """
        MAX_SECTIONS is intentionally lower here (4) than in
        retriever_richcontext.py (8): raising it for the agent measurably
        hurt score|hit at 1000-sample scale, since the agent already carries
        tool-call reasoning overhead on top of reading the context.
        MAX_CONTEXT_CHARS is an added safety cap per document, on top of
        MAX_SECTIONS, kept as extra margin against oversized single sections.
        """
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

        full_text = "\n".join(parts)
        if len(full_text) > MAX_CONTEXT_CHARS:
            full_text = full_text[:MAX_CONTEXT_CHARS] + " [...truncated]"
        return full_text

    def retrieve(
        self,
        image: Image.Image,
        query_text: str = "",
        text_weight: float = None,
    ) -> tuple:
        """Returns (labeled_context: str, retrieved_urls: list, labeled_passages: list[(label, text)])."""
        if text_weight is None:
            text_weight = self.text_weight

        query_embedding = self._embed_multimodal(image, query_text, text_weight)
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

        labeled_passages = [(f"Source {i+1}", part) for i, part in enumerate(context_parts)]
        labeled_context = "\n\n".join(f"[{label}]\n{part}" for label, part in labeled_passages)
        return labeled_context, retrieved_urls, labeled_passages