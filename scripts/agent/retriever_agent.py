"""
Shared retriever for the ReAct agent: single EVA-CLIP-8B (vision+text)
instance, used by retrieve_knowledge via a per-call `text_weight` override.

REDESIGNED per indicazione dei tutor: niente più troncamento a 4 sezioni /
3000 caratteri per documento. Si recuperano TUTTE le sezioni dei top-k
documenti, e sarà il critico (ReAG-Critic) a selezionare quali sezioni sono
utili -- non noi a tagliare a monte. retrieve() restituisce quindi una
lista FLAT di (label, testo_sezione), una entry PER SEZIONE su tutti i
documenti, non una entry per documento intero -- esattamente come
nell'esempio ufficiale di ReAG-Critic, che valuta singole sezioni
("# Description:", "# Distribution:", ...) non articoli interi.

Un tetto per-SEZIONE (non per-documento) resta come rete di sicurezza
contro una singola sezione patologicamente lunga -- non è un troncamento
di routine, ogni sezione viene comunque considerata.
"""

import json
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

EXCLUDE_SECTIONS = {"references", "external links", "see also", "notes"}
MAX_SECTION_CHARS = 2000  # tetto di sicurezza su UNA sezione, non sul numero di sezioni

INDEX_PATH = "/work/cvcs2026/encyclopedic/knn.index"
KNN_PATH   = "/work/cvcs2026/encyclopedic/knn.json"
KB_PATH    = "/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.json"
CACHE_DIR  = "/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf"


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
        """text_query: a SINGLE combined string, embedded in one call."""
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

    def _get_all_sections(self, url: str, source_label: str) -> list:
        """
        Restituisce TUTTE le sezioni del documento a `url` (escluse
        References/External links/See also/Notes), come lista di
        (label, testo) -- una entry PER SEZIONE, non una per l'intero
        documento. label include sia la fonte che il titolo della sezione,
        es. "Source 1 - Description".
        """
        if url not in self.kb:
            return []

        entry = self.kb[url]
        section_texts = entry.get("section_texts", [])
        section_titles = entry.get("section_titles", [""] * len(section_texts))

        sections = []
        for title, text in zip(section_titles, section_texts):
            if title.lower() in EXCLUDE_SECTIONS:
                continue
            if not text.strip():
                continue
            label_title = title.strip() if title.strip() else "Overview"
            section_text = text.strip()
            if len(section_text) > MAX_SECTION_CHARS:
                section_text = section_text[:MAX_SECTION_CHARS] + " [...truncated]"
            sections.append((f"{source_label} - {label_title}", section_text))

        return sections

    def retrieve(
        self,
        image: Image.Image,
        query_text: str = "",
        text_weight: float = None,
    ) -> tuple:
        """
        Restituisce (retrieved_urls: list, labeled_sections: list[(label, testo)]).
        labeled_sections è FLAT su tutti i top_k documenti -- ogni sezione
        di ogni documento è una entry a sé, pronta per essere filtrata
        individualmente da ReAG-Critic.
        """
        if text_weight is None:
            text_weight = self.text_weight

        query_embedding = self._embed_multimodal(image, query_text, text_weight)
        scores, indices = self.index.search(query_embedding, k=self.top_k)

        retrieved_urls = []
        labeled_sections = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            str_idx = str(idx)
            try:
                entry = self.knn[str_idx] if isinstance(self.knn, dict) else self.knn[int(idx)]
                url = entry[0] if isinstance(entry, list) else entry
                retrieved_urls.append(url)
                source_label = f"Source {i + 1}"
                labeled_sections.extend(self._get_all_sections(url, source_label))
            except Exception:
                continue

        return retrieved_urls, labeled_sections