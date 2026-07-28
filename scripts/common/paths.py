"""
Centralized filesystem paths, shared across every retriever variant and the
Qwen loading helper. Previously INDEX_PATH, KNN_PATH, KB_PATH, and CACHE_DIR
were each redefined identically in every retriever_*.py file.
"""

INDEX_PATH = "/work/cvcs2026/encyclopedic/knn.index"
KNN_PATH   = "/work/cvcs2026/encyclopedic/knn.json"
KB_PATH    = "/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.json"
CACHE_DIR  = "/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf"