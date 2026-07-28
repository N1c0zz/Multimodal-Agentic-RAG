"""
Shared helpers for loading the dataset and building
evqa_compute_metrics.py-compatible output records. Previously the
oracle_urls / evidence_in_context computation and the dataset-loading
list(data.values())-if-dict pattern were duplicated across every RAG script.

Two DIFFERENT evidence_in_context methods exist in this project, used in
different situations -- they are NOT interchangeable:

1. compute_evidence_in_context(sample, retrieved_urls): URL-membership check.
   Used by every script that performs real FAISS retrieval (rag, richcontext,
   rerank, rerank_dynamic, guesses, combined) -- checks whether the oracle
   Wikipedia URL is among the URLs actually retrieved.

2. compute_evidence_in_context_by_answer_match(sample, context): substring
   check. Used ONLY by run_inference_rag_oracle.py, which does not perform
   FAISS retrieval at all (it uses the dataset's own pre-computed 'retrieval'
   field directly) -- there are no retrieved URLs to compare against an
   oracle URL, so instead this checks whether the reference answer text
   literally appears in the given context.
"""

import json


def load_dataset(dataset_path: str) -> list:
    """Loads the Encyclopedic-VQA subset, handling both dict and list JSON layouts."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return list(data.values()) if isinstance(data, dict) else data


def compute_evidence_in_context(sample: dict, retrieved_urls: list) -> bool:
    """
    Whether the oracle Wikipedia URL(s) for this sample were among the
    documents actually retrieved. sample['wikipedia_url'] may contain
    multiple alternatives separated by '|' (same convention used throughout
    the project and in evqa_compute_metrics.py itself).

    Use this for any script that performs real FAISS retrieval.
    """
    oracle_urls = [u.strip() for u in sample.get('wikipedia_url', '').split('|') if u.strip()]
    return any(url in retrieved_urls for url in oracle_urls)


def compute_evidence_in_context_by_answer_match(sample: dict, context: str) -> bool:
    """
    Whether the reference answer text appears literally inside the given
    context string. Used ONLY when there is no retrieved-URL list to compare
    against an oracle URL (i.e. run_inference_rag_oracle.py, which uses
    dataset-provided context rather than performing retrieval itself).

    This is a weaker/different proxy than compute_evidence_in_context and is
    NOT directly comparable to it -- do not use both methods across rows of
    the same results table without noting the difference.
    """
    reference = sample.get('answer', "")
    if not reference or not context:
        return False
    ref_lower = str(reference).lower()
    ctx_lower = context.lower()
    if '|' in ref_lower:
        return any(ans.strip() in ctx_lower for ans in ref_lower.split('|'))
    return ref_lower in ctx_lower


def build_result_record(
    sample: dict,
    prediction: str,
    retrieved_urls: list = None,
    extra_fields: dict = None,
) -> dict:
    """
    Builds a single evqa_compute_metrics.py-compatible result record.
    - reference is kept as the RAW string (with | and && separators);
      evqa_compute_metrics.py splits it internally, do not convert to a list.
    - evidence_in_context is only computed here if retrieved_urls is passed
      (leave None for the plain baseline, or for scripts that compute it
      differently and pass it via extra_fields instead, e.g. the oracle
      script).
    - extra_fields lets per-script extras (e.g. high_confidence in the
      dynamic top-k script, or a differently-computed evidence_in_context)
      merge in without needing a separate builder. extra_fields is applied
      AFTER the URL-based evidence_in_context, so it can override it.
    """
    record = {
        "data_id": sample["unique_id"],
        "question": sample["question"],
        "reference": sample.get("answer", ""),
        "answers": prediction,
        "question_type": sample.get("question_type", "automatic"),
    }
    if retrieved_urls is not None:
        record["evidence_in_context"] = compute_evidence_in_context(sample, retrieved_urls)
    if extra_fields:
        record.update(extra_fields)
    return record