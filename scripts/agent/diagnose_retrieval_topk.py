"""
Diagnostic script for pure retrieval evaluation.

Measures the hit rate and the average retrieved section count across different
top_k values, bypassing the Qwen generation and the agent loop. It is designed 
for rapid evaluation of the full-document retrieval interface.

By enforcing text_weight=0.0 (pure image query, no multimodal fusion), this 
script isolates the specific effect of the top_k parameter. While this 
underestimates the absolute hit rate of the complete pipeline, it serves as 
the correct proxy to analyze the system's relative sensitivity to top_k.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import argparse
import torch
from tqdm import tqdm
from PIL import Image

from eval_utils import load_dataset
from retriever_agent import RetrieverAgent

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--top_k", type=int, nargs="+", default=[4, 6, 8, 10, 12, 15])
    args = parser.parse_args()

    samples = load_dataset(args.dataset_path)

    print(f"{'top_k':>6} | {'hit_rate':>10} | {'avg_sections':>12}")
    print("-" * 36)

    for k in args.top_k:
        retriever = RetrieverAgent(top_k=k, text_weight=0.0)

        hits = 0
        total_sections = 0
        for sample in tqdm(samples, desc=f"top_k={k}"):
            image_path = str(IMAGE_ROOT / sample["related_images"])
            image = Image.open(image_path).convert("RGB")

            retrieved_urls, labeled_sections = retriever.retrieve(image, query_text="", text_weight=0.0)
            oracle_urls = [u.strip() for u in sample.get('wikipedia_url', '').split('|') if u.strip()]
            hit = any(url in retrieved_urls for url in oracle_urls)

            hits += int(hit)
            total_sections += len(labeled_sections)

        hit_rate = hits / len(samples)
        avg_sections = total_sections / len(samples)
        print(f"{k:>6} | {hit_rate:>9.1%} | {avg_sections:>12.1f}")

        del retriever
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()