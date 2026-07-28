"""
Two-Stage Multimodal RAG Inference Script (Double-Pass).

Stage 1 (Hypothesis Generation): Qwen generates its top-3 taxonomic guesses
                                  for the entity shown in the image.
Stage 2 (Hybrid Retrieval): Embeds the image and the generated guesses +
                             question, fuses them, and retrieves top-k
                             context passages from the FAISS index.
Stage 3 (Final QA): Feeds the image, question, and retrieved context back to
                     Qwen to generate a short-form answer.

FIXED during repo cleanup: evidence_in_context now uses the same oracle-URL
matching method as every other RAG script (eval_utils.compute_evidence_in_context),
instead of a substring match on the answer text. This script DOES perform
real FAISS retrieval (retriever.retrieve() already returns retrieved_urls),
so there was no reason for it to use the weaker, differently-scaled
substring method -- that made its reported hit rate not directly comparable
to the other rows of the results table. If you need the historical number
for comparison, re-run this script and check the new hit rate against the
previously reported 25.9%.
Also removed a leftover per-sample [DEBUG] print statement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import json
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

from qwen_utils import load_qwen, generate_greedy
from eval_utils import load_dataset, build_result_record
from retriever_guesses import RetrieverGuesses as Retriever

IMAGE_ROOT = Path("/work/cvcs2026/encyclopedic")

GUESS_PROMPT = (
    "Analyze the main subject of this image. Provide your top 3 most probable guesses "
    "for its specific proper name, biological species, or exact identity. "
    "Output ONLY a comma-separated list of these 3 names (e.g., Fuchsia magellanica, Hibiscus rosa-sinensis, Mandevilla sanderi). "
    "Do not write full sentences, background descriptions, or explanations."
)


def run_inference(sample: dict, model, processor, retriever: Retriever, image: Image.Image) -> tuple[str, list[str]]:
    image_path = str(IMAGE_ROOT / sample["related_images"])
    question = sample["question"]

    # Stage 1 -- Hypothesis generation
    messages_p1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": GUESS_PROMPT},
            ],
        }
    ]
    text_p1 = processor.apply_chat_template(messages_p1, tokenize=False, add_generation_prompt=True)
    img_in_1, vid_in_1 = process_vision_info(messages_p1)
    inputs_p1 = processor(
        text=[text_p1], images=img_in_1, videos=vid_in_1, padding=True, return_tensors="pt"
    ).to("cuda")
    guesses = generate_greedy(model, processor, inputs_p1, max_new_tokens=40)

    # Stage 2 -- Hybrid retrieval (fused image + guesses + question embedding)
    combined_query = f"Image tags: {guesses}. Question: {question}"
    context, retrieved_urls = retriever.retrieve(image, query_text=combined_query)

    # Stage 3 -- Final QA
    if context:
        prompt_text = (
            f"Here is some relevant context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )
    else:
        prompt_text = (
            f"{question}\n\n"
            "Answer with the shortest possible response: "
            "a single word, name, or brief phrase. "
            "Do not explain or use full sentences."
        )

    messages_p2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text_p2 = processor.apply_chat_template(messages_p2, tokenize=False, add_generation_prompt=True)
    img_in_2, vid_in_2 = process_vision_info(messages_p2)
    inputs_p2 = processor(
        text=[text_p2], images=img_in_2, videos=vid_in_2, padding=True, return_tensors="pt"
    ).to("cuda")
    prediction = generate_greedy(model, processor, inputs_p2, max_new_tokens=64)

    return prediction, retrieved_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset_path)
    retriever = Retriever(top_k=args.top_k)
    model, processor = load_qwen(args.model_id)

    results = []
    for sample in tqdm(samples, desc="Double-Pass Inference"):
        try:
            image_path = str(IMAGE_ROOT / sample["related_images"])
            image = Image.open(image_path).convert("RGB")
            prediction, retrieved_urls = run_inference(sample, model, processor, retriever, image)
        except Exception as e:
            print(f"Error on {sample['unique_id']}: {e}")
            prediction = ""
            retrieved_urls = []

        results.append(build_result_record(sample, prediction, retrieved_urls))

    out_file = output_dir / "split_0.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()