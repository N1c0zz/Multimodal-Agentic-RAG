#!/bin/bash
#SBATCH --job-name=qwen_rag_rerank_dyn
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=07:00:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/rag_rerank_dyn_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/rag_rerank_dyn_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio RAG Rerank Dynamic inference su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/run_inference_rag_rerank_dynamic.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/rag_rerank_dynamic \
    --top_k 3 \
    --top_k_retrieval 10 \
    --alpha 0.5 \
    --confidence_threshold 0.08
echo "RAG Rerank Dynamic inference terminata."