#!/bin/bash
#SBATCH --job-name=qwen_rag
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/rag_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/rag_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio RAG inference su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/run_inference_rag.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/rag_faiss \
    --top_k 3
echo "RAG inference terminata."