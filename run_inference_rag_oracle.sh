#!/bin/bash
#SBATCH --job-name=qwen_rag_oracle
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/rag_oracle_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/rag_oracle_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio RAG Oracle inference su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/run_inference_rag_oracle.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/rag_oracle
echo "RAG Oracle inference terminata."