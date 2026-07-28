#!/bin/bash
#SBATCH --job-name=rag_double
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/rag_double_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/rag_double_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio RAG Double-Pass su nodo: $SLURMD_NODENAME"

/homes/$USER/cvcs2026/venv/bin/python /homes/$USER/cvcs2026/scripts/rag/run_inference_guesses.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/rag_double_pass_dvulcano/guesses_qwen \
    --top_k 3

echo "RAG Double-Pass terminato."