#!/bin/bash
#SBATCH --job-name=diag_topk_full_curve
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/diag_topk_curve_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/diag_topk_curve_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio diagnostico curva completa top_k su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/agent/diagnose_retrieval_topk.py --top_k 2 4 6 8 10 12 15
echo "Diagnostico terminato."