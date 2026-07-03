#!/bin/bash
#SBATCH --job-name=qwen_inference
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/inference_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/inference_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio inference su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/run_inference.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/baseline_qwen
echo "Inference terminata."