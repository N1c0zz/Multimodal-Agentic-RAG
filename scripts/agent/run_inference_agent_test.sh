#!/bin/bash
#SBATCH --job-name=qwen_agent_test
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=04:30:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/agent_test_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/agent_test_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio Agentic smoke test su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/agent/run_inference_agent.py \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/agent_test \
    --n_samples 100 \
    --top_k 3 \
    --max_steps 7 \
    --max_new_tokens 512 \
    --verbosity_level 2
echo "Smoke test terminato."