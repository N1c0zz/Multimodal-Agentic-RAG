#!/bin/bash
#SBATCH --job-name=qwen3vl_agent_test
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --time=08:30:00
#SBATCH --mem=64G
#SBATCH --output=/homes/%u/cvcs2026/logs/out/qwen3vl_agent_test_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/qwen3vl_agent_test_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/feature_extractors/dati_progetto/.cache_torch

source /work/cvcs2026/feature_extractors/venv_qwen3/bin/activate

echo "Avvio Agentic smoke test (Qwen3-VL) su nodo: $SLURMD_NODENAME"
python /homes/$USER/cvcs2026/scripts/agent/run_inference_agent.py \
    --backbone qwen3vl \
    --output_dir /work/cvcs2026/feature_extractors/dati_progetto/predictions/agent_qwen3_topk8_test \
    --n_samples 5 \
    --top_k 8 \
    --max_steps 5 \
    --max_new_tokens 512 \
    --verbosity_level 2
echo "Smoke test terminato."