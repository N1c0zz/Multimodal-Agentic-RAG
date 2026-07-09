#!/bin/bash
#SBATCH --job-name=infoseek_eval
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --mem=20G
#SBATCH --time=03:10:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/eval_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/eval_%j.err
#SBATCH --account=cvcs2026

source /homes/$USER/cvcs2026/venv_eval/bin/activate

export PYTHONPATH=/homes/$USER/cvcs2026

echo "Avvio eval su nodo: $SLURMD_NODENAME"

python3 /homes/$USER/cvcs2026/infoseek_eval/evaluation_infoseek.py \
    --adjust_score \
    --input_path /work/cvcs2026/feature_extractors/dati_progetto/predictions/rag_oracle \
    --reference_path /work/cvcs2026/feature_extractors/dati_progetto/reference.jsonl \
    --reference_qtype_path /work/cvcs2026/feature_extractors/dati_progetto/reference_qtype.jsonl

echo "Eval terminata."