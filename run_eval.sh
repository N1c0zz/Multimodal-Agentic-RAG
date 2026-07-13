#!/bin/bash
#SBATCH --job-name=evqa_eval
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/evqa_eval_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/evqa_eval_%j.err
#SBATCH --account=cvcs2026

source /homes/$USER/cvcs2026/venv_eval/bin/activate

PRED_DIR=/work/cvcs2026/feature_extractors/dati_progetto/predictions/baseline_qwen

echo "Avvio Encyclopedic-VQA eval su nodo: $SLURMD_NODENAME"
echo "Predictions dir: ${PRED_DIR}"

python /homes/$USER/cvcs2026/scripts/evqa_eval.py \
    --input_path ${PRED_DIR}

echo "Eval terminata."