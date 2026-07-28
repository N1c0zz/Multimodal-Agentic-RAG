#!/bin/bash
#SBATCH --job-name=evqa_eval
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/eval_agent_test_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/eval_agent_test_%j.err
#SBATCH --account=cvcs2026

source /work/cvcs2026/feature_extractors/venv_eval/bin/activate

PRED_DIR=/work/cvcs2026/feature_extractors/dati_progetto/predictions/agent_test

echo "Avvio Encyclopedic-VQA eval su nodo: $SLURMD_NODENAME"
echo "Predictions dir: ${PRED_DIR}"

python /homes/$USER/cvcs2026/evqa_eval/evqa_compute_metrics.py \
    --input_path ${PRED_DIR}

echo "Eval terminata."