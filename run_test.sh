#!/bin/bash

#----USAGE EXAMPLE----
# sbatch run.sh /homes/$USER/cvcs2026/test_qwen_VL.py

#SBATCH --job-name=test_qwen
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/homes/%u/cvcs2026/logs/out/test_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/err/test_%j.err
#SBATCH --account=cvcs2026

export HF_HOME=/work/cvcs2026/dati_progetto/.cache_hf
export TORCH_HOME=/work/cvcs2026/dati_progetto/.cache_torch

module load python/3.11.11-gcc-11.4.0
source /homes/$USER/cvcs2026/venv/bin/activate

echo "Avvio job su nodo: $SLURMD_NODENAME"
python ${1}
echo "Job terminato."