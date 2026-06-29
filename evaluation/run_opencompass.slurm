#!/bin/bash
#SBATCH --job-name=opencompass_meditron3
#SBATCH --time=11:59:59
#SBATCH --gres=gpu:4
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 288
#SBATCH --account=a127
#SBATCH --output=/users/$USER/output/slurm_batch_%j.out
#SBATCH --error=/users/$USER/output/slurm_batch_%j.err
#SBATCH -A a127

# --- env setup ---
export HOME=/users/$USER
cd $HOME/opencompass/opencompass
# --- init conda ---
source $HOME/miniconda3/bin/activate
conda activate opencompass
# --- set data path (OpenCompass read from data/ defaultly)---
export OPENCOMPASS_DATA_DIR=$HOME/opencompass/opencompass/data

# one by one
# opencompass configs/eval_meditron2_ppl.py --debug
opencompass configs/eval_meditron2_ppl_dpo.py --debug

echo "Batch completed!"
