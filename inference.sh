#!/usr/bin/env bash
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-inference-%J.log
sync;sync;sync

echo `hostname`s

source ~/.bashrc

conda deactivate
conda activate glofe

export PYTHONPATH=.

sh scripts/youtubeasl/finetune.sh
