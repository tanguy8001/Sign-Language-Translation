#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-mmpose-extract.log
sync;sync;sync

echo `hostname`s

source ~/.bashrc

conda deactivate
conda activate glofe

export PYTHONPATH=.

sh tools/extract.sh 0 1 0