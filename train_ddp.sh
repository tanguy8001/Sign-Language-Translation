#!/usr/bin/env bash
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-%J.log
#SBATCH --nodelist compute1080ti05

sync;sync;sync

echo `hostname`s

source ~/.bashrc

conda deactivate
conda activate glofe

export PYTHONPATH=.

sh scripts/openasl/train_ddp.sh