#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name generate_word_embd_ytasl
#SBATCH --output generate_word_embd_ytasl-%J.log
sync;sync;sync

echo `hostname`s

source ~/.bashrc

conda deactivate
conda activate glofe

export PYTHONPATH=.

python analyse_word_embd_ytasl.py