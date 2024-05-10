#!/usr/bin/env bash
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-%J.log
#SBATCH --nodelist compute1080ti05

python train_openasl_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/home/grt/GloFE" \
    --work_dir "pre-trained" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --bs 32 \
    --prefix test-vn \
    --phase test --weights "/home/grt/GloFE/OpenASL/train/youtube_asl-024.pt"