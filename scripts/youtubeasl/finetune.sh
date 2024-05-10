#!/usr/bin/env bash
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output finetune-%J.log
#SBATCH --nodelist compute01

# 512_pemb_bs48_ep400_encpenc_maskenc_lr3e4_ddp4_dp01_4pt_ccl10m4_e1
python train_youtubeasl_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/home/grt/GloFE" \
    --work_dir "pre-trained" \
    --tokenizer "notebooks/ytasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --bs 4 \
    --prefix test-vn \
    --phase test --weights "/home/grt/GloFE/OpenASL/train/youtube_asl-000.pt"

