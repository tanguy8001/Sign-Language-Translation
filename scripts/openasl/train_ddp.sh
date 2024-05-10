#!/usr/bin/env bash
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --partition=student
#SBATCH --job-name test
#SBATCH --output test-%J.log
#SBATCH --nodelist compute1080ti05

GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
torchrun --nnodes=1 --nproc_per_node=$GPUS /home/grt/GloFE/train_youtubeasl_pose_DDP_inter_VN.py \
    --ngpus $GPUS \
    --work_dir_prefix "/home/grt/GloFE/" \
    --work_dir "OpenASL/train" \
    --bs 6 --ls 0.11 --epochs 25 \
    --save_every 5 \
    --clip_length 512 --vocab_size 25000 \
    --feat_path "/home/grt/GloFE/OpenASL/mmpose" \
    --label_path "/home/grt/yotube-asl_data/data/tsv_files/new_youtube-asl_v1_1.tsv" \
    --eos_token "</s>" \
    --tokenizer "notebooks/ytasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --pose_backbone "PartedPoseBackbone" \
    --pe_enc --mask_enc --lr 1e-4 --dropout_dec 0.3 --dropout_enc 0.3 \
    --inter_cl --inter_cl_margin 0.4 --inter_cl_alpha 1.0 \
    --inter_cl_vocab 304 \
    --inter_cl_we_path "notebooks/ytasl-v1.0/uncased_filtred_glove_VN_embed.pkl"