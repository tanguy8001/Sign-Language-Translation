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
    --work_dir "OpenASL/train_test" \
    --bs 48 --ls 0.2 --epochs 400 \
    --save_every 5 \
    --clip_length 512 --vocab_size 25000 \
    --feat_path "/home/grt/GloFE/mmpose" \
    --label_path "/home/grt/youtube-asl_data/data/tsv_files/new_youtube-asl_v1_1.tsv" \
    --eos_token "</s>" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --pose_backbone "PartedPoseBackbone" \
    --pe_enc --mask_enc --lr 3e-4 --dropout_dec 0.1 --dropout_enc 0.1 \
    --inter_cl --inter_cl_margin 0.4 --inter_cl_alpha 1.0 \
    --inter_cl_vocab 5523 \
    --inter_cl_we_path "notebooks/openasl-v1.0/uncased_filtred_glove_VN_embed.pkl"