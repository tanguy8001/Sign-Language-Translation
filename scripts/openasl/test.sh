# 512_pemb_bs48_ep400_encpenc_maskenc_lr3e4_ddp4_dp01_4pt_ccl10m4_e1
python train_openasl_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/home/grt/GloFE" \
    --work_dir "pre-trained" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --bs 32 \
    --prefix test-vn \
    --phase test --weights "/home/grt/GloFE/pre-trained/glofe_n_openasl.pt"

