#!/bin/bash
echo "[>] Initiating STATERA-50K-Sigma..."
echo "{}" > sota_50k_sigma_metrics.json

python train.py \
    --dataset_path ../sim/HiddenMass-50K.hdf5 \
    --decoder_type deconv \
    --target_type dot \
    --temporal_dropout 0.25 \
    --finetune_blocks 2 \
    --accumulate_steps 3 \
    --epochs 10 \
    --save_checkpoint_every_epoch \
    --checkpoint_dir /home/animesh_varma/PycharmProjects/STATERA/training/sota_50k_sigma_checkpoints \
    --metrics_file sota_50k_sigma_metrics.json \
    --wandb_name STATERA-50K-Sigma

echo "[✓] FINAL 50K RUN SECURED"sudo /home/animesh_varma/PycharmProjects/Eidolon/.venv/bin/python main.py