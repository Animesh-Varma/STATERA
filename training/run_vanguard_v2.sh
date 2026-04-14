#!/bin/bash
echo "[>] Pre-Flight Check for Vanguard V2..."
python validate_hero.py
if [ $? -ne 0 ]; then
    echo "[!] Pre-flight failed! Aborting."
    exit 1
fi

echo "[>] Initiating Vanguard V2 (Dynamic Two-Stage Curriculum)..."
echo "{}" > vanguard_v2_metrics.json

# Launching Vanguard V2 Run (5000 Videos, 30 Epochs, Dynamic Switch)
python train.py \
    --dataset_path ../sim/statera_poc.hdf5 \
    --decoder_type deconv \
    --target_type dynamic_twostage \
    --temporal_dropout 0.25 \
    --finetune_blocks 2 \
    --accumulate_steps 3 \
    --epochs 30 \
    --save_checkpoint_every_epoch \
    --checkpoint_dir /home/animesh_varma/PycharmProjects/STATERA/training/vanguard_v2_checkpoints \
    --metrics_file vanguard_v2_metrics.json \
    --wandb_name Vanguard-5K-Hero-V2

echo "[✓] Vanguard V2 Run Terminated. Generating Analysis Tables..."
python generate_vanguard_v2_tables.py