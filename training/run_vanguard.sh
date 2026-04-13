#!/bin/bash
echo "[>] Initiating Pre-Flight Check for Vanguard 5K Run..."
python validate_hero.py
if [ $? -ne 0 ]; then
    echo "[!] Pre-flight failed! Aborting."
    exit 1
fi

echo "[>] Starting Vanguard Configuration (Locked)..."
echo "{}" > vanguard_metrics.json

python train.py \
    --dataset_path ../sim/statera_poc.hdf5 \
    --decoder_type deconv \
    --target_type twostage \
    --temporal_dropout 0.25 \
    --finetune_blocks 2 \
    --accumulate_steps 3 \
    --epochs 30 \
    --save_checkpoint_every_epoch \
    --checkpoint_dir /home/animesh_varma/PycharmProjects/STATERA/training/vanguard_checkpoints \
    --metrics_file vanguard_metrics.json \
    --wandb_name Vanguard-5K-Hero

echo "[✓] Vanguard Run Terminated. Generating Analysis Tables..."
python generate_vanguard_tables.py