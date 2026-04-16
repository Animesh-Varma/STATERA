#!/bin/bash
echo "[>] Pre-Flight Check for STATERA 50K SOTA..."
python validate_hero.py
if [ $? -ne 0 ]; then
    echo "[!] Pre-flight failed! Aborting."
    exit 1
fi

echo "[>] Initiating STATERA 50K SOTA (Unified Continuous Funnel)..."
echo "{}" > sota_50k_metrics.json

# Launching 50K SOTA Run (50,000 Videos, 10 Epochs, Unified Crescent)
python train.py \
    --dataset_path ../sim/HiddenMass-50K.hdf5 \
    --decoder_type deconv \
    --target_type crescent \
    --temporal_dropout 0.25 \
    --finetune_blocks 2 \
    --accumulate_steps 3 \
    --epochs 10 \
    --save_checkpoint_every_epoch \
    --checkpoint_dir /home/animesh_varma/PycharmProjects/STATERA/training/sota_50k_checkpoints \
    --metrics_file sota_50k_metrics.json \
    --wandb_name STATERA-50K-SOTA

echo "[✓] STATERA 50K SOTA Run Terminated. Generating Analysis Tables..."
python generate_50k_sota_tables.py