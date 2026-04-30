#!/bin/bash
echo "[>] Initiating STATERA 50K Crescent SOTA (Unified Continuous Funnel)..."
echo "{}" > sota_50k_crescent_metrics.json

python ../statera/train.py \
    --dataset_path ../sim/HiddenMass-50K.hdf5 \
    --decoder_type deconv \
    --target_type crescent \
    --temporal_dropout 0.25 \
    --finetune_blocks 2 \
    --accumulate_steps 3 \
    --epochs 10 \
    --save_checkpoint_every_epoch \
    --checkpoint_dir ./sota_50k_crescent_checkpoints \
    --metrics_file sota_50k_crescent_metrics.json \
    --wandb_name STATERA-50K-Crescent-SOTA

echo "[✓] STATERA 50K Crescent SOTA Run SECURED"