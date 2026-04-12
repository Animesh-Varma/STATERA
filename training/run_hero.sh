#!/bin/bash
echo "🚀 Pre-Flight Check for Hero Phase..."
python validate_hero.py
if [ $? -ne 0 ]; then
    echo "Pre-flight failed! Aborting."
    exit 1
fi

echo "Initiating 4-Run Hero Phase Pipeline (50 Epochs)..."
echo "{}" > hero_metrics.json

# Test Run 1: Baseline-50Ep
python train.py --decoder_type deconv --target_type dot --epochs 50 --static_sigma --metrics_file hero_metrics.json --wandb_name Hero-1-Baseline-50Ep

# Test Run 2: TwoStage-Curriculum
python train.py --decoder_type deconv --target_type twostage --epochs 50 --metrics_file hero_metrics.json --wandb_name Hero-2-TwoStage-Curriculum

# Test Run 3: Hero-Frozen
python train.py --decoder_type deconv --target_type twostage --temporal_dropout 0.25 --epochs 50 --metrics_file hero_metrics.json --wandb_name Hero-3-Hero-Frozen

# Test Run 4: Hero-Unchained
python train.py --decoder_type deconv --target_type twostage --temporal_dropout 0.25 --finetune_blocks 2 --epochs 50 --metrics_file hero_metrics.json --wandb_name Hero-4-Hero-Unchained

echo "🏁 Hero Runs Terminated. Generating Hero Analysis Tables..."
python generate_hero_tables.py