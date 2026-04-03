#!/bin/bash
echo "🚀 Pre-Flight Check..."
python validate_pipeline.py
if [ $? -ne 0 ]; then
    echo "Pre-flight failed! Aborting ablations."
    exit 1
fi

echo "🚀 Initiating 13-Run STATERA Ablation Pipeline..."
echo "{}" > run_metrics.json

# Original 10 Runs
python train.py --decoder_type mlp --target_type dot --wandb_name Run-1-Baseline-MLP
python train.py --decoder_type deconv --target_type dot --wandb_name Run-2-Deconv-Spatial
python train.py --decoder_type deconv --target_type blend --wandb_name Run-3-Curriculum-Blend
python train.py --decoder_type deconv --target_type crescent --wandb_name Run-4-Curriculum-Crescent
python train.py --decoder_type deconv --target_type crescent --temporal_weighting uniform --wandb_name Run-5-Uniform-Temporal
python train.py --decoder_type deconv --target_type crescent --temporal_mixer transformer --wandb_name Run-6-Transformer-Mixer
python train.py --decoder_type deconv --target_type crescent --single_task --wandb_name Run-7-Single-Task
python train.py --decoder_type deconv --target_type crescent --static_sigma --wandb_name Run-8-Static-Sigma
python train.py --decoder_type deconv --target_type crescent --backbone dinov2 --wandb_name Run-9-DINOv2
python train.py --decoder_type deconv --target_type crescent --scratch --wandb_name Run-10-Scratch

python train.py --decoder_type regression --target_type dot --wandb_name Run-11-Direct-Regression
python train.py --decoder_type deconv --target_type crescent --temporal_mixer none --wandb_name Run-12-No-Temporal-Mixer
python train.py --decoder_type deconv --target_type crescent --loss_type focal --wandb_name Run-13-Focal-Loss

echo "🏁 All Ablation Runs Terminated. Generating Analysis Tables..."
python generate_table.py