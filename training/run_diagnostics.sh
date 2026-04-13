#!/bin/bash
echo "Initiating Diagnostic & Accumulation Verification Runs..."
echo "{}" > diagnostic_metrics.json

# 1. The Jitter Proof (No Temporal Mixer)
python train.py --decoder_type deconv --target_type twostage --temporal_mixer none --epochs 30 --metrics_file diagnostic_metrics.json --wandb_name Run-12-No-Temporal-Jitter-Proof

# 2. The 50K Pre-Flight (Testing Gradient Accumulation)
python train.py --decoder_type deconv --target_type twostage --temporal_dropout 0.25 --finetune_blocks 2 --epochs 30 --accumulate_steps 3 --metrics_file diagnostic_metrics.json --wandb_name Hero-5-Accumulation-Test

echo "Diagnostics Terminated. Generating Report..."
python generate_diagnostic_tables.py