#!/bin/bash
echo "Rerunning Failed Ablations..."

python train.py --decoder_type deconv --target_type crescent --single_task --wandb_name Run-7-Single-Task
python train.py --decoder_type deconv --target_type crescent --scratch --wandb_name Run-10-Scratch

echo "Failed Runs Terminated. Generating New Analysis Tables..."
python generate_table.py