#!/bin/bash
echo "[>] STATERA FINAL MANDATE TRAINING INITIATING..."
echo "{}" > final_mandate_metrics.json

# ==========================================
# 1. THE NARRATIVE CORE (The Guillotine Contestants)
# ==========================================

# A. Run-00-Anchor-Saved (The Core Baseline at 1K)
echo "[*] Launching Anchor Baseline..."
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file final_mandate_metrics.json --wandb_name Run-00-Anchor-Saved

# B. Run-04-Static-Dot-Saved (Tests "Expectation Collapse" against a fixed, non-shrinking target)
echo "[*] Launching Static-Dot Simulator Memorizer..."
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --static_sigma --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file final_mandate_metrics.json --wandb_name Run-04-Static-Dot-Saved

# C. Run-05-Standard-Sigma-Saved (Tests traditional shrinking heatmaps against Crescent angular targeting)
echo "[*] Launching Standard Gaussian Sigma Annealer..."
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file final_mandate_metrics.json --wandb_name Run-05-Standard-Sigma-Saved


# ==========================================
# 2. THE HARDWARE-FIXED BASELINES (100% Fair Competition)
# ==========================================

# D. Run-01-DINOv2-Fixed (Testing pure spatial backbone - Now given the 1D Conv Mixer for absolute fairness)
echo "[*] Launching DINOv2 Baseline (Fairness Match: 1D Conv active, Hardware Limit bypassed)..."
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone dinov2 --epochs 30 --funnel_epochs 6 --batch_size_override 1 --accumulate_steps 24 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file final_mandate_metrics.json --wandb_name Run-01-DINOv2-Fixed

# E. Run-03-External-Temporal-ResNet3D (The standard 3D CNN baseline - Now given 2 Finetuning blocks for absolute fairness)
echo "[*] Launching ResNet3D Temporal Baseline (Fairness Match: Finetuning active, Channel bug fixed)..."
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone resnet3d --epochs 30 --funnel_epochs 6 --batch_size_override 2 --accumulate_steps 12 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file final_mandate_metrics.json --wandb_name Run-03-ResNet3D-Fixed

echo "[✓] FINAL MANDATE COMPLETE. ALL WEIGHTS SECURED FOR REAL-WORLD EVALUATION."