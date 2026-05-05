#!/bin/bash
cd "$(dirname "$0")/.." || exit

echo "[>] STATERA 1K MERGED ABLATION (30-Epoch Fast Track)..."

mkdir -p .checkpoints
echo "{}" > .checkpoints/1k_ablation_metrics.json

# ==========================================
# CORE BASELINES & NARRATIVE RUNS
# ==========================================

# 1. Run-00-Anchor-Saved
echo "[*] Launching Anchor Baseline..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-00-Anchor-Saved

# 2. Run-01-DINOv2-Fixed
echo "[*] Launching DINOv2 Baseline (Fairness Match)..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone dinov2 --epochs 30 --funnel_epochs 6 --batch_size_override 1 --accumulate_steps 24 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-01-DINOv2-Fixed

# ==========================================
# MISSING ABLATIONS
# ==========================================

# 3. Run-02-No-Temporal-Mixer
echo "[*] Launching No-Temporal-Mixer Ablation..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer none --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-02-No-Temporal-Mixer --temperature 2.0 --finetune_blocks 2

# ==========================================
# FAIRNESS & TARGETING RUNS
# ==========================================

# 4. Run-03-External-Temporal-ResNet3D
echo "[*] Launching ResNet3D Temporal Baseline (Fairness Match)..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone resnet3d --epochs 30 --funnel_epochs 6 --batch_size_override 2 --accumulate_steps 12 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-03-ResNet3D-Fixed

# 5. Run-04-No-Curriculum-Static-Dot
echo "[*] Launching Static-Dot Simulator Memorizer..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --static_sigma --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-04-Static-Dot-Saved

# 6. Run-05-Standard-Sigma-Only
echo "[*] Launching Standard Gaussian Sigma Annealer..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-05-Standard-Sigma-Saved

# ==========================================
# ADDITIONAL ARCHITECTURAL ABLATIONS
# ==========================================

# 7. Run-06-Standard-Softmax-T1
echo "[*] Launching Standard-Softmax-T1 Ablation..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-06-Standard-Softmax-T1 --temperature 1.0 --finetune_blocks 2

# 8. Run-07-Single-Task-Spatial
echo "[*] Launching Single-Task-Spatial Ablation..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --single_task --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-07-Single-Task-Spatial --temperature 2.0 --finetune_blocks 2

# 9. Run-08-Fully-Frozen-Anchor
echo "[*] Launching Fully Frozen V-JEPA Ablation..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 0 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name Run-08-Fully-Frozen-Anchor

echo "[*] Launching VideoMAE v2 Baseline (Temporal Foundation)..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone videomae --epochs 30 --funnel_epochs 6 --batch_size_override 1 --accumulate_steps 24 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name STATERA-1K-VideoMAE

echo "[*] Launching Z-Depth Paradox Ablation (No Head B)..."
python tools/train.py --dataset_path sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --single_task --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --checkpoint_dir .checkpoints --metrics_file .checkpoints/1k_ablation_metrics.json --wandb_name STATERA-1K-No-Z-Depth

echo "[✓] ALL 1K ABLATIONS COMPLETE"