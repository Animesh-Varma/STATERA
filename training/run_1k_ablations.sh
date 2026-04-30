#!/bin/bash
echo "[>] STATERA 1K MERGED ABLATION (30-Epoch Fast Track)..."
echo "{}" > 1k_ablation_metrics.json

# ==========================================
# CORE BASELINES & NARRATIVE RUNS
# ==========================================

# 1. Run-00-Anchor-Saved
echo "[*] Launching Anchor Baseline..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file 1k_ablation_metrics.json --wandb_name Run-00-Anchor-Saved

# 2. Run-01-DINOv2-Fixed
echo "[*] Launching DINOv2 Baseline (Fairness Match)..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone dinov2 --epochs 30 --funnel_epochs 6 --batch_size_override 1 --accumulate_steps 24 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file 1k_ablation_metrics.json --wandb_name Run-01-DINOv2-Fixed

# ==========================================
# MISSING ABLATIONS
# ==========================================

# 3. Run-02-No-Temporal-Mixer
echo "[*] Launching No-Temporal-Mixer Ablation..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer none --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-02-No-Temporal-Mixer --temperature 2.0 --finetune_blocks 2

# ==========================================
# FAIRNESS & TARGETING RUNS
# ==========================================

# 4. Run-03-External-Temporal-ResNet3D
echo "[*] Launching ResNet3D Temporal Baseline (Fairness Match)..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone resnet3d --epochs 30 --funnel_epochs 6 --batch_size_override 2 --accumulate_steps 12 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file 1k_ablation_metrics.json --wandb_name Run-03-ResNet3D-Fixed

# 5. Run-04-No-Curriculum-Static-Dot
echo "[*] Launching Static-Dot Simulator Memorizer..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --static_sigma --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file 1k_ablation_metrics.json --wandb_name Run-04-Static-Dot-Saved

# 6. Run-05-Standard-Sigma-Only
echo "[*] Launching Standard Gaussian Sigma Annealer..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --temperature 2.0 --finetune_blocks 2 --save_checkpoint_every_epoch --metrics_file 1k_ablation_metrics.json --wandb_name Run-05-Standard-Sigma-Saved

# ==========================================
# ADDITIONAL ABLATIONS
# ==========================================

# 7. Run-06-Standard-Softmax-T1
echo "[*] Launching Standard-Softmax-T1 Ablation..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-06-Standard-Softmax-T1 --temperature 1.0 --finetune_blocks 2

# 8. Run-07-Single-Task-Spatial
echo "[*] Launching Single-Task-Spatial Ablation..."
python ../statera/train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 30 --funnel_epochs 6 --accumulate_steps 3 --single_task --metrics_file 1k_ablation_metrics.json --wandb_name Run-07-Single-Task-Spatial --temperature 2.0 --finetune_blocks 2

echo "[✓] ALL 1K ABLATIONS COMPLETE"