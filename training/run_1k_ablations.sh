#!/bin/bash
echo "[>] STATERA 1K ABLATION SUITE INITIALIZING..."
echo "{}" > 1k_ablation_metrics.json

# 1. Run-00-SOTA-1K-Anchor (The Core Baseline at 1K)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-00-SOTA-1K-Anchor --temperature 2.0 --finetune_blocks 2

# 2. Run-01-Spatial-Only-DINOv2 (Testing pure spatial representation devoid of temporal mechanics)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer none --target_type crescent --backbone dinov2 --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-01-Spatial-Only-DINOv2 --temperature 2.0 --finetune_blocks 2

# 3. Run-02-No-Temporal-Mixer (Testing cross-frame mixing necessity within equivalent architecture)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer none --target_type crescent --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-02-No-Temporal-Mixer --temperature 2.0 --finetune_blocks 2

# 4. Run-03-External-Temporal-ResNet3D (The standard 3D CNN baseline, hardware optimized via accumulated constraints)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone resnet3d --epochs 50 --funnel_epochs 10 --batch_size_override 2 --accumulate_steps 12 --metrics_file 1k_ablation_metrics.json --wandb_name Run-03-External-Temporal-ResNet3D --temperature 2.0 --finetune_blocks 0

# 5. Run-04-No-Curriculum-Static-Dot (Tests "Expectation Collapse" against a fixed, non-shrinking target)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --static_sigma --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-04-No-Curriculum-Static-Dot --temperature 2.0 --finetune_blocks 2

# 6. Run-05-Standard-Sigma-Only (Tests traditional shrinking heatmaps against Crescent angular targeting)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type dot --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-05-Standard-Sigma-Only --temperature 2.0 --finetune_blocks 2

# 7. Run-06-Standard-Softmax-T1 (Testing against Reviewer #2's claims via T=1 Quantization Check)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --metrics_file 1k_ablation_metrics.json --wandb_name Run-06-Standard-Softmax-T1 --temperature 1.0 --finetune_blocks 2

# 8. Run-07-Single-Task-Spatial (Testing dual-head 3D auxiliary loss implications on pure 2D trajectory tracking)
python train.py --dataset_path ../sim/1K-ablation.hdf5 --decoder_type deconv --temporal_mixer conv1d --target_type crescent --backbone vjepa --epochs 50 --funnel_epochs 10 --accumulate_steps 3 --single_task --metrics_file 1k_ablation_metrics.json --wandb_name Run-07-Single-Task-Spatial --temperature 2.0 --finetune_blocks 2

echo "[✓] ALL 1K ABLATIONS COMPLETE. GENERATING REPORT."
python generate_1k_ablation_tables.py