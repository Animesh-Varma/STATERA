#!/bin/bash

# Navigate to the project root folder to ensure module imports (statera.model) work properly
cd "$(dirname "$0")/.." || exit

# -------------------------------------------------------------------------
# Run 1: STATERA 50K Crescent SOTA
# -------------------------------------------------------------------------
echo "=================================================="
echo " Rendering Heatmaps for SOTA Crescent Model..."
echo "=================================================="

python tools/render_heatmaps.py \
    --checkpoint "scripts/sota_50k_crescent_checkpoints/STATERA-50K-Crescent-SOTA_epoch_7.pth" \
    --data_dir "sim2real/output" \
    --out_dir "outputs/heatmaps/SOTA_50k_Crescent" \
    --run_name "SOTA-Crescent"

echo ""

# -------------------------------------------------------------------------
# Run 2: STATERA 50K Sigma SOTA
# -------------------------------------------------------------------------
echo "=================================================="
echo " Rendering Heatmaps for SOTA Sigma Model..."
echo "=================================================="

python tools/render_heatmaps.py \
    --checkpoint "scripts/sota_50k_sigma_checkpoints/STATERA-50K-Sigma_epoch_10.pth" \
    --data_dir "sim2real/output" \
    --out_dir "outputs/heatmaps/SOTA_50k_Sigma" \
    --run_name "SOTA-Sigma"

echo "=================================================="
echo " All SOTA heatmaps generated successfully!"
echo " Results available in outputs/heatmaps/"
echo "=================================================="