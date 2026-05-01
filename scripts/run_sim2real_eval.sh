#!/bin/bash
set -e

# Run evaluate.py from project root
cd "$(dirname "$0")/.."

echo "================================================================="
echo "      STATERA SIM2REAL PHYSICAL BENCHMARK SUITE INITIATED      "
echo "================================================================="

echo ""
echo "[1/7] Running Zero-Shot Real-World Transfer (KECS Metric)..."
python tools/evaluate.py --task real_world

echo ""
echo "[2/7] Running Point Tracker Baselines (CoTracker, TAPIR)..."
python tools/evaluate.py --task point_trackers

echo ""
echo "[3/7] Running Geometric Centroid Baseline..."
python tools/evaluate.py --task centroid

echo ""
echo "[4/7] Evaluating Disentanglement: Geometric Collapse Metric..."
python tools/evaluate.py --task geom_collapse

echo ""
echo "[5/7] Evaluating Temporally-Weighted Euclidean Error..."
python tools/evaluate.py --task euclidean

echo ""
echo "[6/7] Evaluating Expected Spatial Dispersion (ESD)..."
python tools/evaluate.py --task dispersion

echo ""
echo "[7/7] Evaluating Spatial Predictive Entropy..."
python tools/evaluate.py --task entropy

echo ""
echo "================================================================="
echo "                 [✓] SIM2REAL BENCHMARK COMPLETE                "
echo "================================================================="