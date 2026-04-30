#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "[*] Running STATERA Evaluation Suite..."
python tools/evaluate.py --task all

echo "[✓] All evaluations finished successfully."