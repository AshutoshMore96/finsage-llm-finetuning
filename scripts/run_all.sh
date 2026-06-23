#!/usr/bin/env bash
# Full training pipeline — run on a CUDA GPU box (Colab / Kaggle).
# Usage: bash scripts/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> [1/5] Preparing SFT data"
python -m src.data.prepare_sft_data

echo "==> [2/5] QLoRA supervised fine-tuning"
python -m src.train.train_sft_qlora

echo "==> [3/5] Building DPO preference pairs"
python -m src.data.prepare_dpo_data

echo "==> [4/5] DPO alignment"
python -m src.train.train_dpo

echo "==> [5/5] Merge + quantize (and optional Hub push)"
python -m src.merge.merge_and_quantize

echo "Done. Run evaluation with:  python -m src.eval.evaluate"
