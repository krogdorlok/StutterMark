#!/bin/bash
#SBATCH --job-name=stuttermark-eval
#SBATCH --partition=nlp
#SBATCH --account=swabhas_1625
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run eval suite E1-E8 (E7 derives from E6) for one model on one A100; finished conditions are skipped on resubmit.
# sbatch --export=ALL,CONFIG=configs/eval/olmo7b.yaml,STUTTERMARK_HF_HOME=/scratch1/$USER/.cache/huggingface scripts/eval_suite.sbatch
set -euo pipefail

: "${CONFIG:?set CONFIG, e.g. configs/eval/qwen05.yaml}"
# : "${STUTTERMARK_HF_HOME:?set STUTTERMARK_HF_HOME to a scratch path (home quota is too small for model caches)}"
PROJECT="${PROJECT:-$PWD}"
# echo Project "$PROJECT
export STUTTERMARK_HF_HOME=/scratch1/$USER/.cache/huggingface
export HF_HOME="$STUTTERMARK_HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE"

cd "$PROJECT"
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: ${SLURMD_NODENAME:-unknown}  Config: $CONFIG  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# E1-E8, the E5 judge and the report run in order inside suite.py; it stops at the first failure.
uv run python scripts/eval/suite.py --config "$CONFIG" --judge-env-file "${JUDGE_ENV_FILE:-.env}"
echo "=== Job finished: $(date) ==="
