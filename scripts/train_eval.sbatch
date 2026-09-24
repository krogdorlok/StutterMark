#!/bin/bash
#SBATCH --job-name=stuttermark-qwen7b
#SBATCH --partition=nlp
#SBATCH --account=swabhas_1625
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train one LoRA adapter, then evaluate it on the same GPU allocation.
set -euo pipefail

# Override at submission time if needed:
# sbatch --export=ALL,CONFIG=configs/train/qwen7b.yaml scripts/train_eval.sbatch
PROJECT="${PROJECT:-/home1/calebtal/projects/StutterMark}"
CONFIG="${CONFIG:-configs/train/qwen7b.yaml}"
# Home has a tight user quota; Sep 14 jobs wrote to ~/.cache/huggingface and hit errno 122.
# Force scratch for Slurm jobs. Do not inherit login-shell HF_HOME (often /home1 or full /project2).
# Override only via STUTTERMARK_HF_HOME if needed.
export HF_HOME="${STUTTERMARK_HF_HOME:-/scratch2/calebtal/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

cd "$PROJECT"
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

source "$PROJECT/.venv/bin/activate"

uv run python scripts/check_device.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== Training ==="
uv run python scripts/train.py --config "$CONFIG"

echo "=== Evaluation ==="
uv run python scripts/eval.py --config "$CONFIG"

echo "=== Job finished: $(date) ==="
