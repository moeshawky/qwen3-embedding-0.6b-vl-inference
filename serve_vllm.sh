#!/bin/bash
# Launch vLLM embedding server for Qwen3-SigLIP2-VL
#
# Two T4s, single-GPU FP16 deployment. Model is small enough for one GPU.
#
# Usage:
#   ./serve_vllm.sh [port]
#
set -euo pipefail

MODEL_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl"
VLLM_VENV="/root/.venvs/vllm"
PORT="${1:-8000}"

# Register our custom model architecture with vLLM
export VLLM_PLUGINS="qwen3_vl_vllm.register"

# Add vllm package to Python path
export PYTHONPATH="/kaggle/working/qwen-0.6-vl/vllm:/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl:${PYTHONPATH:-}"

echo "Starting vLLM embedding server on port $PORT"
echo "Model: $MODEL_DIR"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

exec "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --hf-overrides '{"architectures": ["Qwen3Siglip2EmbeddingModel"]}' \
    --model-impl vllm \
    --runner pooling \
    --convert embed \
    --port "$PORT" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.5 \
    --dtype float32 \
    --trust-remote-code \
    "$@"
