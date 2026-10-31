#!/bin/bash
# Run native text embedding via qwen3vl-embed (CPU or CUDA)
#
# Usage:
#   ./run_native_text.sh "text to embed" [cuda|cpu]
#
set -euo pipefail

NATIVE_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native"
MODEL_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl"
TEXT_MODEL="$MODEL_DIR/qwen3-embedding-0.6b-q8_0.gguf"
COMPAT_HEAD="$NATIVE_DIR/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf"
BACKEND="${2:-cpu}"

if [ "$BACKEND" = "cuda" ]; then
    BIN="$NATIVE_DIR/build_cuda/qwen3vl-embed"
else
    BIN="$NATIVE_DIR/build_cpu/qwen3vl-embed"
fi

if [ ! -f "$BIN" ]; then
    echo "Error: Native binary not found at $BIN"
    echo "Run: ./build_native.sh $BACKEND"
    exit 1
fi

mkdir -p /tmp/native_run

"$BIN" \
    --text-model "$TEXT_MODEL" \
    --compat-head "$COMPAT_HEAD" \
    --text "$1" \
    --stem text \
    --out /tmp/native_run \
    --threads 2
