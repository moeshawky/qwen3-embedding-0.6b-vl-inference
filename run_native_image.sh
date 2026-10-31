#!/bin/bash
# Run native image embedding via qwen3vl-embed (CPU or CUDA)
#
# Usage:
#   ./run_native_image.sh /path/to/image.png [cuda|cpu]
#
set -euo pipefail

NATIVE_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native"
MODEL_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl"
MMPROJ="$MODEL_DIR/mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf"
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

if [ ! -f "$1" ]; then
    echo "Error: Image file not found: $1"
    exit 1
fi

mkdir -p /tmp/native_run

IMAGE_PATH="$(realpath "$1")"
IMAGE_NAME="$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')"

"$BIN" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE_PATH" \
    --stem "$IMAGE_NAME" \
    --out /tmp/native_run \
    --threads 2
