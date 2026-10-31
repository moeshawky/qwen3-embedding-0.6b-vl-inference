#!/bin/bash
# Build native llama.cpp/GGML embedding executable with CUDA support
#
# Builds qwen3vl-embed for text + image embedding on T4.
#
# Usage:
#   ./build_native.sh [cuda|cpu]
#
set -euo pipefail

NATIVE_DIR="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native"
LLAMA_CPP_DIR="/tmp/opencode/llamacpp"
BACKEND="${1:-cuda}"

echo "Building native embedding executable ($BACKEND backend)"

if [ "$BACKEND" = "cuda" ]; then
    cmake -B "$NATIVE_DIR/build_cuda" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
        -DGGML_CUDA=ON \
        -DCMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64" \
        -DCMAKE_PREFIX_PATH="/usr/local/cuda;/usr/local/nvidia"
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64 cmake --build "$NATIVE_DIR/build_cuda" -j2 --target qwen3vl-embed
    BIN="$NATIVE_DIR/build_cuda/qwen3vl-embed"
else
    cmake -B "$NATIVE_DIR/build_cpu" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR"
    cmake --build "$NATIVE_DIR/build_cpu" -j4 --target qwen3vl-embed
    BIN="$NATIVE_DIR/build_cpu/qwen3vl-embed"
fi

echo "Build complete: $BIN"
