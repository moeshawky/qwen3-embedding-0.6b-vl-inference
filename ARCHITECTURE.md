# Qwen3-SigLIP2-VL Dual-Encoder Embedding Backends

## Overview

Two production inference backends for `moeshawky/qwen3-embedding-0.6b-vl`:

| Backend | Text | Image | Acceleration |
|---------|------|-------|--------------|
| vLLM | Qwen3Model + LastPool + CompatHead | SigLIP2VisionModel (direct CUDA) | CUDA / Triton attention |
| llama.cpp/GGML | Qwen3 GGUF + llama.cpp | SigLIP2 GGML graph | CUDA / CPU |

Both produce **1024-d L2-normalised** embeddings in a shared retrieval space.

## Model Architecture

Dual-encoder (independent branches, no token injection):

- **Text**: Qwen3 0.6B (28 layers, 1024 hidden) → terminal-token pool at `<|endoftext|>` (151643) → L2 → compat head (1024→1024) → L2
- **Vision**: SigLIP2 SO400M NaFlex (27 blocks, 1152 hidden) → MAP pooling → L2 → vision projection (1152→1024) → L2

## Critical Fix: Batched Terminal-Token Pooling

The previous deployment passed `seq_len - 1` as the pooling index for all sequences in a batch. For right-padded input, only the longest sequence pooled its actual terminal token. Shorter sequences pooled padding positions, producing ~0.31-0.37 cosine agreement.

**Fix**: Per-row pooling using `pool_index = (attention_mask != 0).sum(dim=-1) - 1` (index of the last true attention-mask element). vLLM's `LastPool` uses `seq_lens_cpu_upper_bound` which correctly handles batched unequal-length sequences without padding tokens.

Verified: batch vs single cosine ≈ 1.00000000 for unequal-length sequences.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### vLLM Server

```bash
./serve_vllm.sh 8000
```

Endpoints:
- `POST /v1/embeddings` — Text embeddings (OpenAI-compatible)
- `POST /embed/image` — Image embeddings (base64 input)
- `GET /health` — Health check

Example:
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "longer text"], "model": "qwen3-vl"}'
```

### Native (llama.cpp/GGML)

Build:
```bash
./build_native.sh cuda    # CUDA backend (T4)
./build_native.sh cpu     # CPU backend
```

Run text:
```bash
./run_native_text.sh "hello world" cuda
```

Run image:
```bash
./run_native_image.sh /path/to/image.png cuda
```

## Testing

Run the full correctness suite:

```bash
# vLLM backend
python3 test_suite.py --backend vllm

# Native backend
python3 test_suite.py --backend native \
    --native-bin /path/to/qwen3vl-embed \
    --mmproj /path/to/mmproj.gguf \
    --text-model /path/to/text.gguf \
    --compat-head /path/to/compat_head.gguf
```

Test categories:
- **T1**: Text — 16 reference prompts, single and batched (cosine ≈ 1.0 vs reference)
- **T2**: Text adversarial batching — mixed lengths, repeats, position invariance, EOS handling
- **T3**: Image — 6 NaFlex fixtures, end-to-end vs PyTorch reference (cosine ≈ 1.0)
- **T4**: Batch association — pairwise cosine matrix, correct response-to-input mapping
- **T5**: Cross-modal retrieval — similarity matrix (informational; synthetic fixtures are weak)

## Benchmark

```bash
./benchmark.sh 8000
```

Measures single-request latency, batched throughput, and GPU memory usage.

## Numerical Validation Results

### vLLM (float32, T4)

| Test | Min Cosine | Mean Cosine | Status |
|------|-----------|-------------|--------|
| T1 Text single+batched | 0.99999994 | 1.00000005 | PASS |
| T2 Text adversarial | 0.99999994 | — | PASS |
| T3 Image (6 fixtures) | 1.00000000 | 1.00000008 | PASS |
| T4 Batch association | 0.99999994 | 1.00000000 | PASS |

### Native llama.cpp/GGML (CPU, validated against FP32 oracle)

| Test | Min Cosine | Status |
|------|-----------|--------|
| Image (6 fixtures) | 0.99905 (square) | PASS on 5/6 |
| Image (tiny fixture) | 0.99476 | Q8 sidecar floor |
| Text (16 prompts) | 0.99952 | PASS |

The `tiny` fixture falls to the Q8_0 sidecar's quantization floor (0.9947), which is input-dependent and not an implementation defect.

## Weight Integrity

All model weights are preserved unchanged:
- `model.safetensors` — 762 tensors, F32, 4.1 GB (sha256 in SHA256SUMS)
- `qwen3-embedding-0.6b-q8_0.gguf` — text-only GGUF (310 tensors)
- `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` — vision sidecar (450 tensors)
- `compat_head_qwen3_embedding_0.6b.f32.gguf` — opt-in compat head (4.2 MB)

No weights are modified or retrained.

## Rollback

The original model directory is untouched. To rollback:

1. Stop the vLLM server
2. Remove any custom build directories
3. Restore from the original model directory if needed

No checkpoint files have been altered.

## File Layout

```
/kaggle/working/qwen-0.6-vl/
├── vllm/
│   ├── qwen3_vl_vllm/
│   │   ├── __init__.py
│   │   ├── model.py          # vLLM model class (text + image)
│   │   ├── preprocess.py     # NaFlex image preprocessing
│   │   └── register.py       # vLLM ModelRegistry registration
│   ├── server.py             # FastAPI server with /embed/image endpoint
│   ├── test_embedding.py     # vLLM embedding test
│   ├── test_image.py         # vLLM image test
│   └── test_image_direct.py  # Direct model test
├── test_suite.py             # Shared correctness test suite
├── serve_vllm.sh             # vLLM launch script
├── build_native.sh           # Native build script
├── run_native_text.sh        # Native text inference
├── run_native_image.sh       # Native image inference
├── benchmark.sh              # Performance benchmark
├── requirements.txt          # Pinned dependencies
└── ARCHITECTURE.md           # This file
```

## Known Limitations

1. **Native tiny fixture**: Q8_0 sidecar quantization floor at 0.9947 on very smooth images. Not fixable without higher-precision sidecar.
2. **Cross-modal retrieval on synthetic fixtures**: Weak due to all fixtures sharing the same synthetic recipe.
3. **Native CUDA build**: Requires libcuda.so from NVIDIA driver (at /usr/local/nvidia/lib64/).
