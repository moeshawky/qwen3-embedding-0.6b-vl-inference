---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
base_model:
- moeshawky/qwen3-embedding-0.6b-vl
tags:
- feature-extraction
- sentence-similarity
- vision-language
- qwen3
- siglip2
- vllm
- llama.cpp
- ggml
- embedding-inference
---

# Qwen3-SigLIP2-VL Inference Backends

Production inference implementations for the dual-encoder embedding model
`moeshawky/qwen3-embedding-0.6b-vl`. Two backends:

| Backend | Text Path | Image Path | Acceleration |
|---------|-----------|------------|--------------|
| vLLM | Qwen3Model (28L, 1024H) + LastPool + CompatHead | SigLIP2VisionModel (27L, 1152H) direct CUDA | CUDA / Triton attention |
| llama.cpp/GGML | Qwen3 GGUF + llama.cpp + compat head | SigLIP2 GGML graph + MAP pooling | CUDA / CPU |

Both branches produce **1024-dimensional L2-normalised** embeddings in a shared
retrieval space. Text and image embeddings are comparable via a single dot product.

> **Not a model weights repo.** This repository contains inference code, tests,
> scripts, and documentation only. Model weights live at
> `moeshawky/qwen3-embedding-0.6b-vl`.

---

## Critical Fix: Batched Terminal-Token Pooling

A previous deployment passed `seq_len - 1` as the pooling index for every
sequence in a batch. With right-padding, only the longest sequence pooled its
actual terminal token; shorter sequences pooled padding positions, producing
~0.31–0.37 cosine agreement against single-sequence references.

**Fix:** Per-row pooling index computed from the attention mask:

```python
pool_index = (attention_mask != 0).long().sum(dim=-1) - 1
```

vLLM's `LastPool` pooling method uses `seq_lens_cpu_upper_bound`, which
correctly handles batched unequal-length sequences without padding tokens being
scheduled.

**Verified:** batch-vs-single cosine ≈ 1.00000000 for unequal-length sequences
(T2 test, see `test_suite.py`).

---

## Numerical Validation Results

All measurements below were executed against the canonical PyTorch reference
model (`moeshawky/qwen3-embedding-0.6b-vl`, float32, CUDA Tesla T4). Every
claim has a verification command.

### vLLM Backend (float32, single Tesla T4)

| Test | Description | Min Cosine | Mean Cosine | Status | Verify |
|------|-------------|-----------|-------------|--------|--------|
| T1 | Text — 16 reference prompts, single + batched | 0.99999994 | 1.00000005 | PASS | `python3 test_suite.py --backend vllm` |
| T2 | Text adversarial — mixed lengths, repeats, position, EOS | 0.99999994 | — | PASS | `python3 test_suite.py --backend vllm` |
| T3 | Image — 6 NaFlex fixtures vs PyTorch reference | 1.00000000 | 1.00000008 | PASS | `python3 test_suite.py --backend vllm` |
| T4 | Batch association — pairwise cosine matrix, response mapping | 0.99999994 | 1.00000000 | PASS | `python3 test_suite.py --backend vllm` |

All text embeddings have L2 norm ≈ 1.0 (verified per-vector).
All image embeddings have L2 norm ≈ 1.0 (verified per-vector).

### Native llama.cpp/GGML Backend (CPU, validated against FP32 oracle)

| Test | Description | Min Cosine | Status | Verify |
|------|-------------|-----------|--------|--------|
| Text | 16 reference prompts vs PyTorch reference | 0.99952 | PASS | `./run_native_text.sh <text> cpu` + compare |
| Image | 6 NaFlex fixtures vs PyTorch reference | 0.99905 (square) | PASS on 5/6 | `./run_native_image.sh <img> cpu` + compare |
| Image | `tiny` fixture | 0.99476 | Q8 sidecar floor | See below |

The `tiny` fixture (112×112, very smooth content) falls to the Q8_0 sidecar's
quantization floor (0.9947), which is input-dependent — verified by the
`q8_floor_check.py` control showing the FP32 harness reproduces the oracle
bitwise while the Q8 sidecar degrades on low-frequency inputs. Not an
implementation defect.

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Requirements: `torch==2.11.0`, `transformers==5.16.1`, `huggingface-hub>=1.5.0,<2.0`,
`safetensors==0.8.0`, `numpy==2.1.3`, `Pillow==12.3.0`, `vllm==0.32.0`,
`pytest==8.4.2`.

### vLLM Server

```bash
./serve_vllm.sh 8000
```

Endpoints (verified against running server):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/embeddings` | JSON | Text embeddings (OpenAI-compatible) |
| `POST /embed/image` | JSON | Image embeddings (base64-encoded input) |
| `GET /health` | GET | Health check |

Example text embedding:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "longer text"], "model": "qwen3-vl"}'
```

Returns `{"data": [{"index": 0, "embedding": [...]}, ...], "object": "list"}`.

### Native (llama.cpp/GGML) Build & Run

Build (choose CUDA for T4 acceleration, CPU otherwise):

```bash
./build_native.sh cuda    # CUDA backend — requires libcuda.so from NVIDIA driver
./build_native.sh cpu     # CPU backend — no special dependencies
```

Run text:

```bash
./run_native_text.sh "a synthetic square gradient image" cuda
```

Run image:

```bash
./run_native_image.sh /path/to/image.png cuda
```

Both write a `.npy` file to `/tmp/native_run/` with the 1024-d embedding.

---

## Testing

Run the full shared correctness suite:

```bash
# vLLM backend (requires running vLLM engine via test_suite internals)
python3 test_suite.py --backend vllm

# Native backend
python3 test_suite.py --backend native \
    --native-bin ./qwen3-embedding-0.6b-vl-native/native/build_cpu/qwen3vl-embed \
    --mmproj ./mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf \
    --text-model ./qwen3-embedding-0.6b-q8_0.gguf \
    --compat-head ./qwen3-embedding-0.6b-vl-native/native/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf
```

Test categories:

| Test | Coverage |
|------|----------|
| T1 | Text: 16 reference prompts, single + batched |
| T2 | Text adversarial: mixed lengths, repeats, position invariance, EOS |
| T3 | Image: 6 NaFlex fixtures, end-to-end vs PyTorch reference |
| T4 | Batch association: pairwise cosine matrix, response-to-input mapping |
| T5 | Cross-modal retrieval: similarity matrix (informational; synthetic fixtures weak) |

---

## Performance Benchmark

```bash
./benchmark.sh 8000
```

Measures single-request latency, batched throughput (batch sizes 1/2/4/8), and
GPU memory usage on the vLLM server. Results printed to stdout.

Measured on 2× Tesla T4 (15360 MiB each), vLLM float32, single-GPU:
- Text embedding load: ~4s model initialisation
- CUDA graph warmup: ~5s for 35 capture sizes
- KV cache memory: ~1.5 GiB actual at 0.4 utilisation

---

## Weight Integrity

No model weights are stored, modified, or retrained in this repository. All
weights remain at `moeshawky/qwen3-embedding-0.6b-vl`:

| Artifact | Tensors | Size | Role |
|----------|---------|------|------|
| `model.safetensors` | 762 | 4.1 GB | Composite model (F32) |
| `qwen3-embedding-0.6b-q8_0.gguf` | 310 | ~639 MB | Text-only GGUF (loads in stock llama.cpp) |
| `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` | 450 | ~458 MB | Vision-tower sidecar (Q8_0) |
| `compat_head_qwen3_embedding_0.6b.f32.gguf` | 2 | 4.2 MB | Opt-in compat head (F32) |

SHA256 digests for all weight artifacts are in the weights repo's `SHA256SUMS`.

---

## Rollback & Recovery

No checkpoint files have been altered. To rollback:

1. Stop the vLLM server (`Ctrl+C` or kill the process)
2. Remove any custom build directories (`build_cpu/`, `build_cuda/`)
3. The model directory and all weights remain untouched

All scripts are reversible — they only read from the model directory and write
to `/tmp/` or local build directories.

---

## File Layout

```
.
├── README.md                 # This file (model card)
├── ARCHITECTURE.md           # Detailed architecture document
├── requirements.txt          # Pinned Python dependencies
├── serve_vllm.sh             # vLLM server launch script
├── build_native.sh           # Native llama.cpp/GGML build script
├── run_native_text.sh        # Native text inference runner
├── run_native_image.sh       # Native image inference runner
├── benchmark.sh              # Performance benchmark script
├── test_suite.py             # Shared correctness test suite (T1-T5)
├── vllm/
│   ├── qwen3_vl_vllm/
│   │   ├── __init__.py
│   │   ├── model.py          # vLLM model class (text + image)
│   │   ├── preprocess.py     # NaFlex image preprocessing (bit-exact vs HF)
│   │   └── register.py       # vLLM ModelRegistry registration
│   ├── server.py             # FastAPI server (/v1/embeddings, /embed/image)
│   ├── test_embedding.py     # vLLM text embedding test
│   ├── test_image.py         # vLLM image embedding test
│   └── test_image_direct.py  # Direct model component test
└── qwen3-embedding-0.6b-vl-native/  # Native C++/GGML source (submodule layout)
```

---

## Known Limitations

| # | Limitation | Details | Mitigation |
|---|-----------|---------|------------|
| 1 | Native `tiny` fixture Q8 floor | 0.9947 cosine on very smooth 112×112 images; input-dependent Q8 sidecar quantization | Use higher-precision sidecar (requires relaxing "no requantisation" constraint) |
| 2 | Cross-modal on synthetic fixtures | All 6 fixtures share one synthetic gradient+rectangles recipe; weak retrieval signal | Test with real photographs before production deployment |
| 3 | Native CUDA build | Requires `libcuda.so` from NVIDIA driver at `/usr/local/nvidia/lib64/` | Set `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` |
| 4 | Image preprocessing | NaFlex preprocessing is bit-exact vs HF processor (diff ≈ 1.2e-7) | None needed — verified |

---

## Verification

Every numerical claim in this card is backed by an executed test. Re-verify:

```bash
# Full correctness suite (vLLM)
python3 test_suite.py --backend vllm

# Native image embedding vs reference (requires native binary)
./run_native_image.sh ./qwen3-embedding-0.6b-vl-native/native/fixtures/square.png cpu
# Then compare /tmp/native_out/square_image_embeds.npy against oracle_square.npz

# vLLM server health
curl http://localhost:8000/health
```

**Last verified:** 2026-10-31 on 2× Tesla T4, vLLM 0.32.0, torch 2.11.0+cu128,
transformers 5.16.1.

**Verification command:** `python3 test_suite.py --backend vllm` (expects exit code 0,
all T1–T4 passing, T5 informational).

---

## Source Citations

- Model weights/config: `moeshawky/qwen3-embedding-0.6b-vl` — `model.safetensors`,
  `config.json`, `configuration_qwen_siglip2_embedding.py`,
  `modeling_qwen_siglip2_embedding.py`
- Native C++ source: `qwen3-embedding-0.6b-vl-native/native/src/`
- Oracle fixtures: `qwen3-embedding-0.6b-vl-native/native/oracle/`
- vLLM 0.32.0 pooling reference: `vllm/model_executor/layers/pooler/seqwise/methods.py`
  (`LastPool` uses `seq_lens_cpu_upper_bound`)

---

## License

Apache-2.0. The parent `Qwen/Qwen3-Embedding-0.6B` is Apache-2.0, and
`google/siglip2-so400m-patch16-naflex` is Apache-2.0.
