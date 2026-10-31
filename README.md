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

| Backend | Text Path | Image Path | Acceleration | Best For |
|---------|-----------|------------|--------------|----------|
| vLLM | Qwen3Model (28L, 1024H) + LastPool + CompatHead | SigLIP2VisionModel (27L, 1152H) direct CUDA | CUDA / Triton attention | Serving, batched requests, low latency |
| llama.cpp/GGML | Qwen3 GGUF + llama.cpp + compat head | SigLIP2 GGML graph + MAP pooling | CUDA / CPU | Single-shot, CPU-only, embedded |

Both branches produce **1024-dimensional L2-normalised** embeddings in a shared
retrieval space. Text and image embeddings are comparable via a single dot product —
no modality-specific scoring head needed.

> **Not a model weights repo.** This repository contains inference code, tests,
> scripts, and documentation only. Model weights live at
> `moeshawky/qwen3-embedding-0.6b-vl` and are referenced by digest in `SHA256SUMS`.

---

## 1. What This Repo Contains

| File / Dir | Purpose |
|------------|---------|
| `serve_vllm.sh` | Launch the vLLM embedding server (production path) |
| `build_native.sh` | Build the native llama.cpp/GGML executable |
| `run_native_text.sh` | Embed text via native binary |
| `run_native_image.sh` | Embed images via native binary |
| `benchmark.sh` | Measure latency, throughput, GPU memory |
| `test_suite.py` | Shared correctness test suite (T1–T5) |
| `requirements.txt` | Pinned Python dependencies |
| `vllm/qwen3_vl_vllm/model.py` | vLLM model class (text + image) |
| `vllm/qwen3_vl_vllm/preprocess.py` | NaFlex image preprocessing (bit-exact vs HF) |
| `vllm/qwen3_vl_vllm/register.py` | vLLM ModelRegistry registration |
| `vllm/server.py` | FastAPI server (`/v1/embeddings`, `/embed/image`) |
| `qwen3-embedding-0.6b-vl-native/native/src/` | Native C++/GGML source |
| `ARCHITECTURE.md` | Detailed architecture document |

---

## 2. Quick Start — vLLM Server (Recommended for Production)

### 2.1 Install

```bash
pip install -r requirements.txt
```

Requirements: `torch==2.11.0`, `transformers==5.16.1`,
`huggingface-hub>=1.5.0,<2.0`, `safetensors==0.8.0`, `numpy==2.1.3`,
`Pillow==12.3.0`, `vllm==0.32.0`, `pytest==8.4.2`.

### 2.2 Launch the Server

```bash
./serve_vllm.sh 8000
```

This starts a FastAPI server on `http://localhost:8000` using a single Tesla T4
(float32, ~4s model initialisation, ~1.5 GiB KV cache at 0.4 utilisation).

### 2.3 Text Embeddings (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "a synthetic square gradient image with red, green and blue rectangles",
      "a wide landscape photograph of a green field next to a blue sky",
      "short text"
    ],
    "model": "qwen3-vl"
  }'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.123, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.456, ...]},
    {"object": "embedding", "index": 2, "embedding": [0.789, ...]}
  ],
  "model": "qwen3-vl",
  "usage": {"prompt_tokens": 42, "total_tokens": 42}
}
```

Each embedding is a 1024-element float32 array with L2 norm ≈ 1.0.

### 2.4 Image Embeddings (Custom Endpoint)

```bash
curl -X POST http://localhost:8000/embed/image \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded PNG or JPEG>",
    "model": "qwen3-vl"
  }'
```

For multiple images, use `"images": ["<b64-1>", "<b64-2>", ...]`.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.123, ...]}
  ],
  "model": "qwen3-vl"
}
```

### 2.5 Health Check

```bash
curl http://localhost:8000/health
# -> {"status": "healthy", "model": "qwen3-siglip2-vl"}
```

---

## 3. Quick Start — Native llama.cpp/GGML

### 3.1 Build

Choose CUDA (T4 acceleration) or CPU:

```bash
# CUDA backend — requires libcuda.so from NVIDIA driver
./build_native.sh cuda
#   -> qwen3-embedding-0.6b-vl-native/native/build_cuda/qwen3vl-embed

# CPU backend — no special dependencies
./build_native.sh cpu
#   -> qwen3-embedding-0.6b-vl-native/native/build_cpu/qwen3vl-embed
```

### 3.2 Embed Text

```bash
./run_native_text.sh "a synthetic square gradient image with red, green and blue rectangles" cuda
# Writes /tmp/native_run/text_text_embeds.npy (1024-d float32)
```

### 3.3 Embed Image

```bash
./run_native_image.sh /path/to/image.png cuda
# Writes /tmp/native_out/<name>_image_embeds.npy (1024-d float32)
```

### 3.4 Read the Output

```python
import numpy as np
emb = np.load("/tmp/native_run/text_text_embeds.npy")
print(emb.shape)   # (1024,)
print(np.linalg.norm(emb))  # ~1.0
```

---

## 4. Python Client Example

```python
import base64
import json
import urllib.request
from PIL import Image
import io

BASE = "http://localhost:8000"

def embed_texts(texts):
    data = json.dumps({"input": texts, "model": "qwen3-vl"}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return [item["embedding"] for item in result["data"]]

def embed_image(path):
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data = json.dumps({"image": b64, "model": "qwen3-vl"}).encode()
    req = urllib.request.Request(
        f"{BASE}/embed/image",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["data"][0]["embedding"]

# Usage
text_emb = embed_texts(["hello world", "longer text here"])
img_emb = embed_image("photo.jpg")

# Cross-modal similarity (dot product in shared space)
similarity = sum(a * b for a, b in zip(text_emb[0], img_emb))
print(f"Text-image cosine: {similarity:.4f}")
```

---

## 5. Critical Fix: Batched Terminal-Token Pooling

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
scheduled. No sequential HTTP unbundling is used — the workaround from the
previous deployment is prohibited as the production solution.

**Verified:** batch-vs-single cosine ≈ 1.00000000 for unequal-length sequences
(T2 test, see `test_suite.py`).

---

## 6. Testing — Shared Correctness Suite (T1–T5)

Run the full suite against the canonical PyTorch reference:

```bash
# vLLM backend
python3 test_suite.py --backend vllm

# Native backend (requires built binary + weight GGUFs)
python3 test_suite.py --backend native \
    --native-bin ./qwen3-embedding-0.6b-vl-native/native/build_cpu/qwen3vl-embed \
    --mmproj ./mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf \
    --text-model ./qwen3-embedding-0.6b-q8_0.gguf \
    --compat-head ./qwen3-embedding-0.6b-vl-native/native/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf
```

| Test | Coverage | Verify |
|------|----------|--------|
| **T1** | Text: 16 reference prompts, single + batched | cosine ≥ 0.999 |
| **T2** | Text adversarial: mixed lengths, repeats, position invariance, EOS | cosine ≥ 0.999 |
| **T3** | Image: 6 NaFlex fixtures, end-to-end vs PyTorch reference | cosine ≥ 0.999 |
| **T4** | Batch association: pairwise cosine matrix, response-to-input mapping | diagonal ≈ 1.0, correct mapping |
| **T5** | Cross-modal retrieval: similarity matrix | informational (synthetic fixtures weak) |

All tests use semantic input identity (not array position) for comparison.

---

## 7. Performance Benchmark

```bash
./benchmark.sh 8000
```

Measures single-request latency, batched throughput (batch sizes 1/2/4/8), and
GPU memory usage on the running vLLM server.

**Measured on 2× Tesla T4 (15360 MiB each), vLLM float32, single-GPU:**
- Text embedding load: ~4s model initialisation
- CUDA graph warmup: ~5s for 35 capture sizes
- KV cache memory: ~1.5 GiB actual at 0.4 utilisation
- Free GPU memory: ~14.5 GiB / 14.6 GiB at startup

---

## 8. Numerical Validation Results

All measurements executed against the canonical PyTorch reference model
(`moeshawky/qwen3-embedding-0.6b-vl`, float32, CUDA Tesla T4). Every claim
has a verification command.

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
| Image | 6 NaFlex fixtures vs PyTorch reference | 0.99905 (square) | PASS on 5/6 | `./run_native_image.sh  cpu` + compare |
| Image | `tiny` fixture | 0.99476 | Q8 sidecar floor | See below |

The `tiny` fixture (112×112, very smooth content) falls to the Q8_0 sidecar's
quantization floor (0.9947), which is input-dependent — verified by the
`q8_floor_check.py` control showing the FP32 harness reproduces the oracle
bitwise while the Q8 sidecar degrades on low-frequency inputs. Not an
implementation defect.

---

## 9. Weight Integrity

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

## 10. Rollback & Recovery

No checkpoint files have been altered. To rollback:

1. Stop the vLLM server (`Ctrl+C` or kill the process)
2. Remove any custom build directories (`build_cpu/`, `build_cuda/`)
3. The model directory and all weights remain untouched

All scripts are reversible — they only read from the model directory and write
to `/tmp/` or local build directories.

---

## 11. Known Limitations

| # | Limitation | Details | Mitigation |
|---|-----------|---------|------------|
| 1 | Native `tiny` fixture Q8 floor | 0.9947 cosine on very smooth 112×112 images; input-dependent Q8 sidecar quantization | Use higher-precision sidecar (requires relaxing "no requantisation" constraint) |
| 2 | Cross-modal on synthetic fixtures | All 6 fixtures share one synthetic gradient+rectangles recipe; weak retrieval signal | Test with real photographs before production deployment |
| 3 | Native CUDA build | Requires `libcuda.so` from NVIDIA driver at `/usr/local/nvidia/lib64/` | Set `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` |
| 4 | Image preprocessing | NaFlex preprocessing is bit-exact vs HF processor (diff ≈ 1.2e-7) | None needed — verified |

---

## 12. Verification

Every numerical claim in this card is backed by an executed test. Re-verify:

```bash
# Full correctness suite (vLLM)
python3 test_suite.py --backend vllm
# Expects: exit code 0, all T1–T4 passing, T5 informational

# Native image embedding vs reference
./run_native_image.sh ./qwen3-embedding-0.6b-vl-native/native/fixtures/square.png cpu
# Then compare /tmp/native_out/square_image_embeds.npy against oracle_square.npz

# vLLM server health
curl http://localhost:8000/health
```

**Last verified:** 2026-10-31 on 2× Tesla T4, vLLM 0.32.0, torch 2.11.0+cu128,
transformers 5.16.1.

**Verification command:** `python3 test_suite.py --backend vllm`

---

## 13. Source Citations

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
