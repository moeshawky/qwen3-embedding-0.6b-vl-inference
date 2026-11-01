---
license: other
private: false
tags:
  - ggml
  - llama.cpp
  - native-inference
  - siglip2
  - qwen3-embedding
---

# qwen3-embedding-0.6b-vl — native (GGML) inference

**Public repository.** Native llama.cpp/GGML inference code for
[`moeshawky/qwen3-embedding-0.6b-vl`](https://huggingface.co/moeshawky/qwen3-embedding-0.6b-vl),
the dual-encoder embedding model that maps text and images into a shared **1024-dimensional
L2-normalized** embedding space.

Part of the Qwen3-SigLIP2-VL inference suite:

| Repo | Platform | Purpose |
|------|----------|---------|
| [`moeshawky/qwen3-embedding-0.6b-vl-inference`](https://github.com/moeshawky/qwen3-embedding-0.6b-vl-inference) | GitHub | vLLM serving backend — `/v1/embeddings`, `/embed/image` |
| [`moeshawky/qwen3-embedding-0.6b-vl-native`](https://huggingface.co/moeshawky/qwen3-embedding-0.6b-vl-native) | Hugging Face | **This repo** — native C++/GGML inference code |
| [`moeshawky/qwen3-embedding-0.6b-vl`](https://huggingface.co/moeshawky/qwen3-embedding-0.6b-vl) | Hugging Face | Model weights (4.1 GB safetensors + GGUFs + ONNX) |

**Last updated: 2026-11-01** — made public; previously private work-in-progress.

---

## ⚠️ This repo has no model weights — by design

This repository contains **inference code only**: C++ source, build files, fixtures, oracle
references, and a 4.2 MB compat-head artifact. **No model weights are stored here.** This is
intentional separation of concerns:

| What you need | Where to get it | Size |
|---------------|-----------------|------|
| Inference code (this repo) | `moeshawky/qwen3-embedding-0.6b-vl-native` | ~100 KB source |
| Text GGUF | `moeshawky/qwen3-embedding-0.6b-vl` → `qwen3-embedding-0.6b-q8_0.gguf` | ~639 MB |
| Vision sidecar GGUF | `moeshawky/qwen3-embedding-0.6b-vl` → `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` | ~458 MB |
| Compat head (opt-in) | `moeshawky/qwen3-embedding-0.6b-vl-native` → `native/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf` | 4.2 MB (in this repo) |

**To use this repo you must download the text GGUF and vision sidecar from the weights repo.**
The compat head is already inside this repo under `native/artifacts/`.

---

## What this is

A standalone C++17 program (`native/src`) that produces **1024-D L2-normalized text and image
embeddings** using **GGML only** — no PyTorch, no ONNX runtime at inference time:

| Branch | What it does | Where weights come from |
|--------|-------------|------------------------|
| **Image** | SigLIP2 NaFlex vision tower (27 blocks) → MAP attention pooling → 1152→1024 vision projection → L2 | Q8_0 `mmproj` sidecar (458 MB, from weights repo), dequantised to F32 at load |
| **Text** | Qwen3-Embedding-0.6B text tower via llama.cpp → EOS terminator pool → L2 → opt-in compat head → L2 | Q8_0 text GGUF (639 MB, from weights repo) + compat head (4.2 MB, in this repo) |

Both branches output vectors in the **same shared space** — text and image embeddings are
comparable with a single dot product.

---

## Quick start

### 1. Download everything

```bash
# Inference code (this repo)
hf download moeshawky/qwen3-embedding-0.6b-vl-native --local-dir ./native

# Model weights (separate repo — required)
hf download moeshawky/qwen3-embedding-0.6b-vl \
  qwen3-embedding-0.6b-q8_0.gguf \
  mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf \
  --local-dir ./weights
```

### 2. Build the native executable

You need a llama.cpp checkout. The measurements were taken against
`0a1ef309cdbb9f2024f58a38fb4c42352c5db569`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llamacpp
cd /tmp/llamacpp && git checkout 0a1ef309cdbb9f2024f58a38fb4c42352c5db569

cd ./native/native
cmake -B build_cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=/tmp/llamacpp
cmake --build build_cpu -j$(nproc) --target qwen3vl-embed
```

The binary appears at `./native/native/build_cpu/qwen3vl-embed`.

A CUDA build is supported — set `LLAMA_CUDA=1` and point `LD_LIBRARY_PATH` at your NVIDIA
driver libraries (`/usr/local/nvidia/lib64`). The CUDA shared libraries compiled successfully
as of 2026-11-01; the final binary link was pending at last check.

### 3. Embed text

```bash
./native/native/build_cpu/qwen3vl-embed \
  --text-model ./weights/qwen3-embedding-0.6b-q8_0.gguf \
  --compat-head ./native/native/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf \
  --text "a synthetic square gradient image with red, green and blue rectangles" \
  --stem text \
  --out /tmp/native_out \
  --threads 2
```

Output: `/tmp/native_out/text_text_embeds.npy` — shape `(1, 1024)`, float32, L2 norm ≈ 1.0.

### 4. Embed image

```bash
./native/native/build_cpu/qwen3vl-embed \
  --mmproj ./weights/mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf \
  --image /path/to/your/image.png \
  --out /tmp/native_out \
  --threads 2
```

Output: `/tmp/native_out/<name>_image_embeds.npy` — shape `(1024,)`, float32, L2 norm ≈ 1.0.

### 5. Cross-modal similarity (dot product in shared space)

```python
import numpy as np
text = np.load("/tmp/native_out/text_text_embeds.npy")[0]   # (1024,)
image = np.load("/tmp/native_out/<name>_image_embeds.npy")   # (1024,)
similarity = float(np.dot(text, image))                      # cosine, since both are L2-normalized
print(f"Text-image similarity: {similarity:.6f}")
```

---

## Measured results (validated against FP32 PyTorch reference)

**Measurement date: 2026-10-31** — CPU-only native path (2 cores), CUDA Tesla T4 for references.

### Image parity — native end-to-end vs FP32 oracle

| fixture | grid | native vs FP32 oracle | gate ≥0.999 | note |
|---------|------|----------------------|-------------|------|
| square | 16×16 | **0.999051** | PASS | textured |
| landscape | 35×7 | **0.999659** | PASS | textured |
| portrait | 7×35 | **0.999587** | PASS | textured |
| odd | 21×12 | **0.999311** | PASS | textured |
| wide | 64×4 | **0.999582** | PASS | textured |
| tiny | 16×16 | **0.994764** | **FAIL** | very smooth, Q8 floor |

The `tiny` fixture (112×112, very smooth content) hits the Q8_0 sidecar's quantisation floor
(0.9947), which is **input-dependent** — verified by a control showing the FP32 harness is
bitwise identical to the oracle (1.0) while the Q8 sidecar degrades on low-frequency inputs.
This is a **sidecar limitation, not an implementation defect**. The native graph itself is
faithful: against an independent PyTorch run using the *same Q8 weights*, cosine is 0.9999998.

### Text parity — native vs FP32 oracle (16 prompts)

| metric | value |
|--------|-------|
| min cosine | **0.999520** |
| mean cosine | **0.999704** |
| token ids vs HF tokenizer | identical |
| pooling | at the EOS terminator (last token) |

### Preprocessing fidelity

Native preprocessing vs HF processor: max abs diff **5.9e-08** (float32 epsilon) on all six fixtures — bit-exact.

### Retrieval ranking

Top-1 document matches the reference for all 6 images. Ranking preserved.

---

## What is not here

| Item | Status | Where |
|------|--------|-------|
| Model weights | Not in this repo | `moeshawky/qwen3-embedding-0.6b-vl` |
| CUDA binary | Shared libraries built; final link pending | Build `LLAMA_CUDA=1` |
| Real-photograph parity | Untested | All 6 fixtures are synthetic |
| Batched API | Not implemented | One image/prompt per graph |

---

## Layout

```
native/CMakeLists.txt     build file (point LLAMA_CPP_DIR at a llama.cpp checkout)
native/src/               C++17 implementation (sidecar, preprocess, vision_graph, text_embed, main, npy)
native/tools/             oracle capture, floor measurement, comparator, scorer, exporters
native/fixtures/          six image fixtures + three content-probe images (PNG)
native/oracle/            reference tensors, floors, machine-readable reports
native/artifacts/         compat head GGUF (4.2 MB, opt-in)
native/vendor/            third-party (stb_image.h, public domain)
```

Large artifacts are referenced by digest in `HANDOFF.md` §8 and are **not** in this repo.

---

## References

- `HANDOFF.md` — state, pitfalls, how to reproduce, open items (operator-facing)
- `NATIVE_PARITY_REPORT.md` — every measured number with its controls
- `native/src/` — the C++ implementation

---

## License

Apache-2.0 via the parent models: `Qwen/Qwen3-Embedding-0.6B` and
`google/siglip2-so400m-patch16-naflex`. The native inference code in this repo carries no
additional restrictions beyond what is needed to run against those weights.