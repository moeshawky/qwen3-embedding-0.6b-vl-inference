---
license: other
private: true
tags:
  - ggml
  - llama.cpp
  - native-inference
  - siglip2
  - qwen3-embedding
---

# qwen3-embedding-0.6b-vl — native (GGML) inference

**Private work-in-progress repository.** It contains the native llama.cpp/GGML implementation
for [`moeshawky/qwen3-embedding-0.6b-vl`](https://huggingface.co/moeshawky/qwen3-embedding-0.6b-vl),
the measurement harness that validates it, and the full handoff.

> **Start with [`HANDOFF.md`](HANDOFF.md)** (state, pitfalls, how to reproduce, open items) and
> then [`NATIVE_PARITY_REPORT.md`](NATIVE_PARITY_REPORT.md) (every measured number with its control).

## What this is

A standalone C++17 program (`native/src`) that produces **1024-D L2-normalised text and image
embeddings** for the model using **GGML only** — no PyTorch, no ONNX runtime at inference time:

- **Image branch** — the SigLIP2 naflex vision tower (27 blocks, 16 heads × 72 dims), the
  multi-head attention pooling head, the 1152→1024 vision projection, and the *whole* image
  preprocessing (target-size search, antialiased resize, patchify, position-embedding
  interpolation) implemented natively. Weights come from the published Q8_0 `mmproj` sidecar,
  dequantised to F32 at load time so the compute is full FP32.
- **Text branch** — the published Qwen3-Embedding-0.6B text tower executed by llama.cpp, plus
  the model's trained 1024→1024 "compat head" supplied as a **separate opt-in artifact**
  (the published text GGUF is never mutated).

## Measured status

| check | result |
|---|---|
| native image parity vs the FP32 PyTorch oracle (6 fixtures) | 0.99905 – 0.99966 on 5 fixtures; **0.99476 on `tiny`** |
| native preprocessing vs the HF processor | **bit-exact** on all 6 fixtures (max abs diff 5.9e-08) |
| native graph vs an independent PyTorch run on the same Q8 weights | **0.9999998** (faithful) |
| the sidecar's own Q8 floor (PyTorch control, FP32 control bitwise = 1.0) | 0.99931 – 0.99966; **0.99474 on `tiny`**; worst measured 0.99411 on a very smooth image |
| native text parity vs the FP32 oracle (16 prompts) | min 0.99952, mean 0.99970 |
| retrieval ranking vs the reference | top-1 preserved for 6/6 images |
| pre-registered image gate (cosine ≥ 0.999, six fixtures) | **5/6 pass**; the sixth is limited by the published sidecar, not by this implementation |

**The honest summary:** the native code reproduces the sidecar's semantics; the residual
deviation from the FP32 reference is the Q8_0 quantisation of the published artifact, and it is
**input-dependent** (smoother images ⇒ worse parity). See §4 of the parity report.

## What is *not* here

The large artifacts are published upstream and referenced by digest in
[`HANDOFF.md` §8](HANDOFF.md):

| artifact | sha256 |
|---|---|
| `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` (458 MB) | `9e3bb2fbfbc1dd7f43706a70c26534315ad886b0636f90a471391ea6a9c3782a` |
| `qwen3-embedding-0.6b-q8_0.gguf` (639 MB) | `ee029816fb9615e361d9e444abc0059827322e665c3ed9e6bc0531efe20cddda` |
| `model.safetensors` (4.1 GB) | `fd8469d3604876efe2ea92750c7c9fd5a2ab85c4508de9287334c91c566e6a05` |

## Layout

```
native/CMakeLists.txt   build file (point LLAMA_CPP_DIR at a llama.cpp checkout)
native/src/             the native implementation
native/tools/           oracle capture, floor measurement, comparator, scorer, exporters
native/fixtures/        six image fixtures + three content-probe images
native/oracle/          reference tensors, floors, machine-readable reports
native/artifacts/       the opt-in compat head GGUF
```

Build and run instructions, the discovered pitfalls, and the open items are in
[`HANDOFF.md`](HANDOFF.md).
