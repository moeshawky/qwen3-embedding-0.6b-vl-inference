# HANDOFF — native llama.cpp/GGML inference for `moeshawky/qwen3-embedding-0.6b-vl`

**Date:** 2026-10-21 · **Status:** image path complete and measured; text path complete and
measured; retrieval ranking preserved. One acceptance gate is **not** met on one fixture, and
the cause is proven to be the published Q8_0 sidecar rather than the native implementation.

This file is the entry point. Read it, then `native/NATIVE_PARITY_REPORT.md` for the numbers.

---

## 1. Objective and binding constraints

Produce **1024-D L2-normalised text and image embeddings** for
`moeshawky/qwen3-embedding-0.6b-vl` with **native llama.cpp/GGML** compute — no PyTorch, no
ONNX runtime — and reach a **pre-registered image cosine ≥ 0.999 across six fixtures**.

Constraints that still bind:

- No retraining. No artifact requantisation/regeneration without demonstrated corruption.
  **The published `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` sidecar is used byte-identical.**
- No ONNX wrapper substitute, no VLM token-injection as image inference.
- The published text GGUF must not be mutated. The text branch's trained "compat head" is
  supplied as a **separate opt-in artifact** (`native/artifacts/compat_head_...f32.gguf`).
- The acceptance floor is pre-registered at 0.999 and was **not** lowered.
- Original upstream docs in the repo root were not touched.

## 2. What was lost, and what survived

A previous session's entire working tree (`/content/lemonade_qmd_models/qwen3_vl_native/`,
`/tmp/opencode/qwenprobe/`, the native C++, the tools, the oracle) was **wiped**. The model
snapshot in the repo root survived, and the upstream artifacts are still published on the Hub.

Everything in `native/` was **rebuilt from scratch in this session** (native C++ implementation,
Python tools, oracle fixtures, floors, reports). The rebuild is reproducible: see §6.

## 3. What was built

| Component | Path | What it is |
|---|---|---|
| Native executable | `native/src/*.cpp`, `native/CMakeLists.txt` | `qwen3vl-embed`: image branch (full GGML graph) + text branch (llama.cpp) |
| Sidecar loader | `native/src/sidecar.{hpp,cpp}` | GGUF → dequantised F32 ggml tensors (450 tensors) |
| Native preprocessing | `native/src/preprocess.{hpp,cpp}` | target size, antialiased resize, patchify, position interpolation |
| Vision graph | `native/src/vision_graph.{hpp,cpp}` | SigLIP2 naflex tower (27 blocks) + MAP head + projection |
| Text branch | `native/src/text_embed.{hpp,cpp}` | llama.cpp Qwen3 tower + opt-in compat head |
| Oracle capture | `native/tools/capture_oracle.py` | FP32 PyTorch reference (images) |
| Floor measurement | `native/tools/q8_floor_check.py` | FP32 control + Q8-weight run → the true floor |
| Content probe | `native/tools/q8_content_probe.py` | is the floor constant or input-dependent? |
| Text oracle | `native/tools/capture_text_oracle.py` | FP32 text reference + retrieval matrix |
| Scorer | `native/tools/retrieval_score.py` | ranking preservation, with a self-test |
| Comparator | `native/tools/compare_vectors.py` | fail-closed npz parity comparator, with a self-test |
| Sidecar verifier | `native/tools/verify_sidecar_weights.py` | sidecar ↔ source weight equivalence |
| Compat head export | `native/tools/export_compat_head.py` | source checkpoint → opt-in GGUF head |

## 4. Measured results (the numbers that matter)

### 4.1 Image parity — native end-to-end vs the FP32 PyTorch oracle

| fixture | grid | valid/masked patches | native vs FP32 oracle | gate ≥0.999 | native vs PyTorch-Q8 control |
|---|---|---|---|---|---|
| square | 16×16 | 256/0 | **0.9990509950** | PASS | 0.9999998335 |
| landscape | 35×7 | 245/11 | **0.9996589770** | PASS | 0.9999998317 |
| portrait | 7×35 | 245/11 | **0.9995872285** | PASS | 0.9999998151 |
| odd | 21×12 | 252/4 | **0.9993107432** | PASS | 0.9999998592 |
| tiny | 16×16 | 256/0 | **0.9947641425** | **FAIL** | 0.9999996833 |
| wide | 64×4 | 256/0 | **0.9995823673** | PASS | 0.9999998012 |

- The native **preprocessing is bit-exact** with the HF processor on all six fixtures
  (`pixel_values` max abs diff 5.9e-08 = float32 epsilon; masks identical).
- The native **graph is faithful**: against an independent PyTorch run that uses the *same Q8
  weights*, the native output matches to cosine 0.9999998 on every fixture.
- ⇒ the `tiny` shortfall is **not** an implementation defect. It is the sidecar's own floor.

### 4.2 The Q8 floor (the decisive control the previous session never ran)

`native/tools/q8_floor_check.py` loads the strictly-loaded FP32 model, swaps in the 450
dequantised sidecar weights, and re-runs. The **FP32 control is bitwise identical to the stored
oracle on all six fixtures** (max abs diff 0.0), which validates the harness.

| fixture | Q8 floor (PyTorch, same weights) | native (same weights) |
|---|---|---|
| square | 0.999050689 | 0.999050995 |
| landscape | 0.999660183 | 0.999658977 |
| portrait | 0.999589032 | 0.999587229 |
| odd | 0.999312098 | 0.999310743 |
| **tiny** | **0.994736746** | 0.994764143 |
| wide | 0.999578499 | 0.999582367 |

**The floor is not a constant of the sidecar — it depends on the input.** Controlled probe
(three 112×112 images, identical except for pixel noise):

| image | mean abs Laplacian | Q8 floor |
|---|---|---|
| `tiny_flat` (no noise) | 3.51 | 0.994106168 |
| `tiny_smooth` (noise 12) | 30.37 | 0.994736746 |
| `tiny_noisy` (+noise 60) | 128.47 | 0.999322585 |

The floor is monotone in the input's high-frequency content: the smoother the image, the worse
the Q8 parity. Mechanism (visible in the intermediates): the deviation is content-independent at
the input stage (0.999959 for all three) and diverges inside the encoder, peaking at the
`post_layernorm` LayerNorm (0.814 flat vs 0.977 noisy), where low-variance activations amplify
a fixed Q8 weight error. **Worst measured floor: 0.9941** (a very smooth image).

⇒ **The pre-registered 0.999 gate is met on 5/6 fixtures and is unreachable with the published
Q8_0 sidecar on low-frequency inputs.** Reaching it would require a higher-precision sidecar,
which the constraints forbid regenerating. This is a decision for the operator, not a coding gap.

### 4.3 Text parity

| metric | value |
|---|---|
| native text vs FP32 oracle (16 prompts) | min **0.999520**, mean **0.999704** |
| token ids vs HF tokenizer | **identical** for every prompt (EOS terminator appended) |
| compat head | applied (opt-in artifact); without it the vector differs by cos 0.99995 |
| pooling | at the EOS terminator (the last token), matching the frozen recipe |

The text weights in the published text GGUF are **bit-identical** to the composite checkpoint's
text branch (verified by swapping all 310 dequantised tensors into the FP32 model and observing
cosine 1.000000). The residual 3e-4 is the llama.cpp compute path (F16 KV cache / flash
attention) versus PyTorch FP32.

### 4.4 Retrieval ranking

`RANKING_PRESERVED: yes` — the top-1 document matches the reference for all six images.
Mean cosine: text 0.999704, image 0.998659. One tie-order flip in `rank_of_correct` for `wide`
(4 vs 5) between two near-equal documents; all six fixtures share one synthetic recipe, so
that verdict is weak by construction (see `native/oracle/Q8_FLOOR_REPORT.md` for the noise sweep
that calibrates it: target mean text cosine ≳ 0.9995).

## 5. Pitfalls discovered (expensive; do not re-learn)

1. **ggml's allocator reuses a consumed intermediate's buffer.** A tensor read back after
   `ggml_backend_graph_compute` is garbage unless it is a *leaf* of the graph. Fix: `ggml_dup`
   every tensor you intend to read, then `ggml_build_forward_expand` the copy.
2. **`ggml_norm` is a LayerNorm**, not an RMS/L2 norm: it subtracts the mean. For L2
   normalisation use `sqr → sum → sqrt → div`. For LayerNorm it is exactly right (biased
   variance, `eps` inside the sqrt).
3. **The attention softmax runs over `ne[0]`.** `mul_mat(k, q)` yields `ne0 = kv`, so the mask
   must be shaped `{kv, 1, 1}` and added *before* the softmax.
4. **The V operand of an attention needs a transposed, contiguous copy**
   (`ggml_cont(ggml_transpose(v_view))`): the contraction dimension must be contiguous.
5. **PyTorch's antialiased bilinear filter** (used by the position-embedding interpolation) is
   a triangle with `support = max(in/out, 1)`, centre `(j+0.5)*s − 0.5`, weights
   `max(0, 1 − |i−centre|/support)`, renormalised over the taps inside the image. Derived from
   one-hot probes and verified exactly.
6. **Torchvision resizes 8-bit images in two passes and rounds to 8 bit after EACH pass**
   (horizontal → round → vertical → round). Rounding only at the end is wrong by ±1 on ~20 % of
   the pixels.
7. **`llama_decode` does not clear the KV cache.** Embedding a second prompt without
   `llama_memory_clear(llama_get_memory(ctx), true)` makes it attend across prompts: prompt 0
   was correct and every later prompt was wrong (cos 0.53–0.83). This was the whole text-parity
   bug.
8. The HF tokenizer's **noNFC** recipe matters: assert
   `tok.backend_tokenizer.normalizer is None`. Pooling one token early costs ~0.45–0.72 cosine.

## 6. Reproducing everything

```bash
# 0. prerequisites: a llama.cpp checkout at the pinned commit
git clone https://github.com/ggml-org/llama.cpp.git /tmp/opencode/llamacpp
cd /tmp/opencode/llamacpp && git fetch --depth 1 origin 0a1ef309cdbb9f2024f58a38fb4c42352c5db569 && git checkout FETCH_HEAD

# 1. build the native executable (2 cores: ~6 min)
cd /content/qwen3-embedding-0.6b-vl/native
cmake -B build2 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=/tmp/opencode/llamacpp
cmake --build build2 -j2 --target qwen3vl-embed

# 2. image branch, one fixture
./build2/qwen3vl-embed \
  --mmproj ../mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf \
  --image fixtures/square.png --out /tmp/out --threads 2
# writes /tmp/out/square_{image_embeds,post_layernorm,encoder_layer_0..26,map_pooler,
#   map_pooler_norm,linear_projection,patch_plus_position,pixel_values,pixel_attention_mask}.npy

# 3. text branch
./build2/qwen3vl-embed \
  --text-model ../qwen3-embedding-0.6b-q8_0.gguf \
  --compat-head artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf \
  --text "a photo of a cat" --stem text --out /tmp/out

# 4. regenerate the reference + floors (needs the T4; torch 2.11, transformers 5.20)
python3 native/tools/capture_oracle.py
python3 native/tools/q8_floor_check.py
python3 native/tools/q8_content_probe.py
python3 native/tools/capture_text_oracle.py
python3 native/tools/retrieval_score.py --text-npz <native text npz> --image-npz <native npz> \
  --reference native/oracle/retrieval_reference.npz
```

Quality gate (must stay green):

```bash
uvx ruff check --select E,F,I,UP,B,SIM,RUF,S,C4,RET,PTH,TC,FA,FURB --ignore E501 native/tools/
uvx ruff format --check native/tools/
uvx --with numpy --with safetensors mypy --strict --python-executable /usr/bin/python3 native/tools/
```

## 7. Open items / next moves

1. **Operator decision on the 0.999 gate.** The native implementation is at the sidecar's floor.
   Options: (a) accept the measured floor and document it; (b) publish a higher-precision sidecar
   (needs the "no requantisation" constraint relaxed); (c) keep the 0.999 gate and re-scope the
   fixtures to inputs where the Q8 floor allows it. **Do not silently lower the threshold.**
2. **Real photographs.** Every fixture here is synthetic. The parity report in the repo root
   says the same for the ONNX path. Natural-image parity is UNKNOWN and should be measured
   before any deployment claim.
3. **The `tiny` failure is content-driven, not fixture-driven.** Extend the content probe
   (n=3 today) with a frequency→floor curve over more images before generalising the mechanism.
4. **The retrieval verdict is weak by construction** (all fixtures share one recipe → top-1 is
   prompt 0 for every image). Build a fixture set with genuinely distinct content, then re-score.
5. **Batching.** The native image path is one image per graph; the text path is one prompt per
   decode. A batched API would be the next engineering step.
6. **Performance.** ~13 s per image on 2 CPU cores (GGML, F32 compute). The F16/quantised
   compute path was not explored; a GPU or a smaller context would change this materially.
7. **`native/build` and `native/build2/`** are build trees, not deliverables.

## 8. Repository map

```
native/
  CMakeLists.txt            build file (LLAMA_CPP_DIR points at a llama.cpp checkout)
  src/                      the native implementation (C++17, ggml + llama.cpp)
  tools/                    oracle capture, floors, comparator, scorer, exporters
  fixtures/                 six image fixtures + three content-probe images (PNG)
  oracle/                   reference tensors, floors, reports (see NATIVE_PARITY_REPORT.md)
  artifacts/                the opt-in compat head GGUF (4.2 MB)
  vendor/stb/stb_image.h    third-party image decoder (public domain)
NATIVE_PARITY_REPORT.md     every measured number, with the controls
HANDOFF.md                  this file
```

Large artifacts are **not** in this repo; they are published upstream and referenced by digest:

| artifact | sha256 |
|---|---|
| `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` (458 MB) | `9e3bb2fbfbc1dd7f43706a70c26534315ad886b0636f90a471391ea6a9c3782a` |
| `qwen3-embedding-0.6b-q8_0.gguf` (639 MB) | `ee029816fb9615e361d9e444abc0059827322e665c3ed9e6bc0531efe20cddda` |
| `model.safetensors` (4.1 GB) | `fd8469d3604876efe2ea92750c7c9fd5a2ab85c4508de9287334c91c566e6a05` |
| `compat_head_qwen3_embedding_0.6b.f32.gguf` (4.2 MB) | `f0566e62102ba836257c944eea9199d50f3a7be04af85fb5c18230cd2528f65e` |

Environment of the measurements: CPU-only (2 cores) for the native path, CUDA Tesla T4 for the
PyTorch references; torch 2.11.0+cu130, transformers 5.20.1, torchvision 0.26.0, numpy 2.1.3;
llama.cpp pinned at `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`; GCC 13.3.0, CMake 3.31.10.
