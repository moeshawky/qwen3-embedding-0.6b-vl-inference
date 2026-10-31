# Native parity report — `moeshawky/qwen3-embedding-0.6b-vl`

Native llama.cpp/GGML inference for the dual-modality embedding model, measured against a
PyTorch FP32 oracle. Every number below was produced by an executed command; the controls that
make each number interpretable are named.

**Verdict in one line:** the native implementation is faithful (it reproduces an independent
PyTorch run on the *same Q8 weights* to cosine 0.9999998); the pre-registered 0.999 image gate
is met on 5 of 6 fixtures and is **unreachable on the sixth with the published Q8_0 sidecar**,
because the sidecar's own parity floor is 0.9947 on that input and drops to 0.9941 on very
smooth images.

---

## 1. What was measured

| # | Probe | Claim it supports | Control |
|---|---|---|---|
| P1 | native vs FP32 PyTorch oracle, 6 fixtures, end-to-end from PNG | end-to-end parity of the native path | — |
| P2 | native `pixel_values` vs the processor's | preprocessing fidelity | float32 epsilon agreement |
| P3 | native vs PyTorch **with the same Q8 weights** | the native graph is faithful (isolates the weights) | FP32 control = 1.0 bitwise |
| P4 | PyTorch with Q8 weights vs FP32 oracle | the sidecar's floor | FP32 control = 1.0 bitwise |
| P5 | three images differing only in noise | is the floor constant or input-dependent? | negative control (FP32 weights → 1.0) |
| P6 | native text vs FP32 text oracle, 16 prompts | text parity | token ids identical to HF |
| P7 | native retrieval matrix vs reference | ranking preservation | scorer `--self-test` |

## 2. P1/P2 — image parity and preprocessing

| fixture | source px | grid (h×w) | valid/masked | `pixel_values` max abs diff | native vs oracle | gate |
|---|---|---|---|---|---|---|
| square | 256×256 | 16×16 | 256/0 | 5.9e-08 | 0.9990509950 | PASS |
| landscape | 560×112 | 35×7 | 245/11 | 5.9e-08 | 0.9996589770 | PASS |
| portrait | 112×560 | 7×35 | 245/11 | 5.9e-08 | 0.9995872285 | PASS |
| odd | 336×192 | 21×12 | 252/4 | 5.9e-08 | 0.9993107432 | PASS |
| tiny | 112×112 | 16×16 | 256/0 | 5.9e-08 | 0.9947641425 | **FAIL** |
| wide | 1024×64 | 64×4 | 256/0 | 5.9e-08 | 0.9995823673 | PASS |

Preprocessing is bit-exact (5.9e-08 is one float32 ulp on the normalised scale) on **all six**,
including the upscaled `tiny` fixture. The mask matches exactly everywhere.

## 3. P3/P4 — the native graph is faithful; the sidecar is the limit

| fixture | native vs PyTorch-Q8 (P3) | PyTorch-Q8 vs FP32 (P4) = floor | native vs FP32 (P1) |
|---|---|---|---|
| square | 0.9999998335 | 0.999050689 | 0.9990509950 |
| landscape | 0.9999998317 | 0.999660183 | 0.9996589770 |
| portrait | 0.9999998151 | 0.999589032 | 0.9995872285 |
| odd | 0.9999998592 | 0.999312098 | 0.9993107432 |
| tiny | 0.9999996833 | **0.994736746** | 0.9947641425 |
| wide | 0.9999998012 | 0.999578499 | 0.9995823673 |

P3 and P4 agree to ~1e-6 on every fixture, including the failing one. The native path and the
PyTorch path compute the same function of the same weights; the deviation from the FP32 oracle
is the sidecar's quantisation, not the implementation.

**Control for P3/P4:** the FP32 run (no weight swap) is **bitwise identical** to the stored
oracle on all six fixtures (max abs diff 0.0), and the load is strict
(`missing_keys=[] unexpected_keys=[] mismatched_keys=[]`).

## 4. P5 — the floor is input-dependent

Three 112×112 sources, identical seed/gradient/rectangles, differing only in added pixel noise.
All three land on the same 16×16 grid with 256 valid patches.

| image | mean abs Laplacian | patch-mean std | FP32 control | Q8 floor | max abs diff |
|---|---|---|---|---|---|
| `tiny_flat` | 3.51 | 34.89 | 1.000000000 | **0.994106168** | 0.0140862 |
| `tiny_smooth` | 30.37 | 34.82 | 1.000000000 | 0.994736746 | 0.0112129 |
| `tiny_noisy` | 128.47 | 31.23 | 1.000000000 | **0.999322585** | 0.0055266 |

Floor spread 0.005216416 — larger than the spread of the six published fixtures (0.00492).
The floor orders with the Laplacian (smoother ⇒ worse); the patch-mean statistic does **not**
track it (the noisiest image has the *lowest* patch-mean std and the *best* floor).

Where the deviation enters (cosines vs FP32, same image):

| stage | flat | smooth | noisy |
|---|---|---|---|
| patch_plus_position | 0.9999589 | 0.9999590 | 0.9999598 |
| encoder_layer_26 | 0.967749 | 0.986132 | 0.995346 |
| post_layernorm | 0.814407 | 0.929593 | 0.977493 |
| map_pooler | 0.993430 | 0.995711 | 0.999408 |

Content-independent at the input, diverging inside the encoder, peaking at the `post_layernorm`
LayerNorm — where a low-variance activation amplifies a fixed Q8 weight error.

Negative control: the same evaluation on `tiny_noisy` **without** the weight swap returns
cosine 1.000000000, so the probe discriminates Q8 weights from FP32 weights.

**Practical envelope:** with the published Q8_0 sidecar, image parity is ≈0.9993–0.9997 on
textured inputs and degrades to ≈0.994 on very smooth ones. A 0.999 floor is therefore an
input-conditional promise, not a property of the artifact.

## 5. P6 — text parity

| metric | value |
|---|---|
| native vs FP32 oracle, 16 prompts | min 0.999520, mean 0.999704 |
| token ids | identical to the HF tokenizer (content + EOS terminator) |
| pooling | at the terminator (last token) — the frozen recipe |
| compat head | applied from the opt-in artifact; omitting it changes the vector by cos 0.99995 |
| text weights | the published text GGUF's weights are **bit-identical** to the composite checkpoint's text branch (swapping all 310 dequantised tensors into the FP32 model changes nothing: cos 1.000000) |

The residual ~3e-4 is the llama.cpp compute path (F16 KV cache, flash attention) versus PyTorch
FP32 — not a weight or tokenisation difference.

## 6. P7 — retrieval ranking

| metric | value |
|---|---|
| top-1 document per image | matches the reference for 6/6 |
| `RANKING_PRESERVED` | yes |
| mean cosine, text | 0.999704 |
| mean cosine, image | 0.998659 |
| max abs matrix delta | 1.34e-02 |
| min top-1 margin | 1.49e-02 |

`rank_of_correct` flips for `wide` (4 vs 5): all six fixtures share one synthetic recipe, so
prompt 0 legitimately describes every image and the top-1 verdict is weak by construction. The
scorer's noise sweep (200 trials) puts the discriminating threshold at mean text cosine
≳ 0.9995 — the native path clears it at 0.999704.

## 7. Reproducibility and provenance

- Oracle npz files are byte-reproducible (pinned zip member timestamps); two fresh-process runs
  produce identical sha256 for every artifact.
- The PyTorch references were captured with TF32 disabled, float32, CUDA; the native path is
  CPU-only GGML.
- The six published fixtures are synthetic. **Real photographs are untested** — the same
  limitation the repository's own ONNX parity report records.
- Environment: torch 2.11.0+cu130, transformers 5.20.1, torchvision 0.26.0, numpy 2.1.3,
  llama.cpp `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`, GCC 13.3.0, CMake 3.31.10, CPU-only
  native build, Tesla T4 for the references.

Machine-readable companions: `native/oracle/q8_floor_report.json`,
`native/oracle/content_probe_report.json`, `native/oracle/manifest.json`,
`native/oracle/text_oracle_manifest.json`.
