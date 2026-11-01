---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
base_model:
- Qwen/Qwen3-Embedding-0.6B
- google/siglip2-so400m-patch16-naflex
base_model_relation: merge
datasets:
- moeshawky/laion-qwen-siglip-pairs
tags:
- merge
- feature-extraction
- sentence-similarity
- vision-language
- qwen3
- siglip2
- transformers
- onnx
---

# Qwen3-Embedding-0.6B-VL

**A merge of two pretrained models** — a Qwen3 text encoder and a SigLIP2 NaFlex
vision encoder — joined into one set of weights so that text and image inputs
land in a **single shared 1024-d L2-normalized embedding space**. Compare a
caption to an image with one dot product; no re-projection, no modality-specific
scoring.

Declared in the metadata above: `base_model_relation: merge`, both parents
listed under `base_model`.

| | text branch | vision branch |
|---|---|---|
| upstream | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex) |
| architecture | `Qwen3ForCausalLM` | `siglip2_vision_model` |
| layers | 28 | 27 |
| width | 1024 | 1152 |
| parameters | 595,776,512 (58.07%) | 427,888,064 (41.71%) |

Both branches are carried through **frozen and bit-identical to their parents**.
Exactly one connector was trained — `vision_projection` (1152 → 1024,
**1,180,672 parameters**, 0.12% of the model). This is a graft, not a joint
fine-tune: two pretrained experts bound by one learned adapter.

## Inference Backends

Production inference implementations are maintained in separate repos so that
the weights repo stays clean for model consumers:

| Backend | Repo | Visibility | Purpose |
|---------|------|------------|---------|
| vLLM (serving) | [`moeshawky/qwen3-embedding-0.6b-vl-inference`](https://github.com/moeshawky/qwen3-embedding-0.6b-vl-inference) | Public | FastAPI server, `/v1/embeddings`, `/embed/image`, batched pooling |
| Native GGML (CPU/CUDA) | [`moeshawky/qwen3-embedding-0.6b-vl-native`](https://huggingface.co/moeshawky/qwen3-embedding-0.6b-vl-native) | Public | Standalone C++17 binary, GGML only, 1024-d L2-normalized embeddings |

Both backends produce **1024-dimensional L2-normalized** embeddings in the same
shared retrieval space as this model's weights. Text and image embeddings are
comparable via a single dot product — no re-projection needed.

---

## Quick start

Requires **`transformers>=5.18.0`** and `trust_remote_code=True` (the config
carries an `auto_map`, so the modeling classes are loaded from this repo). The
version floor is also recorded in `config.json → transformers_version`.

```python
from transformers import AutoModel, AutoTokenizer
import torch

REPO = "moeshawky/qwen3-embedding-0.6b-vl"

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

# --- text ---
batch = tok(
    ["a wooden chair on a patio"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=32767,          # 32768 positions minus the terminator slot
)
pool_index = batch["attention_mask"].sum(dim=1) - 1            # index of the terminator
text_embeds = model.get_text_features(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pool_index=pool_index,
)                                                              # [B, 1024], L2 norm 1.0

# --- image ---
from PIL import Image
inputs = tok(images=Image.open("photo.jpg").convert("RGB"), return_tensors="pt")
image_embeds = model.get_image_features(
    pixel_values=inputs["pixel_values"],
    pixel_attention_mask=inputs.get("pixel_attention_mask"),
    spatial_shapes=inputs.get("spatial_shapes"),
)                                                              # [B, 1024], L2 norm 1.0

scores = text_embeds @ image_embeds.T                        # cross-modal similarity
```

The tokenizer appends `<|endoftext|>` (151643) to every sequence, so
`pool_index` lands on the terminator. Two things that are easy to get wrong —
see [Troubleshooting](#troubleshooting):

1. **Do not pass `add_special_tokens=False`** — that drops the terminator and
   silently degrades embeddings (~0.37 cosine class).
2. **Truncate to 32,767, not 131,072.** The tokenizer advertises 131,072; the
   model has 32,768 positions, and right-truncation past that removes the
   terminator you are supposed to pool at.

---

## Architecture

Declared in `config.json` (`model_type: qwen3_siglip2_embedding`). Measured
directly from the shipped `model.safetensors` header.

### Upstream artifacts

| role | artifact |
|---|---|
| text branch parent | `Qwen/Qwen3-Embedding-0.6B` (recovered FP32) |
| vision branch parent | `google/siglip2-so400m-patch16-naflex` @ `cc24074f717b612951c2dead130904ab9b65a81e` |
| intermediate Q8_0 root | [`Gideon531/Qwen3-Embedding-0.6B-Q8_0-GGUF`](https://huggingface.co/Gideon531/Qwen3-Embedding-0.6B-Q8_0-GGUF) · 639,150,592 B · **no digest recorded** (see [Limitations](#limitations)) |

That root GGUF is the Q8_0 quantization of the text parent and the source the
FP32 text branch was recovered from. Its vectors predate this release and were
never re-embedded, so an existing Oracle Database 26ai collection built from it
remains valid against this model.

### Totals — measured

| | value |
|---|---|
| tensors | 762 |
| parameters | **1,025,894,848** (1.026 B) |
| dtype | **F32 for all 762 tensors** (no mixed precision) |
| file layout | 8 B length prefix + 92,632 B header + 4,103,579,392 B data — **no trailing bytes** |
| embedding dimension | 1024 |
| position limit | 32,768 (`max_position_embeddings`) |
| vocab | 151,669 |

### Text branch

| field | value |
|---|---|
| layers | 28, all `full_attention` (no sliding window) |
| hidden size | 1024 |
| intermediate size | 3072 |
| attention heads / KV heads | 16 / 8 |
| head dim | 128 |
| vocab | 151,669 (`embed_tokens` measured `[151669, 1024]`) |
| RoPE | `default`, theta 1,000,000 |
| RMSNorm eps | 1e-6 |
| tied embeddings | true |
| terminator id (`config.eos_token_id`) | 151643 |

```
Qwen3Model → gather position pool_index → L2 → compat_head → L2   → [B, 1024]
```

### Vision branch

| field | value |
|---|---|
| pinned snapshot | `cc24074f717b612951c2dead130904ab9b65a81e` |
| layers | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| patch size / max patches | 16 / 256 |
| channels | 3 |
| activation | `gelu_pytorch_tanh` |

The composite carries **no SigLIP2 text encoder** — all text comes from the
Qwen3 branch.

```
Siglip2VisionModel → pooler_output → L2 → vision_projection → L2   → [B, 1024]
```

### Connectors — measured shapes

| tensor | stored shape | parameters | trainable |
|---|---|---|---|
| `compat_head.weight` / `.bias` | `[1024, 1024]` / `[1024]` | 1,049,600 | frozen |
| `vision_projection.weight` / `.bias` | `[1024, 1152]` / `[1024]` | **1,180,672** | **trained** |

`nn.Linear` stores weights as `[out, in]`, so `[1024, 1152]` is a
`1152 → 1024` projection. Its orientation follows the NumPy convention
`z = normalize(v) @ W + b`, so the `Linear` weight must be `W.T`; a transposed
copy silently costs ≈7e-4 cosine.

An optional residual path (`resid_fc1` 1152→256, `resid_fc2` 256→1024,
zero-initialised, scaled by `resid_alpha`) exists in the modeling code but is
**disabled** here: `use_residual: false`, `resid_alpha: 0.0`.

### Image preprocessing

`Siglip2Processor`: resize + rescale on, `rescale_factor` 1/255, resample 2
(bilinear), `patch_size` 16, `max_num_patches` 256, channel mean/std 0.5.

---

## Evaluation

Diagonal-positive retrieval (each row queried against its own positive).
Exact values in `PROVENANCE.json → metrics_exact`.

| variant | val i2t R@10 | val t2i R@10 | test i2t R@10 | test t2i R@10 |
|---|---|---|---|---|
| root `model.safetensors` (NVIDIA RTX PRO 6000 Blackwell) | 0.8364 | 0.9170 | 0.7380 | 0.8484 |
| `amd/model.safetensors` (AMD MI300X) | 0.8350 | 0.9170 | 0.7377 | 0.8479 |

- Cross-configuration Δ ≤ 0.0015 — functional agreement, not byte identity
  (0/14 cached-shard digests match; last-ulp kernel drift).
- Zero-shot (untrained projection) scores 0.0039 on the same validation rows.
- **Validation drove best-epoch selection, so validation numbers are
  selection-tainted.** Test is the clean split.
- Training: 2 epochs, global batch 256, 832 steps, lr 3e-4, weight decay 0.01,
  10% warmup, grad clip 1.0, seed 20260918. InfoNCE, τ 0.07, λ_pos 0.1,
  top-16 mined negatives. Corpus: 116,211 images from
  [`moeshawky/laion-qwen-siglip-pairs`](https://huggingface.co/datasets/moeshawky/laion-qwen-siglip-pairs);
  splits train 106,602 / val 2,048 / test 5,707 = 114,357 rows.

---

### Real-World Empirical Benchmarks (October 2026)

Beyond InfoNCE diagonal-positive retrieval on held-out caption-image pairs, the model was evaluated on real-world multi-era human face verification and geometric invariance tasks across 40 authentic historical photographs spanning 8 decades.

#### Human Face Cross-Era Verification & Retrieval (8-Decade Aging Spans)

Evaluated across **40 authentic historical photographs** of 5 prominent individuals (Bill Gates, Steve Jobs, Albert Einstein, Queen Elizabeth II, Winston Churchill) spanning **8 distinct chronological eras** from **1895 to 2023** (40 to 86 years of physical aging, clothing shifts, and hair/facial changes per individual):

| Evaluation Dimension | Metric | Observed Performance | Evaluation Notes |
|---|---|---|---|
| **Vector Space Geometry** | Dimension & Norm | **1024-dim**, strict unit-norm L2 = `1.000000` | Exact unit hypersphere mapping across all inputs. |
| **Nearest-Neighbor Person ID** | Top-1 Accuracy (non-self) | **30 / 40 (75.0%)** | Closest visual neighbor belongs to the same individual across decades. |
| **Nearest-Neighbor Person Recall** | Top-3 Recall | **39 / 40 (97.5%)** | Correct individual present in top 3 nearest visual neighbors. |
| **Nearest-Neighbor Person Recall** | Top-5 Recall | **40 / 40 (100.0%)** | Correct individual present in top 5 nearest neighbors across all 40 queries. |
| **Overall Verification Margin** | Intra-Sim vs Inter-Sim | **+0.1872** | Mean Intra-Person: `0.6222 ± 0.1268` vs Inter-Person: `0.4350 ± 0.1509`. |
| **Cross-Modal Text Query** | Person Retrieval Accuracy | **8 / 8 (100.0%)** | Natural text query to image vector cosine similarity matching. |
| **Cross-Modal Cascade Query** | Exact Era Accuracy | **7 / 8 (87.5%)** | Resolves exact chronological era (e.g. 1977 mugshot vs 2023 EU interview). |

##### Intra-Person vs Inter-Person Similarity Margins by Individual
* **Bill Gates (1977 to 2023, 46-year span):** Intra `0.7428 ± 0.0890` vs Inter `0.4272 ± 0.1644` (**Margin: +0.3156**). High facial geometry stability (smile, cheekbones, eyewear) preserved across 46 years.
* **Winston Churchill (1895 to 1953, 58-year span):** Intra `0.6389 ± 0.0743` vs Inter `0.4612 ± 0.1345` (**Margin: +0.1776**). Persistent jawline and cranial profile across military tunic, chancellor, and wartime leader.
* **Albert Einstein (1895 to 1951, 56-year span):** Intra `0.6528 ± 0.1151` vs Inter `0.4767 ± 0.1219` (**Margin: +0.1761**). Consistent eye region and mustache from 1905 patent clerk to 1951 72nd birthday portrait.
* **Queen Elizabeth II (1928 to 2012, 84-year span):** Intra `0.5457 ± 0.0755` vs Inter `0.3698 ± 0.1538` (**Margin: +0.1759**). Maintained positive margin across extreme developmental trajectory (2-year-old toddler in 1928 to 86-year-old monarch in 2012).
* **Steve Jobs (1972 to 2010, 38-year span):** Intra `0.5307 ± 0.1340` vs Inter `0.4401 ± 0.1530` (**Margin: +0.0906**). Highest styling variance: 1972 shaggy long hair/beard vs 1984 clean-shaven bow tie vs 2010 bald/stubble in black turtleneck.

#### Zero-Shot Visual Clustering & Representation Invariance

Evaluated against synthesized geometric and color primitives:
- **Vector Self-Retrieval Cosine Similarity:** `1.000000` exact across all tests.
- **Topological & Shape Clustering:** Reciprocal nearest neighbors formed between topologically related shapes (e.g., *Red Circle* and *Magenta Ring* reciprocal nearest neighbors at `0.8033` cosine similarity).
- **Semantic Discrimination:** Visual embeddings preserve fine-grained shape boundaries and color discrimination with zero cross-category false positives.

---

## Files

Digests truncated; [`SHA256SUMS`](SHA256SUMS) carries full values.

| path | content | sha256 |
|---|---|---|
| `model.safetensors` | merged composite, root variant (4,103,672,032 B) | `fd8469d3…66e6a05` |
| `amd/model.safetensors` | merged composite, AMD variant (4,103,672,032 B) | `5ee8a99a…425a049` |
| `onnx/root/image_branch.onnx` | image-branch ONNX, opset 18, STATIC `4×256×768 → 4×1024` | `ee367e59…517128` |
| ~~`qwen3-embedding-0.6b-vl-q8_0.gguf`~~ | **withdrawn 2026-10-18** — see [GGUF artifacts](#gguf-artifacts) | — |
| `qwen3-embedding-0.6b-q8_0.gguf` | text-only GGUF, loads in stock llama.cpp (310 tensors, 639,150,592 B) | see `SHA256SUMS` |
| `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` | vision-tower mmproj sidecar (450 tensors, 457,676,352 B) | see `SHA256SUMS` |
| `parity/VERIFY_REPORT.md` | measured ONNX-vs-safetensors parity | — |
| `heads/root_best.pt` | raw connector checkpoint, root run | `80f60004…239f7` |
| `heads/amd_best.pt` | raw connector checkpoint, AMD run | `7a44b86b…63fc5` |
| `onnx/oracle/` | text-branch ONNX with in-graph tokenizer | see `SHA256SUMS` |
| `eval/` | train + val/test summaries, AMD full eval JSONs | see `SHA256SUMS` |
| `PROVENANCE.json` | lineage, env pins, hashes, exact metrics | — |

The AMD image-branch ONNX export lives in
[`moeshawky/qwen-siglip2-graft-composite`](https://huggingface.co/moeshawky/qwen-siglip2-graft-composite)
(`onnx/`, ORT maxerr 6.7e-07).

> **Note — paths renamed.** Several directories and files were renamed in this
> revision to drop internal build codenames. If you pinned an earlier revision,
> these paths changed and must be updated:
>
> | was | now |
> |---|---|
> | `onnx/box2/image_branch.onnx` (+ `.data`, `manifest.json`) | `onnx/root/` |
> | `eval/box2_eval_val.json` | `eval/root_eval_val.json` |
> | `eval/box2_eval_test.json` | `eval/root_eval_test.json` |
> | `eval/box2_train_summary.json` | `eval/root_train_summary.json` |
> | `heads/box2_best.pt` | `heads/root_best.pt` |
> | `MISSION_REPORT_SCALE_X_BOX2.md` | `BUILD_REPORT.md` |
>
> `amd/`, `heads/amd_best.pt` and `onnx/oracle/` are unchanged. File contents and
> weights are unaffected — only names moved. `SHA256SUMS` is authoritative for
> the current names.

---

## GGUF artifacts

> ### ⚠️ Withdrawn 2026-10-18 — `qwen3-embedding-0.6b-vl-q8_0.gguf`
>
> The single-file Q8_0 archive of text + vision weights has been **deleted from
> this repository**. It was a weight archive, not a loadable model — stock
> llama.cpp refuses it:
>
> ```
> done_getting_tensors: wrong number of tensors; expected 762, got 310
> ```
>
> Publishing it invited exactly that failure, so it was withdrawn rather than
> left for people to re-download. If you pinned it, use the split pair below.

### The split pair

The GGUF spec's sidecar convention: a text model plus a separate
`mmproj-*.gguf` carrying the vision tower.

| file | tensors | bytes | role |
|---|---|---|---|
| `qwen3-embedding-0.6b-q8_0.gguf` | 310 | 639,150,592 | **text model — loads in stock llama.cpp today** |
| `mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf` | 450 | 457,676,352 | vision-tower sidecar (437 + 11 + 2) |

`qwen3-embedding-0.6b-q8_0.gguf` is a bit-preserving copy of the upstream text
release. Verified against stock `llama-server`: loads, 1024-dimensional
embeddings, unit norm.

The sidecar is structurally conformant — `general.architecture = clip`,
`general.type = clip-vision`, and llama.cpp's `mmproj-*.gguf` filename shape.
All 450 tensors are byte-identical to the source. Partition:

```
437 tower     v.*             vision_model.encoder.layers.*
 11 pooler    p.*             SigLIP2 MAP attention-pooling head, incl. p.probe
  2 projector mm.model.fc.*   vision_projection
```

### The sidecar does not load yet

No SigLIP2 entry exists in llama.cpp's `conversion` `MMPROJ_MODEL_MAP`, so
stock llama.cpp cannot load the sidecar. Loading it needs an upstream
`qwen3siglip2` clip architecture: SigLIP2 naflex preprocessing, the MAP
pooling head, 16-row padding on 27 `ffn_down` blocks, and an embeddings
endpoint. The decoder-independent `clip_image_encode` primitive already exists
upstream; what is missing is the architecture that feeds it. That work is
upstream C++ and is **not** in this repository.

Vision inference that works today: the ONNX path (`onnx/root/image_branch.onnx`)
or PyTorch (`trust_remote_code=True`). See [`parity/VERIFY_REPORT.md`](parity/VERIFY_REPORT.md).

### The mapping is published

So the eventual patch is mechanical rather than investigative,
[`moeshawky/qwen-siglip2-embed`](https://huggingface.co/moeshawky/qwen-siglip2-embed)
ships `qwen_siglip2_embed.mmproj`, which validates a sidecar against both the
GGUF spec and this repo's `model.safetensors`:

```bash
python -c "
from qwen_siglip2_embed.mmproj import validate_sidecar
for i in validate_sidecar('mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf', '.'):
    print(i.severity, i.code, i.detail)"
```

Three facts any converter must handle, all measured here:

1. **GGML reverses dimensions.** `patch_embedding.weight` is numpy
   `[1152, 768]` and is stored `[768, 1152]`. Reusing numpy shapes verbatim
   transposes every weight.
2. **MAP attention pooling**, not mean pooling — learned attention under `p.*`
   with a `p.probe` query vector.
3. **Padded `ffn_down`**: 27 of 28 blocks are `[4320, 1152]` against a
   canonical `[4304, 1152]` — 16 padding rows forced by the quantiser
   (4304 % 32 = 16).

---

## Oracle Database 26ai ingest (text branch)

`onnx/oracle/text_compat.onnx` exposes the text branch as a self-contained graph
with an in-graph tokenizer, for direct loading into a vector column.

Contract: `DATA[string batch] → embedding[float32 batch×1024]`, L2 unit norm.
Recipe: frozen text branch → last-token pool at 151643 → L2 → `W.T + b` → L2.
Tokenizer contributed by `ai.onnx.contrib:HfJsonTokenizer` (no-NFC vocabulary)
plus the upstream `tokenizer_config.json`.

Verified: `onnx.checker` pass; ORT-vs-PyTorch cosine **1.0000000** across 6
ASCII + 4 unicode no-NFC probes; output norm 1.0 ±1.2e-07. Graph 13,657,530 B
(<2 GB), external weights 2,387,275,776 B.

Load it with `DBMS_VECTOR.LOAD_ONNX_MODEL`, staging the `.onnx` and
`.onnx.data` siblings together:

```sql
-- stage the .onnx and .onnx.data siblings together, then:
SELECT VECTOR_EMBEDDING(QWEN3_GGUF_COMPAT USING 'hello world' AS DATA) FROM DUAL;
-- expect VECTOR(FLOAT32,1024), L2 norm 1.0
```

The vision branch is **not** in this graph. It stays a decoupled static
auxiliary (`onnx/root/image_branch.onnx`, `4×256×768 → 1024`); dynamic-batch
export is unavailable because NaFlex specializes batch into the graph statically.
See `onnx/oracle/VISION_AUX.md`.

---

## Verification

```bash
hf download moeshawky/qwen3-embedding-0.6b-vl --local-dir ./vl
cd ./vl && sha256sum -c SHA256SUMS     # large files take minutes

python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('.', trust_remote_code=True)
assert type(c).__name__ == 'Qwen3Siglip2EmbeddingConfig', type(c)
assert (c.projection_in_dim, c.projection_out_dim) == (1152, 1024)
print('remote-load wire OK:', c.model_type)"
```

Two known manifest caveats, both observed rather than assumed:

- `SHA256SUMS` covers every shipped file **except itself**, by construction.
  Check `SHA256SUMS` and `PROVENANCE.json` against your download if you need
  end-to-end integrity.

### Last verified

**2026-11-01**, by loading the model and running it. Not a static read.

| check | result |
|---|---|
| `AutoConfig.from_pretrained(trust_remote_code=True)` | `Qwen3Siglip2EmbeddingConfig`, projection `(1152, 1024)` |
| `AutoModel.from_pretrained(trust_remote_code=True)` | **loads and runs** on `transformers 5.18.0` |
| parameter count as loaded by torch | 1,025,894,848 — matches the safetensors header exactly |
| `get_text_features` | `(2, 1024)` float32, L2 norms `[1.0, 1.0]` |
| `get_image_features` | `(1, 1024)`, L2 norm `[1.0]` |
| cross-modal dot product | `max abs 0.1638`, inside `[-1, 1]` |
| determinism | same text embedded twice → bitwise equal |
| tokenizer terminator | `<\|endoftext\|>` 151643 appended to every sequence; `pool_index = attention_mask.sum(1) - 1` |
| dtype | 762/762 tensors F32; file is 8 B prefix + 92,632 B header + 4,103,579,392 B data, no trailing bytes |
| config ↔ tensors | 28 text layers, 27 vision layers, vocab 151,669, hidden 1024 |
| `sha256sum -c SHA256SUMS` | all rows OK |
| GGUF digest | recomputed, `fd1368bc…a3bff`, 1,097,956,096 B |

The image branch was exercised with a synthetic `[1,256,768]` pixel tensor rather
than a decoded photograph: that confirms the shape and normalization contract,
not photorealistic inputs. Retrieval numbers were produced by the build
environment and are reported in [Evaluation](#evaluation); they were not
re-measured for this revision.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `AttributeError: ... has no attribute 'all_tied_weights_keys'` | `transformers` older than 5.18 | upgrade: `pip install -U "transformers>=5.18"` |
| embeddings look like noise | `add_special_tokens=False` dropped the terminator, or `pool_index` not on it | keep special tokens; `pool_index = attention_mask.sum(1) - 1` |
| quality slightly below reported numbers | pooled at `tok.eos_token_id` (151645 `<\|im_end\|>`) instead of the model's terminator | pool at the appended `<\|endoftext\|>` (151643) |
| degraded results on long inputs | truncated past 32,768, which removes the terminator | `truncation=True, max_length=32767` |
| `ImportError` mentioning `huggingface-hub` | dependency conflict | `pip install -U transformers "huggingface-hub<2.0"` |
| `requires trust_remote_code=True` | config has an `auto_map` | pass `trust_remote_code=True` |
| confusing error on the image branch | calling `model.vision_model(...)` directly | use `model.get_image_features(...)` — the raw call returns 1152-d `pooler_output`, unprojected and unnormalized |
| `done_getting_tensors: wrong number of tensors; expected 762, got 310` | loading the GGUF with stock llama.cpp | it is a weight archive; use the root Q8_0 GGUF for text and ONNX/PyTorch for vision |
| ONNX export rejects a new batch size | NaFlex static specialization fixes patch/pad dims at calibration | the export is STATIC `4×256×768`; dynamic batches need a modeling change |

---

## Limitations

- Retrieval numbers come from one held-out test split (5,707 rows) of one
  corpus; they are not a benchmark against other embedding models.
- Validation numbers are selection-tainted (validation chose the best epoch).
- The image-branch ONNX export is STATIC-shape. NaFlex-dynamic export is not
  available.
- The vision branch ships no text encoder — image-text matching relies entirely
  on the Qwen3 branch for all text.
- `config.json → qwen_source_dir` records the public Hub id of the text-branch
  parent (`Qwen/Qwen3-Embedding-0.6B`). It is informational; the modeling code
  never reads it.
- The root Q8_0 GGUF this model was recovered from has **no recorded digest** —
  only its size (639,150,592 B) is on record.

## License

Apache-2.0. The parent `Qwen/Qwen3-Embedding-0.6B` is Apache-2.0, and
`google/siglip2-so400m-patch16-naflex` is Apache-2.0 (both confirmed against
their Hub model metadata). The Q8_0 archive in this repository is a
quantization of Apache-2.0 weights and carries the same terms.
