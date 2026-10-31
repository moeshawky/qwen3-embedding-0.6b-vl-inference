"""PyTorch oracle capture for the Qwen3-SigLIP2-VL TEXT branch.

What this produces
------------------
The reference (ground-truth) tensors for the text tower plus the retrieval
ranking protocol the native (C++) text path is diffed against.

Artifacts written (all paths relative to ``/content/qwen3-embedding-0.6b-vl``):

* ``native/oracle/text_oracle.npz``         -- the text-tower oracle (see below)
* ``native/oracle/retrieval_reference.npz`` -- the retrieval ranking reference

``text_oracle.npz`` keys
-----------------------
=========================  =========================  ==================================
key                       shape / dtype              meaning
=========================  =========================  ==================================
``prompts``               (16,) ``<U``               the 16 reference prompts
``text_embeds``           (16, 1024) ``float32``     FULL reference path:
                                                     L2 -> compat_head -> L2
``text_embeds_no_head``   (16, 1024) ``float32``     the SAME computation with the
                                                     compat head removed
``text_embeds_pool_off_by_one``
                          (16, 1024) ``float32``     DISCRIMINATOR WITNESS: the
                                                     same recipe pooled at
                                                     ``n_content - 1`` (the last
                                                     content token) instead of at
                                                     the terminator
``token_ids``             (16, L) ``int64``          per-prompt token ids, right-padded
                                                     with the terminator (151643)
``token_lengths``         (16,) ``int64``           content length per prompt
``pool_index``            (16,) ``int64``           pooling index (== terminator
                                                     position == content length)
``eos_id``                () ``int64``               the terminator id
=========================  =========================  ==================================

``retrieval_reference.npz`` keys
-------------------------------
=========================  =========================  ==================================
key                       shape / dtype              meaning
=========================  =========================  ==================================
``prompts``               (16,) ``<U``               the 16 reference prompts
``fixture_names``         (6,) ``<U``                square,landscape,portrait,
                                                     odd,tiny,wide
``matrix``                (6, 16) ``float32``        cosine(image_i, text_j) with
                                                     the compat head (reference)
``best_index``            (6,) ``int64``             argmax_j matrix[i, j]
``best_prompt``           (6,) ``<U``                prompts[best_index]
``rank_of_correct``       (6,) ``int64``             1-based rank of the prompt
                                                     written for that fixture
``correct_prompt_index``  (6,) ``int64``             the prompt index per fixture
``text_embeds``           (16, 1024) ``float32``     reference text embeddings
                                                     (self-contained reference)
``image_embeds``          (6, 1024) ``float32``      reference image embeddings
``text_embeds_no_head``   (16, 1024) ``float32``     reference text embeddings
                                                     without the compat head
``matrix_no_head``        (6, 16) ``float32``        the no-head cosine matrix
``best_index_no_head``    (6,) ``int64``             argmax of the no-head matrix
``rank_of_correct_no_head`` (6,) ``int64``           1-based correct-prompt rank
=========================  =========================  ==================================

Numerical contract
------------------
* float32 everywhere, on CUDA, under ``torch.no_grad()``; never bf16/fp16.
* TF32 is pinned OFF (matmul and cuDNN) and the flags are reported.
* The load is strict: non-empty ``missing_keys``/``unexpected_keys``/
  ``mismatched_keys`` raise. (The check is an explicit ``raise``, not an
  ``assert`` -- ``assert`` is stripped under ``python -O`` and is also a
  lint error (``S101``) in this repository.)
* Tokenisation is the FROZEN recipe from
  ``processing_qwen_siglip2_embedding.py``: HF-fast tokenizer with
  ``normalizer = None`` (noNFC), content truncated to 32767 tokens, the
  terminator (151643) appended, right-padding with the terminator, and
  pooling AT the terminator index (``n_content``, not ``n_content - 1``).
  This module does NOT re-implement that recipe; it imports it.
* The text side is the DOCUMENT side of the retrieval pair, so it is encoded
  with the query prefix disabled (``prefix=""``).
* The ``.npz`` files are written with a deterministic zip writer (fixed entry
  timestamps) so their sha256 is stable across runs; the layout is identical
  to ``numpy.savez`` (zip of ``<key>.npy`` members, ZIP_STORED).

Usage
-----
    python3 native/tools/capture_text_oracle.py
    python3 native/tools/capture_text_oracle.py --out-dir /tmp/oracle
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib import format as npy_format
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    import numpy.typing as npt
    from transformers import PreTrainedTokenizerFast

MODEL_DIR = Path("/content/qwen3-embedding-0.6b-vl")
DEFAULT_ORACLE_DIR = MODEL_DIR / "native" / "oracle"

#: The trust_remote_code modeling file imports its sibling modules from the
#: model directory.  When this script is launched by path, ``sys.path[0]`` is
#: this script's directory -- not the model directory -- so the model directory
#: is prepended here.  The frozen processing module is then imported
#: dynamically (see :func:`load_processing_module`) so that the dependency is
#: explicit at runtime and no static import path is invented for the linter.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

#: The 16 reference prompts.  Index 0..5 are the prompts written for the six
#: image fixtures (see :data:`CORRECT_PROMPT_INDEX`); 6..15 are distractors
#: that are semantically distinct from every other prompt and from the
#: fixtures, so the retrieval ranking is not degenerate.
PROMPTS: tuple[str, ...] = (
    "a synthetic square gradient image with red, green and blue rectangles",
    "a wide landscape photograph of a green field next to a blue sky",
    "a tall portrait photograph of a red brick wall",
    "an odd sized image with an unusual aspect ratio",
    "a tiny blurry low resolution image",
    "an extremely wide panoramic strip",
    "a photo of a cat",
    "a photo of a dog",
    "a diagram with text and boxes",
    "a chart showing a bar graph",
    "a screenshot of a terminal window",
    "a plate of spaghetti on a table",
    "a snowy mountain landscape",
    "a city street at night with neon signs",
    "a close up of a sunflower",
    "a book cover with a title on it",
)

#: Fixture order: square, landscape, portrait, odd, tiny, wide.  This is the
#: row order of every matrix in the retrieval reference and in the scorer.
FIXTURE_NAMES: tuple[str, ...] = (
    "square",
    "landscape",
    "portrait",
    "odd",
    "tiny",
    "wide",
)

#: The prompt written for each fixture (index into :data:`PROMPTS`).
CORRECT_PROMPT_INDEX: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: :data:`CORRECT_PROMPT_INDEX` as an int64 array (the ranking helpers take one).
CORRECT_PROMPT_INDEX_ARRAY: npt.NDArray[np.int64] = np.array(
    CORRECT_PROMPT_INDEX, dtype=np.int64
)

#: Embedding width of the shared 1024-d space.
EMBED_DIM = 1024


class TextModelProtocol(Protocol):
    """Structural view of the ``trust_remote_code`` model used by this harness.

    Mirrors ``Qwen3Siglip2EmbeddingModel`` in ``modeling_qwen_siglip2_embedding.py``
    for the text branch only.
    """

    text_model: nn.Module

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_index: torch.Tensor,
    ) -> torch.Tensor: ...


class ProcessingProtocol(Protocol):
    """Structural view of the frozen ``processing_qwen_siglip2_embedding``."""

    EOS: int
    LMAX_CONTENT: int

    def encode_texts(
        self, tok: PreTrainedTokenizerFast, texts: list[str], prefix: str = ...
    ) -> list[list[int]]: ...

    def terminator_aware_batch(
        self, batch_of_ids: list[list[int]], eos: int = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_deterministic(
    path: Path, arrays: dict[str, npt.NDArray[np.generic]]
) -> None:
    """Write an ``.npz`` with ``numpy.savez`` layout but reproducible bytes.

    ``numpy.savez`` stamps every zip member with the current time, so the file
    bytes (and any sha256 over them) change on every run even when the arrays
    are bit-identical.  Pinning the member timestamp to the zip epoch keeps the
    artifact reproducible; the payload format is unchanged.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(
                filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as handle:
                npy_format.write_array(handle, array, allow_pickle=False)  # type: ignore[no-untyped-call]


def load_processing_module() -> ProcessingProtocol:
    """Import the FROZEN tokenisation recipe from the model directory.

    The recipe is *not* re-implemented here.  It is imported dynamically
    because the module lives in the model directory, not next to this script.
    """
    module: ModuleType = importlib.import_module("processing_qwen_siglip2_embedding")
    return cast("ProcessingProtocol", module)


def load_model(device: torch.device) -> tuple[TextModelProtocol, dict[str, Any]]:
    """Load the model strictly in float32 and move it to ``device``."""
    from transformers import AutoModel

    model, loading_info = AutoModel.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    missing = sorted(loading_info.get("missing_keys") or [])
    unexpected = sorted(loading_info.get("unexpected_keys") or [])
    mismatched = sorted(loading_info.get("mismatched_keys") or [])
    print(
        f"[load] strict=True missing_keys={missing} unexpected_keys={unexpected} "
        f"mismatched_keys={mismatched}"
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "non-strict load: "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    moved = model.to(device=device, dtype=torch.float32).eval()
    return cast("TextModelProtocol", moved), loading_info


def load_tokenizer() -> PreTrainedTokenizerFast:
    """Load the HF-fast tokenizer and pin the noNFC recipe.

    ``tok.backend_tokenizer.normalizer = None`` is the frozen invariant; the
    processing module asserts it again inside ``encode_texts``.  The check
    below is an explicit ``raise`` (not ``assert``) so it survives
    ``python -O`` and stays lint-clean under ``S101``.
    """
    from transformers import AutoTokenizer

    tok = cast(
        "PreTrainedTokenizerFast",
        AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True),
    )
    if not tok.is_fast:
        raise RuntimeError("tokenizer is not a fast tokenizer (no backend_tokenizer)")
    tok.backend_tokenizer.normalizer = None
    if tok.backend_tokenizer.normalizer is not None:
        raise RuntimeError(
            "tokenizer normalizer must be None (noNFC recipe) -- got "
            f"{tok.backend_tokenizer.normalizer!r}"
        )
    print("[tokenizer] backend normalizer is None (noNFC recipe pinned)")
    return tok


def normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L2-normalise each row, guarding against zero-length rows."""
    norms: npt.NDArray[np.float64] = np.asarray(
        np.linalg.norm(matrix, axis=1, keepdims=True), dtype=np.float64
    )
    safe: npt.NDArray[np.float64] = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe


def cosine_matrix(
    queries: npt.NDArray[np.float64], docs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Cosine similarity of every query against every document."""
    return normalize_rows(queries) @ normalize_rows(docs).T


def rank_of(
    matrix: npt.NDArray[np.float64], correct_index: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return (best_index, 1-based rank of the correct document) per query.

    Ties are broken towards the lower document index (``kind="stable"`` on a
    negated key), which matches ``argmax`` semantics.
    """
    order = np.argsort(-matrix, axis=1, kind="stable")
    best_index = order[:, 0].astype(np.int64)
    ranks = np.empty(matrix.shape[0], dtype=np.int64)
    for row, correct in enumerate(correct_index):
        hits = np.nonzero(order[row] == correct)[0]
        if hits.size == 0:
            raise RuntimeError(f"correct document {correct} missing from the ranking")
        ranks[row] = int(hits[0]) + 1
    return best_index, ranks


def capture_text_embeddings(
    model: TextModelProtocol,
    tok: PreTrainedTokenizerFast,
    processing: ProcessingProtocol,
    prompts: Sequence[str],
    device: torch.device,
) -> dict[str, npt.NDArray[np.generic]]:
    """Encode the prompts and capture both text-tower variants."""
    ids = processing.encode_texts(tok, list(prompts), prefix="")
    input_ids, attention_mask, pool_index = processing.terminator_aware_batch(ids)
    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)
    pool_index = pool_index.to(device=device)

    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_index=pool_index,
        )
        # Discriminator witness: the same recipe with the pooling position
        # moved to n_content - 1 (the last CONTENT token instead of the
        # terminator).  The frozen recipe pools AT the terminator; this
        # witness proves the captured oracle is sensitive to that rule, so a
        # native path that pools one token early cannot pass by accident.
        off_by_one_index = torch.clamp(pool_index - 1, min=0)
        off_by_one = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_index=off_by_one_index,
        )
        # Same computation as get_text_features, minus the compat head:
        # Qwen3Model last-token hidden state -> L2 normalise.
        hidden = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last = cast("torch.Tensor", hidden.last_hidden_state)
        rows = torch.arange(last.shape[0], device=last.device)
        pooled = last[rows, pool_index]
        no_head = F.normalize(pooled.float(), p=2, dim=-1)

    arrays: dict[str, npt.NDArray[np.generic]] = {
        "prompts": np.array(list(prompts)),
        "text_embeds": text_embeds.detach().to(torch.float32).cpu().numpy(),
        "text_embeds_no_head": no_head.detach().to(torch.float32).cpu().numpy(),
        "text_embeds_pool_off_by_one": off_by_one.detach()
        .to(torch.float32)
        .cpu()
        .numpy(),
        "token_ids": input_ids.detach().to(torch.int64).cpu().numpy(),
        "token_lengths": pool_index.detach().to(torch.int64).cpu().numpy(),
        "pool_index": pool_index.detach().to(torch.int64).cpu().numpy(),
        "eos_id": np.asarray(processing.EOS, dtype=np.int64),
    }
    return arrays


def load_image_embeds(oracle_dir: Path) -> npt.NDArray[np.float64]:
    """Load the six vision-side oracle fixtures, in fixture order."""
    rows: list[npt.NDArray[np.float64]] = []
    for name in FIXTURE_NAMES:
        path = oracle_dir / f"oracle_{name}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing vision oracle fixture: {path}")
        with np.load(path) as data:
            if "image_embeds" not in data.files:
                raise RuntimeError(f"{path.name}: no 'image_embeds' key")
            embeds = data["image_embeds"].astype(np.float64)
        if embeds.shape != (EMBED_DIM,):
            raise RuntimeError(
                f"{path.name}: image_embeds shape {embeds.shape} != ({EMBED_DIM},)"
            )
        rows.append(embeds)
    return np.stack(rows)


def build_retrieval_reference(
    text_embeds: npt.NDArray[np.float64],
    image_embeds: npt.NDArray[np.float64],
    prompts: Sequence[str],
) -> tuple[
    dict[str, npt.NDArray[np.generic]],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.str_],
]:
    """Build the 6x16 retrieval reference for the reference (head) text variant.

    The no-compat-head variant is added by the caller, which owns both text
    embedding sets.
    """
    matrix = cosine_matrix(image_embeds, text_embeds)
    best_index, ranks = rank_of(matrix, CORRECT_PROMPT_INDEX_ARRAY)
    best_prompt = np.array([prompts[int(i)] for i in best_index])

    arrays: dict[str, npt.NDArray[np.generic]] = {
        "prompts": np.array(list(prompts)),
        "fixture_names": np.array(list(FIXTURE_NAMES)),
        "matrix": matrix.astype(np.float32),
        "best_index": best_index,
        "best_prompt": best_prompt,
        "rank_of_correct": ranks,
        "correct_prompt_index": np.array(CORRECT_PROMPT_INDEX, dtype=np.int64),
        "text_embeds": text_embeds.astype(np.float32),
        "image_embeds": image_embeds.astype(np.float32),
    }
    return arrays, matrix, best_index, ranks, best_prompt


def format_matrix(
    matrix: npt.NDArray[np.float64],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
) -> str:
    """Render a labelled cosine matrix as a fixed-width table."""
    header = f"{'image':<11}" + "".join(f"{i:>9d}" for i in range(len(col_labels)))
    lines = [title, header]
    for row, label in enumerate(row_labels):
        cells = "".join(f"{value:>9.4f}" for value in matrix[row])
        lines.append(f"{label:<11}{cells}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the optional command line."""
    parser = argparse.ArgumentParser(
        description="Capture the PyTorch oracle for the SigLIP2-VL text tower."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ORACLE_DIR,
        help="directory for text_oracle.npz / retrieval_reference.npz",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture the text oracle and the retrieval reference."""
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: the oracle must be captured on GPU")
    device = torch.device("cuda")
    print(
        f"[env] torch={torch.__version__} numpy={np.__version__} device={device} "
        f"gpu={torch.cuda.get_device_name(0)} "
        f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32}"
    )

    processing = load_processing_module()
    tok = load_tokenizer()
    model, _ = load_model(device)

    print(f"[capture] encoding {len(PROMPTS)} prompts")
    arrays = capture_text_embeddings(model, tok, processing, PROMPTS, device)

    text_embeds = arrays["text_embeds"].astype(np.float64)
    no_head = arrays["text_embeds_no_head"].astype(np.float64)

    text_oracle_path = out_dir / "text_oracle.npz"
    save_npz_deterministic(text_oracle_path, arrays)

    # -- compat-head impact -------------------------------------------------
    per_prompt_cos = np.sum(text_embeds * no_head, axis=1)
    head_norms = np.linalg.norm(text_embeds, axis=1)
    no_head_norms = np.linalg.norm(no_head, axis=1)

    print()
    print(
        f"{'idx':>3} {'ntok':>5} {'cos(head,no_head)':>18} {'||head||':>12} "
        f"{'||no_head||':>12}  prompt"
    )
    lengths = arrays["token_lengths"]
    for index, prompt in enumerate(PROMPTS):
        print(
            f"{index:>3} {int(lengths[index]):>5} {per_prompt_cos[index]:>18.12f} "
            f"{head_norms[index]:>12.12f} {no_head_norms[index]:>12.12f}  {prompt}"
        )
    print(
        f"[compat-head] cos(text_embeds, text_embeds_no_head): "
        f"min={per_prompt_cos.min():.12f} median={np.median(per_prompt_cos):.12f} "
        f"max={per_prompt_cos.max():.12f}"
    )
    print(
        f"[norms] ||text_embeds|| in [{head_norms.min():.12f}, {head_norms.max():.12f}]  "
        f"||text_embeds_no_head|| in [{no_head_norms.min():.12f}, {no_head_norms.max():.12f}]"
    )

    # -- pooling-rule discriminator -----------------------------------------
    off_by_one = arrays["text_embeds_pool_off_by_one"].astype(np.float64)
    pool_cos = np.sum(text_embeds * off_by_one, axis=1)
    print(
        f"[pool-rule] cos(text_embeds, text_embeds_pool_off_by_one): "
        f"min={pool_cos.min():.12f} median={np.median(pool_cos):.12f} "
        f"max={pool_cos.max():.12f}  "
        "(pooling one token early is NOT equivalent -> the oracle is sensitive "
        "to the frozen terminator-pooling rule)"
    )

    # -- retrieval reference ------------------------------------------------
    image_embeds = load_image_embeds(out_dir)
    (
        reference_arrays,
        matrix,
        best_index,
        ranks,
        best_prompt,
    ) = build_retrieval_reference(text_embeds, image_embeds, PROMPTS)

    # The no-compat-head ranking: same images, same recipe, head removed.
    matrix_no_head = cosine_matrix(image_embeds, no_head)
    best_no_head, ranks_no_head = rank_of(matrix_no_head, CORRECT_PROMPT_INDEX_ARRAY)
    reference_arrays["text_embeds_no_head"] = no_head.astype(np.float32)
    reference_arrays["matrix_no_head"] = matrix_no_head.astype(np.float32)
    reference_arrays["best_index_no_head"] = best_no_head
    reference_arrays["rank_of_correct_no_head"] = ranks_no_head

    retrieval_path = out_dir / "retrieval_reference.npz"
    save_npz_deterministic(retrieval_path, reference_arrays)

    print()
    print(format_matrix(matrix, FIXTURE_NAMES, PROMPTS, "[retrieval] cos(image, text)"))
    print()
    print(f"{'fixture':<11}{'best_idx':>9}  {'rank_ok':>8}  best_prompt")
    for row, name in enumerate(FIXTURE_NAMES):
        print(
            f"{name:<11}{int(best_index[row]):>9}  {int(ranks[row]):>8}  "
            f"{best_prompt[row]!s}"
        )

    print()
    print(
        format_matrix(
            matrix_no_head,
            FIXTURE_NAMES,
            PROMPTS,
            "[retrieval/no-head] cos(image, text) -- compat head removed",
        )
    )
    print()
    print(f"{'fixture':<11}{'best_idx':>9}  {'rank_ok':>8}  best_prompt")
    for row, name in enumerate(FIXTURE_NAMES):
        print(
            f"{name:<11}{int(best_no_head[row]):>9}  {int(ranks_no_head[row]):>8}  "
            f"{PROMPTS[int(best_no_head[row])]}"
        )

    changed = [
        name
        for row, name in enumerate(FIXTURE_NAMES)
        if int(best_no_head[row]) != int(best_index[row])
    ]
    ranks_changed = [
        name
        for row, name in enumerate(FIXTURE_NAMES)
        if int(ranks_no_head[row]) != int(ranks[row])
    ]
    print()
    print(
        f"[compat-head] best-match changes without the head: "
        f"{'none' if not changed else ', '.join(changed)}"
    )
    print(
        f"[compat-head] correct-prompt rank changes without the head: "
        f"{'none' if not ranks_changed else ', '.join(ranks_changed)}"
    )
    print(
        f"[compat-head] without the head the correct prompt is rank 1 for "
        f"{int(np.sum(ranks_no_head == 1))}/6 images "
        f"(with the head: {int(np.sum(ranks == 1))}/6)"
    )

    manifest = {
        "model_dir": str(MODEL_DIR),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "tf32": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "strict_load": {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
        },
        "prompts": list(PROMPTS),
        "fixture_names": list(FIXTURE_NAMES),
        "correct_prompt_index": list(CORRECT_PROMPT_INDEX),
        "eos_id": int(processing.EOS),
        "compat_head_cosine": {
            "min": float(per_prompt_cos.min()),
            "median": float(np.median(per_prompt_cos)),
            "max": float(per_prompt_cos.max()),
        },
        "pool_rule_discriminator_cosine": {
            "min": float(pool_cos.min()),
            "median": float(np.median(pool_cos)),
            "max": float(pool_cos.max()),
        },
        "text_oracle": {
            "path": str(text_oracle_path),
            "sha256": sha256_file(text_oracle_path),
        },
        "retrieval_reference": {
            "path": str(retrieval_path),
            "sha256": sha256_file(retrieval_path),
        },
    }
    manifest_path = out_dir / "text_oracle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print("[sha256] written artifacts")
    for path in (text_oracle_path, retrieval_path, manifest_path):
        print(f"{sha256_file(path)}  {path}")
    print(
        f"[done] {len(PROMPTS)} prompts -> {text_oracle_path.name}; "
        f"retrieval reference -> {retrieval_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
