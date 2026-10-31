"""Score native Qwen3-SigLIP2-VL embeddings against the frozen retrieval reference.

The native (C++) text path writes its embeddings to ``.npz`` files; this tool
compares them against the PyTorch reference and reports whether the retrieval
ranking is preserved.  It is the acceptance gate for the native text path.

Inputs
------
``--text-npz``
    An ``.npz`` holding ``text_embeds`` of shape ``(N, 1024)``: one row per
    prompt, in the SAME order as the reference prompts.
``--image-npz``
    An ``.npz`` holding ``image_embeds`` of shape ``(M, 1024)``: one row per
    image fixture, in the SAME order as
    ``square, landscape, portrait, odd, tiny, wide``.
``--reference``
    The frozen reference, ``native/oracle/retrieval_reference.npz``.

What it prints
--------------
* the 6x16 cosine matrix ``cos(image_i, text_j)``
* the best match per image (prompt index and prompt text)
* the 1-based rank of the prompt written for each image
* ``RANKING_PRESERVED: yes/no`` -- the native best-match indices compared with
  the reference best-match indices
* the mean cosine between the native and the reference embeddings, for both
  modalities
* supplementary diagnostics: the full per-image document order, the
  correct-prompt ranks, the maximum absolute matrix deviation and the
  per-image cosine margins

Exit codes
----------
=====  =====================================================================
0      ranking preserved
2      ranking NOT preserved
1      input / usage error (bad file, wrong shape, missing key)
3      self-test failure (only with ``--self-test``)
=====  =====================================================================

Note on the strength of the verdict
-----------------------------------
The primary verdict compares the per-image BEST-MATCH INDEX, as specified.
On this reference the best-match index is ``0`` for all six images (all six
fixtures are the same synthetic gradient-plus-rectangles recipe, so prompt 0
describes every one of them), which makes the best-index criterion a weak
discriminator on its own.  The supplementary diagnostics
(``FULL_ORDER_PRESERVED``, ``RANK_OF_CORRECT_PRESERVED``, the maximum matrix
deviation and the cosine margins) are therefore printed next to it: they are
strictly stronger and are what actually separates a faithful native
implementation from a degenerate one.  ``--self-test`` proves the probe can
discriminate by perturbing one text embedding and observing the verdict flip.

Usage
-----
    python3 native/tools/retrieval_score.py \
        --text-npz native/out/text_embeds.npz \
        --image-npz native/out/image_embeds.npz

    python3 native/tools/retrieval_score.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

MODEL_DIR = Path("/content/qwen3-embedding-0.6b-vl")
DEFAULT_REFERENCE = MODEL_DIR / "native" / "oracle" / "retrieval_reference.npz"

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_RANKING_BROKEN = 2
EXIT_SELF_TEST_FAILED = 3

#: Fixture row order of every matrix in the reference.
FIXTURE_NAMES: tuple[str, ...] = (
    "square",
    "landscape",
    "portrait",
    "odd",
    "tiny",
    "wide",
)


class ScorerArgumentParser(argparse.ArgumentParser):
    """Argument parser that exits with :data:`EXIT_INPUT_ERROR`, not 2.

    ``argparse`` exits with 2 on a usage error, which would collide with the
    "ranking not preserved" status.  Usage errors must not be mistaken for a
    failed ranking check.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_INPUT_ERROR)


@dataclass(frozen=True)
class Reference:
    """The frozen retrieval reference."""

    prompts: list[str]
    fixture_names: list[str]
    text_embeds: npt.NDArray[np.float64]
    image_embeds: npt.NDArray[np.float64]
    best_index: npt.NDArray[np.int64]
    rank_of_correct: npt.NDArray[np.int64]
    correct_prompt_index: npt.NDArray[np.int64]


@dataclass(frozen=True)
class ScoreResult:
    """Everything the scorer measured."""

    matrix: npt.NDArray[np.float64]
    best_index: npt.NDArray[np.int64]
    best_prompt: list[str]
    rank_of_correct: npt.NDArray[np.int64]
    reference_best_index: npt.NDArray[np.int64]
    reference_rank_of_correct: npt.NDArray[np.int64]
    full_order: npt.NDArray[np.int64]
    reference_full_order: npt.NDArray[np.int64]
    ranking_preserved: bool
    full_order_preserved: bool
    rank_of_correct_preserved: bool
    mean_cos_text: float
    mean_cos_image: float
    max_abs_matrix_delta: float
    min_margin: float


def normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L2-normalise each row, guarding against zero-length rows."""
    norms: npt.NDArray[np.float64] = np.asarray(
        np.linalg.norm(matrix, axis=1, keepdims=True), dtype=np.float64
    )
    safe: npt.NDArray[np.float64] = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe


def row_cosines(
    native: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Cosine of each native row against the matching reference row."""
    cosines: npt.NDArray[np.float64] = np.sum(
        normalize_rows(native) * normalize_rows(reference), axis=1
    )
    return cosines


def load_npz_array(path: Path, key: str) -> npt.NDArray[np.float64]:
    """Load one array from an ``.npz`` as float64, with a clear error message."""
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path.name}: no {key!r} key (has {sorted(data.files)})")
        array = np.asarray(data[key], dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{path.name}:{key} must be 2-D, got shape {array.shape}")
    return array


def load_reference(path: Path) -> Reference:
    """Load the frozen reference artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"reference not found: {path}")
    with np.load(path) as data:
        files = set(data.files)
        required = {
            "prompts",
            "best_index",
            "rank_of_correct",
            "text_embeds",
            "image_embeds",
        }
        missing = sorted(required - files)
        if missing:
            raise KeyError(f"{path.name}: missing keys {missing}")

        prompts = [str(p) for p in data["prompts"].tolist()]
        text_embeds = np.asarray(data["text_embeds"], dtype=np.float64)
        image_embeds = np.asarray(data["image_embeds"], dtype=np.float64)
        best_index = np.asarray(data["best_index"], dtype=np.int64)
        rank_of_correct = np.asarray(data["rank_of_correct"], dtype=np.int64)

        if "fixture_names" in files:
            fixture_names = [str(f) for f in data["fixture_names"].tolist()]
        else:
            fixture_names = list(FIXTURE_NAMES[: image_embeds.shape[0]])

        if "correct_prompt_index" in files:
            correct_prompt_index = np.asarray(
                data["correct_prompt_index"], dtype=np.int64
            )
        else:
            print(
                "[warn] reference has no 'correct_prompt_index'; assuming fixture i "
                "-> prompt i"
            )
            correct_prompt_index = np.arange(len(fixture_names), dtype=np.int64)

    if text_embeds.shape[0] != len(prompts):
        raise ValueError(
            f"{path.name}: text_embeds rows {text_embeds.shape[0]} != "
            f"{len(prompts)} prompts"
        )
    if image_embeds.shape[0] != best_index.shape[0]:
        raise ValueError(
            f"{path.name}: image_embeds rows {image_embeds.shape[0]} != "
            f"{best_index.shape[0]} reference rows"
        )
    return Reference(
        prompts=prompts,
        fixture_names=fixture_names,
        text_embeds=text_embeds,
        image_embeds=image_embeds,
        best_index=best_index,
        rank_of_correct=rank_of_correct,
        correct_prompt_index=correct_prompt_index,
    )


def rank_of(
    matrix: npt.NDArray[np.float64], correct_index: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return (best_index, 1-based correct rank, full document order)."""
    order = np.argsort(-matrix, axis=1, kind="stable")
    best_index = order[:, 0].astype(np.int64)
    ranks = np.empty(matrix.shape[0], dtype=np.int64)
    for row, correct in enumerate(correct_index):
        hits = np.nonzero(order[row] == correct)[0]
        if hits.size == 0:
            raise RuntimeError(f"correct document {correct} missing from the ranking")
        ranks[row] = int(hits[0]) + 1
    return best_index, ranks, order.astype(np.int64)


def score(
    native_text: npt.NDArray[np.float64],
    native_image: npt.NDArray[np.float64],
    reference: Reference,
) -> ScoreResult:
    """Compare the native embeddings with the reference."""
    if native_text.shape[0] != len(reference.prompts):
        raise ValueError(
            f"native text rows {native_text.shape[0]} != reference prompts "
            f"{len(reference.prompts)}"
        )
    if native_image.shape[0] != reference.image_embeds.shape[0]:
        raise ValueError(
            f"native image rows {native_image.shape[0]} != reference images "
            f"{reference.image_embeds.shape[0]}"
        )
    if native_text.shape[1] != reference.text_embeds.shape[1]:
        raise ValueError(
            f"native text width {native_text.shape[1]} != reference width "
            f"{reference.text_embeds.shape[1]}"
        )
    if native_image.shape[1] != reference.image_embeds.shape[1]:
        raise ValueError(
            f"native image width {native_image.shape[1]} != reference width "
            f"{reference.image_embeds.shape[1]}"
        )

    matrix = normalize_rows(native_image) @ normalize_rows(native_text).T
    best_index, ranks, order = rank_of(matrix, reference.correct_prompt_index)

    reference_matrix = (
        normalize_rows(reference.image_embeds) @ normalize_rows(reference.text_embeds).T
    )
    _, _, reference_order = rank_of(reference_matrix, reference.correct_prompt_index)

    sorted_rows = np.sort(matrix, axis=1)
    margins = sorted_rows[:, -1] - sorted_rows[:, -2]
    best_prompt = [reference.prompts[int(i)] for i in best_index]

    return ScoreResult(
        matrix=matrix,
        best_index=best_index,
        best_prompt=best_prompt,
        rank_of_correct=ranks,
        reference_best_index=reference.best_index,
        reference_rank_of_correct=reference.rank_of_correct,
        full_order=order,
        reference_full_order=reference_order,
        ranking_preserved=bool(np.array_equal(best_index, reference.best_index)),
        full_order_preserved=bool(np.array_equal(order, reference_order)),
        rank_of_correct_preserved=bool(
            np.array_equal(ranks, reference.rank_of_correct)
        ),
        mean_cos_text=float(np.mean(row_cosines(native_text, reference.text_embeds))),
        mean_cos_image=float(
            np.mean(row_cosines(native_image, reference.image_embeds))
        ),
        max_abs_matrix_delta=float(np.max(np.abs(matrix - reference_matrix))),
        min_margin=float(np.min(margins)),
    )


def format_matrix(
    matrix: npt.NDArray[np.float64],
    row_labels: Sequence[str],
    title: str,
) -> str:
    """Render a labelled cosine matrix as a fixed-width table."""
    header = f"{'image':<11}" + "".join(f"{i:>9d}" for i in range(matrix.shape[1]))
    lines = [title, header]
    for row, label in enumerate(row_labels):
        cells = "".join(f"{value:>9.4f}" for value in matrix[row])
        lines.append(f"{label:<11}{cells}")
    return "\n".join(lines)


def report(
    result: ScoreResult,
    reference: Reference,
    native_text_path: Path,
    native_image_path: Path,
    verbose: bool = True,
) -> None:
    """Print the scoring report."""
    if verbose:
        print(
            format_matrix(
                result.matrix, reference.fixture_names, "[native] cos(image, text)"
            )
        )
        print()
    print(
        f"{'fixture':<11}{'best_idx':>9}{'ref_idx':>9}{'rank_ok':>9}{'ref_rank':>10}  "
        f"best_prompt"
    )
    for row, name in enumerate(reference.fixture_names):
        print(
            f"{name:<11}{int(result.best_index[row]):>9}"
            f"{int(result.reference_best_index[row]):>9}"
            f"{int(result.rank_of_correct[row]):>9}"
            f"{int(result.reference_rank_of_correct[row]):>10}  "
            f"{result.best_prompt[row]}"
        )
    print()
    print(f"RANKING_PRESERVED: {'yes' if result.ranking_preserved else 'no'}")
    print(
        f"FULL_ORDER_PRESERVED: {'yes' if result.full_order_preserved else 'no'} "
        f"(every image's complete 16-document order matches the reference)"
    )
    print(
        f"RANK_OF_CORRECT_PRESERVED: "
        f"{'yes' if result.rank_of_correct_preserved else 'no'}"
    )
    print(
        f"MEAN_COSINE_TEXT: {result.mean_cos_text:.12f}  "
        f"(native {native_text_path.name} vs reference text_embeds)"
    )
    print(
        f"MEAN_COSINE_IMAGE: {result.mean_cos_image:.12f}  "
        f"(native {native_image_path.name} vs reference image_embeds)"
    )
    print(
        f"MAX_ABS_MATRIX_DELTA: {result.max_abs_matrix_delta:.12e}  "
        f"MIN_TOP1_MARGIN: {result.min_margin:.12e}"
    )


def fabricate_perturbation(reference: Reference) -> npt.NDArray[np.float64]:
    """Fabricate a text embedding set that MUST change the ranking.

    The text row that currently wins for image 0 is overwritten with the
    image-0 embedding itself, so that row has cosine 1.0 against image 0 --
    the maximum attainable -- while every other row stays strictly below 1.0.
    The best match for image 0 therefore necessarily moves to the fabricated
    row, which is what the self-test then observes.
    """
    text = reference.text_embeds.copy()
    target_row = int(reference.best_index[0])
    victim_row = (target_row + 1) % text.shape[0]
    cosines = normalize_rows(text) @ normalize_rows(reference.image_embeds)[0]
    if float(np.max(cosines)) >= 1.0 - 1e-6:
        raise RuntimeError(
            "cannot fabricate a discriminating perturbation: some text row is "
            "already at cosine 1.0 against image 0"
        )
    text[victim_row] = reference.image_embeds[0]
    return text


def run_self_test(reference_path: Path) -> int:
    """Prove the probe discriminates: an unperturbed input passes, a perturbed
    input does not."""
    reference = load_reference(reference_path)
    print(f"[self-test] reference={reference_path}")

    with tempfile.TemporaryDirectory(prefix="retrieval-self-test-") as tmp:
        tmp_dir = Path(tmp)
        clean_text = tmp_dir / "text_clean.npz"
        clean_image = tmp_dir / "image_clean.npz"
        dirty_text = tmp_dir / "text_perturbed.npz"
        np.savez(clean_text, text_embeds=reference.text_embeds)
        np.savez(clean_image, image_embeds=reference.image_embeds)
        np.savez(dirty_text, text_embeds=fabricate_perturbation(reference))

        print("[self-test] control: unperturbed native == reference")
        control_rc = main(
            [
                "--text-npz",
                str(clean_text),
                "--image-npz",
                str(clean_image),
                "--reference",
                str(reference_path),
                "--quiet",
            ]
        )
        print(f"[self-test] control exit={control_rc} (expected 0)")
        if control_rc != EXIT_OK:
            print(
                "[self-test] FAIL: the control run did not preserve the ranking, "
                "so the probe cannot discriminate",
                file=sys.stderr,
            )
            return EXIT_SELF_TEST_FAILED

        print()
        print("[self-test] negative control: one text embedding fabricated")
        dirty_rc = main(
            [
                "--text-npz",
                str(dirty_text),
                "--image-npz",
                str(clean_image),
                "--reference",
                str(reference_path),
                "--quiet",
            ]
        )
        print(f"[self-test] perturbed exit={dirty_rc} (expected 2)")
        if dirty_rc != EXIT_RANKING_BROKEN:
            print(
                "[self-test] FAIL: the perturbed run did not report a ranking "
                "change, so the probe cannot discriminate",
                file=sys.stderr,
            )
            return EXIT_SELF_TEST_FAILED

    print()
    print(
        "[self-test] PASS: the probe reports 'preserved' on the unperturbed "
        "input and 'not preserved' on the perturbed input"
    )
    return EXIT_OK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = ScorerArgumentParser(
        description="Score native embeddings against the frozen retrieval reference."
    )
    parser.add_argument(
        "--text-npz",
        type=Path,
        help="npz with 'text_embeds' [N,1024], one row per reference prompt",
    )
    parser.add_argument(
        "--image-npz",
        type=Path,
        help="npz with 'image_embeds' [M,1024], one row per fixture",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="the frozen retrieval reference npz",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the summary lines (not the cosine matrix)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the discriminating self-test and exit",
    )
    args = parser.parse_args(argv)
    if not args.self_test and (args.text_npz is None or args.image_npz is None):
        parser.error("--text-npz and --image-npz are required (or use --self-test)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Score the native embeddings against the reference."""
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.reference)

    try:
        reference = load_reference(args.reference)
        native_text = load_npz_array(args.text_npz, "text_embeds")
        native_image = load_npz_array(args.image_npz, "image_embeds")
        result = score(native_text, native_image, reference)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    report(result, reference, args.text_npz, args.image_npz, verbose=not args.quiet)
    if result.ranking_preserved:
        return EXIT_OK
    return EXIT_RANKING_BROKEN


if __name__ == "__main__":
    raise SystemExit(main())
