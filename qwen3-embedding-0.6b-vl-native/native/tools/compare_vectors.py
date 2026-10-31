"""Fail-closed comparator for ``.npz`` parity files.

The tool compares a reference archive against a candidate archive key by key and
enforces a fixed set of hard-fail conditions.  It is deliberately fail-closed:
every condition that is not provably satisfied is reported as a failure, and the
process exit status is the contract that downstream automation consumes.

Exit codes
----------
``0``
    Every checked key satisfied every applicable threshold.
``2``
    At least one hard-fail condition was detected (also used for usage errors).

Metric conventions
------------------
* 1-D arrays are compared as a single vector.
* 2-D arrays are compared row by row; the reported ``cosine`` is the *worst*
  row cosine, and the flattened cosine is reported alongside it.
* The cosine that is checked against ``--cos-min`` is the minimum of the worst
  row cosine and the flattened cosine, so neither statistic can hide a
  discrepancy in the other.
* The unit-norm contract is checked on the *candidate* only, for every key whose
  name contains ``embeds``, ``projection`` or ``pooler_norm`` (case-insensitive).
  For a 2-D array every row must satisfy it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

EXIT_OK = 0
EXIT_FAIL = 2

DEFAULT_COS_MIN = 0.999
DEFAULT_MAX_ABS = 1e-3
UNIT_NORM_TOLERANCE = 1e-4
UNIT_NORM_KEY_MARKERS = ("embeds", "projection", "pooler_norm")


@dataclass(frozen=True)
class Thresholds:
    """Acceptance thresholds applied to every compared key."""

    cos_min: float = DEFAULT_COS_MIN
    max_abs: float = DEFAULT_MAX_ABS


@dataclass(frozen=True)
class KeyReport:
    """Outcome of comparing a single key."""

    key: str
    shape: tuple[int, ...]
    status: str
    cosine: float | None
    worst_row_cosine: float | None
    flattened_cosine: float | None
    max_abs: float | None
    unit_norm_checked: bool
    unit_norm_min: float | None
    unit_norm_max: float | None
    failures: tuple[str, ...]

    @property
    def failed(self) -> bool:
        """Whether the key violated at least one hard-fail condition."""
        return self.status == "FAIL"


@dataclass(frozen=True)
class Comparison:
    """Aggregate outcome of a reference/candidate comparison."""

    reports: tuple[KeyReport, ...]
    only_in_reference: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    requested_keys: tuple[str, ...] | None

    @property
    def n_checked(self) -> int:
        """Number of keys considered by the comparison."""
        return len(self.reports)

    @property
    def n_failed(self) -> int:
        """Number of keys that violated a hard-fail condition."""
        return sum(1 for report in self.reports if report.failed)

    @property
    def passed(self) -> bool:
        """Whether every checked key passed."""
        return self.n_failed == 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        prog="compare_vectors.py",
        description="Fail-closed comparator for .npz parity files.",
    )
    parser.add_argument("--reference", type=Path, help="reference .npz archive")
    parser.add_argument("--candidate", type=Path, help="candidate .npz archive")
    parser.add_argument(
        "--keys",
        help="comma separated subset of keys to compare (default: keys present in both archives)",
    )
    parser.add_argument(
        "--cos-min",
        type=float,
        default=DEFAULT_COS_MIN,
        help="minimum acceptable cosine similarity (default: %(default)s)",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=DEFAULT_MAX_ABS,
        help="maximum acceptable absolute difference (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="write the machine readable summary to this path",
    )
    parser.add_argument(
        "--label", default="", help="free form label recorded in the summary"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built in discrimination test and exit",
    )
    return parser


def parse_keys(raw: str | None) -> list[str] | None:
    """Split a ``--keys`` argument into an ordered list of key names."""
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_archive(path: Path) -> dict[str, NDArray[Any]]:
    """Load every array of a ``.npz`` archive into memory."""
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"not a .npz archive: {path}")
    with loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def cosine(reference: NDArray[Any], candidate: NDArray[Any]) -> float:
    """Cosine similarity of two flattened vectors; ``0.0`` when a norm is zero."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 or cand_norm == 0.0:
        return 0.0
    return float(np.dot(ref, cand) / (ref_norm * cand_norm))


def row_cosines(reference: NDArray[Any], candidate: NDArray[Any]) -> NDArray[Any]:
    """Per-row cosine similarity of two 2-D arrays; zero rows score ``0.0``."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    numerator = np.sum(ref * cand, axis=1)
    denominator = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    scores = np.zeros(ref.shape[0], dtype=np.float64)
    usable = denominator > 0.0
    scores[usable] = numerator[usable] / denominator[usable]
    return scores


def unit_norm_scope(key: str) -> bool:
    """Whether the unit-norm contract applies to a key."""
    lowered = key.lower()
    return any(marker in lowered for marker in UNIT_NORM_KEY_MARKERS)


def candidate_norms(candidate: NDArray[Any]) -> NDArray[Any]:
    """L2 norms of the candidate vectors: one vector for 1-D, one per row for 2-D."""
    array = np.asarray(candidate, dtype=np.float64)
    if array.ndim <= 1:
        norms = np.linalg.norm(array.reshape(1, -1), axis=1)
    else:
        norms = np.linalg.norm(array.reshape(array.shape[0], -1), axis=1)
    return np.asarray(norms, dtype=np.float64)


def evaluate_key(
    key: str,
    reference: NDArray[Any],
    candidate: NDArray[Any],
    thresholds: Thresholds,
) -> KeyReport:
    """Compare one key and collect every hard-fail condition it triggers."""
    ref_shape = tuple(int(dim) for dim in reference.shape)
    cand_shape = tuple(int(dim) for dim in candidate.shape)
    failures: list[str] = []

    if ref_shape != cand_shape:
        failures.append(
            f"shape mismatch: reference {ref_shape} != candidate {cand_shape}",
        )
        return KeyReport(
            key=key,
            shape=ref_shape,
            status="FAIL",
            cosine=None,
            worst_row_cosine=None,
            flattened_cosine=None,
            max_abs=None,
            unit_norm_checked=False,
            unit_norm_min=None,
            unit_norm_max=None,
            failures=tuple(failures),
        )

    if not bool(np.all(np.isfinite(reference))):
        failures.append("reference contains NaN or Inf")
    if not bool(np.all(np.isfinite(candidate))):
        failures.append("candidate contains NaN or Inf")
    if failures:
        return KeyReport(
            key=key,
            shape=ref_shape,
            status="FAIL",
            cosine=None,
            worst_row_cosine=None,
            flattened_cosine=None,
            max_abs=None,
            unit_norm_checked=False,
            unit_norm_min=None,
            unit_norm_max=None,
            failures=tuple(failures),
        )

    ref_norm = float(np.linalg.norm(np.asarray(reference, dtype=np.float64).ravel()))
    cand_norm = float(np.linalg.norm(np.asarray(candidate, dtype=np.float64).ravel()))
    if ref_norm == 0.0:
        failures.append("reference has zero L2 norm, cosine is undefined")
    if cand_norm == 0.0:
        failures.append("candidate has zero L2 norm, cosine is undefined")

    flattened = cosine(reference, candidate)
    worst_row: float | None = None
    if reference.ndim == 2:
        worst_row = float(np.min(row_cosines(reference, candidate)))
    effective = flattened if worst_row is None else min(worst_row, flattened)

    max_abs = float(
        np.max(
            np.abs(
                np.asarray(reference, dtype=np.float64)
                - np.asarray(candidate, dtype=np.float64),
            ),
        ),
    )

    norms_checked = unit_norm_scope(key)
    norm_min: float | None = None
    norm_max: float | None = None
    if norms_checked:
        norms = candidate_norms(candidate)
        norm_min = float(np.min(norms))
        norm_max = float(np.max(norms))
        lower = 1.0 - UNIT_NORM_TOLERANCE
        upper = 1.0 + UNIT_NORM_TOLERANCE
        outside = np.where((norms < lower) | (norms > upper))[0]
        for index in outside:
            failures.append(
                f"candidate L2 norm {norms[index]:.9g} outside [{lower:g}, {upper:g}] "
                f"(vector {int(index)})",
            )

    if effective < thresholds.cos_min:
        failures.append(f"cosine {effective:.9g} < --cos-min {thresholds.cos_min:g}")
    if max_abs > thresholds.max_abs:
        failures.append(
            f"max abs diff {max_abs:.9g} > --max-abs {thresholds.max_abs:g}"
        )

    return KeyReport(
        key=key,
        shape=ref_shape,
        status="FAIL" if failures else "PASS",
        cosine=effective,
        worst_row_cosine=worst_row,
        flattened_cosine=flattened,
        max_abs=max_abs,
        unit_norm_checked=norms_checked,
        unit_norm_min=norm_min,
        unit_norm_max=norm_max,
        failures=tuple(failures),
    )


def missing_key_report(key: str, in_reference: bool, in_candidate: bool) -> KeyReport:
    """Build the report for a requested key that is absent from an archive."""
    failures: list[str] = []
    if not in_reference:
        failures.append("key missing from reference archive")
    if not in_candidate:
        failures.append("key missing from candidate archive")
    return KeyReport(
        key=key,
        shape=(),
        status="FAIL",
        cosine=None,
        worst_row_cosine=None,
        flattened_cosine=None,
        max_abs=None,
        unit_norm_checked=False,
        unit_norm_min=None,
        unit_norm_max=None,
        failures=tuple(failures),
    )


def compare_archives(
    reference: dict[str, NDArray[Any]],
    candidate: dict[str, NDArray[Any]],
    requested: list[str] | None,
    thresholds: Thresholds,
) -> Comparison:
    """Compare the two archives and return one report per considered key."""
    reports: list[KeyReport] = []
    only_reference: list[str] = []
    only_candidate: list[str] = []

    if requested is None:
        for key in reference:
            if key in candidate:
                reports.append(
                    evaluate_key(key, reference[key], candidate[key], thresholds),
                )
            else:
                only_reference.append(key)
        for key in candidate:
            if key not in reference:
                only_candidate.append(key)
    else:
        for key in requested:
            in_reference = key in reference
            in_candidate = key in candidate
            if not in_reference or not in_candidate:
                reports.append(missing_key_report(key, in_reference, in_candidate))
                continue
            reports.append(
                evaluate_key(key, reference[key], candidate[key], thresholds),
            )

    return Comparison(
        reports=tuple(reports),
        only_in_reference=tuple(only_reference),
        only_in_candidate=tuple(only_candidate),
        requested_keys=None if requested is None else tuple(requested),
    )


def format_metric(value: float | None, spec: str) -> str:
    """Format an optional metric for the table."""
    if value is None:
        return "-"
    return format(value, spec)


def format_table(reports: Sequence[KeyReport]) -> str:
    """Render the per-key result table."""
    key_width = max((len(report.key) for report in reports), default=4)
    header = (
        f"{'KEY':<{key_width}}  {'SHAPE':<20}  {'COSINE':>14}  {'MAX_ABS':>13}  STATUS"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        shape = "x".join(str(dim) for dim in report.shape) or "-"
        shape_text = f"({shape})" if report.shape else "-"
        lines.append(
            f"{report.key:<{key_width}}  {shape_text:<20}  "
            f"{format_metric(report.cosine, '.9g'):>14}  "
            f"{format_metric(report.max_abs, '.6g'):>13}  {report.status}",
        )
    return "\n".join(lines)


def build_summary(
    comparison: Comparison,
    reference_path: Path,
    candidate_path: Path,
    thresholds: Thresholds,
    label: str,
) -> dict[str, Any]:
    """Build the machine readable summary payload."""
    return {
        "tool": "compare_vectors",
        "label": label,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "thresholds": {
            "cos_min": thresholds.cos_min,
            "max_abs": thresholds.max_abs,
            "unit_norm_tolerance": UNIT_NORM_TOLERANCE,
            "unit_norm_key_markers": list(UNIT_NORM_KEY_MARKERS),
        },
        "requested_keys": (
            None
            if comparison.requested_keys is None
            else list(comparison.requested_keys)
        ),
        "keys_only_in_reference": list(comparison.only_in_reference),
        "keys_only_in_candidate": list(comparison.only_in_candidate),
        "n_checked": comparison.n_checked,
        "n_failed": comparison.n_failed,
        "result": "PASS" if comparison.passed else "FAIL",
        "keys": [
            {
                "key": report.key,
                "shape": list(report.shape),
                "status": report.status,
                "cosine": report.cosine,
                "worst_row_cosine": report.worst_row_cosine,
                "flattened_cosine": report.flattened_cosine,
                "max_abs": report.max_abs,
                "unit_norm_checked": report.unit_norm_checked,
                "unit_norm_min": report.unit_norm_min,
                "unit_norm_max": report.unit_norm_max,
                "failures": list(report.failures),
            }
            for report in comparison.reports
        ],
    }


def report_failures(comparison: Comparison) -> None:
    """Print every hard-fail reason on stderr."""
    for report in comparison.reports:
        for failure in report.failures:
            print(f"FAIL: key '{report.key}': {failure}", file=sys.stderr)


def run_compare(
    reference_path: Path,
    candidate_path: Path,
    thresholds: Thresholds,
    requested: list[str] | None,
    label: str,
    json_path: Path | None,
) -> int:
    """Compare two archives, print the report and return the process exit code."""
    try:
        reference = load_archive(reference_path)
        candidate = load_archive(candidate_path)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAIL

    comparison = compare_archives(reference, candidate, requested, thresholds)

    print(f"LABEL: {label}" if label else "LABEL: -")
    print(f"REFERENCE: {reference_path}")
    print(f"CANDIDATE: {candidate_path}")
    print(f"THRESHOLDS: cos_min={thresholds.cos_min:g} max_abs={thresholds.max_abs:g}")
    if comparison.requested_keys is not None:
        print(f"KEY SELECTION: explicit ({len(comparison.requested_keys)} keys)")
    else:
        print("KEY SELECTION: keys present in both archives")
    print()
    print(format_table(comparison.reports))
    print()
    if comparison.only_in_reference:
        print(
            "NOT COMPARED (only in reference): "
            + ", ".join(comparison.only_in_reference),
        )
    if comparison.only_in_candidate:
        print(
            "NOT COMPARED (only in candidate): "
            + ", ".join(comparison.only_in_candidate),
        )

    report_failures(comparison)

    if json_path is not None:
        summary = build_summary(
            comparison, reference_path, candidate_path, thresholds, label
        )
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"JSON: {json_path}")

    if comparison.passed:
        print(f"RESULT: PASS ({comparison.n_checked}/{comparison.n_checked} keys)")
        return EXIT_OK
    print(f"RESULT: FAIL ({comparison.n_failed} failures)")
    return EXIT_FAIL


def run_cli(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Invoke this tool as a subprocess and capture its status and streams."""
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603


def write_case(directory: Path, name: str, arrays: dict[str, NDArray[Any]]) -> Path:
    """Write a ``.npz`` archive into ``directory`` and return its path."""
    path = directory / f"{name}.npz"
    np.savez(path, **arrays)
    return path


def self_test_case(
    directory: Path,
    name: str,
    reference: dict[str, NDArray[Any]],
    candidate: dict[str, NDArray[Any]],
    expected: int,
    extra_arguments: Sequence[str] = (),
) -> tuple[bool, subprocess.CompletedProcess[str]]:
    """Run one self-test case and report whether the exit status was as expected."""
    directory.mkdir(parents=True, exist_ok=True)
    ref_path = write_case(directory, "ref", reference)
    cand_path = write_case(directory, "cand", candidate)
    completed = run_cli(
        [
            "--reference",
            str(ref_path),
            "--candidate",
            str(cand_path),
            *extra_arguments,
        ],
    )
    ok = completed.returncode == expected
    print(
        f"SELF-TEST case '{name}': exit {completed.returncode} (expected {expected}) "
        f"-> {'OK' if ok else 'MISMATCH'}",
    )
    return ok, completed


def run_self_test() -> int:
    """Prove the tool discriminates: identical archives pass, perturbed archives fail."""
    unit_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    unit_rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    perturbed_vector = np.array([0.99, 0.05, 0.05, 0.05], dtype=np.float32)
    perturbed_rows = np.array(
        [[0.99, 0.05, 0.05], [0.05, 0.99, 0.05]], dtype=np.float32
    )

    outcomes: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="compare_vectors_selftest_") as temporary:
        root = Path(temporary)

        ok, completed = self_test_case(
            root / "identical",
            "identical",
            {"vision_features": unit_vector, "pooler_output": unit_rows},
            {"vision_features": unit_vector, "pooler_output": unit_rows},
            EXIT_OK,
        )
        outcomes.append(ok)
        print("--- stdout of the identical case ---")
        print(completed.stdout.rstrip())
        print("--- end stdout ---")

        ok, completed = self_test_case(
            root / "perturbed",
            "perturbed beyond every threshold",
            {"vision_features": unit_vector, "pooler_output": unit_rows},
            {"vision_features": perturbed_vector, "pooler_output": perturbed_rows},
            EXIT_FAIL,
        )
        outcomes.append(ok)
        print("--- stdout of the perturbed case ---")
        print(completed.stdout.rstrip())
        print("--- end stdout ---")
        print("--- stderr of the perturbed case ---")
        print(completed.stderr.rstrip())
        print("--- end stderr ---")

        ok, _ = self_test_case(
            root / "cosine_only",
            "cosine violation only",
            {"tiny": np.array([0.0005, 0.0], dtype=np.float32)},
            {"tiny": np.array([0.0, 0.0005], dtype=np.float32)},
            EXIT_FAIL,
        )
        outcomes.append(ok)

        ok, _ = self_test_case(
            root / "max_abs_only",
            "max abs violation only",
            {"big": np.array([1000.0, 1000.0], dtype=np.float32)},
            {"big": np.array([1000.002, 1000.0], dtype=np.float32)},
            EXIT_FAIL,
        )
        outcomes.append(ok)

        ok, _ = self_test_case(
            root / "unit_norm",
            "unit norm violation only",
            {"image_embeds": np.array([0.001, 0.0], dtype=np.float32)},
            {"image_embeds": np.array([0.0015, 0.0], dtype=np.float32)},
            EXIT_FAIL,
        )
        outcomes.append(ok)

        ok, _ = self_test_case(
            root / "shape_mismatch",
            "shape mismatch",
            {"shape_probe": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            {"shape_probe": np.array([1.0, 2.0], dtype=np.float32)},
            EXIT_FAIL,
        )
        outcomes.append(ok)

        ok, _ = self_test_case(
            root / "nan",
            "NaN in candidate",
            {"nan_probe": np.array([1.0, 2.0], dtype=np.float32)},
            {"nan_probe": np.array([1.0, float("nan")], dtype=np.float32)},
            EXIT_FAIL,
        )
        outcomes.append(ok)

        ok, _ = self_test_case(
            root / "missing_key",
            "missing key under --keys",
            {"present": np.array([1.0], dtype=np.float32)},
            {"present": np.array([1.0], dtype=np.float32)},
            EXIT_FAIL,
            ("--keys", "present,absent"),
        )
        outcomes.append(ok)

    passed = sum(1 for outcome in outcomes if outcome)
    total = len(outcomes)
    print(f"SELF-TEST: {passed}/{total} cases behaved as expected")
    if passed != total:
        return EXIT_FAIL
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.self_test:
        return run_self_test()

    if arguments.reference is None or arguments.candidate is None:
        parser.error(
            "--reference and --candidate are required unless --self-test is used"
        )
    if not -1.0 <= arguments.cos_min <= 1.0:
        parser.error("--cos-min must be within [-1, 1]")
    if arguments.max_abs < 0.0:
        parser.error("--max-abs must be non-negative")

    thresholds = Thresholds(cos_min=arguments.cos_min, max_abs=arguments.max_abs)
    return run_compare(
        reference_path=arguments.reference,
        candidate_path=arguments.candidate,
        thresholds=thresholds,
        requested=parse_keys(arguments.keys),
        label=arguments.label,
        json_path=arguments.json_path,
    )


if __name__ == "__main__":
    sys.exit(main())
