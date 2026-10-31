"""Verify that a GGUF vision sidecar faithfully represents the source weights.

The sidecar stores the vision tower, the attention-pooling head and the
``1152 -> 1024`` projection of the source model.  Every sidecar tensor is
dequantized in numpy and compared against the matching tensor of the source
``model.safetensors`` file.

Conventions implemented here
----------------------------
* A ``nn.Linear`` weight ``W`` of shape ``[out, in]`` is stored transposed, so
  the sidecar GGUF shape is ``[in, out]``.  The numpy view of the sidecar data
  therefore has shape ``(ne1, ne0)``, which equals the torch shape.
* The ``+16`` padded tensors (``v.blk.<i>.ffn_down.weight`` and
  ``p.mlp.fc2.weight``) carry ``4320`` columns; the trailing 16 columns are
  zero padding and are trimmed away before the comparison.
* ``v.position_embd.weight`` is an ``nn.Embedding`` weight of shape
  ``[num_positions, embed_dim]`` and needs no transposition.

Exit codes
----------
``0``
    Every sidecar tensor matched its source tensor within the configured
    cosine threshold and every completeness check held.
``1``
    At least one check failed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from safetensors import safe_open

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

EXIT_OK = 0
EXIT_FAIL = 1

QK8_0 = 32
Q8_0_BLOCK_BYTES = 34
BLOCK_COUNT = 27
DEFAULT_EXPECTED_TENSORS = 450
DEFAULT_MIN_COSINE = 0.999
DEFAULT_TOP_N = 5
PADDED_WIDTH = 4304
PADDING_COLUMNS = 16
SOURCE_NAMESPACES = ("vision_model.", "vision_projection.")


@dataclass(frozen=True)
class TensorSpec:
    """Explicit binding between a sidecar tensor and a source tensor.

    ``trim_width`` is the number of columns the source tensor really has; the
    sidecar stores that many columns plus ``PADDING_COLUMNS`` zero columns.
    """

    source_name: str
    trim_width: int | None = None


def build_tensor_specs() -> dict[str, TensorSpec]:
    """Build the explicit sidecar-name to source-name mapping.

    The mapping is written out by hand: no fuzzy or prefix matching is used
    anywhere, so a renamed or renamed-away tensor can never be silently
    rebound to the wrong source weight.
    """
    specs: dict[str, TensorSpec] = {
        "mm.model.fc.weight": TensorSpec("vision_projection.weight"),
        "mm.model.fc.bias": TensorSpec("vision_projection.bias"),
        "p.probe": TensorSpec("vision_model.head.probe"),
        "p.attention.in_proj_weight": TensorSpec(
            "vision_model.head.attention.in_proj_weight"
        ),
        "p.attention.in_proj_bias": TensorSpec(
            "vision_model.head.attention.in_proj_bias"
        ),
        "p.attention.out_proj.weight": TensorSpec(
            "vision_model.head.attention.out_proj.weight"
        ),
        "p.attention.out_proj.bias": TensorSpec(
            "vision_model.head.attention.out_proj.bias"
        ),
        "p.layernorm.weight": TensorSpec("vision_model.head.layernorm.weight"),
        "p.layernorm.bias": TensorSpec("vision_model.head.layernorm.bias"),
        "p.mlp.fc1.weight": TensorSpec("vision_model.head.mlp.fc1.weight"),
        "p.mlp.fc1.bias": TensorSpec("vision_model.head.mlp.fc1.bias"),
        "p.mlp.fc2.weight": TensorSpec(
            "vision_model.head.mlp.fc2.weight",
            trim_width=PADDED_WIDTH,
        ),
        "p.mlp.fc2.bias": TensorSpec("vision_model.head.mlp.fc2.bias"),
        "v.patch_embd.weight": TensorSpec(
            "vision_model.embeddings.patch_embedding.weight"
        ),
        "v.patch_embd.bias": TensorSpec("vision_model.embeddings.patch_embedding.bias"),
        "v.position_embd.weight": TensorSpec(
            "vision_model.embeddings.position_embedding.weight"
        ),
        "v.post_ln.weight": TensorSpec("vision_model.post_layernorm.weight"),
        "v.post_ln.bias": TensorSpec("vision_model.post_layernorm.bias"),
    }

    block_tensors = (
        ("ln1.weight", "layer_norm1.weight"),
        ("ln1.bias", "layer_norm1.bias"),
        ("ln2.weight", "layer_norm2.weight"),
        ("ln2.bias", "layer_norm2.bias"),
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_q.bias", "self_attn.q_proj.bias"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_k.bias", "self_attn.k_proj.bias"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_v.bias", "self_attn.v_proj.bias"),
        ("attn_out.weight", "self_attn.out_proj.weight"),
        ("attn_out.bias", "self_attn.out_proj.bias"),
        ("ffn_up.weight", "mlp.fc1.weight"),
        ("ffn_up.bias", "mlp.fc1.bias"),
        ("ffn_down.bias", "mlp.fc2.bias"),
    )
    for index in range(BLOCK_COUNT):
        sidecar_prefix = f"v.blk.{index}."
        source_prefix = f"vision_model.encoder.layers.{index}."
        for sidecar_suffix, source_suffix in block_tensors:
            specs[sidecar_prefix + sidecar_suffix] = TensorSpec(
                source_prefix + source_suffix
            )
        specs[sidecar_prefix + "ffn_down.weight"] = TensorSpec(
            source_prefix + "mlp.fc2.weight",
            trim_width=PADDED_WIDTH,
        )

    return specs


def dequantize_q8_0(raw: NDArray[Any], dims: tuple[int, ...]) -> NDArray[Any]:
    """Dequantize a Q8_0 buffer into float32 with shape ``reversed(dims)``.

    A Q8_0 block is 34 bytes: a 2 byte little-endian ``ggml_half`` scale
    followed by 32 ``int8`` values.  Blocks run along the fastest (ggml ``ne0``)
    dimension, so the buffer is read in C order over ``reversed(dims)``.
    """
    reversed_dims = tuple(reversed(dims))
    n_blocks = reversed_dims[-1] // QK8_0
    blocks = np.ascontiguousarray(raw).reshape(
        (*reversed_dims[:-1], n_blocks, Q8_0_BLOCK_BYTES),
    )
    scales = (
        blocks[..., :2]
        .copy()
        .view(np.float16)
        .reshape((*reversed_dims[:-1], n_blocks))
        .astype(np.float32)
    )
    quants = (
        blocks[..., 2:]
        .copy()
        .view(np.int8)
        .reshape((*reversed_dims[:-1], n_blocks, QK8_0))
        .astype(np.float32)
    )
    return (quants * scales[..., np.newaxis]).reshape(reversed_dims)


def sidecar_to_numpy(tensor: Any) -> NDArray[Any]:
    """Return the float32 numpy view of a sidecar tensor."""
    dims = tuple(int(dim) for dim in tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q8_0:
        return dequantize_q8_0(tensor.data, dims)
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    raise ValueError(
        f"unsupported sidecar tensor type {tensor.tensor_type} for {tensor.name}",
    )


def expected_shape(dims: tuple[int, ...], trim_width: int | None) -> tuple[int, ...]:
    """Torch shape of the source tensor for the given ggml dimensions."""
    reversed_dims = tuple(reversed(dims))
    if trim_width is None:
        return reversed_dims
    if len(reversed_dims) != 2:
        raise ValueError("trimming is only defined for 2-D tensors")
    return (reversed_dims[0], trim_width)


def trim_padding(array: NDArray[Any], trim_width: int) -> tuple[NDArray[Any], bool]:
    """Trim the zero padded columns and report whether the padding was zero."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D padded tensor, got shape {array.shape}")
    if array.shape[1] != trim_width + PADDING_COLUMNS:
        raise ValueError(
            f"expected {trim_width + PADDING_COLUMNS} columns, got {array.shape[1]}",
        )
    padding = array[:, trim_width:]
    padding_is_zero = bool(np.all(padding == 0.0))
    return np.ascontiguousarray(array[:, :trim_width]), padding_is_zero


@dataclass(frozen=True)
class TensorReport:
    """Outcome of comparing one sidecar tensor against its source tensor."""

    sidecar_name: str
    source_name: str
    shape: tuple[int, ...]
    status: str
    cosine: float | None
    max_abs: float | None
    relative_frobenius: float | None
    failures: tuple[str, ...]


@dataclass(frozen=True)
class Verification:
    """Aggregate outcome of a sidecar verification run."""

    reports: tuple[TensorReport, ...]
    failures: tuple[str, ...]
    tensor_count: int
    mapping_size: int
    expected_tensors: int

    @property
    def cosines(self) -> list[float]:
        """Cosine similarities of every successfully compared tensor."""
        return [report.cosine for report in self.reports if report.cosine is not None]

    @property
    def min_cosine(self) -> float | None:
        """Smallest cosine similarity across all compared tensors."""
        values = self.cosines
        return min(values) if values else None

    @property
    def median_cosine(self) -> float | None:
        """Median cosine similarity across all compared tensors."""
        values = self.cosines
        return float(statistics.median(values)) if values else None

    @property
    def worst(self) -> tuple[TensorReport, ...]:
        """Reports ordered from the worst to the best cosine similarity."""
        return tuple(
            sorted(
                (report for report in self.reports if report.cosine is not None),
                key=lambda report: report.cosine if report.cosine is not None else 1.0,
            ),
        )


def compare_arrays(
    sidecar_name: str,
    source_name: str,
    sidecar: NDArray[Any],
    source: NDArray[Any],
) -> TensorReport:
    """Compare one dequantized sidecar tensor against its source tensor."""
    sidecar_shape = tuple(int(dim) for dim in sidecar.shape)
    source_shape = tuple(int(dim) for dim in source.shape)
    if sidecar_shape != source_shape:
        return TensorReport(
            sidecar_name=sidecar_name,
            source_name=source_name,
            shape=sidecar_shape,
            status="FAIL",
            cosine=None,
            max_abs=None,
            relative_frobenius=None,
            failures=(
                f"shape mismatch: sidecar {sidecar_shape} != source {source_shape}",
            ),
        )

    side = np.asarray(sidecar, dtype=np.float64)
    origin = np.asarray(source, dtype=np.float64)
    difference = side - origin
    flat_side = side.ravel()
    flat_origin = origin.ravel()
    side_norm = float(np.linalg.norm(flat_side))
    origin_norm = float(np.linalg.norm(flat_origin))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    if origin_norm == 0.0:
        relative = 0.0 if difference_norm == 0.0 else float("inf")
        cosine = 1.0 if difference_norm == 0.0 else 0.0
    else:
        relative = difference_norm / origin_norm
        cosine = (
            1.0
            if side_norm == 0.0
            else float(np.dot(flat_side, flat_origin) / (side_norm * origin_norm))
        )
    max_abs = float(np.max(np.abs(difference)))

    return TensorReport(
        sidecar_name=sidecar_name,
        source_name=source_name,
        shape=sidecar_shape,
        status="PASS",
        cosine=cosine,
        max_abs=max_abs,
        relative_frobenius=relative,
        failures=(),
    )


def verify(
    gguf_path: Path,
    safetensors_path: Path,
    expected_tensors: int,
    min_cosine: float,
    max_abs: float | None = None,
    rel_fro_max: float | None = None,
) -> Verification:
    """Verify every sidecar tensor against the source weights."""
    specs = build_tensor_specs()
    reader = GGUFReader(str(gguf_path))
    sidecars = {tensor.name: tensor for tensor in reader.tensors}

    failures: list[str] = []
    unmapped = sorted(name for name in sidecars if name not in specs)
    absent = sorted(name for name in specs if name not in sidecars)
    for name in unmapped:
        failures.append(
            f"mismatched key: sidecar tensor '{name}' has no explicit mapping"
        )
    for name in absent:
        failures.append(
            f"missing key: mapped sidecar tensor '{name}' is absent from the GGUF"
        )

    reports: list[TensorReport] = []
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        missing_source = sorted(
            spec.source_name
            for name, spec in specs.items()
            if spec.source_name not in source_keys
        )
        for name in missing_source:
            failures.append(
                f"missing key: source tensor '{name}' is absent from the safetensors file"
            )
        consumed = {spec.source_name for spec in specs.values()}
        unexpected = sorted(
            key
            for key in source_keys
            if key.startswith(SOURCE_NAMESPACES) and key not in consumed
        )
        for name in unexpected:
            failures.append(
                f"unexpected key: source tensor '{name}' is not covered by the mapping"
            )

        for name in sorted(sidecars):
            if name not in specs:
                continue
            spec = specs[name]
            dims = tuple(int(dim) for dim in sidecars[name].shape)
            try:
                array = sidecar_to_numpy(sidecars[name])
                if spec.trim_width is not None:
                    array, padding_is_zero = trim_padding(array, spec.trim_width)
                    if not padding_is_zero:
                        failures.append(
                            f"padding region of '{name}' is not exactly zero",
                        )
                if spec.source_name not in source_keys:
                    continue
                source_tensor = handle.get_tensor(spec.source_name)
                source = np.asarray(source_tensor, dtype=np.float32)
                report = compare_arrays(name, spec.source_name, array, source)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report = TensorReport(
                    sidecar_name=name,
                    source_name=spec.source_name,
                    shape=expected_shape(dims, spec.trim_width),
                    status="FAIL",
                    cosine=None,
                    max_abs=None,
                    relative_frobenius=None,
                    failures=(f"could not compare: {error}",),
                )
            reasons: list[str] = list(report.failures)
            if report.cosine is not None and report.cosine < min_cosine:
                reasons.append(f"cosine {report.cosine:.9g} < threshold {min_cosine:g}")
            if (
                max_abs is not None
                and report.max_abs is not None
                and report.max_abs > max_abs
            ):
                reasons.append(
                    f"max abs diff {report.max_abs:.9g} > threshold {max_abs:g}"
                )
            if (
                rel_fro_max is not None
                and report.relative_frobenius is not None
                and report.relative_frobenius > rel_fro_max
            ):
                reasons.append(
                    f"relative Frobenius error {report.relative_frobenius:.9g} "
                    f"> threshold {rel_fro_max:g}",
                )
            report = TensorReport(
                sidecar_name=report.sidecar_name,
                source_name=report.source_name,
                shape=report.shape,
                status="FAIL" if reasons else "PASS",
                cosine=report.cosine,
                max_abs=report.max_abs,
                relative_frobenius=report.relative_frobenius,
                failures=tuple(reasons),
            )
            if report.failures:
                failures.extend(
                    f"tensor '{report.sidecar_name}': {reason}"
                    for reason in report.failures
                )
            reports.append(report)

    if len(specs) != expected_tensors:
        failures.append(
            f"mapping holds {len(specs)} tensors, expected {expected_tensors}"
        )
    if len(sidecars) != expected_tensors:
        failures.append(
            f"GGUF holds {len(sidecars)} tensors, expected {expected_tensors}"
        )
    if len(reports) != expected_tensors:
        failures.append(f"compared {len(reports)} tensors, expected {expected_tensors}")

    return Verification(
        reports=tuple(reports),
        failures=tuple(failures),
        tensor_count=len(sidecars),
        mapping_size=len(specs),
        expected_tensors=expected_tensors,
    )


def format_table(reports: Sequence[TensorReport]) -> str:
    """Render the per-tensor comparison table."""
    name_width = max((len(report.sidecar_name) for report in reports), default=4)
    source_width = max((len(report.source_name) for report in reports), default=4)
    header = (
        f"{'SIDECAR TENSOR':<{name_width}}  {'SOURCE TENSOR':<{source_width}}  "
        f"{'SHAPE':<14}  {'COSINE':>14}  {'MAX_ABS':>13}  {'REL_FRO':>13}  STATUS"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        shape = "x".join(str(dim) for dim in report.shape) or "-"
        shape_text = f"({shape})"
        cosine = "-" if report.cosine is None else format(report.cosine, ".9g")
        max_abs = "-" if report.max_abs is None else format(report.max_abs, ".6g")
        relative = (
            "-"
            if report.relative_frobenius is None
            else format(report.relative_frobenius, ".6g")
        )
        lines.append(
            f"{report.sidecar_name:<{name_width}}  {report.source_name:<{source_width}}  "
            f"{shape_text:<14}  {cosine:>14}  {max_abs:>13}  {relative:>13}  {report.status}",
        )
    return "\n".join(lines)


def build_summary(
    verification: Verification,
    gguf_path: Path,
    safetensors_path: Path,
    min_cosine: float,
    max_abs: float | None = None,
    rel_fro_max: float | None = None,
) -> dict[str, Any]:
    """Build the machine readable summary payload."""
    return {
        "tool": "verify_sidecar_weights",
        "gguf": str(gguf_path),
        "safetensors": str(safetensors_path),
        "min_cosine": min_cosine,
        "max_abs": max_abs,
        "rel_fro_max": rel_fro_max,
        "tensor_count": verification.tensor_count,
        "mapping_size": verification.mapping_size,
        "expected_tensors": verification.expected_tensors,
        "min_cosine_observed": verification.min_cosine,
        "median_cosine": verification.median_cosine,
        "failures": list(verification.failures),
        "result": "PASS" if not verification.failures else "FAIL",
        "tensors": [
            {
                "sidecar_name": report.sidecar_name,
                "source_name": report.source_name,
                "shape": list(report.shape),
                "status": report.status,
                "cosine": report.cosine,
                "max_abs": report.max_abs,
                "relative_frobenius": report.relative_frobenius,
                "failures": list(report.failures),
            }
            for report in verification.reports
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="verify_sidecar_weights.py",
        description="Verify that the GGUF sidecar faithfully represents the source weights.",
    )
    parser.add_argument(
        "--gguf", type=Path, required=True, help="path to the sidecar GGUF"
    )
    parser.add_argument(
        "--safetensors",
        type=Path,
        required=True,
        help="path to the source model.safetensors",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=DEFAULT_MIN_COSINE,
        help="minimum acceptable cosine similarity (default: %(default)s)",
    )
    parser.add_argument(
        "--expected-tensors",
        type=int,
        default=DEFAULT_EXPECTED_TENSORS,
        help="number of tensors the sidecar must hold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="optional maximum absolute difference gate (disabled by default)",
    )
    parser.add_argument(
        "--rel-fro-max",
        type=float,
        default=None,
        help="optional maximum relative Frobenius error gate (disabled by default)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="number of worst tensors to list (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="write the machine readable summary to this path",
    )
    arguments = parser.parse_args(argv)

    print("SIDECAR WEIGHT VERIFICATION")
    print(f"GGUF:        {arguments.gguf}")
    print(f"SAFETENSORS: {arguments.safetensors}")
    print(
        f"THRESHOLD:   min cosine {arguments.min_cosine:g}, "
        f"expected tensors {arguments.expected_tensors}",
    )
    if arguments.max_abs is not None:
        print(f"THRESHOLD:   max abs diff {arguments.max_abs:g}")
    if arguments.rel_fro_max is not None:
        print(f"THRESHOLD:   max relative Frobenius error {arguments.rel_fro_max:g}")
    print()

    verification = verify(
        gguf_path=arguments.gguf,
        safetensors_path=arguments.safetensors,
        expected_tensors=arguments.expected_tensors,
        min_cosine=arguments.min_cosine,
        max_abs=arguments.max_abs,
        rel_fro_max=arguments.rel_fro_max,
    )

    print(format_table(verification.reports))
    print()

    worst = verification.worst[: arguments.top_n]
    print(f"WORST {len(worst)} TENSORS BY COSINE")
    for report in worst:
        cosine = "-" if report.cosine is None else format(report.cosine, ".9g")
        max_abs = "-" if report.max_abs is None else format(report.max_abs, ".6g")
        relative = (
            "-"
            if report.relative_frobenius is None
            else format(report.relative_frobenius, ".6g")
        )
        print(
            f"  {report.sidecar_name:<46} {cosine:>14}  {max_abs:>13}  {relative:>13}  "
            f"{report.status}",
        )
    print()

    min_cosine = verification.min_cosine
    median_cosine = verification.median_cosine
    print(
        f"TENSOR COUNT: {verification.tensor_count} (mapping {verification.mapping_size}, "
        f"expected {verification.expected_tensors})",
    )
    print(
        f"MIN COSINE:    {min_cosine if min_cosine is None else format(min_cosine, '.17g')}"
    )
    print(
        f"MEDIAN COSINE: {median_cosine if median_cosine is None else format(median_cosine, '.17g')}",
    )
    print(f"FAILURES:      {len(verification.failures)}")
    for failure in verification.failures:
        print(f"FAIL: {failure}")

    if arguments.json_path is not None:
        summary = build_summary(
            verification,
            arguments.gguf,
            arguments.safetensors,
            arguments.min_cosine,
            arguments.max_abs,
            arguments.rel_fro_max,
        )
        arguments.json_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON: {arguments.json_path}")

    if verification.failures:
        print("RESULT: FAIL")
        return EXIT_FAIL
    print("RESULT: PASS")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
