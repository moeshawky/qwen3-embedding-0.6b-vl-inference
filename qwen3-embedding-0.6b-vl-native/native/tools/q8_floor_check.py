"""Establish the true Q8-vs-FP32 parity floor of the SigLIP2-VL image embedding.

Why this tool exists
--------------------
A previous session reported a Q8-vs-FP32 image-embedding cosine of ``0.9926``
without ever running the FP32 control, so that number cannot be attributed to
quantization: a broken harness produces exactly the same kind of number.  This
tool is fail-closed and refuses to publish a Q8 number it cannot defend.

* ``(A)`` FP32 CONTROL.  The model is loaded strictly (missing / unexpected /
  mismatched keys must be empty), and every fixture is replayed from the stored
  oracle inputs (pixel_values, pixel_attention_mask and spatial_shapes come
  from the ``npz``, never from the processor).  The replay must reproduce the
  oracle to ``cosine >= 0.999999``.  If it does not, the harness is broken:
  the tool reports the failure, writes the report and exits non-zero WITHOUT
  running the Q8 measurement at all.
* ``(B)`` Q8 MEASUREMENT.  Every vision-branch weight is replaced by the
  dequantised Q8_0 sidecar tensor.  The shape of each replacement is checked
  against the target parameter BEFORE the write, and the orientation is proved
  per tensor (see ``build_tensor_swaps``).  The same replay then runs.
* ``(C)`` INTERMEDIATES.  The cosine of every intermediate activation against
  the oracle is reported for both runs, so that a layout bug in the native
  implementation can be localised by stage rather than by guessing.

The sidecar stores the torch weights transposed into ggml order, so the numpy
view of a sidecar tensor already has the torch shape: a torch ``nn.Linear``
weight ``W`` of shape ``[out, in]`` appears as ``(out, in)`` and needs NO
transpose.  The only exception is the 28 zero-padded tensors (the 27
``v.blk.<i>.ffn_down.weight`` plus ``p.mlp.fc2.weight``) whose numpy view has 16
extra columns that are trimmed away.  Both facts are *verified*, never assumed:
the shape of every replacement is compared against the target parameter and the
cosine of the dequantised tensor against the original weight must be at least
``SWAP_MIN_COSINE``.

Exit codes
----------
``0``
    Control passed, Q8 measurement completed, both reports written.
``1``
    The FP32 control failed: the harness is invalid, so no Q8 number exists.
``2``
    An unexpected error aborted the run (e.g. a shape mismatch in a swap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from numpy.lib import format as npy_format
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.lib.npyio import NpzFile
    from numpy.typing import NDArray

MODEL_DIR = Path("/content/qwen3-embedding-0.6b-vl")
TOOLS_DIR = MODEL_DIR / "native" / "tools"
ORACLE_DIR = MODEL_DIR / "native" / "oracle"
SIDECAR_PATH = MODEL_DIR / "mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"
DEFAULT_JSON_PATH = ORACLE_DIR / "q8_floor_report.json"
DEFAULT_MD_PATH = ORACLE_DIR / "Q8_FLOOR_REPORT.md"

#: The fixtures, in the order used by the oracle manifest.
FIXTURE_NAMES: tuple[str, ...] = (
    "square",
    "landscape",
    "portrait",
    "odd",
    "tiny",
    "wide",
)

#: (dotted module path on the vision model, oracle key) hook pairs.
HOOK_TARGETS: tuple[tuple[str, str], ...] = (
    ("vision_model.embeddings", "patch_plus_position"),
    ("vision_model.encoder.layers.0", "encoder_layer_0"),
    ("vision_model.encoder.layers.1", "encoder_layer_1"),
    ("vision_model.encoder.layers.13", "encoder_layer_13"),
    ("vision_model.encoder.layers.26", "encoder_layer_26"),
    ("vision_model.post_layernorm", "post_layernorm"),
    ("vision_model.head", "map_pooler"),
)

#: Every tensor compared against the oracle, in report order.
COMPARED_KEYS: tuple[str, ...] = (
    "patch_plus_position",
    "encoder_layer_0",
    "encoder_layer_1",
    "encoder_layer_13",
    "encoder_layer_26",
    "post_layernorm",
    "map_pooler",
    "map_pooler_norm",
    "linear_projection",
    "image_embeds",
)

#: The FP32 control must reproduce the oracle at least this well.
CONTROL_MIN_COSINE = 0.999999

#: A dequantised sidecar tensor must match its source weight at least this well.
SWAP_MIN_COSINE = 0.999

#: The claimed parity floor threshold under test.
FLOOR_THRESHOLD = 0.999

EXIT_OK = 0
EXIT_CONTROL_FAILED = 1
EXIT_ERROR = 2

# The trust_remote_code modeling file imports its sibling
# ``configuration_qwen_siglip2_embedding``.  When this script is launched by
# path, ``sys.path[0]`` is this script's directory -- not the model directory --
# so the sibling module would not be importable and the dynamic-module loader
# would fail with "requires the following packages that were not found".
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


class VisionModelProtocol(Protocol):
    """Structural view of the ``trust_remote_code`` model used by this harness.

    Mirrors ``Qwen3Siglip2EmbeddingModel`` in ``modeling_qwen_siglip2_embedding.py``.
    """

    vision_model: nn.Module
    vision_projection: nn.Linear

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor: ...

    def get_parameter(self, target: str) -> nn.Parameter: ...


@dataclass(frozen=True)
class Fixture:
    """One oracle fixture: its name and the arrays stored in the ``npz``."""

    name: str
    arrays: dict[str, NDArray[np.generic]]


@dataclass(frozen=True)
class Comparison:
    """Cosine similarity and maximum absolute difference for one tensor."""

    cosine: float
    max_abs: float


@dataclass(frozen=True)
class FixtureResult:
    """Every comparison of one fixture against the oracle."""

    name: str
    comparisons: dict[str, Comparison]


@dataclass(frozen=True)
class RunResult:
    """The outcome of one full pass over the six fixtures."""

    label: str
    fixtures: list[FixtureResult]

    @property
    def image_embeds_cosines(self) -> dict[str, float]:
        """Per-fixture cosine of the final image embedding."""
        return {
            result.name: result.comparisons["image_embeds"].cosine
            for result in self.fixtures
        }

    @property
    def min_image_embeds_cosine(self) -> float:
        """Smallest image-embedding cosine over the fixtures."""
        return min(self.image_embeds_cosines.values())

    @property
    def min_image_embeds_fixture(self) -> str:
        """Fixture that carries the smallest image-embedding cosine."""
        cosines = self.image_embeds_cosines
        return min(cosines, key=lambda name: cosines[name])

    @property
    def max_image_embeds_max_abs(self) -> float:
        """Largest absolute deviation from the oracle over the fixtures."""
        return max(
            result.comparisons["image_embeds"].max_abs for result in self.fixtures
        )


@dataclass(frozen=True)
class RunOutcome:
    """One run: the comparisons against the oracle plus the raw tensors."""

    result: RunResult
    tensors: list[dict[str, NDArray[np.float32]]]


@dataclass(frozen=True)
class TensorSwap:
    """One sidecar tensor bound to a torch parameter, with its shape evidence."""

    sidecar_name: str
    target_path: str
    array: NDArray[np.float32]
    direct_cosine: float
    transposed_cosine: float | None
    padded: bool


@dataclass(frozen=True)
class SwapStats:
    """What the sidecar swap actually did to the model."""

    total: int
    changed: int
    unchanged: int
    padded: int
    min_direct_cosine: float
    min_direct_cosine_tensor: str
    max_transposed_cosine: float
    square_tensors_checked: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="q8_floor_check.py",
        description=(
            "Measure the Q8-vs-FP32 parity floor of the SigLIP2-VL image embedding."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="machine readable report path (default: %(default)s)",
    )
    parser.add_argument(
        "--md",
        dest="md_path",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="human readable report path (default: %(default)s)",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_similarity(left: NDArray[np.generic], right: NDArray[np.generic]) -> float:
    """Cosine similarity of two equally shaped arrays, computed in float64."""
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {left.shape} != {right.shape}")
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise RuntimeError("cannot take the cosine of a zero vector")
    return float(np.dot(a, b) / (norm_a * norm_b))


def max_abs_difference(left: NDArray[np.generic], right: NDArray[np.generic]) -> float:
    """Largest absolute element-wise difference of two equally shaped arrays."""
    if left.shape != right.shape:
        raise RuntimeError(f"shape mismatch: {left.shape} != {right.shape}")
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def load_fixture(name: str) -> Fixture:
    """Load one oracle fixture from its ``npz``."""
    path = ORACLE_DIR / f"oracle_{name}.npz"
    with cast("NpzFile", np.load(path)) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return Fixture(name=name, arrays=arrays)


def resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted module path, indexing into ``nn.ModuleList`` on digits."""
    obj: object = root
    for part in dotted.split("."):
        if isinstance(obj, nn.ModuleList):
            obj = obj[int(part)]
        elif isinstance(obj, nn.Module):
            obj = getattr(obj, part)
        else:
            raise RuntimeError(f"cannot resolve {dotted!r}: {part!r} is not a module")
    if not isinstance(obj, nn.Module):
        raise RuntimeError(f"{dotted!r} did not resolve to a module")
    return obj


def make_hook(key: str, sink: dict[str, torch.Tensor]) -> Callable[..., None]:
    """Build a forward hook that stores the module output under ``key``."""

    def _hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"hook {key!r} captured a non-tensor output: {type(output)!r}"
            )
        sink[key] = tensor.detach()

    return _hook


def load_model(device: torch.device) -> tuple[VisionModelProtocol, dict[str, object]]:
    """Load the model strictly in float32 and move it onto ``device``."""
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
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "non-strict load: "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    moved = cast("nn.Module", model).to(device=device, dtype=torch.float32).eval()
    return cast("VisionModelProtocol", moved), {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "error_msgs": list(loading_info.get("error_msgs") or []),
    }


def evaluate_fixture(
    model: VisionModelProtocol,
    fixture: Fixture,
    device: torch.device,
) -> dict[str, NDArray[np.float32]]:
    """Replay one fixture and return every intermediate plus the final embedding.

    The processor is deliberately not used: the oracle already stores the exact
    processor inputs, so replaying them isolates the model from the
    preprocessing path.
    """
    pixel_values = (
        torch.from_numpy(
            np.ascontiguousarray(fixture.arrays["pixel_values"], dtype=np.float32)
        )
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
    )
    pixel_attention_mask = (
        torch.from_numpy(
            np.ascontiguousarray(fixture.arrays["pixel_attention_mask"], dtype=np.int64)
        )
        .to(device=device, dtype=torch.int64)
        .unsqueeze(0)
    )
    spatial_shapes = (
        torch.from_numpy(
            np.ascontiguousarray(fixture.arrays["spatial_shapes"], dtype=np.int64)
        )
        .to(device=device, dtype=torch.int64)
        .unsqueeze(0)
    )

    captured: dict[str, torch.Tensor] = {}
    handles = []
    root = cast("nn.Module", model)
    for dotted, key in HOOK_TARGETS:
        module = resolve_module(root, dotted)
        handles.append(module.register_forward_hook(make_hook(key, captured)))
    try:
        with torch.no_grad():
            image_embeds = model.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [key for _, key in HOOK_TARGETS if key not in captured]
    if missing:
        raise RuntimeError(f"hooks never fired for: {missing}")

    map_pooler = captured["map_pooler"]
    map_pooler_norm = F.normalize(map_pooler.float(), p=2, dim=-1)
    with torch.no_grad():
        linear_projection = model.vision_projection(map_pooler_norm)

    outputs: dict[str, NDArray[np.float32]] = {
        "image_embeds": cast(
            "NDArray[np.float32]",
            image_embeds.detach().to(torch.float32).cpu().numpy().reshape(-1),
        ),
        "map_pooler": cast(
            "NDArray[np.float32]",
            map_pooler.detach().to(torch.float32).cpu().numpy().reshape(-1),
        ),
        "map_pooler_norm": cast(
            "NDArray[np.float32]",
            map_pooler_norm.detach().to(torch.float32).cpu().numpy().reshape(-1),
        ),
        "linear_projection": cast(
            "NDArray[np.float32]",
            linear_projection.detach().to(torch.float32).cpu().numpy().reshape(-1),
        ),
    }
    for _, key in HOOK_TARGETS:
        if key == "map_pooler":
            continue
        outputs[key] = cast(
            "NDArray[np.float32]",
            captured[key].detach().to(torch.float32).cpu().numpy()[0],
        )
    return outputs


def evaluate_run(
    model: VisionModelProtocol,
    fixtures: list[Fixture],
    device: torch.device,
    label: str,
) -> RunOutcome:
    """Replay every fixture, compare every tensor against the oracle.

    The raw tensors are kept so that the run can be persisted verbatim.
    """
    results: list[FixtureResult] = []
    tensors: list[dict[str, NDArray[np.float32]]] = []
    for fixture in fixtures:
        outputs = evaluate_fixture(model, fixture, device)
        comparisons: dict[str, Comparison] = {}
        for key in COMPARED_KEYS:
            if key not in outputs:
                raise RuntimeError(f"fixture {fixture.name}: missing output {key!r}")
            expected = fixture.arrays[key]
            actual = outputs[key]
            comparisons[key] = Comparison(
                cosine=cosine_similarity(actual, expected),
                max_abs=max_abs_difference(actual, expected),
            )
        results.append(FixtureResult(name=fixture.name, comparisons=comparisons))
        tensors.append(outputs)
    return RunOutcome(result=RunResult(label=label, fixtures=results), tensors=tensors)


def build_tensor_swaps(
    model: VisionModelProtocol, reader: GGUFReader
) -> list[TensorSwap]:
    """Bind every sidecar tensor to its torch parameter and verify the binding.

    Three independent checks per tensor:

    1. SHAPE.  The numpy view of the sidecar tensor must have exactly the
       shape of the target parameter (after trimming the 16 zero-padded
       columns of the two padded tensor families).  A mismatch raises.
    2. VALUE.  The cosine of the dequantised tensor against the original torch
       weight must be at least ``SWAP_MIN_COSINE``; this proves both that the
       dequantisation is faithful and that the values are not garbage.
    3. ORIENTATION.  For square 2-D tensors the shape check cannot
       discriminate a transposed binding, so the cosine of the transposed
       variant is computed and must be strictly worse than the direct one.
    """
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    from verify_sidecar_weights import (
        build_tensor_specs,
        sidecar_to_numpy,
        trim_padding,
    )

    specs = build_tensor_specs()
    sidecars = {tensor.name: tensor for tensor in reader.tensors}
    swaps: list[TensorSwap] = []
    for sidecar_name in sorted(specs):
        spec = specs[sidecar_name]
        if sidecar_name not in sidecars:
            raise RuntimeError(
                f"sidecar tensor {sidecar_name!r} is missing from the GGUF"
            )
        array = np.asarray(sidecar_to_numpy(sidecars[sidecar_name]), dtype=np.float32)
        padded = spec.trim_width is not None
        if spec.trim_width is not None:
            array, padding_is_zero = trim_padding(array, spec.trim_width)
            if not padding_is_zero:
                raise RuntimeError(f"{sidecar_name!r}: the padded columns are not zero")
        target = model.get_parameter(spec.source_name)
        array_shape = tuple(int(dim) for dim in array.shape)
        target_shape = tuple(int(dim) for dim in target.shape)
        if array_shape != target_shape:
            raise RuntimeError(
                f"{sidecar_name!r}: sidecar shape {array_shape} does not match "
                f"{spec.source_name!r} shape {target_shape}"
            )
        original = cast(
            "NDArray[np.float32]",
            target.detach().to(device="cpu", dtype=torch.float32).numpy(),
        )
        direct = cosine_similarity(array, original)
        if direct < SWAP_MIN_COSINE:
            raise RuntimeError(
                f"{sidecar_name!r}: dequantised tensor does not match "
                f"{spec.source_name!r} (cosine {direct:.9g} < {SWAP_MIN_COSINE:g})"
            )
        transposed: float | None = None
        if array.ndim == 2 and array_shape[0] == array_shape[1]:
            transposed = cosine_similarity(array.T, original)
            if transposed >= direct:
                raise RuntimeError(
                    f"{sidecar_name!r}: the transposed binding is not worse "
                    f"(transposed cosine {transposed:.9g} >= direct {direct:.9g})"
                )
        swaps.append(
            TensorSwap(
                sidecar_name=sidecar_name,
                target_path=spec.source_name,
                array=array,
                direct_cosine=direct,
                transposed_cosine=transposed,
                padded=padded,
            )
        )
    return swaps


def apply_tensor_swaps(
    model: VisionModelProtocol, swaps: list[TensorSwap]
) -> dict[str, int]:
    """Write every swapped tensor into its torch parameter.

    Returns how many parameters actually changed: tensors whose dequantised
    value is bit-identical to the source (the LayerNorm ones, whose weights
    are exactly 1.0 and biases exactly 0.0) are counted as unchanged.
    """
    changed = 0
    unchanged = 0
    with torch.no_grad():
        for swap in swaps:
            target = model.get_parameter(swap.target_path)
            original = cast(
                "NDArray[np.float32]",
                target.detach().to(device="cpu", dtype=torch.float32).numpy(),
            )
            if np.array_equal(original, swap.array):
                unchanged += 1
                continue
            changed += 1
            replacement = torch.from_numpy(
                np.ascontiguousarray(swap.array, dtype=np.float32)
            )
            target.copy_(replacement.to(device=target.device, dtype=target.dtype))
    return {"changed": changed, "unchanged": unchanged, "total": len(swaps)}


def save_npz_deterministic(path: Path, arrays: dict[str, NDArray[np.float32]]) -> None:
    """Write a compressed npz whose bytes are reproducible across runs.

    ``numpy.savez_compressed`` stamps every zip member with the current time,
    so two byte-identical runs produce different archives (and different
    sha256 digests).  Pinning the member timestamp to the zip epoch removes
    that source of variation; the payload format is unchanged and
    ``numpy.load`` reads the archive exactly as it reads a ``savez_compressed``
    archive.  The member order is fixed by the caller's key order.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(
                filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as handle:
                npy_format.write_array(handle, array, allow_pickle=False)  # type: ignore[no-untyped-call]


def persist_run_tensors(
    outcome: RunOutcome,
    prefix: str,
    fixtures: list[Fixture],
) -> dict[str, object]:
    """Write one archive per fixture and return their manifest entries.

    The archives mirror the oracle npz files exactly: same key names, same
    shapes, same orientation, float32 throughout.  Every shape is checked
    against the oracle before the archive is written, so a silent layout drift
    cannot reach disk.
    """
    manifest: dict[str, object] = {}
    for fixture, tensors in zip(fixtures, outcome.tensors, strict=True):
        for key in COMPARED_KEYS:
            if key not in tensors:
                raise RuntimeError(f"fixture {fixture.name}: missing tensor {key!r}")
            expected_shape = tuple(int(dim) for dim in fixture.arrays[key].shape)
            actual_shape = tuple(int(dim) for dim in tensors[key].shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"fixture {fixture.name}: {key} shape {actual_shape} != "
                    f"oracle shape {expected_shape}"
                )
            if tensors[key].dtype != np.float32:
                raise RuntimeError(
                    f"fixture {fixture.name}: {key} dtype {tensors[key].dtype} "
                    "!= float32"
                )
        path = ORACLE_DIR / f"{prefix}_{fixture.name}.npz"
        save_npz_deterministic(path, {key: tensors[key] for key in COMPARED_KEYS})
        manifest[f"{prefix}_{fixture.name}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "keys": list(COMPARED_KEYS),
        }
    return manifest


def _mapping(value: object) -> dict[str, object]:
    """Narrow a JSON value to a mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value)!r}")
    return cast("dict[str, object]", value)


def _sequence(value: object) -> list[object]:
    """Narrow a JSON value to a list."""
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON list, got {type(value)!r}")
    return cast("list[object]", value)


def _text(container: dict[str, object], key: str) -> str:
    """Read a string field out of a JSON mapping."""
    value = container.get(key)
    if not isinstance(value, str):
        raise TypeError(f"expected a string at {key!r}, got {type(value)!r}")
    return value


def _number(container: dict[str, object], key: str) -> float:
    """Read a numeric field out of a JSON mapping."""
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number at {key!r}, got {type(value)!r}")
    return float(value)


def _optional_number(container: dict[str, object], key: str) -> float | None:
    """Read a numeric field that may be absent."""
    if key not in container:
        return None
    return _number(container, key)


def load_previous_report(path: Path) -> dict[str, object] | None:
    """Load a previously written report, if any, for the drift check."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return cast("dict[str, object]", data)


def previous_cosines(
    previous: dict[str, object], section: str
) -> dict[str, float] | None:
    """Extract the per-fixture image-embedding cosines of a previous report."""
    try:
        payload = previous.get(section)
        if not isinstance(payload, dict):
            return None
        cosines: dict[str, float] = {}
        for raw in _sequence(payload.get("fixtures")):
            row = _mapping(raw)
            cosines[_text(row, "name")] = _number(
                _mapping(row.get("image_embeds")), "cosine"
            )
        return cosines
    except (TypeError, KeyError):
        return None


def previous_floor(previous: dict[str, object]) -> float | None:
    """Extract the Q8 floor of a previous report."""
    try:
        q8 = previous.get("q8")
        if not isinstance(q8, dict):
            return None
        floor = q8.get("floor")
        if not isinstance(floor, dict):
            return None
        return _number(cast("dict[str, object]", floor), "min_cosine")
    except TypeError:
        return None


def max_abs_drift(drift: dict[str, float]) -> float:
    """Largest absolute drift of a cosine map."""
    return max((abs(value) for value in drift.values()), default=0.0)


def cosine_drift(
    previous_map: dict[str, float] | None, current: dict[str, float]
) -> dict[str, float]:
    """Per-fixture cosine drift against a previous run."""
    if previous_map is None:
        return {}
    return {
        name: current[name] - old
        for name, old in previous_map.items()
        if name in current
    }


def environment_report(device: torch.device) -> dict[str, object]:
    """Collect the environment facts that make the numbers reproducible."""
    from transformers import __version__ as transformers_version

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else "none"
        ),
        "dtype": "float32",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def run_payload(run: RunResult, fixtures: list[Fixture]) -> dict[str, object]:
    """Serialise one run (control or Q8) into the JSON payload."""
    return {
        "label": run.label,
        "min_image_embeds_cosine": run.min_image_embeds_cosine,
        "min_image_embeds_fixture": run.min_image_embeds_fixture,
        "fixtures": [
            {
                "name": result.name,
                "image_embeds": {
                    "cosine": result.comparisons["image_embeds"].cosine,
                    "max_abs_diff": result.comparisons["image_embeds"].max_abs,
                },
                "intermediates": {
                    key: {
                        "cosine": result.comparisons[key].cosine,
                        "max_abs_diff": result.comparisons[key].max_abs,
                    }
                    for key in COMPARED_KEYS
                    if key != "image_embeds"
                },
            }
            for result in run.fixtures
        ],
        "oracle_inputs": {
            fixture.name: {
                "pixel_values_shape": list(fixture.arrays["pixel_values"].shape),
                "pixel_attention_mask_shape": list(
                    fixture.arrays["pixel_attention_mask"].shape
                ),
                "spatial_shapes": [
                    int(value) for value in fixture.arrays["spatial_shapes"].ravel()
                ],
            }
            for fixture in fixtures
        },
    }


def render_markdown(
    environment: dict[str, object],
    load_info: dict[str, object],
    control: RunResult,
    control_passed: bool,
    q8_run: RunResult,
    stats: SwapStats,
    artifacts: dict[str, str],
    persisted: dict[str, object],
    confirmation: dict[str, object],
) -> str:
    """Render the human readable report from the typed results."""
    floor = q8_run.min_image_embeds_cosine
    meets = floor >= FLOOR_THRESHOLD
    lines: list[str] = []
    lines.append("# Q8 vs FP32 Parity Floor -- SigLIP2-VL Image Embedding")
    lines.append("")
    lines.append("Generated by `native/tools/q8_floor_check.py`.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- FP32 control: **{'PASS' if control_passed else 'FAIL'}** "
        f"(min image-embedding cosine {control.min_image_embeds_cosine:.12f}, "
        f"threshold {CONTROL_MIN_COSINE:g})."
    )
    lines.append(f"- Q8 floor (minimum image-embedding cosine): **{floor:.12f}**.")
    lines.append(
        f"- Is the floor >= {FLOOR_THRESHOLD:g}? **{'YES' if meets else 'NO'}**."
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for key, value in environment.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## (A) FP32 control")
    lines.append("")
    lines.append(
        f"Strict load: missing={load_info['missing_keys']} "
        f"unexpected={load_info['unexpected_keys']} "
        f"mismatched={load_info['mismatched_keys']}."
    )
    lines.append("")
    lines.append("| fixture | cosine vs oracle | max abs diff |")
    lines.append("| --- | ---: | ---: |")
    for result in control.fixtures:
        comparison = result.comparisons["image_embeds"]
        lines.append(
            f"| {result.name:<10} | {comparison.cosine:>18.12f} | "
            f"{comparison.max_abs:>18.9g} |"
        )
    lines.append("")
    lines.append(
        f"Minimum control cosine: {control.min_image_embeds_cosine:.12f} "
        f"on fixture {control.min_image_embeds_fixture}."
    )
    lines.append("")

    lines.append("## (B) Q8 measurement")
    lines.append("")
    lines.append(
        f"Swapped {stats.total} sidecar tensors into the model "
        f"({stats.changed} changed, {stats.unchanged} bit-identical to the "
        f"source, {stats.padded} zero-padded and trimmed). Minimum per-tensor "
        f"cosine against the source weight: {stats.min_direct_cosine:.12f} "
        f"({stats.min_direct_cosine_tensor}). Worst transposed-variant cosine "
        f"over the {stats.square_tensors_checked} square tensors: "
        f"{stats.max_transposed_cosine:.12f}."
    )
    lines.append("")
    lines.append("| fixture | cosine vs FP32 oracle | max abs diff |")
    lines.append("| --- | ---: | ---: |")
    for result in q8_run.fixtures:
        comparison = result.comparisons["image_embeds"]
        lines.append(
            f"| {result.name:<10} | {comparison.cosine:>18.12f} | "
            f"{comparison.max_abs:>18.9g} |"
        )
    lines.append("")
    lines.append(
        f"**Q8 floor = {floor:.12f}** (minimum over the six fixtures, on "
        f"{q8_run.min_image_embeds_fixture})."
    )
    lines.append("")
    lines.append(
        f"The floor is {'at or above' if meets else 'below'} the required "
        f"{FLOOR_THRESHOLD:g}."
    )
    lines.append("")
    lines.append(f"### Floor-setting fixture: {q8_run.min_image_embeds_fixture}")
    lines.append("")
    lines.append("| stage | cosine vs FP32 oracle | max abs diff |")
    lines.append("| --- | ---: | ---: |")
    floor_result = next(
        result
        for result in q8_run.fixtures
        if result.name == q8_run.min_image_embeds_fixture
    )
    for key in COMPARED_KEYS:
        comparison = floor_result.comparisons[key]
        lines.append(
            f"| {key} | {comparison.cosine:>18.12f} | {comparison.max_abs:>18.9g} |"
        )
    lines.append("")

    lines.append("## (C) Intermediate cosines against the oracle")
    lines.append("")
    for label, run in (("FP32 control", control), ("Q8", q8_run)):
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| fixture | " + " | ".join(COMPARED_KEYS) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in COMPARED_KEYS) + " |")
        for result in run.fixtures:
            cells = " | ".join(
                f"{result.comparisons[key].cosine:.12f}" for key in COMPARED_KEYS
            )
            lines.append(f"| {result.name:<10} | {cells} |")
        lines.append("")

    lines.append("## (D) Persisted tensors and confirmation")
    lines.append("")
    lines.append(
        "| fixture | fp32 cosine vs oracle | q8 cosine vs oracle | "
        "fp32 max abs diff | q8 max abs diff |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for control_result, q8_result in zip(
        control.fixtures, q8_run.fixtures, strict=True
    ):
        control_comparison = control_result.comparisons["image_embeds"]
        q8_comparison = q8_result.comparisons["image_embeds"]
        lines.append(
            f"| {control_result.name:<10} | {control_comparison.cosine:>18.12f} | "
            f"{q8_comparison.cosine:>18.12f} | {control_comparison.max_abs:>14.9g} | "
            f"{q8_comparison.max_abs:>14.9g} |"
        )
    lines.append("")
    previous_floor_value = _optional_number(confirmation, "previous_floor")
    floor_drift = _optional_number(confirmation, "floor_drift")
    max_control_drift = _number(confirmation, "max_control_cosine_drift")
    max_q8_drift = _number(confirmation, "max_q8_cosine_drift")
    lines.append(
        "- Floor now: "
        f"**{floor:.12f}**; previous: "
        f"{'n/a' if previous_floor_value is None else format(previous_floor_value, '.12f')}"
        f"; drift: {'n/a' if floor_drift is None else format(floor_drift, '.3g')}"
        f"; **{'UNCHANGED' if confirmation.get('floor_unchanged') else 'MOVED -- INVESTIGATE'}**."
    )
    lines.append(
        "- Max cosine drift against the previous report: control "
        f"{format(max_control_drift, '.3g')}, Q8 {format(max_q8_drift, '.3g')}."
    )
    lines.append("")
    lines.append("| archive | path | sha256 | keys |")
    lines.append("| --- | --- | --- | --- |")
    for name, entry in persisted.items():
        archive = _mapping(entry)
        keys = ", ".join(str(key) for key in cast("list[object]", archive["keys"]))
        lines.append(
            f"| {name} | `{archive['path']}` | `{archive['sha256']}` | {keys} |"
        )
    lines.append("")
    lines.append(
        "Every archive holds the same ten keys as the oracle npz files, with the "
        "same shapes and the same orientation, float32 throughout."
    )
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("| artifact | sha256 |")
    lines.append("| --- | --- |")
    for name, digest in artifacts.items():
        lines.append(f"| {name} | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace) -> int:
    """Run the control, then the Q8 measurement, then write both reports."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 could not be disabled; the oracle is TF32-free")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = environment_report(device)
    print("Q8 FLOOR CHECK")
    print(f"[env] {json.dumps(environment, sort_keys=True)}")

    model, load_info = load_model(device)
    print(
        f"[load] missing={load_info['missing_keys']} "
        f"unexpected={load_info['unexpected_keys']} "
        f"mismatched={load_info['mismatched_keys']} -> STRICT"
    )

    fixtures = [load_fixture(name) for name in FIXTURE_NAMES]
    previous = load_previous_report(args.json_path)
    control_outcome = evaluate_run(model, fixtures, device, label="fp32_control")
    control = control_outcome.result
    control_passed = control.min_image_embeds_cosine >= CONTROL_MIN_COSINE
    control_payload = run_payload(control, fixtures)
    control_payload["passed"] = control_passed
    control_payload["threshold_cosine"] = CONTROL_MIN_COSINE
    persisted: dict[str, object] = dict(
        persist_run_tensors(control_outcome, "fp32", fixtures)
    )
    print()
    print("(A) FP32 CONTROL")
    for result in control.fixtures:
        comparison = result.comparisons["image_embeds"]
        print(
            f"  {result.name:<10} cosine={comparison.cosine:.12f} "
            f"max_abs_diff={comparison.max_abs:.9g}"
        )
    print(
        f"  minimum control cosine = {control.min_image_embeds_cosine:.12f} "
        f"(threshold {CONTROL_MIN_COSINE:g})"
    )

    artifacts = {
        SIDECAR_PATH.name: sha256_file(SIDECAR_PATH),
        SAFETENSORS_PATH.name: sha256_file(SAFETENSORS_PATH),
    }
    payload: dict[str, object] = {
        "tool": "q8_floor_check",
        "generated_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "load": load_info,
        "control_fp32": control_payload,
        "artifacts": artifacts,
    }

    if not control_passed:
        print()
        print("CONTROL FAILED: the harness does not reproduce the oracle.")
        print("The Q8 measurement is NOT attempted; no floor can be claimed.")
        payload["q8"] = {
            "status": "NOT_RUN",
            "reason": "the FP32 control did not reproduce the oracle",
        }
        args.json_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        args.md_path.write_text(
            "# Q8 vs FP32 Parity Floor -- CONTROL FAILED\n\n"
            "The FP32 control did not reproduce the stored oracle, so the harness "
            "is invalid and no Q8 floor was measured.\n",
            encoding="utf-8",
        )
        return EXIT_CONTROL_FAILED

    reader = GGUFReader(str(SIDECAR_PATH))
    swaps = build_tensor_swaps(model, reader)
    swap_counts = apply_tensor_swaps(model, swaps)
    if swap_counts["changed"] == 0:
        raise RuntimeError(
            "the sidecar swap changed no weight; the Q8 run would be a copy of "
            "the FP32 control"
        )
    transposed = [
        swap.transposed_cosine for swap in swaps if swap.transposed_cosine is not None
    ]
    stats = SwapStats(
        total=swap_counts["total"],
        changed=swap_counts["changed"],
        unchanged=swap_counts["unchanged"],
        padded=sum(1 for swap in swaps if swap.padded),
        min_direct_cosine=min(swap.direct_cosine for swap in swaps),
        min_direct_cosine_tensor=min(
            swaps, key=lambda swap: swap.direct_cosine
        ).sidecar_name,
        max_transposed_cosine=max(transposed) if transposed else float("nan"),
        square_tensors_checked=len(transposed),
    )

    q8_outcome = evaluate_run(model, fixtures, device, label="q8")
    q8_run = q8_outcome.result
    q8_payload = run_payload(q8_run, fixtures)
    persisted.update(persist_run_tensors(q8_outcome, "q8", fixtures))
    q8_payload["swaps"] = {
        "total": stats.total,
        "changed": stats.changed,
        "unchanged": stats.unchanged,
        "padded": stats.padded,
        "min_direct_cosine": stats.min_direct_cosine,
        "min_direct_cosine_tensor": stats.min_direct_cosine_tensor,
        "max_transposed_cosine": stats.max_transposed_cosine,
        "square_tensors_checked": stats.square_tensors_checked,
    }
    floor = q8_run.min_image_embeds_cosine
    q8_payload["floor"] = {
        "min_cosine": floor,
        "min_cosine_fixture": q8_run.min_image_embeds_fixture,
        "threshold": FLOOR_THRESHOLD,
        "meets_threshold": floor >= FLOOR_THRESHOLD,
    }
    q8_payload["negative_control"] = {
        "weights_changed": stats.changed,
        "q8_max_abs_diff_vs_fp32_oracle": q8_run.max_image_embeds_max_abs,
        "passed": (stats.changed > 0 and q8_run.max_image_embeds_max_abs > 0.0),
    }
    payload["q8"] = q8_payload

    control_drift = cosine_drift(
        previous_cosines(previous, "control_fp32") if previous else None,
        control.image_embeds_cosines,
    )
    q8_drift = cosine_drift(
        previous_cosines(previous, "q8") if previous else None,
        q8_run.image_embeds_cosines,
    )
    previous_floor_value = previous_floor(previous) if previous else None
    floor_drift = (
        floor - previous_floor_value if previous_floor_value is not None else None
    )
    floor_unchanged = previous_floor_value is None or floor == previous_floor_value
    confirmation: dict[str, object] = {
        "previous_report_available": previous is not None,
        "control_cosine_drift": control_drift,
        "q8_cosine_drift": q8_drift,
        "max_control_cosine_drift": max_abs_drift(control_drift),
        "max_q8_cosine_drift": max_abs_drift(q8_drift),
        "previous_floor": previous_floor_value,
        "current_floor": floor,
        "floor_drift": floor_drift,
        "floor_unchanged": floor_unchanged,
    }
    payload["persisted_tensors"] = persisted
    payload["confirmation"] = confirmation

    print()
    print("(B) Q8 MEASUREMENT")
    print(
        f"  swapped {stats.total} tensors ({stats.changed} changed, "
        f"{stats.unchanged} identical), min per-tensor cosine "
        f"{stats.min_direct_cosine:.12f} on {stats.min_direct_cosine_tensor}, "
        f"worst transposed-variant cosine {stats.max_transposed_cosine:.12f}"
    )
    for result in q8_run.fixtures:
        comparison = result.comparisons["image_embeds"]
        print(
            f"  {result.name:<10} cosine={comparison.cosine:.12f} "
            f"max_abs_diff={comparison.max_abs:.9g}"
        )
    print(f"  Q8 FLOOR = {floor:.12f} (on {q8_run.min_image_embeds_fixture})")
    print(
        f"  floor >= {FLOOR_THRESHOLD:g}? {'YES' if floor >= FLOOR_THRESHOLD else 'NO'}"
    )

    print()
    print("(C) INTERMEDIATE COSINES")
    for label, run in (("fp32", control), ("q8", q8_run)):
        print(f"  {label}")
        header = (
            "    " + f"{'fixture':<10}" + "".join(f"{key:>22}" for key in COMPARED_KEYS)
        )
        print(header)
        for result in run.fixtures:
            cells = "".join(
                f"{result.comparisons[key].cosine:>22.12f}" for key in COMPARED_KEYS
            )
            print(f"    {result.name:<10}" + cells)

    print()
    print("(D) PERSISTED TENSORS AND CONFIRMATION TABLE")
    print(
        f"  {'fixture':<10} {'fp32 cosine':>18} {'q8 cosine':>18} "
        f"{'fp32 max_abs':>14} {'q8 max_abs':>14}"
    )
    for control_result, q8_result in zip(
        control.fixtures, q8_run.fixtures, strict=True
    ):
        control_comparison = control_result.comparisons["image_embeds"]
        q8_comparison = q8_result.comparisons["image_embeds"]
        print(
            f"  {control_result.name:<10} {control_comparison.cosine:>18.12f} "
            f"{q8_comparison.cosine:>18.12f} "
            f"{control_comparison.max_abs:>14.9g} {q8_comparison.max_abs:>14.9g}"
        )
    print(
        f"  floor now {floor:.12f} | previous "
        f"{'n/a' if previous_floor_value is None else f'{previous_floor_value:.12f}'}"
        f" | drift {'n/a' if floor_drift is None else f'{floor_drift:.3g}'}"
        f" | {'UNCHANGED' if floor_unchanged else 'MOVED -- INVESTIGATE'}"
    )
    print(
        f"  max control cosine drift "
        f"{max_abs_drift(control_drift):.3g} | max q8 cosine drift "
        f"{max_abs_drift(q8_drift):.3g}"
    )
    print(f"  wrote {len(persisted)} archives into {ORACLE_DIR}")

    args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_path.write_text(
        render_markdown(
            environment,
            load_info,
            control,
            control_passed,
            q8_run,
            stats,
            artifacts,
            persisted,
            confirmation,
        )
        + "\n",
        encoding="utf-8",
    )
    print()
    print(f"JSON: {args.json_path}")
    print(f"MD:   {args.md_path}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        return run_experiment(args)
    except Exception:
        traceback.print_exc()
        args.json_path.write_text(
            json.dumps(
                {
                    "tool": "q8_floor_check",
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "status": "ERROR",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
