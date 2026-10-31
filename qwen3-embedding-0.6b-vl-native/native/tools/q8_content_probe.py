"""Content-dependence probe for the Q8 image-embedding parity floor.

The question
------------
Is the Q8 parity floor a property of the **sidecar alone** (a constant that any
input must respect) or of the **input's spatial frequency content**?

The published fixture floor is 0.994736746 for ``tiny`` -- a 112x112 source
that the HF processor upscales to 256x256 -- against 0.9990..0.9997 for the
other five fixtures, which are natively larger.  ``tiny`` is the smoothest
fixture, so the two explanations are confounded by the fixture set.  This probe
breaks the confound with three images that share *everything* except their
noise:

* ``tiny_smooth`` -- the construction of the published ``tiny`` fixture
  (smooth gradient + rectangles, per-pixel Gaussian noise std 12).  It is
  byte-identical to ``native/fixtures/tiny.png`` and therefore reproduces the
  published floor; that is the calibration anchor of the whole probe.
* ``tiny_noisy``  -- ``tiny_smooth`` plus an additional per-pixel noise
  component of std 60 (high frequency).
* ``tiny_flat``   -- ``tiny_smooth`` with the noise removed (std 0): a purely
  smooth, very low frequency image.

All three are 112x112, so all three land on the same 16x16 / 256-patch grid.

Method
------
For every image the FP32 embedding is computed FIRST with the strictly loaded
FP32 model; the Q8 run then replaces every vision weight with the dequantised
sidecar tensor and recomputes the embedding.  The reported floor is the cosine
of the Q8 embedding against that image's own FP32 embedding -- never against a
stored oracle.  The negative control runs the identical evaluation path with
the sidecar weights NOT swapped in and must return cosine 1.0, which proves
the probe discriminates "Q8 weights" from "FP32 weights" on the same input.

Exit codes
----------
``0``
    Probe completed, all three floors measured, negative control passed.
``1``
    The negative control failed: the probe cannot discriminate, so the floors
    it reports are meaningless.
``2``
    An unexpected error aborted the run.
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
from gguf import GGUFReader
from numpy.lib import format as npy_format
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from transformers import BaseImageProcessor

MODEL_DIR = Path("/content/qwen3-embedding-0.6b-vl")
TOOLS_DIR = MODEL_DIR / "native" / "tools"
FIXTURES_DIR = MODEL_DIR / "native" / "fixtures"
ORACLE_DIR = MODEL_DIR / "native" / "oracle"
SIDECAR_PATH = MODEL_DIR / "mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"
DEFAULT_JSON_PATH = ORACLE_DIR / "content_probe_report.json"

#: The three probe images: name -> (base noise std, extra high-frequency std).
PROBE_SPECS: dict[str, tuple[float, float]] = {
    "tiny_smooth": (12.0, 0.0),
    "tiny_noisy": (12.0, 60.0),
    "tiny_flat": (0.0, 0.0),
}

#: Every probe image shares the geometry of the published ``tiny`` fixture.
SOURCE_HEIGHT = 112
SOURCE_WIDTH = 112
SOURCE_SEED = 1005

#: The grid every probe image must land on.
EXPECTED_GRID = (16, 16)
EXPECTED_VALID_PATCHES = 256
PATCH_SIZE = 16
GRID = 16
NUM_CHANNELS = 3

#: Rectangles as (x0, y0, x1, y1) fractions of (width, height) -- copied
#: verbatim from ``native/tools/capture_oracle.py`` so that ``tiny_smooth`` is
#: byte-identical to the published ``tiny`` fixture.
RECTANGLES: tuple[tuple[float, float, float, float], ...] = (
    (0.10, 0.10, 0.40, 0.45),
    (0.50, 0.55, 0.85, 0.90),
    (0.55, 0.05, 0.95, 0.40),
)

#: Solid RGB colour of each rectangle, in the order of :data:`RECTANGLES`.
RECTANGLE_COLOURS: tuple[tuple[int, int, int], ...] = (
    (220, 30, 30),
    (30, 220, 30),
    (30, 30, 220),
)

#: The image whose Q8 path is re-run with FP32 weights as the negative control.
NEGATIVE_CONTROL_IMAGE = "tiny_noisy"

#: (dotted module path on the model, captured key) hook pairs -- identical to
#: ``native/tools/q8_floor_check.py``.
HOOK_TARGETS: tuple[tuple[str, str], ...] = (
    ("vision_model.embeddings", "patch_plus_position"),
    ("vision_model.encoder.layers.0", "encoder_layer_0"),
    ("vision_model.encoder.layers.1", "encoder_layer_1"),
    ("vision_model.encoder.layers.13", "encoder_layer_13"),
    ("vision_model.encoder.layers.26", "encoder_layer_26"),
    ("vision_model.post_layernorm", "post_layernorm"),
    ("vision_model.head", "map_pooler"),
)

#: The published floor of the ``tiny`` fixture, used as a calibration anchor.
PUBLISHED_TINY_FLOOR = 0.994736746231073

#: A dequantised sidecar tensor must match its source weight at least this well.
SWAP_MIN_COSINE = 0.999

EXIT_OK = 0
EXIT_NEGATIVE_CONTROL_FAILED = 1
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
class ProbeImage:
    """One probe image: its source, its processor inputs and its statistics."""

    name: str
    png_path: Path
    source: NDArray[np.uint8]
    pixel_values: NDArray[np.float32]
    pixel_attention_mask: NDArray[np.int32]
    spatial_shapes: NDArray[np.int64]
    laplacian_mean_abs: float
    patch_mean_std: float
    model_input_std: float


@dataclass(frozen=True)
class ProbeOutcome:
    """The FP32 and Q8 tensors of one probe image."""

    name: str
    fp32: dict[str, NDArray[np.float32]]
    q8: dict[str, NDArray[np.float32]]
    negative_control: dict[str, object] | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="q8_content_probe.py",
        description=(
            "Measure whether the Q8 parity floor depends on the input's frequency "
            "content."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="summary report path (default: %(default)s)",
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


def save_npz_deterministic(path: Path, arrays: dict[str, NDArray[np.generic]]) -> None:
    """Write a compressed npz whose bytes are reproducible across runs.

    ``numpy.savez_compressed`` stamps every zip member with the current time,
    so two byte-identical runs produce different archives.  Pinning the member
    timestamp to the zip epoch removes that variation; the payload format is
    unchanged and ``numpy.load`` reads the archive exactly as it reads a
    ``savez_compressed`` archive.
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


def build_source_image(
    base_noise_std: float, extra_noise_std: float
) -> NDArray[np.uint8]:
    """Build one probe source image.

    The recipe is the one of ``native/tools/capture_oracle.py``: a per-channel
    linear gradient, three axis-aligned rectangles, then additive per-pixel
    Gaussian noise, then a clip to ``[0, 255]`` and a round to ``uint8``.  The
    extra high-frequency component (if any) is drawn from the same generator
    after the base noise, so the base draw is identical across the three probe
    images and only the noise level differs.
    """
    height, width = SOURCE_HEIGHT, SOURCE_WIDTH
    rng = np.random.RandomState(SOURCE_SEED)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float64).reshape(height, 1)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float64).reshape(1, width)

    image = np.empty((height, width, 3), dtype=np.float64)
    for channel in range(3):
        image[:, :, channel] = (
            (40.0 + 30.0 * channel)
            + (60.0 + 20.0 * channel) * xs
            + (40.0 + 15.0 * channel) * ys
        )

    for rect, colour in zip(RECTANGLES, RECTANGLE_COLOURS, strict=True):
        x0, y0, x1, y1 = rect
        ix0, ix1 = round(x0 * width), round(x1 * width)
        iy0, iy1 = round(y0 * height), round(y1 * height)
        image[iy0:iy1, ix0:ix1, :] = colour

    image += rng.normal(0.0, base_noise_std, size=(height, width, 3))
    if extra_noise_std > 0.0:
        image += rng.normal(0.0, extra_noise_std, size=(height, width, 3))
    np.clip(image, 0.0, 255.0, out=image)
    return cast("NDArray[np.uint8]", np.rint(image).astype(np.uint8))


def to_gray(image: NDArray[np.generic]) -> NDArray[np.float64]:
    """Rec.601 luma of an RGB array, returned in the array's own scale."""
    rgb = np.asarray(image, dtype=np.float64)
    return cast(
        "NDArray[np.float64]",
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
    )


def laplacian_mean_abs(gray: NDArray[np.float64]) -> float:
    """Mean absolute 4-neighbour Laplacian of a 2-D array (borders cropped).

    ``L[i, j] = x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1] - 4 x[i, j]``.
    A flat image scores 0; per-pixel noise scores roughly ``8 * sigma`` for
    Gaussian noise, so the statistic measures how much high-frequency energy
    the image carries.
    """
    core = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * core
    )
    return float(np.mean(np.abs(laplacian)))


def patch_mean_std(gray_hw: NDArray[np.float64]) -> float:
    """Standard deviation of the per-patch mean intensity over the patch grid."""
    means = gray_hw.reshape(GRID, PATCH_SIZE, GRID, PATCH_SIZE).mean(axis=(1, 3))
    return float(np.std(means.ravel()))


def patchify_model_input(image_hw_c: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reproduce the processor's patch layout from a ``[H, W, C]`` image."""
    patches = image_hw_c.reshape(
        GRID, PATCH_SIZE, GRID, PATCH_SIZE, NUM_CHANNELS
    ).transpose(0, 2, 1, 3, 4)
    return patches.reshape(GRID * GRID, PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS)


def reconstruct_model_input(
    pixel_values: NDArray[np.float32],
) -> NDArray[np.float64]:
    """Recover the 256x256 RGB model input from the normalised patches.

    The processor normalises with ``(x - 0.5) / 0.5`` and patches the image
    into ``[grid_h, grid_w, patch_h, patch_w, channel]`` row-major order.  The
    round trip is verified by re-patching the reconstruction and comparing it
    with the processor output.
    """
    patches = np.asarray(pixel_values, dtype=np.float64).reshape(
        GRID, GRID, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS
    )
    image = patches.transpose(0, 2, 1, 3, 4).reshape(
        GRID * PATCH_SIZE, GRID * PATCH_SIZE, NUM_CHANNELS
    )
    return image * 0.5 + 0.5


def prepare_probe_image(
    name: str,
    base_noise_std: float,
    extra_noise_std: float,
    processor: BaseImageProcessor,
) -> ProbeImage:
    """Render one probe image, run the processor on it and measure it."""
    source = build_source_image(base_noise_std, extra_noise_std)
    png_path = FIXTURES_DIR / f"{name}.png"
    Image.fromarray(source).save(png_path, format="PNG")

    with Image.open(png_path) as opened:
        opened.load()
        inputs = processor(images=[opened], return_tensors="pt")
    pixel_values = np.asarray(inputs["pixel_values"][0], dtype=np.float32)
    pixel_attention_mask = np.asarray(inputs["pixel_attention_mask"][0], dtype=np.int32)
    spatial_shapes = np.asarray(inputs["spatial_shapes"][0], dtype=np.int64)

    grid = (int(spatial_shapes[0]), int(spatial_shapes[1]))
    valid = int(pixel_attention_mask.sum())
    if grid != EXPECTED_GRID:
        raise RuntimeError(f"{name}: grid {grid} != {EXPECTED_GRID}")
    if valid != EXPECTED_VALID_PATCHES:
        raise RuntimeError(f"{name}: {valid} valid patches != {EXPECTED_VALID_PATCHES}")
    if pixel_values.shape != (EXPECTED_VALID_PATCHES, 768):
        raise RuntimeError(f"{name}: pixel_values shape {pixel_values.shape}")

    model_input = reconstruct_model_input(pixel_values)
    denormalised = pixel_values.astype(np.float64) * 0.5 + 0.5
    layout_error = float(
        np.max(np.abs(patchify_model_input(model_input) - denormalised))
    )
    if layout_error > 1e-5:
        raise RuntimeError(
            f"{name}: the patch layout round trip failed (max |diff| {layout_error:.3g})"
        )
    # Independent, discriminating layout check: the reconstruction must match
    # the source resized the way the processor resizes it (bilinear to 256x256).
    # A transposed patch grid misses by ~26/255 here, so the tolerance can be
    # tight.
    with Image.open(png_path) as opened:
        opened.load()
        resized = (
            np.asarray(
                opened.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            / 255.0
        )
    resize_error = (
        float(np.mean(np.abs(to_gray(model_input) - to_gray(resized)))) * 255.0
    )
    if resize_error > 0.5:
        raise RuntimeError(
            f"{name}: the reconstructed model input does not match the resized "
            f"source (mean |diff| {resize_error:.4g}/255); the patch layout is wrong"
        )

    gray_source = to_gray(source)
    gray_input = to_gray(model_input * 255.0)
    return ProbeImage(
        name=name,
        png_path=png_path,
        source=source,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        laplacian_mean_abs=laplacian_mean_abs(gray_source),
        patch_mean_std=patch_mean_std(gray_input),
        model_input_std=float(np.std(gray_input)),
    )


def evaluate_image(
    model: VisionModelProtocol,
    image: ProbeImage,
    device: torch.device,
) -> dict[str, NDArray[np.float32]]:
    """Run one probe image through the model and return every intermediate.

    This is the same evaluation path as ``q8_floor_check.py``: the processor
    inputs of the image are replayed through ``get_image_features`` under
    ``torch.no_grad()`` with the intermediate activations captured by hooks.
    """
    pixel_values = (
        torch.from_numpy(np.ascontiguousarray(image.pixel_values, dtype=np.float32))
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
    )
    pixel_attention_mask = (
        torch.from_numpy(
            np.ascontiguousarray(image.pixel_attention_mask, dtype=np.int64)
        )
        .to(device=device, dtype=torch.int64)
        .unsqueeze(0)
    )
    spatial_shapes = (
        torch.from_numpy(np.ascontiguousarray(image.spatial_shapes, dtype=np.int64))
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
        raise RuntimeError(f"{image.name}: hooks never fired for {missing}")

    outputs: dict[str, NDArray[np.float32]] = {
        "image_embeds": cast(
            "NDArray[np.float32]",
            image_embeds.detach().to(torch.float32).cpu().numpy().reshape(-1),
        ),
    }
    for _, key in HOOK_TARGETS:
        outputs[key] = cast(
            "NDArray[np.float32]",
            captured[key].detach().to(torch.float32).cpu().numpy(),
        )
    return outputs


def build_tensor_swaps(
    model: VisionModelProtocol, reader: GGUFReader
) -> list[tuple[str, str, NDArray[np.float32], float]]:
    """Bind every sidecar tensor to its torch parameter and verify the binding.

    Same three checks as ``q8_floor_check.py``: the shape of the numpy view
    must equal the shape of the target parameter, the cosine of the
    dequantised tensor against the source weight must be at least
    ``SWAP_MIN_COSINE``, and for square tensors the transposed binding must be
    strictly worse.
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
    swaps: list[tuple[str, str, NDArray[np.float32], float]] = []
    for sidecar_name in sorted(specs):
        spec = specs[sidecar_name]
        if sidecar_name not in sidecars:
            raise RuntimeError(
                f"sidecar tensor {sidecar_name!r} is missing from the GGUF"
            )
        array = np.asarray(sidecar_to_numpy(sidecars[sidecar_name]), dtype=np.float32)
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
        if array.ndim == 2 and array_shape[0] == array_shape[1]:
            transposed = cosine_similarity(array.T, original)
            if transposed >= direct:
                raise RuntimeError(
                    f"{sidecar_name!r}: the transposed binding is not worse "
                    f"(transposed cosine {transposed:.9g} >= direct {direct:.9g})"
                )
        swaps.append((sidecar_name, spec.source_name, array, direct))
    return swaps


def apply_tensor_swaps(
    model: VisionModelProtocol,
    swaps: list[tuple[str, str, NDArray[np.float32], float]],
) -> dict[str, int]:
    """Write every swapped tensor into its torch parameter."""
    changed = 0
    unchanged = 0
    with torch.no_grad():
        for _, target_path, array, _ in swaps:
            target = model.get_parameter(target_path)
            original = cast(
                "NDArray[np.float32]",
                target.detach().to(device="cpu", dtype=torch.float32).numpy(),
            )
            if np.array_equal(original, array):
                unchanged += 1
                continue
            changed += 1
            replacement = torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32)
            )
            target.copy_(replacement.to(device=target.device, dtype=target.dtype))
    return {"changed": changed, "unchanged": unchanged, "total": len(swaps)}


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


def run_probe(args: argparse.Namespace) -> int:
    """Measure the floor for the three probe images and write the report."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 could not be disabled; the oracle is TF32-free")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = environment_report(device)
    print("Q8 CONTENT PROBE")
    print(f"[env] {json.dumps(environment, sort_keys=True)}")

    model, load_info = load_model(device)
    print(
        f"[load] missing={load_info['missing_keys']} "
        f"unexpected={load_info['unexpected_keys']} "
        f"mismatched={load_info['mismatched_keys']} -> STRICT"
    )

    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        str(MODEL_DIR)
    )

    images: list[ProbeImage] = []
    for name, (base_noise_std, extra_noise_std) in PROBE_SPECS.items():
        image = prepare_probe_image(name, base_noise_std, extra_noise_std, processor)
        images.append(image)
        print(
            f"[image] {name:<12} png={image.png_path.name} "
            f"grid={EXPECTED_GRID} valid={EXPECTED_VALID_PATCHES} "
            f"laplacian_mean_abs={image.laplacian_mean_abs:.6f} "
            f"patch_mean_std={image.patch_mean_std:.6f}"
        )

    published_tiny = FIXTURES_DIR / "tiny.png"
    anchor = {
        "tiny_png_sha256": sha256_file(published_tiny),
        "tiny_smooth_png_sha256": sha256_file(images[0].png_path),
        "tiny_smooth_is_byte_identical_to_tiny": (
            sha256_file(published_tiny) == sha256_file(images[0].png_path)
        ),
    }
    print(
        f"[anchor] tiny_smooth == published tiny.png: "
        f"{anchor['tiny_smooth_is_byte_identical_to_tiny']}"
    )

    fp32_outputs: dict[str, dict[str, NDArray[np.float32]]] = {}
    for image in images:
        fp32_outputs[image.name] = evaluate_image(model, image, device)

    # Repeat the FP32 evaluation for every image.  This is the FP32 control
    # (it must be exactly 1.0) and, for the designated image, it is also the
    # negative control of the Q8 path: the identical evaluation with the
    # sidecar weights NOT swapped in.
    fp32_repeat: dict[str, dict[str, NDArray[np.float32]]] = {}
    fp32_control: dict[str, float] = {}
    for image in images:
        repeat = evaluate_image(model, image, device)
        fp32_repeat[image.name] = repeat
        fp32_control[image.name] = cosine_similarity(
            repeat["image_embeds"], fp32_outputs[image.name]["image_embeds"]
        )
    control_image = next(
        image for image in images if image.name == NEGATIVE_CONTROL_IMAGE
    )
    negative_control: dict[str, object] = {
        "image": control_image.name,
        "image_embeds_cosine": fp32_control[control_image.name],
        "post_layernorm_cosine": cosine_similarity(
            fp32_repeat[control_image.name]["post_layernorm"],
            fp32_outputs[control_image.name]["post_layernorm"],
        ),
    }
    negative_control_passed = fp32_control[control_image.name] == 1.0
    negative_control["passed"] = negative_control_passed
    print(
        f"[negative control] {control_image.name} re-run with FP32 weights: "
        f"image_embeds cosine {fp32_control[control_image.name]:.12f}"
    )

    reader = GGUFReader(str(SIDECAR_PATH))
    swaps = build_tensor_swaps(model, reader)
    swap_counts = apply_tensor_swaps(model, swaps)
    if swap_counts["changed"] == 0:
        raise RuntimeError("the sidecar swap changed no weight")
    print(
        f"[swap] {swap_counts['total']} tensors, {swap_counts['changed']} changed, "
        f"{swap_counts['unchanged']} identical"
    )

    q8_outputs: dict[str, dict[str, NDArray[np.float32]]] = {}
    for image in images:
        q8_outputs[image.name] = evaluate_image(model, image, device)

    outcomes: list[ProbeOutcome] = []
    for image in images:
        fp32 = fp32_outputs[image.name]
        q8 = q8_outputs[image.name]
        outcomes.append(
            ProbeOutcome(
                name=image.name,
                fp32=fp32,
                q8=q8,
                negative_control=(
                    negative_control if image.name == NEGATIVE_CONTROL_IMAGE else None
                ),
            )
        )

    print()
    print("RESULTS")
    header = (
        f"  {'image':<12} {'laplacian':>12} {'patch_std':>12} "
        f"{'fp32 control':>14} {'Q8 floor':>14} {'max_abs':>12}"
    )
    print(header)
    rows: list[dict[str, object]] = []
    floors: dict[str, float] = {}
    for image, outcome in zip(images, outcomes, strict=True):
        floor_cosine = cosine_similarity(
            outcome.fp32["image_embeds"], outcome.q8["image_embeds"]
        )
        max_abs = max_abs_difference(
            outcome.fp32["image_embeds"], outcome.q8["image_embeds"]
        )
        floors[image.name] = floor_cosine
        rows.append(
            {
                "name": image.name,
                "laplacian_mean_abs": image.laplacian_mean_abs,
                "patch_mean_std": image.patch_mean_std,
                "model_input_std": image.model_input_std,
                "fp32_control_cosine": fp32_control[image.name],
                "q8_floor_cosine": floor_cosine,
                "q8_max_abs_diff": max_abs,
                "intermediate_cosines": {
                    key: cosine_similarity(outcome.fp32[key], outcome.q8[key])
                    for key in sorted(outcome.fp32)
                    if key != "image_embeds"
                },
            }
        )
        print(
            f"  {image.name:<12} {image.laplacian_mean_abs:>12.6f} "
            f"{image.patch_mean_std:>12.6f} {fp32_control[image.name]:>14.12f} "
            f"{floor_cosine:>14.12f} {max_abs:>12.6g}"
        )

    persisted: dict[str, object] = {}
    for image, outcome in zip(images, outcomes, strict=True):
        path = ORACLE_DIR / f"content_probe_{image.name}.npz"
        save_npz_deterministic(
            path,
            {
                "image_embeds_fp32": outcome.fp32["image_embeds"],
                "image_embeds_q8": outcome.q8["image_embeds"],
                "pixel_values": image.pixel_values,
                "pixel_attention_mask": image.pixel_attention_mask,
                "spatial_shapes": image.spatial_shapes,
            },
        )
        persisted[image.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "keys": [
                "image_embeds_fp32",
                "image_embeds_q8",
                "pixel_values",
                "pixel_attention_mask",
                "spatial_shapes",
            ],
        }

    floor_values = list(floors.values())
    laplacians = [image.laplacian_mean_abs for image in images]
    patch_stds = [image.patch_mean_std for image in images]
    by_laplacian = [
        image.name
        for _, image in sorted(
            zip(laplacians, images, strict=True), key=lambda pair: pair[0]
        )
    ]
    by_patch_std = [
        image.name
        for _, image in sorted(
            zip(patch_stds, images, strict=True), key=lambda pair: pair[0]
        )
    ]
    by_floor = sorted(floors, key=lambda name: floors[name])
    monotone = by_laplacian == by_floor or by_laplacian == list(reversed(by_floor))
    verdict = {
        "min_floor": min(floor_values),
        "max_floor": max(floor_values),
        "spread": max(floor_values) - min(floor_values),
        "worst_image": by_floor[0],
        "best_image": by_floor[-1],
        "order_by_laplacian": by_laplacian,
        "order_by_patch_mean_std": by_patch_std,
        "order_by_floor": by_floor,
        "floor_is_monotone_in_laplacian": monotone,
        "published_tiny_floor": PUBLISHED_TINY_FLOOR,
        "tiny_smooth_reproduces_published_tiny": (
            abs(floors["tiny_smooth"] - PUBLISHED_TINY_FLOOR) <= 1e-9
        ),
    }
    print()
    print(
        f"  floor spread = {verdict['spread']:.9g} "
        f"(min {verdict['min_floor']:.12f} on {verdict['worst_image']}, "
        f"max {verdict['max_floor']:.12f} on {verdict['best_image']})"
    )
    print(f"  order by laplacian: {by_laplacian}")
    print(f"  order by floor:     {by_floor}")
    print(f"  monotone in the laplacian: {monotone}")
    print(
        f"  tiny_smooth reproduces the published tiny floor: "
        f"{verdict['tiny_smooth_reproduces_published_tiny']}"
    )

    payload: dict[str, object] = {
        "tool": "q8_content_probe",
        "generated_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "load": load_info,
        "construction": {
            "source_size": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "seed": SOURCE_SEED,
            "recipe": (
                "gradient + 3 rectangles + per-pixel Gaussian noise, clip [0,255]"
            ),
            "noise_stds": {
                name: {"base": base, "extra": extra}
                for name, (base, extra) in PROBE_SPECS.items()
            },
            "grid": list(EXPECTED_GRID),
            "valid_patches": EXPECTED_VALID_PATCHES,
        },
        "anchor": anchor,
        "swap": swap_counts,
        "negative_control": negative_control,
        "images": rows,
        "verdict": verdict,
        "persisted_tensors": persisted,
        "artifacts": {
            SIDECAR_PATH.name: sha256_file(SIDECAR_PATH),
            SAFETENSORS_PATH.name: sha256_file(SAFETENSORS_PATH),
            **{image.png_path.name: sha256_file(image.png_path) for image in images},
        },
    }
    args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"JSON: {args.json_path}")

    if not negative_control_passed:
        print("NEGATIVE CONTROL FAILED: the probe cannot discriminate.")
        return EXIT_NEGATIVE_CONTROL_FAILED
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        return run_probe(args)
    except Exception:
        traceback.print_exc()
        args.json_path.write_text(
            json.dumps(
                {
                    "tool": "q8_content_probe",
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
