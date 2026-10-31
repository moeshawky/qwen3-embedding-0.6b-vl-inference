"""PyTorch oracle capture for the Qwen3-SigLIP2-VL image tower.

What this produces
------------------
Six deterministic synthetic RGB fixtures, the processor inputs derived from
them, and the intermediate activations captured from the *real* PyTorch
model, so that a native (C++) reimplementation of the same vision branch can
be diffed against the reference implementation.

Artifacts written (all paths relative to ``/content/qwen3-embedding-0.6b-vl``):

* ``native/fixtures/<name>.png``      -- the original RGB fixture, unmodified
* ``native/oracle/oracle_<name>.npz`` -- captured tensors (see ``EXPECTED_KEYS``)
* ``native/oracle/manifest.json``     -- per-fixture provenance + digests

Fixture content recipe (bit-for-bit reproducible; re-derivable from this
docstring alone)
--------------------------------------------------------------------------
For a fixture of size H x W (rows x cols) and per-fixture seed S:

    rng = numpy.random.RandomState(S)        # legacy MT19937, one RNG/fixture
    ys  = linspace(0, 1, H) -> column vector (H, 1)
    xs  = linspace(0, 1, W) -> row vector    (1, W)

    for c in (0, 1, 2):                       # per-channel linear gradient
        channel[c] = 40.0 + 30.0 * c          # channel offset
                   + (60.0 + 20.0 * c) * xs   # horizontal slope
                   + (40.0 + 15.0 * c) * ys   # vertical slope

    then 3 axis-aligned solid-colour rectangles, positioned as FRACTIONS of
    (W, H) so they scale with the image; pixel bounds are
    int(round(fraction * size)):

        R0  x in [0.10, 0.40) W   y in [0.10, 0.45) H   colour (220,  30,  30)
        R1  x in [0.50, 0.85) W   y in [0.55, 0.90) H   colour ( 30, 220,  30)
        R2  x in [0.55, 0.95) W   y in [0.05, 0.40) H   colour ( 30,  30, 220)

    then additive Gaussian noise: rng.normal(0.0, 12.0, size=(H, W, 3))
    then clip to [0, 255] and round to uint8 (RGB).

The operation order is exactly: gradient -> rectangles -> noise -> clip.

Numerical contract
------------------
* float32 everywhere, on CUDA, under ``torch.no_grad()``; never bf16/fp16.
* TF32 is pinned OFF (matmul and cuDNN) and the flags are recorded in the
  manifest so the oracle's numerics stay interpretable.
* The load is strict: non-empty ``missing_keys``/``unexpected_keys`` raise.
* ``image_embeds`` is checked against ``normalize(linear_projection)``
  bit-for-bit, which *measures* (rather than assumes) that the captured head
  output is the tensor the model normalises internally.
* ``.npz`` files are written with a deterministic zip writer (fixed entry
  timestamps) so their sha256 is stable across runs; the layout is identical
  to ``numpy.savez`` (zip of ``<key>.npy`` members, ZIP_STORED).

Usage
-----
    python3 native/tools/capture_oracle.py
    python3 native/tools/capture_oracle.py --out-dir /tmp/oracle --fixtures-dir /tmp/fx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib import format as npy_format
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

MODEL_DIR = Path("/content/qwen3-embedding-0.6b-vl")
DEFAULT_FIXTURES_DIR = MODEL_DIR / "native" / "fixtures"
DEFAULT_OUT_DIR = MODEL_DIR / "native" / "oracle"

# The trust_remote_code modeling file imports its sibling
# ``configuration_qwen_siglip2_embedding``.  When this script is launched by
# path, ``sys.path[0]`` is this script's directory -- not the model directory --
# so the sibling module would not be importable and the dynamic-module loader
# would fail with "requires the following packages that were not found".
# Prepending the model directory makes the script runnable from any cwd.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

#: Rectangles as (x0, y0, x1, y1) fractions of (width, height).
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


@dataclass(frozen=True)
class FixtureSpec:
    """One synthetic fixture: name, original pixel size (H, W) and RNG seed."""

    name: str
    height: int
    width: int
    seed: int


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec("square", 256, 256, 1001),
    FixtureSpec("landscape", 560, 112, 1002),
    FixtureSpec("portrait", 112, 560, 1003),
    FixtureSpec("odd", 336, 192, 1004),
    FixtureSpec("tiny", 112, 112, 1005),
    FixtureSpec("wide", 1024, 64, 1006),
)

#: (dotted module path on the model, capture key) pairs, in capture order.
HOOK_TARGETS: tuple[tuple[str, str], ...] = (
    ("vision_model.embeddings", "patch_plus_position"),
    ("vision_model.encoder.layers.0", "encoder_layer_0"),
    ("vision_model.encoder.layers.1", "encoder_layer_1"),
    ("vision_model.encoder.layers.13", "encoder_layer_13"),
    ("vision_model.encoder.layers.26", "encoder_layer_26"),
    ("vision_model.post_layernorm", "post_layernorm"),
    ("vision_model.head", "map_pooler"),
)

#: Every key stored in ``oracle_<name>.npz``.
EXPECTED_KEYS: tuple[str, ...] = (
    "pixel_values",
    "pixel_attention_mask",
    "spatial_shapes",
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

#: key -> (expected shape after the batch dim is squeezed, expected dtype).
EXPECTED_ARRAYS: dict[str, tuple[tuple[int, ...], str]] = {
    "pixel_values": ((256, 768), "float32"),
    "pixel_attention_mask": ((256,), "int32"),
    "spatial_shapes": ((2,), "int64"),
    "patch_plus_position": ((256, 1152), "float32"),
    "encoder_layer_0": ((256, 1152), "float32"),
    "encoder_layer_1": ((256, 1152), "float32"),
    "encoder_layer_13": ((256, 1152), "float32"),
    "encoder_layer_26": ((256, 1152), "float32"),
    "post_layernorm": ((256, 1152), "float32"),
    "map_pooler": ((1152,), "float32"),
    "map_pooler_norm": ((1152,), "float32"),
    "linear_projection": ((1024,), "float32"),
    "image_embeds": ((1024,), "float32"),
}

#: Fixture re-captured twice in-process to prove run-to-run determinism.
DETERMINISM_FIXTURE = "odd"


class VisionModelProtocol(Protocol):
    """Structural view of the ``trust_remote_code`` model used by this harness.

    Mirrors ``Qwen3Siglip2EmbeddingModel`` in ``modeling_qwen_siglip2_embedding.py``.
    """

    vision_model: nn.Module
    vision_projection: nn.Linear
    use_residual: bool

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class FixtureCapture:
    """Everything captured for one fixture."""

    spec: FixtureSpec
    arrays: dict[str, npt.NDArray[np.generic]]
    grid: tuple[int, int]
    valid_patches: int
    masked_patches: int
    embeds_norm: float
    projection_max_diff: float


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture_image(spec: FixtureSpec) -> npt.NDArray[np.uint8]:
    """Build the deterministic RGB fixture described in the module docstring."""
    height, width = spec.height, spec.width
    rng = np.random.RandomState(spec.seed)
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

    image += rng.normal(0.0, 12.0, size=(height, width, 3))
    np.clip(image, 0.0, 255.0, out=image)
    return np.rint(image).astype(np.uint8)


def write_fixture_png(spec: FixtureSpec, path: Path) -> npt.NDArray[np.uint8]:
    """Render the fixture and persist it as an 8-bit RGB PNG."""
    array = build_fixture_image(spec)
    Image.fromarray(array).save(path, format="PNG")
    return array


def resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted module path, indexing into ``nn.ModuleList`` on digits."""
    obj: object = root
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, nn.ModuleList) else getattr(obj, part)
    if not isinstance(obj, nn.Module):
        raise TypeError(f"hook target {dotted!r} did not resolve to an nn.Module")
    return obj


def _make_hook(
    key: str, sink: dict[str, torch.Tensor]
) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    """Build a forward hook that stores the module output under ``key``."""

    def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"hook {key!r} captured a non-tensor output: {type(output)!r}"
            )
        sink[key] = tensor.detach()

    return _hook


def _to_numpy(tensor: torch.Tensor, dtype: torch.dtype) -> npt.NDArray[np.generic]:
    """Detach, cast and move a tensor to a host numpy array."""
    return tensor.detach().to(dtype).cpu().numpy()


def capture_fixture(
    model: VisionModelProtocol,
    processor: Any,
    spec: FixtureSpec,
    png_path: Path,
    device: torch.device,
) -> FixtureCapture:
    """Run one fixture end to end and capture every oracle tensor."""
    inputs = processor(images=[str(png_path)], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
    pixel_attention_mask = inputs["pixel_attention_mask"].to(device=device)
    spatial_shapes = inputs["spatial_shapes"].to(device=device, dtype=torch.int64)

    captured: dict[str, torch.Tensor] = {}
    handles = []
    model_root = cast("nn.Module", model)
    for dotted, key in HOOK_TARGETS:
        module = resolve_module(model_root, dotted)
        handles.append(module.register_forward_hook(_make_hook(key, captured)))
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
    recomposed = F.normalize(linear_projection.float(), p=2, dim=-1)
    projection_max_diff = float((recomposed - image_embeds.float()).abs().max().item())
    if not torch.equal(recomposed, image_embeds):
        raise RuntimeError(
            "captured head output does not reproduce image_embeds: "
            f"max|diff|={projection_max_diff:.3e}"
        )

    arrays: dict[str, npt.NDArray[np.generic]] = {
        "pixel_values": _to_numpy(pixel_values, torch.float32)[0],
        "pixel_attention_mask": _to_numpy(pixel_attention_mask, torch.int32)[0],
        "spatial_shapes": _to_numpy(spatial_shapes, torch.int64)[0],
        "map_pooler": _to_numpy(map_pooler, torch.float32)[0],
        "map_pooler_norm": _to_numpy(map_pooler_norm, torch.float32)[0],
        "linear_projection": _to_numpy(linear_projection, torch.float32)[0],
        "image_embeds": _to_numpy(image_embeds, torch.float32)[0],
    }
    for _, key in HOOK_TARGETS:
        if key == "map_pooler":
            continue
        arrays[key] = _to_numpy(captured[key], torch.float32)[0]

    grid = (int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item()))
    valid_patches = int(pixel_attention_mask.sum().item())
    masked_patches = int(pixel_attention_mask.numel() - valid_patches)
    embeds_norm = float(image_embeds.to(torch.float64).norm().item())

    return FixtureCapture(
        spec=spec,
        arrays=arrays,
        grid=grid,
        valid_patches=valid_patches,
        masked_patches=masked_patches,
        embeds_norm=embeds_norm,
        projection_max_diff=projection_max_diff,
    )


def save_npz_deterministic(
    path: Path, arrays: dict[str, npt.NDArray[np.generic]]
) -> None:
    """Write an ``.npz`` with ``numpy.savez`` layout but reproducible bytes.

    ``numpy.savez`` stamps every zip member with the current time, so the file
    bytes (and any sha256 over them) change on every run even when the arrays
    are bit-identical.  Pinning the member timestamp to the zip epoch keeps
    the artifact reproducible; the payload format is unchanged.
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


def verify_npz(path: Path) -> None:
    """Reload an ``.npz`` and assert its key set, shapes and dtypes."""
    with np.load(path) as data:
        keys = set(data.files)
        expected = set(EXPECTED_KEYS)
        if keys != expected:
            raise RuntimeError(
                f"{path.name}: key mismatch; missing={sorted(expected - keys)} "
                f"unexpected={sorted(keys - expected)}"
            )
        for key, (shape, dtype) in EXPECTED_ARRAYS.items():
            array = data[key]
            if tuple(array.shape) != shape:
                raise RuntimeError(
                    f"{path.name}:{key}: shape {tuple(array.shape)} != {shape}"
                )
            if array.dtype.name != dtype:
                raise RuntimeError(
                    f"{path.name}:{key}: dtype {array.dtype.name} != {dtype}"
                )


def load_model(device: torch.device) -> tuple[VisionModelProtocol, dict[str, Any]]:
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
    print(f"[load] missing_keys={missing} unexpected_keys={unexpected}")
    print(f"[load] mismatched_keys={mismatched}")
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "non-strict load: "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    moved = cast("nn.Module", model).to(device=device, dtype=torch.float32).eval()
    return cast("VisionModelProtocol", moved), loading_info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the optional command line."""
    parser = argparse.ArgumentParser(
        description="Capture the PyTorch oracle for the SigLIP2-VL image tower."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="npz/manifest output dir"
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="directory for the generated PNG fixtures",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate fixtures, capture the oracle, write the manifest."""
    args = parse_args(argv)
    fixtures_dir: Path = args.fixtures_dir
    out_dir: Path = args.out_dir
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[env] torch={torch.__version__} device={device} "
        f"cuda={torch.cuda.is_available()} "
        f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn_tf32={torch.backends.cudnn.allow_tf32}"
    )

    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR))  # type: ignore[no-untyped-call]
    model, _ = load_model(device)

    print("[fixtures] rendering synthetic RGB fixtures")
    png_paths: dict[str, Path] = {}
    for spec in FIXTURES:
        png_path = fixtures_dir / f"{spec.name}.png"
        write_fixture_png(spec, png_path)
        png_paths[spec.name] = png_path

    captures: list[FixtureCapture] = []
    for spec in FIXTURES:
        capture = capture_fixture(model, processor, spec, png_paths[spec.name], device)
        captures.append(capture)

        npz_path = out_dir / f"oracle_{spec.name}.npz"
        save_npz_deterministic(npz_path, capture.arrays)
        verify_npz(npz_path)
        print(
            f"[capture] {spec.name:<10} grid={capture.grid} "
            f"valid={capture.valid_patches} masked={capture.masked_patches} "
            f"||image_embeds||={capture.embeds_norm:.12f} "
            f"max|normalize(proj)-embeds|={capture.projection_max_diff:.3e} "
            f"-> {npz_path.name}"
        )

    spec = next(s for s in FIXTURES if s.name == DETERMINISM_FIXTURE)
    first = capture_fixture(model, processor, spec, png_paths[spec.name], device)
    second = capture_fixture(model, processor, spec, png_paths[spec.name], device)
    mismatched = [
        key
        for key in EXPECTED_KEYS
        if first.arrays[key].tobytes() != second.arrays[key].tobytes()
    ]
    if mismatched:
        raise RuntimeError(f"non-deterministic capture for {spec.name}: {mismatched}")
    print(
        f"[determinism] fixture={spec.name} in-process double capture: "
        f"BITWISE IDENTICAL=True (all {len(EXPECTED_KEYS)} arrays)"
    )

    manifest: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "tf32": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "recipe": {
            "rng": "numpy.random.RandomState(seed) -- legacy MT19937",
            "order": "linear gradient -> 3 rectangles -> gaussian noise -> clip",
            "noise_std": 12.0,
            "clip": [0, 255],
            "gradient": "40+30c + (60+20c)*xs + (40+15c)*ys",
            "rectangles": [
                {"xy": list(rect), "rgb": list(colour)}
                for rect, colour in zip(RECTANGLES, RECTANGLE_COLOURS, strict=True)
            ],
        },
        "fixtures": [],
    }
    for capture in captures:
        spec_ = capture.spec
        png_path = png_paths[spec_.name]
        npz_path = out_dir / f"oracle_{spec_.name}.npz"
        manifest["fixtures"].append(
            {
                "name": spec_.name,
                "png": str(png_path),
                "png_sha256": sha256_file(png_path),
                "original_size": {"height": spec_.height, "width": spec_.width},
                "seed": spec_.seed,
                "spatial_shapes": list(capture.grid),
                "valid_patches": capture.valid_patches,
                "masked_patches": capture.masked_patches,
                "npz": str(npz_path),
                "npz_sha256": sha256_file(npz_path),
                "image_embeds_l2_norm": capture.embeds_norm,
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(
        f"{'fixture':<10} {'spatial_shapes':<16} {'valid':>5} {'masked':>6} "
        f"{'||image_embeds||':>16}"
    )
    for capture in captures:
        print(
            f"{capture.spec.name:<10} {capture.grid!s:<16} "
            f"{capture.valid_patches:>5} {capture.masked_patches:>6} "
            f"{capture.embeds_norm:>16.12f}"
        )
    print()

    embeds = np.stack([c.arrays["image_embeds"] for c in captures])
    norms = np.linalg.norm(embeds, axis=1)
    gram = np.dot(embeds, embeds.T)
    off_diagonal = gram[~np.eye(len(captures), dtype=bool)]
    print(
        f"[sanity] pairwise image_embeds cosine: min={off_diagonal.min():.6f} "
        f"max={off_diagonal.max():.6f} (distinct fixtures -> distinct embeddings)"
    )
    print(f"[sanity] ||image_embeds|| range: [{norms.min():.12f}, {norms.max():.12f}]")

    print("[sha256] written artifacts")
    for spec_ in FIXTURES:
        print(f"{sha256_file(png_paths[spec_.name])}  {png_paths[spec_.name]}")
    for spec_ in FIXTURES:
        npz_path = out_dir / f"oracle_{spec_.name}.npz"
        print(f"{sha256_file(npz_path)}  {npz_path}")
    print(f"{sha256_file(manifest_path)}  {manifest_path}")

    print(
        f"[done] {len(captures)} fixtures, {len(EXPECTED_KEYS)} arrays each; "
        f"manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
