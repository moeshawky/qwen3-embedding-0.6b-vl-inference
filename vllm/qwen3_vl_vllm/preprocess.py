"""NaFlex image preprocessing for the SigLIP2 vision tower.

Reproduces the HuggingFace ``Siglip2ImageProcessor`` NaFlex pipeline
without PyTorch, using NumPy + PIL:

1. Target size: ``get_image_size_for_max_num_patches`` binary search
2. Resize: bilinear with antialias (torchvision semantics, 2-pass with
   8-bit rounding after each pass)
3. Patchify: 16x16 patches, (x/255 - 0.5) / 0.5 normalisation
4. Pad: zero rows up to 256 patches + 0/1 attention mask
5. Spatial shapes: (grid_h, grid_w) per image

Output tensors are ready for ``SigLIP2VisionModel``.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np
from PIL import Image

PATCH_SIZE = 16
MAX_NUM_PATCHES = 256
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
RESCALE_FACTOR = 1 / 255.0


def _scaled_image_size(scale: float, size: int, patch_size: int = PATCH_SIZE) -> int:
    scaled = size * scale
    ceiled = math.ceil(scaled / patch_size) * patch_size
    v = int(ceiled)
    return v if v > patch_size else patch_size


def compute_target_size(
    height: int,
    width: int,
    patch_size: int = PATCH_SIZE,
    max_num_patches: int = MAX_NUM_PATCHES,
) -> tuple[int, int]:
    """Port of transformers' ``get_image_size_for_max_num_patches`` (eps=1e-5)."""
    eps = 1e-5
    scale_min = eps / 10.0
    scale_max = 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2.0
        th = _scaled_image_size(scale, height, patch_size)
        tw = _scaled_image_size(scale, width, patch_size)
        num_patches = (th / patch_size) * (tw / patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale
    scale = scale_min
    out_h = _scaled_image_size(scale, height, patch_size)
    out_w = _scaled_image_size(scale, width, patch_size)
    return out_h, out_w


def _aa_filter(in_size: int, out_size: int):
    """Antialias filter weights (torchvision bilinear antialias semantics)."""
    s = in_size / out_size
    support = max(s, 1.0)
    idx_list = []
    w_list = []
    for j in range(out_size):
        center = (j + 0.5) * s - 0.5
        lo = int(math.ceil(center - support))
        hi = int(math.floor(center + support))
        lo = max(lo, 0)
        hi = min(hi, in_size - 1)
        ws = []
        total = 0.0
        for i in range(lo, hi + 1):
            w = 1.0 - abs(i - center) / support
            if w < 0.0:
                w = 0.0
            ws.append(w)
            total += w
        if total <= 0.0:
            nearest = min(max(int(round(center)), 0), in_size - 1)
            idx_list.append([nearest])
            w_list.append([1.0])
        else:
            idx_list.append([lo + k for k in range(len(ws))])
            w_list.append([w / total for w in ws])
    return idx_list, w_list


def resize_bilinear_aa(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize with antialias, 8-bit rounding after each pass.

    Matches torchvision's ``resize(antialias=True)`` on uint8 input:
    horizontal pass -> round to uint8 -> vertical pass -> round to uint8.
    """
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img.copy()

    # Horizontal pass
    idx_x, w_x = _aa_filter(in_w, out_w)
    tmp = np.empty((in_h, out_w, img.shape[2]), dtype=np.uint8)
    for y in range(in_h):
        for x in range(out_w):
            acc = np.zeros(3, dtype=np.float64)
            for i, w in zip(idx_x[x], w_x[x]):
                acc += w * img[y, i].astype(np.float64)
            tmp[y, x] = np.clip(np.floor(acc + 0.5), 0, 255).astype(np.uint8)

    # Vertical pass
    idx_y, w_y = _aa_filter(in_h, out_h)
    out = np.empty((out_h, out_w, img.shape[2]), dtype=np.uint8)
    for y in range(out_h):
        for x in range(out_w):
            acc = np.zeros(3, dtype=np.float64)
            for i, w in zip(idx_y[y], w_y[y]):
                acc += w * tmp[i, x].astype(np.float64)
            out[y, x] = np.clip(np.floor(acc + 0.5), 0, 255).astype(np.uint8)

    return out


def preprocess_image(
    image: Image.Image,
    patch_size: int = PATCH_SIZE,
    max_num_patches: int = MAX_NUM_PATCHES,
) -> dict:
    """Preprocess one PIL image into NaFlex inputs.

    Returns a dict with:
      - pixel_values: [max_patches, patch_dim] float32 (patch-major)
      - pixel_attention_mask: [max_patches] int64 (1=real, 0=pad)
      - spatial_shapes: [2] int64 (grid_h, grid_w)
    """
    img = image.convert("RGB")
    arr = np.array(img)  # [H, W, 3] uint8
    h, w = arr.shape[:2]

    # 1. Target size
    th, tw = compute_target_size(h, w, patch_size, max_num_patches)

    # 2. Resize with antialias
    resized = resize_bilinear_aa(arr, th, tw)  # [th, tw, 3] uint8

    # 3. Grid dimensions
    grid_h = th // patch_size
    grid_w = tw // patch_size
    patch_dim = 3 * patch_size * patch_size  # 768

    # 4. Patchify + normalize
    values = np.zeros((max_num_patches, patch_dim), dtype=np.float32)
    mask = np.zeros(max_num_patches, dtype=np.int64)

    for py in range(grid_h):
        for px in range(grid_w):
            p = py * grid_w + px
            patch = resized[
                py * patch_size : (py + 1) * patch_size,
                px * patch_size : (px + 1) * patch_size,
            ]  # [patch_size, patch_size, 3] H,W,C
            # Flatten in pixel-major order (H, W, C) to match HF's
            # Siglip2VisionEmbeddings conv2d patch extraction order.
            flat = patch.reshape(-1)  # [patch_size*patch_size*3] pixel-major
            # Normalize: (x/255 - 0.5) / 0.5
            norm = (flat.astype(np.float32) * RESCALE_FACTOR - IMAGE_MEAN) / IMAGE_STD
            values[p] = norm
            mask[p] = 1

    spatial_shapes = np.array([grid_h, grid_w], dtype=np.int64)

    return {
        "pixel_values": values,
        "pixel_attention_mask": mask,
        "spatial_shapes": spatial_shapes,
    }


def preprocess_images(
    images: list[Image.Image],
    patch_size: int = PATCH_SIZE,
    max_num_patches: int = MAX_NUM_PATCHES,
) -> dict:
    """Preprocess a batch of PIL images.

    All images are padded to ``max_num_patches`` patches.  Returns batched
    tensors ready for the vision tower.

    Returns:
      - pixel_values: [B, max_patches, patch_dim] float32
      - pixel_attention_mask: [B, max_patches] int64
      - spatial_shapes: [B, 2] int64
    """
    per_image = [preprocess_image(im, patch_size, max_num_patches) for im in images]
    pixel_values = np.stack([p["pixel_values"] for p in per_image])  # [B, 256, 768]
    pixel_attention_mask = np.stack(
        [p["pixel_attention_mask"] for p in per_image]
    )  # [B, 256]
    spatial_shapes = np.stack([p["spatial_shapes"] for p in per_image])  # [B, 2]
    return {
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
        "spatial_shapes": spatial_shapes,
    }


def load_image(path: str) -> Image.Image:
    """Load an image file as RGB."""
    return Image.open(path).convert("RGB")


def b64_to_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image."""
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")
