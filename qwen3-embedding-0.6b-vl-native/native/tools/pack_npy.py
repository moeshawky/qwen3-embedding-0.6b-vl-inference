#!/usr/bin/env python3
"""Pack the native binary's `.npy` outputs into a single `.npz`.

The native executable writes one `.npy` per tensor, named `<stem>_<key>.npy`.
This tool collects those files into an `.npz` whose keys match the PyTorch
oracle, so `compare_vectors.py` can diff them directly.

Usage:
    python3 pack_npy.py --stem square --npy-dir <dir> --out <file.npz>
"""

from __future__ import annotations

import argparse
import pathlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Native output suffix -> oracle key.
KEYS: dict[str, str] = {
    "patch_plus_position": "patch_plus_position",
    "encoder_layer_0": "encoder_layer_0",
    "encoder_layer_1": "encoder_layer_1",
    "encoder_layer_13": "encoder_layer_13",
    "encoder_layer_26": "encoder_layer_26",
    "post_layernorm": "post_layernorm",
    "map_pooler": "map_pooler",
    "map_pooler_norm": "map_pooler_norm",
    "linear_projection": "linear_projection",
    "image_embeds": "image_embeds",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stem", required=True, help="output prefix used by the native binary"
    )
    ap.add_argument("--npy-dir", required=True, help="directory holding the .npy files")
    ap.add_argument("--out", required=True, help="destination .npz")
    args = ap.parse_args()

    src = pathlib.Path(args.npy_dir)
    arrays: dict[str, NDArray[np.generic]] = {}
    for suffix, key in KEYS.items():
        path = src / f"{args.stem}_{suffix}.npy"
        if not path.exists():
            print(f"missing: {path}")
            return 2
        arrays[key] = np.load(path)

    np.savez(args.out, **arrays)
    print(f"wrote {args.out} with {len(arrays)} keys")
    for key, arr in arrays.items():
        print(f"  {key:20s} {arr.shape} {arr.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
