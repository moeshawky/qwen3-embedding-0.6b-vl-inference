#!/usr/bin/env python3
"""Export the text branch's compatibility head as a standalone GGUF.

The published text archive (`qwen3-embedding-0.6b-q8_0.gguf`) is the frozen
Qwen3-Embedding-0.6B text tower; the composite model additionally applies a
trained 1024->1024 affine "compat head" before the final L2 normalisation.
That head is NOT part of the published GGUF, and the published GGUF must not be
mutated, so it is exported here as its own opt-in artifact.

Source tensors (torch names in `model.safetensors`):
    compat_head.weight  [1024, 1024]  (out_features, in_features)
    compat_head.bias    [1024]

Layout: the GGUF stores a torch [out, in] weight as ggml ne {in, out}, i.e. the
bytes are unchanged and only the declared shape is transposed. The native
loader therefore uploads the array as-is.

Usage:
    python3 export_compat_head.py --safetensors model.safetensors --out compat_head.f32.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np
from gguf import GGUFWriter
from safetensors import safe_open


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safetensors", default="model.safetensors")
    ap.add_argument(
        "--out", default="native/artifacts/compat_head_qwen3_embedding_0.6b.f32.gguf"
    )
    args = ap.parse_args()

    src = pathlib.Path(args.safetensors)
    with safe_open(str(src), framework="np") as f:
        if "compat_head.weight" not in f:
            print("ERROR: compat_head.weight not found in the source checkpoint")
            return 2
        weight = np.asarray(f.get_tensor("compat_head.weight"), dtype=np.float32)
        bias = np.asarray(f.get_tensor("compat_head.bias"), dtype=np.float32)

    if weight.ndim != 2 or weight.shape[0] != bias.shape[0]:
        print(
            f"ERROR: unexpected compat head shapes: weight {weight.shape}, bias {bias.shape}"
        )
        return 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(out), arch="qwen3")
    writer.add_tensor("compat_head.weight", weight)
    writer.add_tensor("compat_head.bias", bias)
    writer.add_string("general.name", "qwen3-embedding-0.6b compat head")
    writer.add_string(
        "general.description", "opt-in compatibility head for the text branch"
    )
    writer.add_string("compat_head.source", str(src))
    writer.add_string("compat_head.source_sha256", sha256(src))
    writer.add_uint32("compat_head.weight_shape_out", int(weight.shape[0]))
    writer.add_uint32("compat_head.weight_shape_in", int(weight.shape[1]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"  compat_head.weight {weight.shape} float32")
    print(f"  compat_head.bias   {bias.shape} float32")
    print(f"  sha256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
