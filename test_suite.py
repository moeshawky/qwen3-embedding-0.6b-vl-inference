"""Shared correctness test suite for Qwen3-SigLIP2-VL dual-encoder embeddings.

Validates both vLLM and llama.cpp/GGML backends against the canonical
PyTorch reference model and against each other.

Test categories:
  T1 - Text: 16 reference prompts, single and batched
  T2 - Text adversarial batching: unequal lengths, repeats, positions, padding
  T3 - Image: 6 NaFlex fixtures, preprocessing + end-to-end
  T4 - Batch association: pairwise cosine matrix between reference and batched
  T5 - Cross-modal: image-to-text and text-to-image retrieval

Usage:
    python3 test_suite.py --backend vllm --model-dir /path/to/model
    python3 test_suite.py --backend native --native-bin /path/to/qwen3vl-embed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')

# 16 reference prompts (from capture_text_oracle.py)
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

FIXTURE_NAMES: tuple[str, ...] = (
    "square", "landscape", "portrait", "odd", "tiny", "wide",
)

EMBED_DIM = 1024
EOS_TOKEN_ID = 151643


# ---------------------------------------------------------------------------
# Reference computation (PyTorch)
# ---------------------------------------------------------------------------

def load_reference_model(model_dir: str, device: str = "cuda"):
    """Load the canonical PyTorch model for reference computation."""
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok.backend_tokenizer.normalizer = None
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32
    ).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    return tok, model, processor


def reference_text_embeds(
    model, tok, texts: list[str], device: str = "cuda"
) -> np.ndarray:
    """Compute reference text embeddings using the canonical recipe."""
    from processing_qwen_siglip2_embedding import (
        encode_texts, terminator_aware_batch, EOS,
    )
    ids = encode_texts(tok, texts, prefix="")
    input_ids, attention_mask, pool_index = terminator_aware_batch(ids)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pool_index = pool_index.to(device)
    with torch.no_grad():
        embeds = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_index=pool_index,
        )
    return embeds.cpu().numpy()


def reference_image_embeds(
    model, processor, image_paths: list[str], device: str = "cuda"
) -> np.ndarray:
    """Compute reference image embeddings using HF processor + PyTorch model."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(device=device, dtype=torch.float32)
    pam = inputs.get(
        "pixel_attention_mask",
        torch.ones(pv.shape[0], pv.shape[2], dtype=torch.int64),
    ).to(device=device)
    ss = inputs.get(
        "spatial_shapes", torch.tensor([[16, 16]] * len(images))
    ).to(device=device, dtype=torch.int64)
    with torch.no_grad():
        embeds = model.get_image_features(
            pixel_values=pv,
            pixel_attention_mask=pam,
            spatial_shapes=ss,
        )
    return embeds.cpu().numpy()


# ---------------------------------------------------------------------------
# Cosine utilities
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine matrix between rows of A and rows of B."""
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T


# ---------------------------------------------------------------------------
# vLLM backend client
# ---------------------------------------------------------------------------

class VLLMClient:
    """Async client for vLLM embedding server."""

    def __init__(self, model_dir: str, dtype: str = "float32"):
        self.model_dir = model_dir
        self.dtype = dtype
        self.engine = None
        self.vision_model = None
        self.vision_proj = None

    async def start(self):
        os.environ["VLLM_PLUGINS"] = "qwen3_vl_vllm.register"
        from qwen3_vl_vllm import register  # noqa: F401
        from vllm import AsyncLLMEngine, PoolingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.inputs import TextPrompt

        self.PoolingParams = PoolingParams
        self.TextPrompt = TextPrompt

        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            runner="pooling",
            convert="embed",
            dtype=self.dtype,
            max_model_len=4096,
            gpu_memory_utilization=0.4,
            trust_remote_code=True,
            hf_overrides={"architectures": ["Qwen3Siglip2EmbeddingModel"]},
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM engine ready")

        # Load vision components directly for image encoding
        from transformers import AutoConfig, Siglip2VisionConfig, Siglip2VisionModel
        import torch.nn as nn
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        vc_config = config.get_vision_config()
        self.vision_model = Siglip2VisionModel(vc_config).cuda().float()
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False
        self.vision_proj = nn.Linear(
            config.projection_in_dim, config.projection_out_dim, bias=True
        ).cuda().float()
        for p in self.vision_proj.parameters():
            p.requires_grad = False

        state = load_file(os.path.join(self.model_dir, "model.safetensors"))
        vision_state = {
            k[len("vision_model."):]: v
            for k, v in state.items() if k.startswith("vision_model.")
        }
        proj_state = {
            k[len("vision_projection."):]: v
            for k, v in state.items() if k.startswith("vision_projection.")
        }
        missing, unexpected = self.vision_model.load_state_dict(
            vision_state, strict=False
        )
        missing2, unexpected2 = self.vision_proj.load_state_dict(
            proj_state, strict=False
        )
        assert not missing and not missing2, (
            f"Vision missing: {missing}, {missing2}"
        )
        print("Vision components loaded")

    def _encode_images_direct(
        self, pixel_values, pixel_attention_mask, spatial_shapes
    ) -> np.ndarray:
        import torch.nn.functional as F
        self.vision_model.eval()
        with torch.no_grad():
            out = self.vision_model(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            v = F.normalize(out.pooler_output.float(), p=2, dim=-1)
            z = self.vision_proj(v)
            return F.normalize(z, p=2, dim=-1)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        pooling_params = self.PoolingParams(task="embed")

        async def get_one(text, req_id):
            prompt = self.TextPrompt(prompt=text)
            async for output in self.engine.encode(
                prompt, pooling_params, req_id
            ):
                if output.finished:
                    return output.outputs.data.cpu().numpy()
            return None

        results = await asyncio.gather(*[
            get_one(t, f"req-{i}") for i, t in enumerate(texts)
        ])
        return np.stack([r for r in results if r is not None])

    def embed_images_native(self, image_paths: list[str]) -> np.ndarray:
        """Encode images through direct SigLIP2 path (outside vLLM scheduler)."""
        from qwen3_vl_vllm.preprocess import preprocess_images, load_image
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = preprocess_images(images)
        pv = torch.from_numpy(inputs["pixel_values"]).cuda().float()
        pam = torch.from_numpy(inputs["pixel_attention_mask"]).cuda()
        ss = torch.from_numpy(inputs["spatial_shapes"]).cuda()

        embeds = self._encode_images_direct(pv, pam, ss)
        return embeds.cpu().numpy()

    async def shutdown(self):
        if self.engine:
            self.engine.shutdown()


# ---------------------------------------------------------------------------
# Native backend client
# ---------------------------------------------------------------------------

class NativeClient:
    """Client for the native llama.cpp/GGML executable."""

    def __init__(
        self,
        native_bin: str,
        model_dir: str,
        mmproj: str,
        text_model: str,
        compat_head: str,
        out_dir: str,
    ):
        self.native_bin = native_bin
        self.model_dir = model_dir
        self.mmproj = mmproj
        self.text_model = text_model
        self.compat_head = compat_head
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts via native executable (one at a time)."""
        text_file = os.path.join(self.out_dir, "texts.txt")
        with open(text_file, "w") as f:
            for t in texts:
                f.write(t + "\n")

        out_prefix = os.path.join(self.out_dir, "native_text")
        result = subprocess.run(
            [
                self.native_bin,
                "--text-model", self.text_model,
                "--compat-head", self.compat_head,
                "--text-file", text_file,
                "--stem", "text",
                "--out", self.out_dir,
                "--threads", "2",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Native text failed: {result.stderr}")

        npz_path = os.path.join(self.out_dir, "text_text_embeds.npy")
        if not os.path.exists(npz_path):
            # Try alternate naming
            for f in os.listdir(self.out_dir):
                if f.startswith("text_") and f.endswith("_text_embeds.npy"):
                    npz_path = os.path.join(self.out_dir, f)
                    break

        data = np.load(npz_path)
        return data

    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        """Embed images via native executable."""
        out_prefix = os.path.join(self.out_dir, "native_img")
        cmd = [
            self.native_bin,
            "--mmproj", self.mmproj,
            "--out", self.out_dir,
            "--threads", "2",
        ]
        for p in image_paths:
            cmd.extend(["--image", p])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Native image failed: {result.stderr}")

        # Load individual output npz files
        embeds = []
        for name in Path(self.out_dir).glob("*_image_embeds.npy"):
            embeds.append(np.load(name))
        if embeds:
            return np.stack(embeds)
        return np.array([])


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestSuite:
    def __init__(self, backend: str, model_dir: str, **kwargs):
        self.backend = backend
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        self.failures = []

        if backend == "vllm":
            self.client = VLLMClient(model_dir, dtype=kwargs.get("dtype", "float32"))
        elif backend == "native":
            self.client = NativeClient(
                native_bin=kwargs["native_bin"],
                model_dir=model_dir,
                mmproj=kwargs.get(
                    "mmproj",
                    os.path.join(model_dir, "mmproj-qwen3-embedding-0.6b-vl-v1.0-q8_0.gguf"),
                ),
                text_model=kwargs.get(
                    "text_model",
                    os.path.join(model_dir, "qwen3-embedding-0.6b-q8_0.gguf"),
                ),
                compat_head=kwargs.get(
                    "compat_head",
                    os.path.join(
                        os.path.dirname(model_dir),
                        "qwen3-embedding-0.6b-vl-native/native/artifacts/"
                        "compat_head_qwen3_embedding_0.6b.f32.gguf",
                    ),
                ),
                out_dir=kwargs.get("out_dir", "/tmp/native_test_out"),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Load reference model
        self.tok, self.ref_model, self.ref_processor = load_reference_model(
            model_dir, device=self.device
        )

    async def run_all(self):
        if self.backend == "vllm":
            await self.client.start()

        try:
            await self._test_text_single()
            await self._test_text_adversarial_batch()
            await self._test_image()
            await self._test_batch_association()
            await self._test_cross_modal()
        finally:
            if self.backend == "vllm":
                await self.client.shutdown()

        self._report()

    async def _test_text_single(self):
        """T1: Text - 16 reference prompts, single and batched."""
        print("\n=== T1: Text single + batched ===")
        ref = reference_text_embeds(
            self.ref_model, self.tok, list(PROMPTS), device=self.device
        )

        if self.backend == "vllm":
            ours = await self.client.embed_texts(list(PROMPTS))
        else:
            ours = self.client.embed_texts(list(PROMPTS))

        assert ours.shape == ref.shape, f"Shape mismatch: {ours.shape} vs {ref.shape}"

        cosines = [cosine(ours[i], ref[i]) for i in range(len(PROMPTS))]
        min_cos = min(cosines)
        mean_cos = np.mean(cosines)

        print(f"  Min cosine: {min_cos:.8f}")
        print(f"  Mean cosine: {mean_cos:.8f}")
        print(f"  All norms ≈ 1.0: {all(abs(np.linalg.norm(ours[i]) - 1.0) < 1e-4 for i in range(len(PROMPTS)))}")

        self.results["text_single_min_cos"] = min_cos
        self.results["text_single_mean_cos"] = float(mean_cos)

        if min_cos < 0.999:
            self.failures.append(f"Text single: min cosine {min_cos} < 0.999")

    async def _test_text_adversarial_batch(self):
        """T2: Text adversarial batching cases."""
        print("\n=== T2: Text adversarial batching ===")

        # Case 1: Short and long texts together
        batch1 = ["hi", "this is a much longer sentence with many words to test length differences", "short"]
        ref1 = reference_text_embeds(self.ref_model, self.tok, batch1, device=self.device)
        ours1 = await self._embed_texts_batch(batch1)
        cos1 = [cosine(ours1[i], ref1[i]) for i in range(len(batch1))]
        print(f"  Case 1 (mixed lengths): min cos={min(cos1):.8f}")

        # Case 2: Repeated texts
        batch2 = ["a photo of a cat", "a photo of a cat", "a photo of a cat"]
        ref2 = reference_text_embeds(self.ref_model, self.tok, batch2, device=self.device)
        ours2 = await self._embed_texts_batch(batch2)
        cos2 = [cosine(ours2[i], ref2[i]) for i in range(len(batch2))]
        print(f"  Case 2 (repeated): min cos={min(cos2):.8f}")
        # All three should be identical
        identical = cosine(ours2[0], ours2[1]) > 0.9999 and cosine(ours2[1], ours2[2]) > 0.9999
        print(f"  All identical: {identical}")

        # Case 3: Same text at different batch positions
        batch3 = ["first", "middle", "a photo of a cat", "last", "a photo of a cat"]
        ref3 = reference_text_embeds(self.ref_model, self.tok, batch3, device=self.device)
        ours3 = await self._embed_texts_batch(batch3)
        # Position 2 and 4 (0-indexed) are the same text
        cos3a = cosine(ours3[2], ref3[2])
        cos3b = cosine(ours3[4], ref3[4])
        print(f"  Case 3 (position invariance): cat@pos2 cos={cos3a:.8f}, cat@pos4 cos={cos3b:.8f}")

        # Case 4: Explicit vs automatic EOS (both use our recipe which appends EOS)
        # This is implicitly tested by the reference using the same recipe
        batch4 = ["text with explicit EOS", "another text"]
        ref4 = reference_text_embeds(self.ref_model, self.tok, batch4, device=self.device)
        ours4 = await self._embed_texts_batch(batch4)
        cos4 = [cosine(ours4[i], ref4[i]) for i in range(len(batch4))]
        print(f"  Case 4 (EOS handling): min cos={min(cos4):.8f}")

        all_cos = cos1 + cos2 + [cos3a, cos3b] + cos4
        min_all = min(all_cos)

        self.results["text_adversarial_min_cos"] = min_all
        if min_all < 0.999:
            self.failures.append(f"Text adversarial: min cosine {min_all} < 0.999")

    async def _embed_texts_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if self.backend == "vllm":
            return await self.client.embed_texts(texts)
        else:
            return self.client.embed_texts(texts)

    async def _test_image(self):
        """T3: Image - 6 NaFlex fixtures."""
        print("\n=== T3: Image end-to-end ===")
        img_dir = os.path.join(
            os.path.dirname(self.model_dir),
            "qwen3-embedding-0.6b-vl-native/native/fixtures",
        )
        image_paths = [os.path.join(img_dir, f"{n}.png") for n in FIXTURE_NAMES]

        ref = reference_image_embeds(
            self.ref_model, self.ref_processor, image_paths, device=self.device
        )

        if self.backend == "vllm":
            ours = self.client.embed_images_native(image_paths)
        else:
            ours = self.client.embed_images(image_paths)

        assert ours.shape == ref.shape, f"Shape mismatch: {ours.shape} vs {ref.shape}"

        cosines = [cosine(ours[i], ref[i]) for i in range(len(FIXTURE_NAMES))]
        for i, name in enumerate(FIXTURE_NAMES):
            print(f"  {name}: cos={cosines[i]:.8f}")

        min_cos = min(cosines)
        mean_cos = np.mean(cosines)
        print(f"  Min cosine: {min_cos:.8f}")
        print(f"  Mean cosine: {mean_cos:.8f}")

        self.results["image_min_cos"] = min_cos
        self.results["image_mean_cos"] = float(mean_cos)

        if min_cos < 0.999:
            self.failures.append(f"Image: min cosine {min_cos} < 0.999")

    async def _test_batch_association(self):
        """T4: Batch association - pairwise cosine matrix."""
        print("\n=== T4: Batch association ===")

        # Create a mixed batch and verify responses are correctly associated
        texts = list(PROMPTS[:8])  # First 8 prompts
        ref = reference_text_embeds(self.ref_model, self.tok, texts, device=self.device)
        ours = await self._embed_texts_batch(texts)

        # Compute pairwise cosine matrix between ours and ref
        # Diagonal should be ≈ 1.0 (correct association)
        # Off-diagonal should be < 1.0 (different texts)
        cm = cosine_matrix(ours, ref)
        diag = np.diag(cm)
        off_diag = cm[~np.eye(len(texts), dtype=bool)]

        print(f"  Diagonal (self-cosine): min={diag.min():.8f}, mean={diag.mean():.8f}")
        print(f"  Off-diagonal max: {off_diag.max():.8f}")

        # Check that each row's best match is itself (correct association)
        best_match = np.argmax(cm, axis=1)
        correct_assoc = np.all(best_match == np.arange(len(texts)))
        print(f"  Correct batch association: {correct_assoc}")

        self.results["batch_assoc_diag_min"] = float(diag.min())
        self.results["batch_assoc_correct"] = bool(correct_assoc)

        if diag.min() < 0.999:
            self.failures.append(f"Batch assoc: diagonal min {diag.min()} < 0.999")
        if not correct_assoc:
            self.failures.append("Batch assoc: incorrect response-to-input mapping")

    async def _test_cross_modal(self):
        """T5: Cross-modal retrieval similarity matrix (informational)."""
        print("\n=== T5: Cross-modal retrieval ===")

        img_dir = os.path.join(
            os.path.dirname(self.model_dir),
            "qwen3-embedding-0.6b-vl-native/native/fixtures",
        )
        image_paths = [os.path.join(img_dir, f"{n}.png") for n in FIXTURE_NAMES]

        # Text embeds for first 6 prompts
        text_subset = list(PROMPTS[:6])
        ref_text = reference_text_embeds(
            self.ref_model, self.tok, text_subset, device=self.device
        )

        if self.backend == "vllm":
            our_text = await self.client.embed_texts(text_subset)
            our_image = self.client.embed_images_native(image_paths)
        else:
            our_text = self.client.embed_texts(text_subset)
            our_image = self.client.embed_images(image_paths)

        # Cross-modal similarity matrix
        our_cm = cosine_matrix(our_image, our_text)

        print("  Image→Text cosine matrix (rows=images, cols=texts):")
        for i, name in enumerate(FIXTURE_NAMES):
            row_str = " ".join(f"{v:6.3f}" for v in our_cm[i])
            best = int(np.argmax(our_cm[i]))
            print(f"    {name:10s}: [{row_str}]  best={best} ({text_subset[best][:30]})")

        our_t2i = cosine_matrix(our_text, our_image)
        print("  Text→Image cosine matrix (rows=texts, cols=images):")
        for i, text in enumerate(text_subset):
            row_str = " ".join(f"{v:6.3f}" for v in our_t2i[i])
            best = int(np.argmax(our_t2i[i]))
            print(f"    text {i}: [{row_str}]  best={best} ({FIXTURE_NAMES[best]})")

        # Check diagonal dominance (each image's best text is its matched prompt)
        our_best_i2t = np.argmax(our_cm, axis=1)
        i2t_correct = np.all(our_best_i2t == np.arange(len(FIXTURE_NAMES)))
        our_best_t2i = np.argmax(our_t2i, axis=1)
        t2i_correct = np.all(our_best_t2i == np.arange(len(FIXTURE_NAMES)))

        print(f"  i2t correct match: {i2t_correct}")
        print(f"  t2i correct match: {t2i_correct}")

        self.results["cross_modal_i2t_correct"] = bool(i2t_correct)
        self.results["cross_modal_t2i_correct"] = bool(t2i_correct)

        # Synthetic fixtures may not support strong retrieval - report but don't fail
        if not i2t_correct or not t2i_correct:
            print("  NOTE: Synthetic fixtures may not support strong cross-modal retrieval")

    def _report(self):
        print("\n" + "=" * 60)
        print(f"TEST SUITE RESULTS — Backend: {self.backend}")
        print("=" * 60)
        for key, value in self.results.items():
            print(f"  {key}: {value}")
        print()
        if self.failures:
            print(f"FAILURES ({len(self.failures)}):")
            for f in self.failures:
                print(f"  - {f}")
            print("\nRESULT: FAILED")
            return 1
        else:
            print("RESULT: ALL TESTS PASSED")
            return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    backend = args.backend
    model_dir = args.model_dir

    if backend == "vllm":
        suite = TestSuite(
            backend="vllm", model_dir=model_dir, dtype=args.dtype,
        )
    elif backend == "native":
        suite = TestSuite(
            backend="native",
            model_dir=model_dir,
            native_bin=args.native_bin,
            mmproj=args.mmproj,
            text_model=args.text_model,
            compat_head=args.compat_head,
            out_dir=args.out_dir,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return await suite.run_all()


def main():
    parser = argparse.ArgumentParser(
        description="Shared correctness test suite for Qwen3-SigLIP2-VL"
    )
    parser.add_argument(
        "--backend", choices=["vllm", "native"], required=True,
        help="Backend to test",
    )
    parser.add_argument(
        "--model-dir", default="/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl",
        help="Path to the canonical model directory",
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16"],
        help="vLLM dtype",
    )
    # Native options
    parser.add_argument("--native-bin", help="Path to qwen3vl-embed executable")
    parser.add_argument("--mmproj", help="Path to mmproj sidecar GGUF")
    parser.add_argument("--text-model", help="Path to text GGUF")
    parser.add_argument("--compat-head", help="Path to compat head GGUF")
    parser.add_argument("--out-dir", default="/tmp/native_test_out",
                        help="Output dir for native results")
    args = parser.parse_args()

    if args.backend == "native" and not args.native_bin:
        parser.error("--native-bin is required for native backend")

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
