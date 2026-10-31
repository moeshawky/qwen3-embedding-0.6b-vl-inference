"""vLLM server for Qwen3-SigLIP2-VL dual-encoder embedding model.

Serves both text and image embeddings:
- Text: vLLM AsyncLLMEngine (GPU-accelerated, batched, correct terminal pooling)
- Image: Direct SigLIP2VisionModel on CUDA (NaFlex preprocessing → MAP pool → projection → norm)

Endpoints:
  POST /v1/embeddings          - Text embeddings (OpenAI-compatible)
  POST /embed/image            - Image embeddings (custom, base64 or URL)
  GET  /health                 - Health check
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from pydantic import Field

import sys
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')

os_env = __import__('os')
os_env.environ['VLLM_PLUGINS'] = 'qwen3_vl_vllm.register'

from qwen3_vl_vllm import register  # noqa: F401
from qwen3_vl_vllm.preprocess import preprocess_images, load_image, b64_to_image

from vllm import AsyncLLMEngine, PoolingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class TextEmbedRequest(BaseModel):
    input: list[str] = Field(..., description="List of texts to embed")
    model: Optional[str] = None


class TextEmbedResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    model: str
    usage: dict


class ImageEmbedRequest(BaseModel):
    image: Optional[str] = None        # base64-encoded image
    images: Optional[list[str]] = None  # list of base64-encoded images
    model: Optional[str] = None


class ImageEmbedResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    model: str


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Qwen3VLEmbeddingServer:
    def __init__(
        self,
        model_dir: str = "/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl",
        dtype: str = "float32",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.4,
    ):
        self.model_dir = model_dir
        self.engine = None
        self.model = None  # Direct model reference for encode_images

    async def start(self):
        """Start the vLLM engine and load the model."""
        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            runner='pooling',
            convert='embed',
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            hf_overrides={'architectures': ['Qwen3Siglip2EmbeddingModel']},
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # Get direct model reference for image encoding
        self.model = self.engine.model_driver.model_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts through vLLM."""
        pooling_params = PoolingParams(task="embed")

        async def get_embedding(text, req_id):
            prompt = TextPrompt(prompt=text)
            async for output in self.engine.encode(prompt, pooling_params, req_id):
                if output.finished:
                    return output.outputs.data.cpu().numpy().tolist()
            return None

        results = await asyncio.gather(*[
            get_embedding(t, f"req-{i}") for i, t in enumerate(texts)
        ])
        return [r for r in results if r is not None]

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed images through direct SigLIP2 vision tower on CUDA."""
        if not images:
            return []
        inputs = preprocess_images(images)
        pixel_values = torch.from_numpy(inputs["pixel_values"]).cuda().float()
        pixel_attention_mask = torch.from_numpy(inputs["pixel_attention_mask"]).cuda()
        spatial_shapes = torch.from_numpy(inputs["spatial_shapes"]).cuda()

        with torch.no_grad():
            embeds = self.model.encode_images(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
        return embeds.cpu().numpy().tolist()

    async def shutdown(self):
        if self.engine:
            await self.engine.shutdown()


# Global server instance
server: Optional[Qwen3VLEmbeddingServer] = None

app = FastAPI(title="Qwen3-SigLIP2-VL Embedding Server")


@app.on_event("startup")
async def startup():
    global server
    server = Qwen3VLEmbeddingServer()
    await server.start()


@app.on_event("shutdown")
async def shutdown():
    global server
    if server:
        await server.shutdown()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "qwen3-siglip2-vl"}


@app.post("/v1/embeddings", response_model=TextEmbedResponse)
async def create_embeddings(req: TextEmbedRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    texts = req.input
    embeddings = await server.embed_texts(texts)
    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": emb,
        }
        for i, emb in enumerate(embeddings)
    ]
    return TextEmbedResponse(
        data=data,
        model=req.model or "qwen3-siglip2-vl",
        usage={"prompt_tokens": sum(len(t) for t in texts), "total_tokens": sum(len(t) for t in texts)},
    )


@app.post("/embed/image", response_model=ImageEmbedResponse)
async def embed_image(req: ImageEmbedRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    b64_images = []
    if req.image:
        b64_images.append(req.image)
    if req.images:
        b64_images.extend(req.images)

    if not b64_images:
        raise HTTPException(status_code=400, detail="No image provided")

    pil_images = []
    for b64 in b64_images:
        try:
            img = b64_to_image(b64)
            pil_images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    embeddings = server.embed_images(pil_images)
    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": emb,
        }
        for i, emb in enumerate(embeddings)
    ]
    return ImageEmbedResponse(
        data=data,
        model=req.model or "qwen3-siglip2-vl",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
