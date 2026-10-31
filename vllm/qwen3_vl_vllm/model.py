"""Qwen3-SigLIP2-VL vLLM model implementation.

Dual-encoder embedding model for moeshawky/qwen3-embedding-0.6b-vl.

Text branch (vLLM-accelerated, batched correctly):
    Qwen3Model (28 layers, 1024 hidden) -> LastPool (terminal token per seq) ->
    L2 -> compat_head (1024->1024) -> L2  == [B, 1024]

Image branch (CUDA-accelerated SigLIP2, outside vLLM scheduler):
    SigLIP2VisionModel (27 blocks, 1152 hidden) -> MAP pool ->
    L2 -> vision_projection (1152->1024) -> L2  == [B, 1024]

Text and image embeddings land in the same shared 1024-d L2-normalised
retrieval space.

The model is registered with vLLM's ``ModelRegistry`` so it can be loaded
via ``vllm serve``.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Siglip2VisionConfig, Siglip2VisionModel

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.seqwise.heads import (
    EmbeddingPoolerHead,
)
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.activations import PoolerIdentity
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    default_pooling_type,
)
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

EOS_TOKEN_ID = 151643


# ---------------------------------------------------------------------------
# Compat head: L2 -> Linear(W.T + b) -> L2
# ---------------------------------------------------------------------------

class CompatHead(nn.Module):
    """1024->1024 affine head matching the reference recipe::

        v = F.normalize(hidden, p=2, dim=-1)
        z = linear(v)         # x @ W.T + b
        return F.normalize(z, p=2, dim=-1)

    The safetensors stores ``compat_head.weight`` as [1024, 1024] in the
    NumPy convention ``z = x @ W + b``.  ``nn.Linear`` computes
    ``x @ weight.T + bias`` with weight shaped [out, in], so we load the
    stored matrix directly as the Linear weight (no transpose).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1)
        z = self.linear(x)
        return F.normalize(z, p=2, dim=-1)

    def load_from_state_dict(self, state_dict: dict, prefix: str = "compat_head."):
        w_key = prefix + "weight"
        b_key = prefix + "bias"
        if w_key not in state_dict:
            raise KeyError(f"compat_head weight not found: {w_key}")
        self.linear.weight.data = (
            state_dict[w_key]
            .clone()
            .to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        )
        if b_key in state_dict and self.linear.bias is not None:
            self.linear.bias.data = (
                state_dict[b_key]
                .clone()
                .to(device=self.linear.bias.device, dtype=self.linear.bias.dtype)
            )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class Qwen3Siglip2EmbeddingModel(nn.Module, VllmModelForPooling):
    """Dual-encoder embedding model for vLLM.

    Text: vLLM Qwen3Model + LastPool + CompatHead (GPU, batched).
    Image: SigLIP2VisionModel on CUDA (direct, outside vLLM scheduler).
    """

    is_pooling_model: bool = True
    attn_type: str = "decoder"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        composite_config = vllm_config.model_config.hf_config

        # --- Text branch: Qwen3Model (vLLM-optimized CUDA) ------------------
        self.text_model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=f"{prefix}.text_model" if prefix else "text_model",
        )
        self.text_config = self.text_model.config

        # --- Compat head ----------------------------------------------------
        self.compat_head = CompatHead(
            in_dim=self.text_config.hidden_size,
            out_dim=self.text_config.hidden_size,  # 1024 -> 1024
            bias=True,
        )

        # --- Vision branch: SigLIP2VisionModel (direct CUDA) ----------------
        vision_cfg_dict = composite_config.vision_config or {}
        if not isinstance(vision_cfg_dict, Siglip2VisionConfig):
            vision_cfg_dict = Siglip2VisionConfig(**vision_cfg_dict)
        self.vision_config = vision_cfg_dict

        self.vision_model = Siglip2VisionModel(vision_cfg_dict)
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False

        # Vision projection: 1152 -> 1024
        self.vision_projection = nn.Linear(
            composite_config.projection_in_dim,
            composite_config.projection_out_dim,
            bias=True,
        )
        for p in self.vision_projection.parameters():
            p.requires_grad = False

        # --- Pooler: LastPool -> CompatHead (no extra activation) -----------
        # LastPool selects hidden_states[last_token_indices_gpu] for each seq.
        # CompatHead applies L2 -> Linear -> L2.
        self.pooler = SequencePooler(
            pooling=LastPool(),
            head=EmbeddingPoolerHead(
                projector=self.compat_head,
                activation=PoolerIdentity(),
                head_dtype=torch.float32,
            ),
        )

        # Move to device/dtype
        self.to(
            device=vllm_config.device_config.device,
            dtype=vllm_config.model_config.dtype,
        )

    # -- VllmModelForPooling interface --------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the Qwen3 text tower forward.

        Returns hidden states [total_tokens, 1024] for the pooler to consume.
        """
        if input_ids is None and inputs_embeds is None:
            # Dummy for image-only requests (image path uses encode_images).
            return torch.zeros(
                1,
                self.text_config.hidden_size,
                device=positions.device,
                dtype=torch.float32,
            )

        return self.text_model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    # -- Image encoding (CUDA-accelerated, outside vLLM scheduler) ----------

    def encode_images(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images through SigLIP2 vision tower on CUDA.

        Args:
            pixel_values: [B, num_patches, patch_dim] float32
            pixel_attention_mask: [B, num_patches] int64
            spatial_shapes: [B, 2] int64

        Returns:
            [B, 1024] L2-normalised image embeddings
        """
        self.vision_model.eval()
        with torch.no_grad():
            out = self.vision_model(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            v = F.normalize(out.pooler_output.float(), p=2, dim=-1)
            z = self.vision_projection(v)
            return F.normalize(z, p=2, dim=-1)

    # -- Weight loading ------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from safetensors into text model + compat head + vision."""
        loaded: set[str] = set()

        # Separate weights by prefix
        text_items = []
        compat_items = {}
        vision_items = {}
        vision_proj_items = {}

        for name, tensor in weights:
            if name.startswith("text_model."):
                text_items.append((name[len("text_model."):], tensor))
            elif name.startswith("compat_head."):
                compat_items[name] = tensor
            elif name.startswith("vision_model."):
                vision_items[name] = tensor
            elif name.startswith("vision_projection."):
                vision_proj_items[name] = tensor

        # Load text model weights via vLLM's AutoWeightsLoader
        if text_items:
            text_loaded = self.text_model.load_weights(text_items)
            loaded.update({"text_model." + k for k in text_loaded})

        # Load compat head
        if compat_items:
            self.compat_head.load_from_state_dict(compat_items)
            # Register with vLLM's weight tracker using parameter names.
            loaded.update({"compat_head.linear.weight", "compat_head.linear.bias"})

        # Load vision tower weights into SigLIP2VisionModel
        if vision_items:
            # Strip the "vision_model." prefix for SigLIP2VisionModel's state_dict
            vision_state = {
                k[len("vision_model."):]: v
                for k, v in vision_items.items()
                if k.startswith("vision_model.")
            }
            missing, unexpected = self.vision_model.load_state_dict(
                vision_state, strict=False, assign=True
            )
            if missing:
                logger.warning("Vision missing keys (first 10): %s", missing[:10])
            if unexpected:
                logger.warning("Vision unexpected keys (first 10): %s", unexpected[:10])
            loaded.update(vision_items.keys())

        # Load vision projection
        if vision_proj_items:
            # Strip the "vision_projection." prefix for the Linear layer
            proj_state = {
                k[len("vision_projection."):]: v
                for k, v in vision_proj_items.items()
                if k.startswith("vision_projection.")
            }
            missing, unexpected = self.vision_projection.load_state_dict(
                proj_state, strict=False, assign=True
            )
            if missing:
                logger.warning("Vision proj missing keys (first 10): %s", missing[:10])
            if unexpected:
                logger.warning("Vision proj unexpected keys (first 10): %s", unexpected[:10])
            loaded.update(vision_proj_items.keys())

        return loaded

    # -- Device/dtype --------------------------------------------------------

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.vision_model = self.vision_model.to(*args, **kwargs)
        self.vision_projection = self.vision_projection.to(*args, **kwargs)
        return self
