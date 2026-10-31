"""Register Qwen3Siglip2EmbeddingModel with vLLM's ModelRegistry.

This module is imported by the vLLM server launch script to make the
custom dual-encoder embedding model available via ``vllm serve``.

It registers:
1. A config subclass (VLLMConfig) that inherits the patched
   ``get_text_config()`` / ``get_vision_config()`` from the base
   ``Qwen3Siglip2EmbeddingConfig`` (see
   ``qwen3-embedding-0.6b-vl/configuration_qwen_siglip2_embedding.py``).
2. The model class with vLLM's ``ModelRegistry``.

Usage
-----
    VLLM_PLUGINS=qwen3_vl_vllm.register vllm serve /path/to/model \
        --hf-overrides '{"architectures": ["Qwen3Siglip2EmbeddingModel"]}' \
        --model-impl vllm \
        --runner pooling \
        --convert embed
"""

import sys
from pathlib import Path

# Ensure the model directory is on sys.path for trust_remote_code imports.
_MODEL_DIR = Path(__file__).resolve().parents[2] / "qwen3-embedding-0.6b-vl"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from transformers import AutoConfig

from vllm.model_executor.models import ModelRegistry
from qwen3_vl_vllm.model import Qwen3Siglip2EmbeddingModel

# ---------------------------------------------------------------------------
# Load the base composite config class from the model repo
# ---------------------------------------------------------------------------

from configuration_qwen_siglip2_embedding import Qwen3Siglip2EmbeddingConfig  # noqa: E402


class Qwen3Siglip2EmbeddingVLLMConfig(Qwen3Siglip2EmbeddingConfig):
    """Composite config subclass.

    Inherits the patched get_text_config() / get_vision_config() from the
    base class, which wrap the dict-based sub-configs in typed Qwen3Config /
    Siglip2VisionConfig objects that vLLM requires.
    """
    pass


# Register the subclass so AutoConfig.from_pretrained() loads it.
AutoConfig.register(
    "qwen3_siglip2_embedding",
    Qwen3Siglip2EmbeddingVLLMConfig,
)

# ---------------------------------------------------------------------------
# Register the model architecture with vLLM
# ---------------------------------------------------------------------------

ModelRegistry.register_model(
    "Qwen3Siglip2EmbeddingModel",
    Qwen3Siglip2EmbeddingModel,
)

__all__ = ["Qwen3Siglip2EmbeddingModel"]
