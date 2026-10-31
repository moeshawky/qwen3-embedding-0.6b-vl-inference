"""Test vLLM image encoding using HF processor pixel_values (isolates vision tower)."""
import asyncio
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')
os.environ['VLLM_PLUGINS'] = 'qwen3_vl_vllm.register'

from qwen3_vl_vllm import register  # noqa: F401

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


async def main():
    engine_args = AsyncEngineArgs(
        model='/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl',
        runner='pooling',
        convert='embed',
        dtype='float32',
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
        hf_overrides={'architectures': ['Qwen3Siglip2EmbeddingModel']},
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    model = engine.model_driver.model_model
    print("Engine created!")

    # Use HF processor for pixel_values (isolates vision tower from preprocessing)
    from transformers import AutoModel, AutoImageProcessor
    model_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl'
    ref_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval().cuda()
    processor = AutoImageProcessor.from_pretrained(model_dir)

    img_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl-native/native/fixtures'
    names = ['square', 'landscape', 'portrait', 'odd', 'tiny', 'wide']

    ref_inputs = processor(images=[f"{img_dir}/{n}.png" for n in names], return_tensors="pt")
    pv = ref_inputs['pixel_values'].cuda().float()
    pam = ref_inputs.get('pixel_attention_mask', torch.ones(pv.shape[0], pv.shape[2], dtype=torch.int64)).cuda()
    ss = ref_inputs.get('spatial_shapes', torch.tensor([[16, 16]] * 6)).cuda()

    print(f"HF pixel_values shape: {pv.shape}")
    print(f"HF pixel_attention_mask shape: {pam.shape}")
    print(f"HF spatial_shapes: {ss}")

    # Encode through vLLM model's encode_images (direct SigLIP2 on CUDA)
    with torch.no_grad():
        vllm_embeds = model.encode_images(
            pixel_values=pv,
            pixel_attention_mask=pam,
            spatial_shapes=ss,
        )

    # Reference through PyTorch
    with torch.no_grad():
        ref_embeds = ref_model.get_image_features(
            pixel_values=pv,
            pixel_attention_mask=pam,
            spatial_shapes=ss,
        )

    print(f"vLLM embeds shape: {vllm_embeds.shape}")
    print(f"Ref embeds shape: {ref_embeds.shape}")

    for i, name in enumerate(names):
        cos = F.cosine_similarity(vllm_embeds[i:i+1], ref_embeds[i:i+1])
        norm_v = torch.norm(vllm_embeds[i]).item()
        norm_r = torch.norm(ref_embeds[i]).item()
        print(f"{name}: vLLM vs PyTorch cosine = {cos.item():.8f}, norm_vllm={norm_v:.6f}, norm_ref={norm_r:.6f}")

    await engine.shutdown()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
