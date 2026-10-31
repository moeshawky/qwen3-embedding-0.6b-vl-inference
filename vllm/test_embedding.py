"""Test vLLM text embedding through AsyncLLMEngine - batched unequal lengths."""
import asyncio
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
os.environ['VLLM_PLUGINS'] = 'qwen3_vl_vllm.register'

from qwen3_vl_vllm import register  # noqa: F401

from vllm import AsyncLLMEngine, PoolingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt


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
    print("Engine created!")

    pooling_params = PoolingParams(task="embed")

    texts = ["hi", "this is a longer sentence for testing batch pooling", "short"]
    
    async def get_embedding(text, req_id):
        prompt = TextPrompt(prompt=text)
        async for output in engine.encode(prompt, pooling_params, req_id):
            if output.finished:
                emb = output.outputs.data.clone()
                norm = (emb ** 2).sum() ** 0.5
                return emb, norm.item()
        return None, None

    results = await asyncio.gather(*[
        get_embedding(t, f"test-batch-{i}") for i, t in enumerate(texts)
    ])

    for i, (t, (emb, norm)) in enumerate(zip(texts, results)):
        print(f"Item {i} '{t}': norm={norm:.6f}, first5={emb[:5]}")

    # Reference comparison - use same tokenizer path as vLLM
    print("\n--- Reference comparison ---")
    from transformers import AutoModel, AutoTokenizer
    model_dir = '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl'
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok.backend_tokenizer.normalizer = None
    
    # Load model directly without AutoConfig (to avoid config class mismatch)
    import importlib
    sys.path.insert(0, model_dir)
    from modeling_qwen_siglip2_embedding import Qwen3Siglip2EmbeddingModel
    from configuration_qwen_siglip2_embedding import Qwen3Siglip2EmbeddingConfig
    from safetensors.torch import load_file
    
    config = Qwen3Siglip2EmbeddingConfig.from_pretrained(model_dir)
    model = Qwen3Siglip2EmbeddingModel(config)
    state = load_file(model_dir + '/model.safetensors')
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    model = model.eval().cuda().float()
    
    # Tokenize like vLLM does - with add_special_tokens=True (adds EOS 151643)
    all_ids = [tok(t, add_special_tokens=True)['input_ids'] for t in texts]
    m = max(len(x) for x in all_ids)
    I = torch.tensor([x + [151643] * (m - len(x)) for x in all_ids]).cuda()
    n_content = torch.tensor([len(x) for x in all_ids]).cuda()
    ar = torch.arange(m, device=I.device).unsqueeze(0)
    mask = (ar <= n_content.unsqueeze(1)).long()  # attend through EOS terminator
    pool_index = n_content  # index of EOS terminator (0-indexed)
    with torch.no_grad():
        ref_embeds = model.get_text_features(input_ids=I, attention_mask=mask, pool_index=pool_index)

    for i, t in enumerate(texts):
        vllm_emb = results[i][0].float().cuda()
        ref_emb = ref_embeds[i]
        cos = F.cosine_similarity(vllm_emb.unsqueeze(0), ref_emb.unsqueeze(0))
        print(f"Text '{t}': vLLM vs PyTorch cosine = {cos.item():.8f}")

    await engine.shutdown()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
