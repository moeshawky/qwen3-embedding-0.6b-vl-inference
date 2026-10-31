#!/bin/bash
# Benchmark vLLM embedding server performance
#
# Measures latency and throughput for text + image embeddings under batching.
#
# Usage:
#   ./benchmark.sh [port]
#
set -euo pipefail

PORT="${1:-8000}"
BASE_URL="http://localhost:$PORT"

echo "============================================="
echo "vLLM Embedding Benchmark"
echo "============================================="
echo "Server: $BASE_URL"
echo "Date: $(date)"
echo ""

# Check server is running
if ! curl -s "$BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "ERROR: Server not running at $BASE_URL"
    echo "Start it with: ./serve_vllm.sh $PORT"
    exit 1
fi
echo "Server is running"
echo ""

# ---- Text embedding benchmarks ----
echo "---------------------------------------------"
echo "Text Embedding Benchmarks"
echo "---------------------------------------------"

# Single request latency
echo ""
echo "--- Single request latency ---"
for i in 1 2 3 4 5; do
    START=$(python3 -c "import time; print(time.time())")
    curl -s -X POST "$BASE_URL/v1/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"input\": [\"a synthetic square gradient image with red, green and blue rectangles\"], \"model\": \"qwen3-vl\"}" > /dev/null
    END=$(python3 -c "import time; print(time.time())")
    LATENCY=$(echo "$END - $START" | python3 -c "import sys; print(f'{float(sys.stdin.read().strip()):.3f}')")
    echo "  Run $i: ${LATENCY}s"
done

# Batched requests
echo ""
echo "--- Batched requests (different batch sizes) ---"
python3 -c "
import asyncio, sys, time
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/vllm')
sys.path.insert(0, '/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl')
import os
os.environ['VLLM_PLUGINS'] = 'qwen3_vl_vllm.register'
from qwen3_vl_vllm import register
from vllm import AsyncLLMEngine, PoolingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt

async def bench():
    engine_args = AsyncEngineArgs(
        model='/kaggle/working/qwen-0.6-vl/qwen3-embedding-0.6b-vl',
        runner='pooling', convert='embed', dtype='float32',
        max_model_len=4096, gpu_memory_utilization=0.4,
        trust_remote_code=True,
        hf_overrides={'architectures': ['Qwen3Siglip2EmbeddingModel']},
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    pooling_params = PoolingParams(task='embed')
    
    texts = [
        'a synthetic square gradient image with red, green and blue rectangles',
        'a wide landscape photograph of a green field next to a blue sky',
        'a tall portrait photograph of a red brick wall',
        'an odd sized image with an unusual aspect ratio',
        'a tiny blurry low resolution image',
        'an extremely wide panoramic strip',
        'a photo of a cat',
        'a photo of a dog',
    ]
    
    for batch_size in [1, 2, 4, 8]:
        batch = texts[:batch_size]
        prompts = [TextPrompt(prompt=t) for t in batch]
        
        # Warmup
        async for _ in engine.encode(prompts[0], pooling_params, 'warmup'):
            pass
        
        # Benchmark
        start = time.time()
        tasks = []
        for i, p in enumerate(prompts):
            tasks.append(engine.encode(p, pooling_params, f'bench-{batch_size}-{i}'))
        
        results = await asyncio.gather(*[
            _collect(t) for t in tasks
        ])
        elapsed = time.time() - start
        
        per_item = elapsed / batch_size
        print(f'  Batch {batch_size}: {elapsed:.3f}s total, {per_item:.3f}s per item')

async def _collect(coro):
    async for output in coro:
        if output.finished:
            return output.outputs.data
    return None

import time
asyncio.run(bench())
"

# ---- Memory ----
echo ""
echo "--- GPU Memory ---"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while read used total; do
    echo "  GPU memory: ${used}MiB / ${total}MiB used"
done

echo ""
echo "============================================="
echo "Benchmark complete"
echo "============================================="
