# vLLM Architecture: High-Throughput LLM Serving

## What is vLLM?

vLLM is an open-source library for fast LLM inference and serving. It achieves 2-24x higher throughput than HuggingFace Transformers by combining:

1. **PagedAttention** for memory efficiency
2. **Continuous batching** for GPU utilization
3. **Optimized CUDA kernels** for speed

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         vLLM Server                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   API       │    │  Scheduler  │    │   Block Manager     │  │
│  │  (OpenAI    │───▶│             │◀──▶│                     │  │
│  │  compatible)│    │             │    │  Physical Blocks    │  │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Model Runner                            ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ││
│  │  │   Model     │  │  Attention  │  │    KV Cache         │  ││
│  │  │   Weights   │  │   (Paged)   │  │    (GPU Blocks)     │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. The Scheduler

The scheduler is the brain of vLLM. It decides:
- Which requests to process in each iteration
- When to preempt requests (if memory is tight)
- How to handle priorities

```python
# Simplified scheduler logic
class Scheduler:
    def schedule(self):
        # 1. Get requests that can run
        running = []
        waiting = self.waiting_queue

        for request in waiting:
            blocks_needed = self.estimate_blocks(request)
            if self.block_manager.can_allocate(blocks_needed):
                self.block_manager.allocate(request)
                running.append(request)

        # 2. Check if running requests need more blocks
        for request in self.running:
            if request.needs_new_block():
                if self.block_manager.can_allocate(1):
                    self.block_manager.allocate_block(request)
                else:
                    # Preempt: swap to CPU or recompute later
                    self.preempt(request)

        return running
```

### 2. Block Manager

Manages the physical block allocation:

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # request_id -> list of block_ids

    def allocate(self, request_id, num_blocks):
        if len(self.free_blocks) < num_blocks:
            return False
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.block_tables[request_id] = blocks
        return True

    def free(self, request_id):
        blocks = self.block_tables.pop(request_id)
        self.free_blocks.extend(blocks)
```

### 3. Model Runner

Executes the actual model forward pass:

```python
class ModelRunner:
    def execute_model(self, requests):
        # 1. Prepare inputs
        input_ids = self.prepare_inputs(requests)
        block_tables = [r.block_table for r in requests]
        positions = [r.position for r in requests]

        # 2. Forward pass with paged attention
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_cache=self.kv_cache,
                block_tables=block_tables
            )

        # 3. Sample next tokens
        next_tokens = self.sampler.sample(logits, requests)

        return next_tokens
```

## Continuous Batching in vLLM

vLLM implements iteration-level batching:

```
Iteration 1: [Req1_prefill, Req2_prefill]
             ↓
Iteration 2: [Req1_decode, Req2_decode, Req3_prefill]
             ↓
Iteration 3: [Req1_decode, Req2_done✓, Req3_decode, Req4_prefill]
             ↓
Iteration 4: [Req1_decode, Req3_decode, Req4_decode]
```

No waiting! As soon as a request finishes, its slot is filled.

## Using vLLM

### Installation

```bash
pip install vllm
```

### Offline Inference

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
prompts = [
    "The capital of France is",
    "Machine learning is",
    "To be or not to be,"
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### Online Serving

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

```python
# Client (OpenAI-compatible API)
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require real key
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "What is PagedAttention?"}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Key Configuration Options

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # Memory settings
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=4096,          # Max sequence length

    # Performance settings
    tensor_parallel_size=1,      # Number of GPUs for tensor parallelism
    dtype="auto",                # Data type (auto, float16, bfloat16)

    # Batching settings
    max_num_batched_tokens=None, # Max tokens per batch (auto)
    max_num_seqs=256,            # Max concurrent sequences

    # Quantization
    quantization=None,           # "awq", "gptq", "squeezellm"
)
```

## Performance Tuning

### 1. Maximize GPU Memory Usage

```python
# Use more memory for KV cache = higher throughput
llm = LLM(model=model, gpu_memory_utilization=0.95)
```

### 2. Tune Batch Size

```python
# For throughput: larger batches
llm = LLM(model=model, max_num_seqs=512)

# For latency: smaller batches
llm = LLM(model=model, max_num_seqs=32)
```

### 3. Use Tensor Parallelism for Large Models

```python
# Split model across 2 GPUs
llm = LLM(model="meta-llama/Llama-2-70b-hf", tensor_parallel_size=2)
```

### 4. Enable Quantization

```python
# 4-bit AWQ quantization
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq"
)
```

## Benchmarking vLLM

```python
from vllm import LLM, SamplingParams
import time

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate many requests
prompts = ["Hello, how are you?"] * 100
sampling_params = SamplingParams(max_tokens=100)

start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed

print(f"Throughput: {throughput:.1f} tokens/sec")
```

## When to Use vLLM

**Best for**:
- High-throughput serving (many concurrent requests)
- Production deployments
- GPU-based inference
- Long sequences (benefits most from PagedAttention)

**Not ideal for**:
- CPU-only environments (use llama.cpp)
- Single-request latency optimization
- Very small models (overhead not worth it)

## Comparison with HuggingFace

| Feature | HuggingFace | vLLM |
|---------|-------------|------|
| Ease of use | ⭐⭐⭐ | ⭐⭐ |
| Throughput | ⭐ | ⭐⭐⭐ |
| Memory efficiency | ⭐ | ⭐⭐⭐ |
| Flexibility | ⭐⭐⭐ | ⭐⭐ |
| Production ready | ⭐⭐ | ⭐⭐⭐ |

## What's Next

vLLM is great for GPU serving. But what about running on CPU, or with extreme quantization? See `03_llama_cpp.md` for llama.cpp.
