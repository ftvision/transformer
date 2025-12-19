# llama.cpp: Efficient CPU Inference

## What is llama.cpp?

llama.cpp is a C/C++ implementation of LLaMA inference, designed for:
- **CPU inference** without requiring a GPU
- **Aggressive quantization** (2-bit, 3-bit, 4-bit, etc.)
- **Portability** across platforms (Mac, Linux, Windows, mobile)
- **Low memory usage** for running large models on consumer hardware

## Why llama.cpp Matters

Before llama.cpp, running a 7B model required a high-end GPU. Now:

```
Model: Llama-2-7B
Original: 14 GB (fp16)
Quantized: 3.5 GB (4-bit)  ← Fits in 8GB RAM!
```

This democratized access to LLMs for hobbyists, researchers, and edge deployments.

## The GGUF Format

llama.cpp uses **GGUF** (GPT-Generated Unified Format), which includes:
- Model weights in various quantization formats
- Model architecture metadata
- Tokenizer data

```
model.gguf
├── Header
│   ├── Model name
│   ├── Architecture (llama, mistral, etc.)
│   ├── Context length
│   └── Vocab size
├── Tokenizer
│   ├── Token vocabulary
│   └── Merge rules (for BPE)
└── Weights
    ├── Layer 0: Q, K, V, O projections
    ├── Layer 0: FFN weights
    ├── ...
    └── Layer N
```

### Quantization Types in GGUF

| Type | Bits | Description | Size (7B) | Quality |
|------|------|-------------|-----------|---------|
| F16 | 16 | Full fp16 | 14 GB | Best |
| Q8_0 | 8 | 8-bit symmetric | 7 GB | Excellent |
| Q5_K_M | 5.5 | K-quant mixed | 5 GB | Very Good |
| Q4_K_M | 4.5 | K-quant mixed | 4 GB | Good |
| Q4_0 | 4 | 4-bit symmetric | 3.5 GB | Good |
| Q3_K_M | 3.5 | K-quant mixed | 3 GB | Acceptable |
| Q2_K | 2.5 | 2-bit k-quant | 2.5 GB | Degraded |

**K-quants** use different precisions for different layers based on importance.

## Installation

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Or with GPU support (CUDA)
make -j LLAMA_CUDA=1

# Or with Metal support (Mac)
make -j LLAMA_METAL=1
```

## Converting Models to GGUF

### From HuggingFace

```bash
# Download conversion script dependencies
pip install torch transformers sentencepiece

# Convert to fp16 GGUF
python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf

# Quantize to 4-bit
./llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
```

### Download Pre-Quantized

Many models are pre-quantized on HuggingFace:

```bash
# Example: TheBloke's quantized models
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
```

## Running Inference

### CLI Usage

```bash
# Simple completion
./llama-cli -m model-q4_k_m.gguf -p "The capital of France is"

# Interactive chat
./llama-cli -m model-q4_k_m.gguf -i

# With parameters
./llama-cli -m model-q4_k_m.gguf \
    -p "Write a poem about AI" \
    -n 200 \           # Max tokens
    --temp 0.7 \       # Temperature
    --top-p 0.9 \      # Top-p sampling
    --threads 8        # CPU threads
```

### Server Mode

```bash
# Start OpenAI-compatible server
./llama-server -m model-q4_k_m.gguf --port 8080

# Use with curl
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-7b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Python Bindings

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="model-q4_k_m.gguf",
    n_ctx=2048,        # Context length
    n_threads=8,       # CPU threads
    n_gpu_layers=0     # GPU layers (0 for CPU-only)
)

# Generate
output = llm(
    "The meaning of life is",
    max_tokens=100,
    temperature=0.7,
    stop=[".", "\n"]
)
print(output["choices"][0]["text"])

# Chat format
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=200
)
print(output["choices"][0]["message"]["content"])
```

## Performance Optimization

### 1. Thread Count

```python
# Match to physical cores (not hyperthreads)
llm = Llama(model_path="model.gguf", n_threads=8)
```

### 2. GPU Offloading

Offload some layers to GPU while keeping others on CPU:

```python
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=20  # Offload first 20 layers to GPU
)
```

For full GPU: `n_gpu_layers=-1`

### 3. Memory Mapping

llama.cpp memory-maps the model file, allowing:
- Faster loading (no full read into RAM)
- Shared memory between processes
- Lower peak memory usage

```python
llm = Llama(
    model_path="model.gguf",
    use_mmap=True,    # Default: True
    use_mlock=False   # Lock in RAM (prevents swapping)
)
```

### 4. Batch Size

```python
llm = Llama(
    model_path="model.gguf",
    n_batch=512  # Tokens per batch (affects prefill speed)
)
```

## Quantization Quality Comparison

Perplexity on WikiText-2 (lower is better):

| Model | F16 | Q8_0 | Q4_K_M | Q4_0 | Q3_K_M |
|-------|-----|------|--------|------|--------|
| Llama-2-7B | 5.47 | 5.48 | 5.54 | 5.68 | 5.75 |
| Mistral-7B | 5.25 | 5.26 | 5.32 | 5.45 | 5.52 |

**Takeaway**: Q4_K_M offers excellent quality with 4x compression.

## Memory Requirements

For a 7B model with 4K context:

| Quantization | Model | KV Cache | Total (approx) |
|--------------|-------|----------|----------------|
| Q4_K_M | 4 GB | 1 GB | 5 GB |
| Q8_0 | 7 GB | 1 GB | 8 GB |
| F16 | 14 GB | 1 GB | 15 GB |

KV cache size = `n_ctx × n_layers × 2 × n_heads × head_dim × 2 bytes`

## Comparison with Other Frameworks

| Feature | llama.cpp | vLLM | HuggingFace |
|---------|-----------|------|-------------|
| CPU inference | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| GPU inference | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Quantization | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Memory efficiency | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Throughput | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Portability | ⭐⭐⭐ | ⭐ | ⭐⭐ |

## When to Use llama.cpp

**Best for**:
- CPU-only environments
- Consumer hardware (laptops, desktops)
- Edge deployment (mobile, embedded)
- Maximum quantization (2-4 bit)
- Local/private inference

**Not ideal for**:
- High-throughput serving (use vLLM)
- Training or fine-tuning
- Very large batch sizes

## Example: Running on a MacBook

```python
from llama_cpp import Llama

# Load 7B model quantized to 4-bit
# Fits in 8GB M1 MacBook!
llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=-1  # Use Metal GPU acceleration
)

# Chat
response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Explain quantum computing simply."}
    ],
    max_tokens=500
)
print(response["choices"][0]["message"]["content"])
```

Expected performance on M1 MacBook Pro:
- ~20-30 tokens/sec generation
- ~2 sec time to first token
- Peak memory: ~5GB

## What's Next

We've covered efficient inference with llama.cpp. But what if you need structured output (like JSON)? See `04_sglang.md` for SGLang.
