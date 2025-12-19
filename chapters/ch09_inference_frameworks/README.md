# Chapter 9: Production Inference Frameworks

## Overview

Building efficient inference systems is one thing—using battle-tested production frameworks is another. In this chapter, you'll learn the production inference ecosystem: HuggingFace for rapid prototyping, vLLM for high-throughput serving, llama.cpp for efficient CPU inference, and SGLang for structured generation.

## Learning Objectives

By the end of this chapter, you will:
- Use HuggingFace Transformers for model loading, generation, and fine-tuning
- Deploy models with vLLM and understand PagedAttention
- Run quantized models on CPU with llama.cpp
- Generate structured outputs (JSON) with SGLang
- Benchmark and compare frameworks for different use cases

## Key Concepts

- **PagedAttention**: Virtual memory for KV-cache, enabling efficient memory management
- **vLLM**: High-throughput serving with continuous batching and PagedAttention
- **llama.cpp**: GGUF format, CPU inference, aggressive quantization
- **SGLang**: Structured generation with RadixAttention for constraint decoding
- **Speculative decoding**: Using a small model to accelerate a large model
- **Framework selection**: Matching the right tool to the job

## Reading Order

1. Start with `docs/01_paged_attention.md` - understand the memory innovation
2. Read `docs/02_vllm_architecture.md` - how vLLM achieves high throughput
3. Read `docs/03_llama_cpp.md` - efficient CPU inference and GGUF format
4. Read `docs/04_sglang.md` - structured generation and RadixAttention
5. Read `docs/05_framework_comparison.md` - when to use what

## Labs

| Lab | Title | What you build |
|-----|-------|----------------|
| lab01 | HuggingFace Basics | Load models, generate text, basic fine-tuning |
| lab02 | vLLM Serving | Deploy and serve with vLLM, measure throughput |
| lab03 | llama.cpp | Convert to GGUF, quantize, run on CPU |
| lab04 | SGLang | Structured JSON generation with constraints |
| lab05 | Benchmark | Compare frameworks on throughput, latency, memory |

## How to Work Through This Chapter

```bash
# 1. Read the docs
cat docs/01_paged_attention.md

# 2. Start lab 1
cd lab01_huggingface
cat README.md  # Read instructions

# 3. Implement the code
# Edit src/hf_basics.py

# 4. Run tests until green
uv run pytest tests/

# 5. Move to next lab
cd ../lab02_vllm
```

## Milestone

Serve a 7B model with vLLM, achieving >100 tokens/sec throughput on appropriate hardware.

## Prerequisites

- Completed Chapter 8 (Memory & Inference Optimization)
- Understanding of KV-cache and batching strategies
- Basic familiarity with REST APIs (for serving)

## Hardware Notes

- **Lab 01-04**: Can run on laptop with small models (e.g., GPT-2, TinyLlama)
- **Lab 05 (Benchmark)**: GPU recommended for full benchmarking
- **vLLM**: Requires GPU for optimal performance (CPU mode available but slow)
- **llama.cpp**: Designed for CPU, works well on laptops
