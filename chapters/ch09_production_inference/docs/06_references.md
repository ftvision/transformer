# Chapter 9: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### PagedAttention Paper

**"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023)
- Paper: https://arxiv.org/abs/2309.06180
- **Priority**: Must read
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 3: The virtual memory analogy and block-based KV cache
  - Section 4: Memory sharing and copy-on-write
  - Figure 2: Memory layout comparison
- **Key insight**: The core innovation is treating KV cache like OS virtual memory - this single idea enables massive throughput improvements.

---

## Core Libraries & Documentation

### vLLM

**Official Documentation**: https://docs.vllm.ai/
- **Priority**: Must read for serving
- **What to focus on**:
  - Getting Started guide
  - Distributed inference section
  - Quantization options
  - OpenAI-compatible server setup

**GitHub**: https://github.com/vllm-project/vllm
- **Files to read**: `vllm/engine/llm_engine.py` (core loop), `vllm/core/scheduler.py` (scheduling)

### llama.cpp

**GitHub**: https://github.com/ggerganov/llama.cpp
- **Priority**: Must read for CPU inference
- **What to focus on**:
  - README for build instructions
  - Quantization options in `examples/quantize/`
  - GGUF format specification

**GGUF Format Spec**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Priority**: Important if you're converting models
- **Time**: 30 minutes

### SGLang

**Official Documentation**: https://sgl-project.github.io/
- **Priority**: Must read for structured generation
- **What to focus on**:
  - Programming model (the `@sgl.function` decorator)
  - Constrained generation examples
  - RadixAttention explanation

**Paper**: "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al., 2024)
- Paper: https://arxiv.org/abs/2312.07104
- **Priority**: Recommended
- **Time**: 1-2 hours

---

## Recommended Reading

### Continuous Batching

**"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (Yu et al., 2022)
- Paper: https://www.usenix.org/conference/osdi22/presentation/yu
- **Priority**: Recommended
- **Time**: 1-2 hours
- **Why read it**: Introduces iteration-level scheduling (continuous batching). vLLM and others build on these ideas.

### Speculative Decoding

**"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2023)
- Paper: https://arxiv.org/abs/2211.17192
- **Priority**: Recommended
- **Time**: 1 hour
- **What to focus on**: The acceptance/rejection mechanism, when speculative decoding helps.

**"Medusa: Simple LLM Inference Acceleration Framework"** (Cai et al., 2024)
- Paper: https://arxiv.org/abs/2401.10774
- **Priority**: Optional but interesting
- **Why read it**: An alternative to speculative decoding that uses multiple prediction heads.

### Quantization

**"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** (Frantar et al., 2023)
- Paper: https://arxiv.org/abs/2210.17323
- **Priority**: Recommended for quantization
- **Time**: 1-2 hours
- **What to focus on**: Sections 2-3, the optimal brain quantization approach.

**"AWQ: Activation-aware Weight Quantization"** (Lin et al., 2023)
- Paper: https://arxiv.org/abs/2306.00978
- **Priority**: Recommended
- **Why read it**: Shows that not all weights are equally important for quantization.

---

## Deep Dives (Optional)

### TensorRT-LLM

**Documentation**: https://nvidia.github.io/TensorRT-LLM/
- **Priority**: Optional (NVIDIA-specific)
- **When to read**: If deploying on NVIDIA GPUs at scale

**GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **What to focus on**: Examples directory for common model configurations

### Constrained Decoding Theory

**"Efficient Guided Generation for Large Language Models"** (Willard & Louf, 2023)
- Paper: https://arxiv.org/abs/2307.09702
- **Priority**: Optional deep dive
- **Why read it**: The theory behind regex/grammar-based constrained decoding. Used in SGLang.

### KV Cache Compression

**"Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time"** (Liu et al., 2023)
- Paper: https://arxiv.org/abs/2305.17118
- **Priority**: Optional
- **Why read it**: Techniques for reducing KV cache beyond PagedAttention.

---

## Benchmarking & Evaluation

### LLMPerf

**Repository**: https://github.com/ray-project/llmperf
- **Priority**: Useful for benchmarking
- **What it does**: Standardized benchmarks for LLM serving systems
- **When to use**: Comparing frameworks for your specific workload

### vLLM Benchmarks

**Location**: `benchmarks/` in vLLM repo
- **Scripts to know**: `benchmark_throughput.py`, `benchmark_latency.py`
- **What to measure**: Tokens/second, time-to-first-token, memory usage

---

## Video Resources

### vLLM Talk (OSDI 2023)

**URL**: https://www.youtube.com/watch?v=5ZlavKF_98U
- **Priority**: Recommended
- **Time**: 20 minutes
- **Why watch**: Clear explanation of PagedAttention by the authors

### llama.cpp Deep Dive

**Karpathy's "Let's build llama3 from scratch"**: https://www.youtube.com/watch?v=BUmJYxKuIJM
- **Priority**: Highly recommended
- **Time**: ~2 hours
- **Why watch**: Builds intuition for how inference works at a low level

---

## Quick Reference Card

| Topic | Primary Resource | Time |
|-------|------------------|------|
| PagedAttention | vLLM paper Section 3-4 | 1 hour |
| Continuous batching | Orca paper | 1 hour |
| Quantization overview | GPTQ paper Sections 2-3 | 1 hour |
| Constrained decoding | SGLang paper | 1 hour |
| llama.cpp usage | GitHub README | 30 min |
| vLLM deployment | Official docs | 1 hour |

---

## What's NOT Covered Here

These topics are covered in other chapters:

- **KV cache basics** → Chapter 8 (`kv_cache.md`)
- **Flash Attention** → Chapter 10
- **Distributed training** → Chapter 11
- **Custom CUDA kernels** → Chapter 12

Stay focused on understanding production inference frameworks first. The optimization techniques build on this foundation.

---

## Lab Connections

After reading these docs, you'll be ready for the labs:

| Lab | Relevant Reading |
|-----|------------------|
| Lab 01: HuggingFace Basics | HuggingFace docs |
| Lab 02: vLLM Serving | vLLM paper, docs |
| Lab 03: llama.cpp | llama.cpp GitHub, GGUF spec |
| Lab 04: SGLang | SGLang paper, docs |
| Lab 05: Benchmark | LLMPerf, vLLM benchmarks |
