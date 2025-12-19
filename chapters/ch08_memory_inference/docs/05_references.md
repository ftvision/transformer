# Chapter 8: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Memory-Bound Inference

**"LLM Inference Unveiled"** (Understanding the bottleneck)
- Blog: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 30-45 minutes
- **What to focus on**:
  - Memory bandwidth vs compute analysis
  - The roofline model explanation
  - Prefill vs decode phase differences
- **Key insight**: Understanding why inference is memory-bound changes how you think about optimization.

**"Making Deep Learning Go Brrrr From First Principles"** (Horace He)
- Blog: https://horace.io/brrr_intro.html
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Bandwidth-bound vs compute-bound operations
  - The arithmetic intensity concept
  - When fusion helps and why
- **Key insight**: This is the mental model for understanding GPU performance.

---

## Core Concepts

### KV-Cache

**"The KV Cache"** (HuggingFace documentation)
- Docs: https://huggingface.co/docs/transformers/kv_cache
- **Priority**: ⭐⭐⭐ Must understand
- **Time**: 20 minutes
- **What to focus on**:
  - How HuggingFace implements KV-cache
  - The `use_cache` parameter
  - Different cache types (DynamicCache, StaticCache)

**Grouped-Query Attention Paper** (GQA)
- Paper: https://arxiv.org/abs/2305.13245
- **Priority**: ⭐⭐ Important
- **Time**: 1 hour
- **What to focus on**: Section 2 (the GQA mechanism)
- **Why read it**: Explains how Llama-2-70B and others reduce KV-cache memory

### Batching & Serving

**"Orca: A Distributed Serving System for Transformer-Based Generative Models"**
- Paper: https://www.usenix.org/conference/osdi22/presentation/yu
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Iteration-level scheduling (continuous batching)
  - Selective batching for prefill vs decode
- **Key insight**: This paper introduced continuous batching to the LLM serving community.

---

## Quantization Deep Dives

### Survey and Overview

**"A Survey of Quantization Methods for Efficient Neural Network Inference"**
- Paper: https://arxiv.org/abs/2103.13630
- **Priority**: ⭐⭐ Important for depth
- **Time**: 2-3 hours (skim sections)
- **What to focus on**:
  - Sections 2-3: Quantization fundamentals
  - Section 5: Post-training quantization
- **What to skim**: Hardware-specific sections

### Specific Methods

**GPTQ: "Accurate Post-Training Quantization for Generative Pre-trained Transformers"**
- Paper: https://arxiv.org/abs/2210.17323
- **Priority**: ⭐⭐ Important
- **Time**: 1 hour
- **What to focus on**: The OBQ algorithm adaptation for LLMs
- **Code**: https://github.com/IST-DASLab/gptq

**AWQ: "AWQ: Activation-aware Weight Quantization"**
- Paper: https://arxiv.org/abs/2306.00978
- **Priority**: ⭐⭐ Important
- **Time**: 1 hour
- **What to focus on**: Why activations matter for weight quantization
- **Code**: https://github.com/mit-han-lab/llm-awq

**SmoothQuant: "Accurate and Efficient Post-Training Quantization"**
- Paper: https://arxiv.org/abs/2211.10438
- **Priority**: ⭐⭐ Important
- **Time**: 1 hour
- **What to focus on**: The smoothing transformation
- **Key insight**: Moving quantization difficulty from activations to weights

---

## Code References

### Reference Implementations

**llama.cpp**
- Code: https://github.com/ggerganov/llama.cpp
- **Priority**: ⭐⭐⭐ Must explore
- **What to focus on**:
  - `ggml.c`: The inference engine
  - Quantization formats (Q4_0, Q4_K, Q8_0, etc.)
  - KV-cache implementation
- **Why use it**: The gold standard for efficient CPU inference

**vLLM**
- Code: https://github.com/vllm-project/vllm
- **Priority**: ⭐⭐⭐ Must explore
- **What to focus on**:
  - `vllm/core/scheduler.py`: Continuous batching logic
  - `vllm/attention/`: PagedAttention implementation
- **Note**: We'll dive deeper in Chapter 9

**HuggingFace Transformers**
- KV-cache: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
- Generation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
- **Priority**: ⭐⭐ Useful reference
- **What to focus on**: The `generate()` method and cache handling

### Libraries

**bitsandbytes**
- Code: https://github.com/TimDettmers/bitsandbytes
- Docs: https://huggingface.co/docs/bitsandbytes/
- **Priority**: ⭐⭐⭐ Must know
- **What it does**: INT8 and INT4 quantization for PyTorch

**AutoGPTQ**
- Code: https://github.com/PanQiWei/AutoGPTQ
- **Priority**: ⭐⭐ Important
- **What it does**: GPTQ quantization with easy API

**llm-compressor** (formerly sparseml)
- Code: https://github.com/vllm-project/llm-compressor
- **Priority**: ⭐ Optional
- **What it does**: Comprehensive quantization and pruning toolkit

---

## Recommended Reading

### Blogs and Tutorials

**"Efficient Inference on a Single GPU"** (HuggingFace)
- Blog: https://huggingface.co/docs/transformers/perf_infer_gpu_one
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 30 minutes
- **What you'll learn**: Practical optimization techniques

**"A Gentle Introduction to Quantization"** (HuggingFace)
- Blog: https://huggingface.co/blog/merve/quantization
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 20 minutes
- **What you'll learn**: High-level overview of quantization landscape

**"The Full Stack LLM Inference Optimization"** (PyTorch)
- Blog: https://pytorch.org/blog/accelerating-generative-ai-2/
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 45 minutes
- **What you'll learn**: How PyTorch optimizes inference

---

## Deep Dives (Optional)

### Speculative Decoding

**"Fast Inference from Transformers via Speculative Decoding"**
- Paper: https://arxiv.org/abs/2211.17192
- **Priority**: ⭐ Optional but interesting
- **Time**: 1 hour
- **What you'll learn**: Using a small model to speed up a large model

### KV-Cache Compression

**"Scissorhands: Exploiting the Persistence of Importance"**
- Paper: https://arxiv.org/abs/2305.17118
- **Priority**: ⭐ Optional
- **What you'll learn**: Token eviction strategies for KV-cache

**"H2O: Heavy-Hitter Oracle for Efficient Generative Inference"**
- Paper: https://arxiv.org/abs/2306.14048
- **Priority**: ⭐ Optional
- **What you'll learn**: Keeping only important KV pairs

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Memory-bound basics | "Making Deep Learning Go Brrrr" | 2 hours |
| KV-cache | HuggingFace docs | 20 min |
| Continuous batching | Orca paper | 1-2 hours |
| INT8 quantization | bitsandbytes docs | 30 min |
| INT4 quantization | GPTQ/AWQ papers | 1 hour each |
| Practical optimization | HuggingFace inference guide | 30 min |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **PagedAttention** → Chapter 9
- **vLLM deep dive** → Chapter 9
- **Flash Attention** → Chapter 10
- **Distributed inference** → Chapter 11

Stay focused on understanding the fundamentals first:
1. Why inference is memory-bound
2. How KV-cache works
3. Batching strategies
4. Basic quantization

These concepts are prerequisites for the advanced serving techniques in Chapter 9.
