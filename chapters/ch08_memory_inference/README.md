# Chapter 8: Memory & Inference Optimization

## Overview

LLM inference is fundamentally different from training—it's memory-bound, not compute-bound. In this chapter, you'll understand why, and implement the key optimizations that make fast inference possible.

## Learning Objectives

By the end of this chapter, you will:
- Understand why LLM inference is memory-bound and calculate theoretical throughput limits
- Implement KV-cache to eliminate redundant computation in autoregressive generation
- Compare static vs continuous batching and understand their tradeoffs
- Implement basic quantization (INT8) to reduce model memory footprint

## Key Concepts

- **Memory bandwidth bottleneck**: Why loading weights dominates inference time
- **Arithmetic intensity**: Operations per byte loaded—the key metric for understanding performance
- **KV-cache**: Caching key and value tensors to avoid recomputation
- **Prefill vs decode**: Two phases with different characteristics
- **Continuous batching**: Iteration-level scheduling for maximum throughput
- **Quantization**: Trading precision for speed and memory

## Reading Order

1. Start with `docs/01_memory_bound_inference.md` - understand *why* inference is slow
2. Read `docs/02_kv_cache.md` - the key optimization for autoregressive generation
3. Read `docs/03_batching_strategies.md` - how to serve multiple requests efficiently
4. Read `docs/04_quantization_basics.md` - making models smaller and faster
5. (Optional) See `docs/05_references.md` - papers, libraries, and further reading

## Labs

| Lab | Title | What you build |
|-----|-------|----------------|
| lab01 | Memory Bandwidth Analysis | Calculate roofline limits and arithmetic intensity |
| lab02 | KV-Cache | Implement KV-cache for attention |
| lab03 | Batching Strategies | Simulate static vs continuous batching |
| lab04 | Quantization | Implement INT8 quantization from scratch |

## How to Work Through This Chapter

```bash
# 1. Read the docs
cat docs/01_memory_bound_inference.md

# 2. Start lab 1
cd lab01_memory_bandwidth
cat README.md  # Read instructions

# 3. Implement the code
# Edit src/memory_analysis.py

# 4. Run tests until green
uv run pytest tests/

# 5. Move to next lab
cd ../lab02_kv_cache
```

## Milestone

Your KV-cache implementation achieves O(N) complexity for generating N tokens (instead of O(N²) without cache), and your INT8 quantization achieves <0.1% error on random weights.

## Prerequisites

- Completed Chapter 1 (attention mechanism)
- Understanding of matrix multiplication complexity
- Basic understanding of data types (float16, int8)
