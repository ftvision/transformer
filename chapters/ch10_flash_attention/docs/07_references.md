# Chapter 10: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### The Flash Attention Papers

**"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
- Paper: https://arxiv.org/abs/2205.14135
- **Priority**: Must read
- **Time**: ~2 hours
- **What to focus on**:
  - Section 2: Background on GPU memory hierarchy
  - Section 3.1: The core algorithm (Algorithm 1)
  - Section 3.2: Analysis of I/O complexity
  - Figure 1: Memory access pattern comparison
- **What to skim**: Section 4-5 (experiments, specific benchmarks)
- **Key insight**: Understanding WHY they made design choices is more important than implementation details.

**"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (Dao, 2023)
- Paper: https://arxiv.org/abs/2307.08691
- **Priority**: Highly recommended
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 3.1: Reversed loop order
  - Section 3.2: Work partitioning changes
  - Section 3.3: Causal masking optimization
- **Key insight**: The speedup comes from better GPU utilization, not algorithmic changes.

**"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"** (Shah et al., 2024)
- Paper: https://arxiv.org/abs/2407.08608
- **Priority**: Important (if using H100)
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 3: Hopper-specific optimizations
  - Section 4: FP8 attention
- **Prerequisite**: Familiarity with H100 architecture helps.

---

## GPU Architecture References

### NVIDIA Documentation

**CUDA C++ Programming Guide**
- Link: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Priority**: Reference (don't read cover-to-cover)
- **What to focus on**:
  - Chapter 5: Memory Hierarchy
  - Chapter 5.3: Shared Memory
  - Appendix K: Compute Capabilities

**NVIDIA A100 Whitepaper**
- Link: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- **Priority**: Recommended
- **Time**: ~1 hour for relevant sections
- **What to focus on**:
  - Pages 15-20: Memory hierarchy
  - Pages 25-30: Tensor Cores

**NVIDIA H100 Whitepaper**
- Link: https://resources.nvidia.com/en-us-tensor-core
- **Priority**: Important (for v3)
- **What to focus on**:
  - Tensor Memory Accelerator (TMA)
  - Warp Group operations

---

## Online Softmax References

**"Online Normalizer Calculation for Softmax"** (Milakov & Gimelshein, 2018)
- Paper: https://arxiv.org/abs/1805.02867
- **Priority**: Highly recommended
- **Time**: ~45 minutes
- **Key insight**: The mathematical foundation for Flash Attention's incremental softmax.

**"Self-Attention Does Not Need O(n²) Memory"** (Rabe & Staats, 2021)
- Paper: https://arxiv.org/abs/2112.05682
- **Priority**: Recommended
- **Time**: ~1 hour
- **What to focus on**: The memory-efficient attention algorithm that preceded Flash Attention.

---

## Code References

### Official Implementations

**flash-attention (Dao-AILab)**
- Repository: https://github.com/Dao-AILab/flash-attention
- **Priority**: Must explore
- **What to focus on**:
  - `flash_attn/flash_attn_interface.py`: Python API
  - `csrc/flash_attn/`: CUDA kernels (advanced)
- **Lab connection**: You'll use this library in Lab 04.

**xFormers (Meta)**
- Repository: https://github.com/facebookresearch/xformers
- **Priority**: Recommended
- **What to focus on**:
  - Memory-efficient attention implementation
  - Comparison with Flash Attention

### PyTorch Integration

**torch.nn.functional.scaled_dot_product_attention**
- Docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **Priority**: Must understand
- **What to focus on**:
  - Backend selection (flash, math, memory-efficient)
  - `is_causal` parameter
  - Context manager for backend selection

---

## Gradient Checkpointing References

**"Training Deep Nets with Sublinear Memory Cost"** (Chen et al., 2016)
- Paper: https://arxiv.org/abs/1604.06174
- **Priority**: Highly recommended
- **Time**: ~1 hour
- **What to focus on**:
  - Section 2: The sqrt(n) checkpointing strategy
  - Trade-off analysis

**PyTorch Checkpoint Documentation**
- Link: https://pytorch.org/docs/stable/checkpoint.html
- **Priority**: Must read
- **Time**: 20 minutes
- **What to focus on**: `use_reentrant` parameter and when to use each mode.

---

## Recommended Tutorials

### Blog Posts

**"Making Deep Learning Go Brrrr From First Principles"** (Horace He)
- Link: https://horace.io/brrr_intro.html
- **Priority**: Highly recommended
- **Time**: ~1 hour
- **Why read it**: Excellent introduction to GPU performance optimization from first principles.

**"FlashAttention Explained"** (Aleksa Gordic)
- Link: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
- **Priority**: Recommended
- **Time**: 30 minutes
- **Why read it**: Accessible explanation with good diagrams.

### Video Lectures

**"Flash Attention"** (Tri Dao, Stanford MLSys Seminar)
- Link: Available on YouTube
- **Priority**: Highly recommended
- **Time**: ~1 hour
- **Why watch**: Hear the algorithm explained by its creator.

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Kernel Optimization

**"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** (Tillet et al., 2019)
- Paper: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- **Why read it**: Background on the Triton compiler used in many Flash Attention implementations.

### Memory-Efficient Transformers

**"Efficient Transformers: A Survey"** (Tay et al., 2022)
- Paper: https://arxiv.org/abs/2009.06732
- **Why read it**: Comprehensive overview of efficient attention mechanisms (not just Flash Attention).

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Flash Attention algorithm | Flash Attention v1 paper, Section 3 | 1 hour |
| GPU memory hierarchy | CUDA Programming Guide Ch. 5 | 30 min |
| Online softmax | Milakov & Gimelshein 2018 | 45 min |
| PyTorch integration | SDPA documentation | 20 min |
| v2 improvements | Flash Attention v2 paper | 1 hour |
| Checkpointing | Chen et al. 2016 | 1 hour |
| Practical usage | flash-attention repo examples | 30 min |

---

## What's NOT Covered Here

These topics are covered in other chapters:

- **Triton kernel programming** -> Chapter 12
- **Distributed training** -> Chapter 11
- **Linear attention** -> Chapter 5-6
- **Sparse attention** -> Chapter 7

---

## Tools for the Labs

### Required

```bash
# Flash Attention library
pip install flash-attn --no-build-isolation

# For profiling
pip install torch-tb-profiler

# Memory profiling
pip install pytorch-memlab
```

### Optional (for deeper exploration)

```bash
# NVIDIA Nsight for detailed profiling
# Download from NVIDIA website

# Triton for custom kernels
pip install triton
```

---

## Debugging and Profiling

**NVIDIA Nsight Systems**
- Download: https://developer.nvidia.com/nsight-systems
- **Why use it**: Detailed timeline of GPU operations, memory transfers.

**PyTorch Profiler**
- Docs: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **Why use it**: Easy integration with PyTorch, TensorBoard visualization.

```python
# Example profiling setup
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Community Resources

- **Flash Attention GitHub Issues**: Good for implementation questions
- **PyTorch Forums**: For integration questions
- **r/MachineLearning**: Discussions on new developments
- **Twitter/X**: Follow @tri_dao for updates

Stay focused on understanding the core concepts first. The optimizations and variants build on this foundation.
