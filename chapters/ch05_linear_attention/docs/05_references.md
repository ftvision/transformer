# Chapter 5: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### The Linear Attention Paper

**"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"** (Katharopoulos et al., 2020)
- Paper: https://arxiv.org/abs/2006.16236
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 3: The kernel formulation of attention
  - Section 3.2: Causal masking via cumulative sums
  - Figure 1: The computational complexity comparison
- **What to skim**: Sections 5-6 (experiments on specific tasks)
- **Key insight**: Linear attention can be viewed as an RNN with a specific state update rule, enabling O(1) inference per token.

### The Performers Paper

**"Rethinking Attention with Performers"** (Choromanski et al., 2020)
- Paper: https://arxiv.org/abs/2009.14794
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~2 hours
- **What to focus on**:
  - Section 2: FAVOR+ mechanism for unbiased softmax approximation
  - Section 3: Positive random features
  - Algorithm 1: The Performer attention algorithm
- **Key insight**: Random Fourier features can approximate softmax attention with theoretical guarantees.

---

## Core Concepts

### Random Fourier Features

**"Random Features for Large-Scale Kernel Machines"** (Rahimi & Recht, 2007)
- Paper: https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html
- **Priority**: ⭐⭐ Important for theory
- **Time**: ~1 hour
- **What to focus on**:
  - Theorem 1: The approximation guarantee
  - Section 2: How random features approximate kernels
- **Why read it**: This is the theoretical foundation for Performers-style feature maps.

### Kernel Methods Background

**"A Primer on Kernel Methods"** (Scholkopf & Smola)
- Resource: Chapter 2 of "Learning with Kernels" or online tutorials
- **Priority**: ⭐ Optional background
- **Time**: 1-2 hours
- **Why read it**: If the kernel terminology is unfamiliar, this provides mathematical grounding.

---

## Modern Extensions

### Gated Linear Attention

**"Gated Linear Attention Transformers with Hardware-Efficient Training"** (Yang et al., 2023)
- Paper: https://arxiv.org/abs/2312.06635
- **Priority**: ⭐⭐⭐ Highly recommended
- **Time**: ~1.5 hours
- **What to focus on**:
  - The gating mechanism for data-dependent forgetting
  - Hardware-efficient training via chunkwise computation
- **Connection**: This is the foundation for Chapter 6's labs.

### DeltaNet

**"DeltaNet: Conditional State Space Models with Selective State Spaces"** (Yang et al., 2024)
- Paper: https://arxiv.org/abs/2310.18780
- **Priority**: ⭐⭐ Important
- **Time**: ~1.5 hours
- **What to focus on**:
  - The delta rule update mechanism
  - Connection to fast weight programmers
- **Key insight**: Adding decay/gating to linear attention significantly improves quality.

### Flash Linear Attention

**"Flash Linear Attention"** (Yang et al., 2024)
- Paper: https://arxiv.org/abs/2401.17555
- **Priority**: ⭐⭐⭐ Highly recommended
- **Time**: ~1 hour
- **What to focus on**:
  - Chunkwise parallel algorithm
  - Memory-efficient training via tiling
- **Connection**: Builds directly on this chapter's concepts for efficient GPU implementation.

---

## Code References

### Reference Implementations

**`fla` (Flash Linear Attention) Library**
- Code: https://github.com/sustcsonglin/flash-linear-attention
- **Priority**: ⭐⭐⭐ Must explore
- **Why use it**: Production-quality implementations of linear attention variants
- **Files to read**:
  - `fla/ops/linear_attn/` - Core linear attention operations
  - `fla/layers/` - Layer implementations

**Linear Transformer (Official)**
- Code: https://github.com/idiap/fast-transformers
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Original implementation from the "Transformers are RNNs" paper
- **Files to read**: `fast_transformers/attention/linear_attention.py`

**Performers (Google)**
- Code: https://github.com/google-research/google-research/tree/master/performer
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Official Google implementation with FAVOR+ features
- **Files to read**: `fast_attention.py`

---

## Benchmarks and Analysis

### Complexity Comparisons

**"Long Range Arena: A Benchmark for Efficient Transformers"** (Tay et al., 2020)
- Paper: https://arxiv.org/abs/2011.04006
- **Priority**: ⭐⭐ Important
- **Time**: ~1 hour (skim)
- **What to focus on**:
  - Table 1: Tasks requiring long-range dependencies
  - Table 3: Comparison of efficient attention methods
- **Key insight**: Different efficient attention methods have different strengths across tasks.

**"Scaling Laws for Neural Language Models"** (Kaplan et al., 2020)
- Paper: https://arxiv.org/abs/2001.08361
- **Priority**: ⭐ Optional
- **Why read it**: Understanding scaling helps contextualize when linear attention matters.

---

## Historical Context

### Fast Weight Programmers

**"Learning to Control Fast-Weight Memories"** (Schmidhuber, 1992)
- Paper: Neural Computation, 1992
- **Priority**: ⭐ Optional but fascinating
- **Why read it**: Linear attention's recurrent view was actually discovered 30+ years ago!
- **Key insight**: The idea of using one network to quickly program another's weights.

### Attention Complexity Analysis

**"Efficient Transformers: A Survey"** (Tay et al., 2020)
- Paper: https://arxiv.org/abs/2009.06732
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours (survey)
- **What to focus on**:
  - Section 3: Taxonomy of efficient attention
  - Table 1: Complexity comparison of methods
- **Key insight**: Comprehensive overview of approaches to the O(n²) problem.

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Linear attention formula | Katharopoulos et al. §3 | 45 min |
| Random features | Performers paper §2 | 45 min |
| Causal formulation | Katharopoulos et al. §3.2 | 30 min |
| Gating mechanisms | GLA paper §3 | 45 min |
| Implementation | `fla` library code | 1 hour |
| Efficient methods survey | Tay et al. survey | 1 hour |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **Flash Linear Attention details** → Chapter 6
- **Gated Linear Attention (GLA)** → Chapter 6
- **State-space connections (Mamba)** → Chapter 6
- **Sparse attention patterns** → Chapter 7
- **DeepSeek MLA** → Chapter 7
- **Flash Attention (standard)** → Chapter 10

Master linear attention fundamentals here before moving to the variants.

---

## Lab Connections

| Lab | Relevant Resources |
|-----|-------------------|
| Lab 01: Complexity Analysis | Long Range Arena, Scaling Laws |
| Lab 02: Kernel Trick | Katharopoulos §3, Rahimi & Recht |
| Lab 03: Feature Maps | Performers §3, fast-transformers code |
| Lab 04: Causal Linear | Katharopoulos §3.2, `fla` library |
