# Chapter 6: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Linear Attention Foundations

**"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"** (Katharopoulos et al., 2020)
- Paper: https://arxiv.org/abs/2006.16236
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 3: The kernel trick and associativity
  - Section 4: Causal masking in linear attention
  - The recurrent formulation (Equation 8)
- **Key insight**: Linear attention can be viewed as an RNN with a matrix-valued state.

### Flash Linear Attention

**"Gated Linear Attention Transformers with Hardware-Efficient Training"** (Yang et al., 2023)
- Paper: https://arxiv.org/abs/2312.06635
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 2-3 hours
- **What to focus on**:
  - Section 3: The GLA architecture
  - Section 4: Chunkwise parallel algorithm
  - Appendix A: Implementation details
- **Why it matters**: This paper introduces both GLA and the Flash Linear Attention training algorithm.

---

## State-Space Models

### S4 and Structured SSMs

**"Efficiently Modeling Long Sequences with Structured State Spaces"** (Gu et al., 2022)
- Paper: https://arxiv.org/abs/2111.00396
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 3-4 hours (dense paper)
- **What to focus on**:
  - Section 2: State space model basics
  - Section 3.2: The HiPPO matrix
  - Section 4: Efficient computation via convolutions
- **What to skim**: The derivations in Section 3.1 (unless you want the full math)

### Mamba

**"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** (Gu & Dao, 2023)
- Paper: https://arxiv.org/abs/2312.00752
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 2-3 hours
- **What to focus on**:
  - Section 3.1: Selection mechanism (data-dependent parameters)
  - Section 3.3: Hardware-aware algorithm
  - Figure 2: The selection intuition
- **Key insight**: Making SSM parameters input-dependent dramatically improves expressiveness.

**"Transformers are SSMs"** (Dao & Gu, 2024)
- Paper: https://arxiv.org/abs/2405.21060
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 2 hours
- **What to focus on**:
  - Section 2: The duality between attention and SSMs
  - Section 3: Structured State Space Duality (SSD)
  - Algorithm 1: The unified algorithm
- **Why it matters**: Unifies linear attention and Mamba under a common framework.

---

## Gated Linear Attention Variants

### DeltaNet

**"DeltaNet: Conditional State Space Models for Scalable Sequence Modeling"** (Yang et al., 2024)
- Paper: https://arxiv.org/abs/2310.18780
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 3: The delta rule formulation
  - Section 4: Efficient implementation
- **Key insight**: The delta rule provides implicit forgetting through error correction.

### RetNet

**"Retentive Network: A Successor to Transformer for Large Language Models"** (Sun et al., 2023)
- Paper: https://arxiv.org/abs/2307.08621
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2: The retention mechanism
  - Section 2.3: Multi-scale retention
  - The three computation paradigms (parallel, recurrent, chunked)

### RWKV

**"RWKV: Reinventing RNNs for the Transformer Era"** (Peng et al., 2023)
- Paper: https://arxiv.org/abs/2305.13048
- **Priority**: ⭐ Optional
- **Time**: 1-2 hours
- **Why read it**: Alternative approach to efficient sequence modeling with strong practical results.

---

## Code Libraries

### Flash Linear Attention Library (fla)

**Repository**: https://github.com/sustcsonglin/flash-linear-attention
- **Priority**: ⭐⭐⭐ Essential for labs
- **What it provides**:
  - Optimized Triton kernels for GLA, DeltaNet, RetNet
  - Reference implementations
  - Benchmarking utilities
- **Key files**:
  - `fla/ops/gla/`: GLA implementation
  - `fla/ops/linear_attn/`: Basic linear attention
  - `fla/ops/delta_rule/`: DeltaNet implementation

### Mamba Repository

**Repository**: https://github.com/state-spaces/mamba
- **Priority**: ⭐⭐ Highly recommended
- **What it provides**:
  - Official Mamba implementation
  - CUDA kernels for selective scan
  - Model checkpoints

### S4 Repository

**Repository**: https://github.com/state-spaces/s4
- **Priority**: ⭐ Optional
- **Why use it**: Reference implementation of S4 and variants

---

## Blog Posts and Tutorials

### Mamba Explained

**"Mamba: The Hard Way"** (Aleksa Gordić)
- Video: https://www.youtube.com/watch?v=9dSkvxS2EB0
- **Priority**: ⭐⭐ Highly recommended
- **Time**: ~2 hours
- **Why watch**: Step-by-step derivation with code

**"A Visual Guide to Mamba and State Space Models"** (Maarten Grootendorst)
- Blog: https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 30-45 minutes
- **Why read**: Excellent visualizations

### Linear Attention Deep Dives

**"Linear Attention Survey"** (Karan Praharaj)
- Blog: https://karan-praharaj.github.io/posts/linear-attention/
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 30 minutes
- **Why read**: Comprehensive overview of linear attention variants

---

## Academic Deep Dives (Optional)

For those wanting theoretical depth:

### Feature Maps and Kernels

**"Random Feature Attention"** (Peng et al., 2021)
- Paper: https://arxiv.org/abs/2103.02143
- **Why read**: Explains the kernel interpretation of attention

**"Rethinking Attention with Performers"** (Choromanski et al., 2020)
- Paper: https://arxiv.org/abs/2009.14794
- **Why read**: FAVOR+ mechanism for approximating attention

### State Space Theory

**"HiPPO: Recurrent Memory with Optimal Polynomial Projections"** (Gu et al., 2020)
- Paper: https://arxiv.org/abs/2008.07669
- **Why read**: The mathematical foundation of S4's initialization

**"How to Train Your HiPPO"** (Gu et al., 2022)
- Paper: https://arxiv.org/abs/2206.12037
- **Why read**: Improvements to the original HiPPO framework

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Linear attention basics | Katharopoulos et al. 2020 | 1-2 hours |
| RNN interpretation | GLA paper Section 2 | 30 min |
| Chunkwise parallel | GLA paper Section 4 | 1 hour |
| GLA architecture | GLA paper Section 3 | 1 hour |
| State-space basics | S4 paper Section 2 | 1 hour |
| Selective SSMs | Mamba paper Section 3 | 1 hour |
| SSM-Attention duality | Mamba-2 paper | 2 hours |
| Code implementation | fla library | 2-3 hours |

---

## Lab Prerequisites

Before starting the labs, ensure you've:

1. **Read**: GLA paper Sections 1-4
2. **Understood**: The RNN view of linear attention
3. **Installed**: `fla` library (`pip install fla`)
4. **Reviewed**: Chapter 5 (Linear Attention basics)

---

## What's NOT Covered Here

These topics are covered in other chapters:

- **Basic linear attention** → Chapter 5
- **Flash Attention (softmax)** → Chapter 10
- **KV-cache optimization** → Chapter 8
- **Distributed training** → Chapter 11

Master the fundamentals from this chapter before moving to hardware optimization in later chapters.

---

## Staying Current

This is a fast-moving field. Key venues to follow:

- **arXiv cs.LG**: New papers daily
- **Twitter/X**: @_albertgu (Mamba author), @taboryang (GLA author)
- **GitHub**: Watch the `fla` and `mamba` repositories
- **NeurIPS, ICML, ICLR**: Major ML conferences

The fla library is actively maintained and often includes implementations of new architectures within weeks of publication.
