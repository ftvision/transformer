# Chapter 7: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Sparse Attention Patterns

**"Generating Long Sequences with Sparse Transformers"** (Child et al., 2019)
- Paper: https://arxiv.org/abs/1904.10509
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2: The factorized attention patterns (strided, fixed)
  - Figure 2: Visualization of sparse patterns
  - Section 4: How O(n√n) complexity is achieved
- **Key insight**: You don't need full attention—carefully designed sparse patterns preserve quality while dramatically reducing compute.

**"Longformer: The Long-Document Transformer"** (Beltagy et al., 2020)
- Paper: https://arxiv.org/abs/2004.05150
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 3.1: Sliding window attention
  - Section 3.2: Global attention mechanism
  - Figure 1: The attention pattern visualization
- **Lab connection**: Lab 02 implements Longformer-style attention

---

### DeepSeek MLA & KV Compression

**"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"**
- Paper: https://arxiv.org/abs/2405.04434
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 2 hours
- **What to focus on**:
  - Section 2.1: Multi-head Latent Attention (MLA)
  - Section 2.1.1: Low-rank KV compression
  - Table 2: Memory and throughput comparisons
- **Key insight**: Compressing KV to a latent space dramatically reduces memory while maintaining quality.

**"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (Ainslie et al., 2023)
- Paper: https://arxiv.org/abs/2305.13245
- **Priority**: ⭐⭐ Important
- **Time**: 45 minutes
- **What to focus on**:
  - Section 2: GQA vs MHA vs MQA comparison
  - Figure 1: The grouping concept
  - Section 4: Uptraining from MHA to GQA
- **Lab connection**: Understanding GQA is prerequisite for Lab 03

---

### Mixture of Experts

**"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"** (Shazeer et al., 2017)
- Paper: https://arxiv.org/abs/1701.06538
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2: The MoE layer architecture
  - Section 2.1: The gating network (router)
  - Section 4: Load balancing losses
- **Historical note**: This is the foundational MoE paper that modern architectures build upon.

**"Switch Transformers: Scaling to Trillion Parameter Models"** (Fedus et al., 2022)
- Paper: https://arxiv.org/abs/2101.03961
- **Priority**: ⭐⭐ Important
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2.1: Simplified top-1 routing
  - Section 2.2: Expert capacity and load balancing
  - Table 1: Scaling results
- **Key insight**: Simpler routing (top-1) can work well with proper load balancing.

**"Mixtral of Experts"** (Jiang et al., 2024)
- Paper: https://arxiv.org/abs/2401.04088
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1 hour
- **What to focus on**:
  - Section 2: Architecture details (8 experts, top-2)
  - Section 3.1: Expert analysis (what different experts learn)
- **Lab connection**: Lab 05 implements a simplified Mixtral-style MoE

---

## Recommended Reading

### Sliding Window Implementations

**"Mistral 7B"** (Jiang et al., 2023)
- Paper: https://arxiv.org/abs/2310.06825
- **Priority**: ⭐⭐ Important
- **Time**: 30 minutes
- **What to focus on**:
  - Section 2.1: Sliding window attention
  - Section 2.2: Rolling KV-cache
- **Why read it**: Clean, practical implementation of sliding window in production.

**"BigBird: Transformers for Longer Sequences"** (Zaheer et al., 2020)
- Paper: https://arxiv.org/abs/2007.14062
- **Priority**: ⭐ Optional but enriching
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2: The three attention patterns (local + global + random)
  - Theorem 1: Theoretical expressiveness guarantees
- **Key insight**: Random attention + local + global is theoretically as powerful as full attention.

---

### KV-Cache Optimization

**"Efficient Streaming Language Models with Attention Sinks"** (Xiao et al., 2023)
- Paper: https://arxiv.org/abs/2309.17453
- **Priority**: ⭐⭐ Important
- **Time**: 45 minutes
- **What to focus on**:
  - Section 3: The attention sink phenomenon
  - Section 4: StreamingLLM design
- **Key insight**: First few tokens receive disproportionate attention and should be kept in cache.

**"Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs"** (Ge et al., 2023)
- Paper: https://arxiv.org/abs/2310.01801
- **Priority**: ⭐ Optional
- **Time**: 1 hour
- **What to focus on**:
  - Section 3: Importance-based token selection
  - Algorithm 1: Adaptive compression

---

### Advanced MoE

**"MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"** (Gale et al., 2023)
- Paper: https://arxiv.org/abs/2211.15841
- Code: https://github.com/databricks/megablocks
- **Priority**: ⭐⭐ Important for implementation
- **What to focus on**:
  - Section 3: Efficient GPU implementation of MoE
  - Section 4: Block-sparse operations
- **Lab connection**: Understanding these optimizations helps for Lab 05

**"DeepSeekMoE: Towards Ultimate Expert Specialization"** (Dai et al., 2024)
- Paper: https://arxiv.org/abs/2401.06066
- **Priority**: ⭐ Optional
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2.2: Fine-grained expert segmentation
  - Section 2.3: Shared experts

---

## Code References

### Reference Implementations

**Mistral AI Official Code**
- Code: https://github.com/mistralai/mistral-src
- **Priority**: ⭐⭐ Highly recommended
- **Files to read**: `model.py` - sliding window attention implementation

**HuggingFace Transformers - Mixtral**
- Code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
- **Priority**: ⭐⭐ Highly recommended
- **What to focus on**: `MixtralSparseMoeBlock` class, the router implementation

**Megablocks**
- Code: https://github.com/databricks/megablocks
- **Priority**: ⭐ Optional
- **Why use it**: Production-quality MoE kernels for efficient training

**vLLM MLA Implementation**
- Code: https://github.com/vllm-project/vllm
- **Priority**: ⭐ Optional
- **Why use it**: See how MLA is implemented for efficient inference

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Theoretical Foundations

**"Are Transformers universal approximators of sequence-to-sequence functions?"** (Yun et al., 2019)
- Paper: https://arxiv.org/abs/1912.10077
- **Why read it**: Theoretical analysis of transformer expressiveness, relevant to understanding what sparse patterns can/cannot approximate.

**"On the Expressive Power of Self-Attention"** (Pérez et al., 2019)
- Paper: https://arxiv.org/abs/1906.06755
- **Why read it**: Formal analysis of self-attention computational properties.

### Expert Analysis

**"Towards Understanding Mixture of Experts in Deep Learning"** (Chen et al., 2022)
- Paper: https://arxiv.org/abs/2208.02813
- **Why read it**: Theoretical analysis of when and why MoE helps.

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Sparse attention patterns | Sparse Transformers §2 | 30 min |
| Sliding window | Longformer §3.1 | 30 min |
| DeepSeek MLA | DeepSeek-V2 §2.1 | 45 min |
| GQA | GQA paper §2 | 30 min |
| MoE basics | Shazeer et al. §2 | 45 min |
| Load balancing | Switch Transformers §2.2 | 30 min |
| Mixtral architecture | Mixtral paper §2 | 30 min |
| KV-cache optimization | Attention Sinks paper | 30 min |

---

## What's NOT Covered Here

These topics are covered in other chapters:

- **Standard attention** → Chapter 1
- **Flash Attention** (memory-efficient dense) → Chapter 10
- **Linear Attention** → Chapters 5-6
- **Quantization details** → Chapter 8
- **Distributed MoE training** → Chapter 11

Focus on sparse patterns, compression, and MoE here. The other efficiency techniques have dedicated chapters.

---

## Lab Connections

| Lab | Key References |
|-----|----------------|
| Lab 01: Sparse Patterns | Sparse Transformers, Longformer |
| Lab 02: Sliding Window | Mistral, Longformer |
| Lab 03: KV Compression | DeepSeek-V2, GQA paper |
| Lab 04: DeepSeek MLA | DeepSeek-V2 |
| Lab 05: Basic MoE | Shazeer et al., Switch, Mixtral |
