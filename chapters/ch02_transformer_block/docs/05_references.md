# Chapter 2: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Residual Networks

**"Deep Residual Learning for Image Recognition"** (He et al., 2015)
- Paper: https://arxiv.org/abs/1512.03385
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1 hour
- **What to focus on**:
  - Section 3.1: The residual learning framework
  - Figure 2: The skip connection diagram
  - Section 4.1: Why deeper networks were harder to train
- **Key insight**: The residual formulation changes the problem from "learn f(x)" to "learn f(x) - x", which is easier when the optimal function is close to identity.

### Layer Normalization

**"Layer Normalization"** (Ba, Kiros, Hinton, 2016)
- Paper: https://arxiv.org/abs/1607.06450
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 45 minutes
- **What to focus on**:
  - Section 2: Batch norm limitations for RNNs
  - Section 3: Layer norm definition and implementation
  - Section 4: Why it works for sequence models
- **What to skim**: Experimental results sections (specific to their tasks)

---

## Core Libraries

### PyTorch `nn.LayerNorm`

**Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - The `normalized_shape` parameter (usually just d_model)
  - The `elementwise_affine` parameter (gamma and beta)
  - The formula: `y = (x - mean) / sqrt(var + eps) * gamma + beta`
- **Lab connection**: In Lab 01, you'll implement this from scratch

### HuggingFace GPT-2 Implementation

**Code**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

- **Priority**: ⭐⭐⭐ Must understand for Lab 04
- **What to focus on**:
  - `GPT2Block` class: The transformer block implementation
  - `GPT2Attention` class: Where normalization is applied
  - `GPT2MLP` class: The feed-forward network
- **Gotcha**: GPT-2 uses a hybrid norm placement. Study the code carefully!

---

## Positional Encodings

### RoPE (Rotary Position Embedding)

**"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021)
- Paper: https://arxiv.org/abs/2104.09864
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 1.5 hours
- **What to focus on**:
  - Section 3.2: The rotation formulation
  - Section 3.4: Why rotation encodes relative position
  - Figure 1: The rotation visualization
- **Key insight**: Rotation in 2D subspaces lets the dot product depend only on relative position.

**"A Length-Extrapolatable Transformer"** (Press et al., 2021)
- Paper: https://arxiv.org/abs/2108.12409
- Blog: https://ofir.io/The-Positional-Encoding-Spectrum/
- **Priority**: ⭐ Optional but enriching
- **Why read it**: Explains ALiBi, another positional encoding approach, and compares many methods.

### Position Interpolation

**"Extending Context Window of Large Language Models via Positional Interpolation"** (Chen et al., 2023)
- Paper: https://arxiv.org/abs/2306.15595
- **Priority**: ⭐ Optional
- **Why read it**: How to extend context length without retraining from scratch.

---

## Activation Functions

### GELU

**"Gaussian Error Linear Units (GELUs)"** (Hendrycks & Gimpel, 2016)
- Paper: https://arxiv.org/abs/1606.08415
- **Priority**: ⭐ Optional
- **Time**: 30 minutes
- **What to focus on**: Section 2 (the motivation) and Figure 1 (the curve)
- **Key insight**: GELU is like dropout that depends on input magnitude.

### SwiGLU / GLU Variants

**"GLU Variants Improve Transformer"** (Shazeer, 2020)
- Paper: https://arxiv.org/abs/2002.05202
- **Priority**: ⭐⭐ Recommended
- **Time**: 30 minutes
- **What to focus on**:
  - Section 2: The GLU formulation
  - Table 1: Comparing different activation functions
  - Section 3: Why gating helps
- **Key insight**: Gating provides adaptive feature selection.

---

## Pre-Norm vs Post-Norm

**"On Layer Normalization in the Transformer Architecture"** (Xiong et al., 2020)
- Paper: https://arxiv.org/abs/2002.04745
- **Priority**: ⭐⭐ Recommended
- **Time**: 45 minutes
- **What to focus on**:
  - Section 2: Analysis of gradient flow
  - Section 3: Why pre-norm is more stable
  - Figure 1: Gradient magnitude comparison
- **Key insight**: Pre-norm keeps gradients bounded, enabling larger learning rates.

---

## RMSNorm

**"Root Mean Square Layer Normalization"** (Zhang & Sennrich, 2019)
- Paper: https://arxiv.org/abs/1910.07467
- **Priority**: ⭐ Optional
- **Time**: 20 minutes
- **What to focus on**: Section 2 (the simplification) and Table 1 (speed comparison)
- **Key insight**: Mean centering may be unnecessary; RMS alone provides sufficient normalization.

---

## Code References

### LLaMA Implementation

**LLaMA Reference Code**
- Code: https://github.com/facebookresearch/llama
- **Priority**: ⭐⭐ Highly recommended
- **What to focus on**:
  - `model.py`: The `TransformerBlock` class
  - `model.py`: `RMSNorm` and `FeedForward` implementations
  - `model.py`: The `precompute_freqs_cis` function for RoPE

**LitGPT (Lightning AI)**
- Code: https://github.com/Lightning-AI/litgpt
- **Priority**: ⭐⭐ Recommended
- **Why use it**: Clean, readable implementations of many model architectures
- **Files to read**: `litgpt/model.py` for the block structure

### Annotated Implementations

**The Annotated Transformer** (Harvard NLP)
- Code: https://nlp.seas.harvard.edu/annotated-transformer/
- **Priority**: ⭐⭐ Highly recommended
- **What to focus on**:
  - `PositionwiseFeedForward` class
  - `SublayerConnection` class (residual + norm)
  - `EncoderLayer` class (full block)

---

## Deep Dives (Optional)

### Understanding Transformers Theoretically

**"A Mathematical Framework for Transformer Circuits"** (Elhage et al., 2021)
- Blog: https://transformer-circuits.pub/2021/framework/index.html
- **Priority**: ⭐ Advanced/Optional
- **Why read it**: Deep analysis of how information flows through transformer blocks
- **Time**: 2-3 hours
- **What to focus on**: The "Residual Stream" concept

### Initialization and Training Dynamics

**"Improving Transformer Models by Reordering their Sublayers"** (Press et al., 2020)
- Paper: https://arxiv.org/abs/1911.03864
- **Priority**: ⭐ Optional
- **Why read it**: Explores alternative orderings of attention and FFN

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Residual connections | ResNet paper §3.1 | 30 min |
| Layer normalization | Ba et al. paper §3 | 45 min |
| Pre-norm vs post-norm | Xiong et al. paper | 45 min |
| RoPE positions | Su et al. paper §3 | 1 hour |
| GELU/SwiGLU | Shazeer GLU paper | 30 min |
| Full implementation | LLaMA code + Annotated Transformer | 2 hours |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **Causal masking** → Chapter 3
- **Training dynamics** → Chapter 4
- **Flash Attention / memory efficiency** → Chapter 10
- **Distributed training** → Chapter 11

Master the transformer block first - it's the foundation for everything that follows.
