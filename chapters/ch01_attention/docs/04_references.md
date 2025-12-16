# Chapter 1: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### The Original Paper

**"Attention Is All You Need"** (Vaswani et al., 2017)
- Paper: https://arxiv.org/abs/1706.03762
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~2 hours
- **What to focus on**:
  - Section 3.2: Scaled Dot-Product Attention (the formula we implement)
  - Section 3.2.2: Multi-Head Attention (why multiple heads)
  - Figure 2: The architecture diagram
- **What to skim**: Section 5-6 (training details, specific to their experiments)
- **Key insight**: The entire paper is about replacing recurrence with attention. Understanding *why* they made this choice is more important than the specific hyperparameters.

---

## Core Libraries

### PyTorch `nn.MultiheadAttention`

**Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - The `forward()` signature: `query, key, value, key_padding_mask, attn_mask`
  - Return values: `(attn_output, attn_output_weights)`
  - The `batch_first` parameter (default False, but we usually want True)
- **Lab connection**: In Lab 04, you'll verify your implementation matches this

### PyTorch `F.scaled_dot_product_attention`

**Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

- **Priority**: ⭐⭐ Important
- **Why it matters**: This is the modern, optimized primitive. It can automatically use Flash Attention when available.
- **What to focus on**:
  - The function signature and input shapes
  - The `is_causal` parameter for decoder attention
  - The automatic backend selection (math, flash, memory-efficient)

---

## Recommended Reading

### Visualizing Attention

**"A Visual Guide to Attention"** (Jay Alammar)
- Blog: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 30-45 minutes
- **Why read it**: Excellent visualizations of attention mechanics. Good for building intuition before diving into code.

**"The Illustrated Transformer"** (Jay Alammar)
- Blog: https://jalammar.github.io/illustrated-transformer/
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1 hour
- **What to focus on**: The step-by-step walkthrough of multi-head attention with actual numbers

### Historical Context

**"Neural Machine Translation by Jointly Learning to Align and Translate"** (Bahdanau et al., 2014)
- Paper: https://arxiv.org/abs/1409.0473
- **Priority**: ⭐ Optional but enriching
- **Time**: 1-2 hours
- **Why read it**: This is the paper that introduced attention for seq2seq. Reading it helps you understand what problem transformers were solving.
- **What to focus on**: Section 3.1 (the attention mechanism), Figure 3 (attention visualizations)

---

## Code References

### Reference Implementations

**Annotated Transformer** (Harvard NLP)
- Code: https://nlp.seas.harvard.edu/annotated-transformer/
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Line-by-line implementation of the original paper with explanations
- **What to focus on**: The `attention()` and `MultiHeadedAttention` functions

**minGPT** (Karpathy)
- Code: https://github.com/karpathy/minGPT
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Clean, minimal implementation of GPT. Good for understanding how attention fits into a full model.
- **Files to read**: `mingpt/model.py` - the `CausalSelfAttention` class

**nanoGPT** (Karpathy)
- Code: https://github.com/karpathy/nanoGPT
- **Priority**: ⭐ Optional
- **Why use it**: Even more minimal than minGPT, good for training small models

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Attention Patterns Analysis

**"What Do Attention Heads Do?"** (Clark et al., 2019)
- Paper: https://arxiv.org/abs/1906.04341
- **Why read it**: Analyzes what different attention heads learn (syntactic, positional, etc.)
- **Key sections**: Section 3 (probing individual heads)

**"A Multiscale Visualization of Attention"** (Vig, 2019)
- Paper: https://arxiv.org/abs/1906.05714
- Tool: https://github.com/jessevig/bertviz
- **Why read it**: Tools for visualizing attention across layers and heads

### Softmax & Numerical Stability

**"The Softmax Function and Its Derivative"**
- Blog: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
- **Why read it**: If you want to understand the math behind softmax gradients

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Attention formula | "Attention Is All You Need" §3.2 | 30 min |
| Multi-head intuition | Illustrated Transformer | 1 hour |
| PyTorch API | `nn.MultiheadAttention` docs | 15 min |
| Clean implementation | Annotated Transformer | 1 hour |
| Attention patterns | BertViz tool | 30 min |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **Positional encodings** → Chapter 2
- **Causal masking** → Chapter 3
- **Flash Attention** → Chapter 10
- **Linear Attention** → Chapter 5

Stay focused on understanding the core attention mechanism first. The variants build on this foundation.
