# Chapter 4: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Optimization Fundamentals

**"Decoupled Weight Decay Regularization"** (Loshchilov & Hutter, 2017)
- Paper: https://arxiv.org/abs/1711.05101
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~1 hour
- **What to focus on**:
  - Section 2: Why L2 regularization ≠ weight decay for adaptive optimizers
  - Algorithm 2: The AdamW algorithm
  - Section 4: Experimental results showing AdamW is better
- **Key insight**: This paper fixed a widespread bug in how Adam was being used with regularization.

### Learning Rate Schedules

**"On the Variance of the Adaptive Learning Rate and Beyond"** (Liu et al., 2019)
- Paper: https://arxiv.org/abs/1908.03265
- **Priority**: ⭐⭐ Important
- **Time**: ~1 hour
- **What to focus on**:
  - Section 3: Analysis of why warmup is needed (variance reduction)
  - The RAdam optimizer (Adam with rectified variance)
- **Key insight**: Warmup compensates for high variance in Adam's early estimates.

---

## Core Libraries

### PyTorch Optimizers

**AdamW Documentation**: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - The `weight_decay` parameter and how it differs from Adam
  - Parameter groups for different decay rates

**LR Schedulers**: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - `CosineAnnealingLR` for cosine decay
  - `LinearLR` for warmup
  - `SequentialLR` for combining schedulers

### HuggingFace Transformers

**Training and Optimization**: https://huggingface.co/docs/transformers/training
- **Priority**: ⭐⭐ Important
- **What to focus on**:
  - The `Trainer` class and its configuration
  - Built-in learning rate schedules

**get_scheduler API**: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
- **Priority**: ⭐⭐ Important
- **Why use it**: Convenient API for standard schedules

---

## Recommended Reading

### Understanding Training Dynamics

**"An Empirical Model of Large-Batch Training"** (McCandlish et al., 2018)
- Paper: https://arxiv.org/abs/1812.06162
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - The "critical batch size" concept
  - How learning rate should scale with batch size
- **Lab connection**: Understanding when to use larger batches

**"Scaling Laws for Neural Language Models"** (Kaplan et al., 2020)
- Paper: https://arxiv.org/abs/2001.08361
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 4: How loss scales with compute, data, and parameters
  - Guidance on optimal model size given a compute budget
- **Key insight**: Bigger models are more sample-efficient.

### Gradient Flow

**"Deep Residual Learning for Image Recognition"** (He et al., 2015)
- Paper: https://arxiv.org/abs/1512.03385
- **Priority**: ⭐⭐ Important for background
- **Time**: 1 hour
- **What to focus on**:
  - Section 3: The residual learning framework
  - Why identity shortcuts help gradient flow
- **Key insight**: The foundational paper for residual connections.

---

## Code References

### Reference Implementations

**nanoGPT Training Loop** (Karpathy)
- Code: https://github.com/karpathy/nanoGPT/blob/master/train.py
- **Priority**: ⭐⭐⭐ Must study
- **What to focus on**:
  - Learning rate schedule implementation
  - Gradient clipping and accumulation
  - The `configure_optimizers()` function
- **Lab connection**: Your Lab 03 training loop should mirror this structure

**minGPT Training** (Karpathy)
- Code: https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py
- **Priority**: ⭐⭐ Recommended
- **What to focus on**:
  - The Trainer class design
  - Callback structure

### Production Training Code

**GPT-NeoX Training** (EleutherAI)
- Code: https://github.com/EleutherAI/gpt-neox
- **Priority**: ⭐ Advanced/Optional
- **What to focus on**:
  - How large-scale training is configured
  - The optimizer configuration files

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Optimizer Theory

**"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, 2014)
- Paper: https://arxiv.org/abs/1412.6980
- **Why read it**: The original Adam paper, explains the theory behind momentum and variance estimates
- **Key sections**: Section 2 (the algorithm), Section 4 (convergence analysis)

**"On the Convergence of Adam and Beyond"** (Reddi et al., 2018)
- Paper: https://arxiv.org/abs/1904.09237
- **Why read it**: Analyzes when Adam can fail and proposes AMSGrad
- **Key insight**: Adam's non-convergence in some cases, though rarely matters in practice

### Loss Landscape

**"Visualizing the Loss Landscape of Neural Nets"** (Li et al., 2018)
- Paper: https://arxiv.org/abs/1712.09913
- **Why read it**: Beautiful visualizations of what we're optimizing
- **Key figures**: Figures 1, 6 (skip connections dramatically smooth the landscape)

**"The Loss Surfaces of Multilayer Networks"** (Choromanska et al., 2015)
- Paper: https://arxiv.org/abs/1412.0233
- **Why read it**: Theoretical analysis of loss landscape structure
- **Key insight**: Local minima in deep networks are often good enough

### Perplexity and Evaluation

**"Perplexity of Language Models"** - Stanford CS224N Notes
- Link: https://web.stanford.edu/class/cs224n/
- **Why read it**: Clear explanation of perplexity as a metric

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| AdamW | "Decoupled Weight Decay" paper §2 | 30 min |
| Why warmup | RAdam paper §3 | 30 min |
| PyTorch optimizers | PyTorch docs | 20 min |
| LR schedules | HuggingFace docs | 15 min |
| Training loop | nanoGPT train.py | 1 hour |
| Batch size scaling | McCandlish et al. | 1 hour |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **Mixed precision training** → Chapter 11
- **Distributed training** → Chapter 11
- **Gradient checkpointing** → Chapter 10
- **Quantization-aware training** → Chapter 8
- **Fine-tuning strategies** → Later chapters

Focus on understanding the fundamentals first. The advanced training techniques build on this foundation.

---

## Datasets for Practice

For Lab 04 (training a small model), consider:

**tiny_shakespeare** (Karpathy)
- Link: https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
- Size: ~1MB
- Good for: Quick experiments, character-level modeling

**WikiText-2** (Merity et al.)
- Link: https://huggingface.co/datasets/wikitext
- Size: ~10MB
- Good for: Small-scale word/subword modeling

**WikiText-103**
- Link: https://huggingface.co/datasets/wikitext
- Size: ~500MB
- Good for: More serious training runs

**OpenWebText** (for GPT-2 style)
- Link: https://huggingface.co/datasets/openwebtext
- Size: ~40GB
- Good for: Reproducing GPT-2 scale experiments
