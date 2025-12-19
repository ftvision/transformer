# Chapter 3: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### The Original Paper (Revisited)

**"Attention Is All You Need"** (Vaswani et al., 2017)
- Paper: https://arxiv.org/abs/1706.03762
- **Priority**: ⭐⭐⭐ Must read (if not read in Chapter 1)
- **Time**: ~2 hours
- **What to focus on for this chapter**:
  - Section 3.1: Encoder and Decoder Stacks
  - Section 3.2.3: Applications of Attention (self vs cross-attention)
  - Section 3.3: Position-wise Feed-Forward Networks
  - Section 3.4: Embeddings and Softmax (tied weights)
  - Figure 1: The full architecture diagram (now you can understand all parts!)

### GPT-2 Paper

**"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
- Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 2.1: Input Representation (BPE tokenization)
  - Section 2.2: Model (decoder-only architecture details)
  - Section 2.3: Training specifics
- **Key insight**: This paper explains why decoder-only models work well for many tasks.

---

## Core Libraries

### HuggingFace Transformers

**Documentation**: https://huggingface.co/docs/transformers
- **Priority**: ⭐⭐⭐ Must understand
- **Key pages**:
  - [Quick tour](https://huggingface.co/docs/transformers/quicktour): Get started fast
  - [Model loading](https://huggingface.co/docs/transformers/main_classes/model): `from_pretrained()` details
  - [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2): Specific to Lab 04

### HuggingFace Tokenizers

**Documentation**: https://huggingface.co/docs/tokenizers
- **Priority**: ⭐⭐ Important
- **What to focus on**:
  - BPE tokenization explanation
  - `encode()` vs `tokenize()` vs `encode_plus()`
  - Special tokens handling

### HuggingFace Hub

**Documentation**: https://huggingface.co/docs/hub
- **Priority**: ⭐⭐ Important
- **Why it matters**: This is where you'll find and download models
- **Key features**: Model cards, browsing models, downloading weights

---

## Recommended Reading

### Tokenization Deep Dive

**"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich et al., 2016)
- Paper: https://arxiv.org/abs/1508.07909
- **Priority**: ⭐⭐ Highly recommended
- **Time**: ~1 hour
- **What to focus on**: Section 3 (BPE algorithm explained)
- **Key insight**: This paper introduced BPE to NLP. Understanding it helps you debug tokenization issues.

**Google's SentencePiece**
- Code: https://github.com/google/sentencepiece
- Paper: https://arxiv.org/abs/1808.06226
- **Priority**: ⭐ Optional
- **Why read it**: Used by LLaMA, T5, and many modern models

### Architecture Evolution

**"Improving Language Understanding by Generative Pre-Training"** (GPT-1, Radford et al., 2018)
- Paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- **Priority**: ⭐⭐ Recommended
- **Time**: ~1 hour
- **Why read it**: Shows the transition from encoder-only (BERT era) to decoder-only dominance

**"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2019)
- Paper: https://arxiv.org/abs/1810.04805
- **Priority**: ⭐⭐ Recommended
- **Time**: ~1.5 hours
- **What to focus on**:
  - Section 3.1: Input representation ([CLS], [SEP] tokens)
  - Section 3.3: Pre-training tasks (MLM, NSP)
- **Key insight**: Understanding BERT helps you appreciate why decoder-only became preferred

---

## Code References

### Reference Implementations

**nanoGPT** (Karpathy)
- Code: https://github.com/karpathy/nanoGPT
- **Priority**: ⭐⭐⭐ Must study
- **Why use it**: Clean, minimal GPT implementation with training code
- **Key files**:
  - `model.py`: The entire GPT architecture in ~300 lines
  - Look at `CausalSelfAttention` for causal masking implementation
  - Look at weight loading code for Lab 04

**minGPT** (Karpathy)
- Code: https://github.com/karpathy/minGPT
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Slightly more documented than nanoGPT
- **Key file**: `mingpt/model.py`

**Annotated Transformer** (Harvard NLP)
- Code: https://nlp.seas.harvard.edu/annotated-transformer/
- **Priority**: ⭐⭐ Highly recommended
- **What to focus on**: Encoder-decoder implementation with cross-attention

### HuggingFace Model Implementations

**GPT-2 Source Code**
- Code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
- **Priority**: ⭐⭐ Important for Lab 04
- **Why read it**: This is what you're matching against in Lab 04
- **What to focus on**:
  - `GPT2Attention` class: How attention is implemented
  - `GPT2Block` class: A single transformer block
  - `GPT2LMHeadModel` class: The full model with LM head

---

## Deep Dives (Optional)

### Causal Masking and Efficiency

**"Generating Long Sequences with Sparse Transformers"** (Child et al., 2019)
- Paper: https://arxiv.org/abs/1904.10509
- **Why read it**: Introduces efficient attention patterns beyond dense causal attention
- **Key insight**: Foundation for later work on efficient attention

### Embedding Analysis

**"Word2Vec Explained"** (Blog post)
- Blog: Various good explanations online
- **Why read it**: Embeddings in transformers work similarly to Word2Vec
- **Key concepts**: Skip-gram, CBOW, similarity in embedding space

### Weight Tying Analysis

**"Using the Output Embedding to Improve Language Models"** (Press & Wolf, 2017)
- Paper: https://arxiv.org/abs/1608.05859
- **Why read it**: Explains why tying input/output embeddings helps
- **Time**: 30 minutes
- **Key insight**: Theoretical and empirical justification for tied embeddings

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Encoder vs decoder | "Attention Is All You Need" §3.1 | 30 min |
| GPT-2 architecture | GPT-2 paper §2 | 45 min |
| BPE tokenization | Sennrich 2016 paper §3 | 30 min |
| HuggingFace basics | Transformers quicktour | 30 min |
| Clean GPT code | nanoGPT model.py | 1 hour |
| Weight loading | HuggingFace GPT2 source | 1 hour |

---

## Model Zoo

### Good Models for Learning

| Model | Why Use It | HuggingFace ID |
|-------|------------|----------------|
| GPT-2 (small) | Simple, well-documented | `gpt2` |
| GPT-2 Medium | When you need more capacity | `gpt2-medium` |
| DistilGPT-2 | Faster, good for testing | `distilgpt2` |

### Modern Models (For Reference)

| Model | Parameters | Notes |
|-------|------------|-------|
| LLaMA 2 7B | 7B | Good open model for learning |
| Mistral 7B | 7B | Efficient, sliding window attention |
| Phi-2 | 2.7B | Strong performance for size |

---

## What's NOT Covered Here

These topics are covered in later chapters:

- **Training from scratch** → Chapter 4
- **Efficient inference** → Chapter 8
- **PagedAttention, vLLM** → Chapter 9
- **Flash Attention** → Chapter 10
- **Distributed training** → Chapter 11

---

## Lab Preparation Checklist

Before starting the labs, make sure you can:

- [ ] Explain why causal masking uses `-inf` (not 0)
- [ ] Draw the attention pattern for a 4-token sequence (encoder vs decoder)
- [ ] Explain what BPE tokenization does and why
- [ ] List the components of a GPT-2 model (embeddings, blocks, head)
- [ ] Load a HuggingFace model and generate text
- [ ] Access model weights with `.state_dict()`

If any of these are unclear, revisit the relevant doc or reference before proceeding to the labs.
