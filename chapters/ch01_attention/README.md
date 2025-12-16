# Chapter 1: The Attention Mechanism

## Overview

Attention is the core innovation that makes transformers work. In this chapter, you'll build attention from scratch and develop deep intuition for how it works.

## Learning Objectives

By the end of this chapter, you will:
- Understand attention as a soft lookup / weighted retrieval mechanism
- Implement scaled dot-product attention from scratch
- Understand why we use multiple heads and how they specialize
- Be able to match PyTorch's implementation exactly

## Key Concepts

- **Query, Key, Value**: The database analogy - queries look up relevant keys, values get retrieved
- **Dot product as similarity**: How we measure "relevance" between tokens
- **Softmax normalization**: Converting scores to attention weights
- **Scaling factor √d_k**: Why we divide by the square root of dimension
- **Multi-head attention**: Running parallel attention in different subspaces

## Reading Order

1. Start with `docs/01_attention_intuition.md` - understand *what* attention does
2. Read `docs/02_scaled_dot_product.md` - understand the math
3. Read `docs/03_multihead_attention.md` - understand why multiple heads help

## Labs

| Lab | Title | What you build |
|-----|-------|----------------|
| lab01 | Dot-Product Attention | `attention(Q, K, V)` from scratch |
| lab02 | Attention Visualization | Heatmaps of attention weights |
| lab03 | Multi-Head Attention | `MultiHeadAttention` class |
| lab04 | PyTorch Comparison | Match `nn.MultiheadAttention` output |

## How to Work Through This Chapter

```bash
# 1. Read the docs
cat docs/01_attention_intuition.md

# 2. Start lab 1
cd lab01_dot_product
cat README.md  # Read instructions

# 3. Implement the code
# Edit src/attention.py

# 4. Run tests until green
uv run pytest tests/

# 5. Check solution if stuck
cat solutions/attention.py

# 6. Move to next lab
cd ../lab02_visualization
```

## Milestone

Your multi-head attention matches PyTorch's `nn.MultiheadAttention` output within 1e-5 tolerance.

## Prerequisites

- Python basics (functions, classes, numpy arrays)
- Linear algebra (matrix multiplication, dot products)
- No PyTorch experience required (we'll learn it along the way)
