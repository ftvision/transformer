# The Quadratic Bottleneck: Why O(n²) Attention Hurts

![Quadratic Complexity](vis/quadratic_complexity.svg)

## The Problem in Numbers

Recall from Chapter 1 that standard attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The `QK^T` operation creates an `(n, n)` attention matrix where n = sequence length.

**Memory and compute scale quadratically**:

| Sequence Length | Attention Matrix Size | Memory (fp16) | FLOPs |
|-----------------|----------------------|---------------|-------|
| 512 | 262K | 0.5 MB | ~134M |
| 2,048 | 4.2M | 8 MB | ~8.6B |
| 8,192 | 67M | 134 MB | ~137B |
| 32,768 | 1.07B | 2.1 GB | ~2.2T |
| 131,072 | 17.2B | 34 GB | ~35T |

This is **per layer, per head**. A model with 32 layers and 32 heads multiplies these numbers by 1024.

## Real-World Consequences

### 1. Long Documents Are Impossible

Consider processing a research paper:
- Average paper: ~8,000 words ≈ 10,000 tokens
- GPT-2's context: 1,024 tokens
- Even "long context" models struggle beyond 4K-8K

```
Your 50-page PDF
      ↓
  [Only first 1-2 pages fit in context]
      ↓
  "I don't have enough context to answer that"
```

### 2. GPU Memory Is the Hard Limit

A typical 24GB GPU with a model using 32 heads:

```
Maximum sequence length ≈ √(available_memory / (heads × layers × 2))

For 24GB with 32 heads, 32 layers:
≈ √(24GB / (32 × 32 × 2 bytes)) ≈ 6,000 tokens
```

Even with Flash Attention's memory optimization, compute still scales O(n²).

### 3. Latency Grows Quadratically

For real-time applications (chatbots, coding assistants):

```
Sequence length:  1000   →   2000   →   4000
Attention time:    1x    →    4x    →   16x
```

Doubling the context **quadruples** the attention computation.

## Why Can't We Just Chunk?

A naive solution: split long sequences into chunks, process separately.

**Problem**: Each chunk can't see the others.

```
Document: "The CEO announced... [2000 tokens later] ...she resigned"

Chunk 1: "The CEO announced..."     Chunk 2: "...she resigned"
         [No connection between chunks!]

Model can't resolve "she" → "CEO" across chunk boundary
```

Solutions like sliding windows help (Chapter 7), but they're band-aids on the fundamental O(n²) problem.

## Where the Time Goes

Profiling a transformer on a 4096-token sequence:

```
┌─────────────────────────────────────────┐
│ Attention (QK^T + softmax + AV)  ~70%  │████████████████████████
├─────────────────────────────────────────┤
│ Feed-Forward Networks            ~20%  │██████
├─────────────────────────────────────────┤
│ Layer Norm, Embeddings, etc.     ~10%  │███
└─────────────────────────────────────────┘
```

Attention dominates, and it gets worse as sequences get longer.

## The Two Culprits

### Culprit 1: The Attention Matrix

```python
# This creates an (n, n) matrix
attention_scores = Q @ K.T  # O(n²) memory, O(n²d) compute
```

For n=32768, d=128: that's 32768² × 128 = 137 trillion operations.

### Culprit 2: Softmax Requires Full Rows

Softmax normalizes across the entire row:

```python
softmax(x) = exp(x) / sum(exp(x))  # Need ALL values to compute the sum
```

This means we can't compute attention incrementally—we must see all keys before producing output.

## What Would Linear Attention Give Us?

If we could make attention O(n):

| Sequence Length | O(n²) Time | O(n) Time | Speedup |
|-----------------|------------|-----------|---------|
| 2,048 | 1x | 1x | baseline |
| 8,192 | 16x | 4x | 4x |
| 32,768 | 256x | 16x | 16x |
| 131,072 | 4096x | 64x | 64x |

The gap widens as sequences grow. This is why linear attention matters for:
- Book-length documents
- Long conversations
- Code repositories
- Video/audio processing

## The Key Insight

The quadratic cost comes from computing **all pairwise interactions**:

```
Every query must compare to every key:

Q[0] → K[0], K[1], K[2], ..., K[n-1]
Q[1] → K[0], K[1], K[2], ..., K[n-1]
...
Q[n-1] → K[0], K[1], K[2], ..., K[n-1]

Total: n × n = n² comparisons
```

**The linear attention insight**: We don't need all pairwise scores explicitly. If we change how similarity is computed, we can use **associativity** to avoid the n² matrix entirely.

## What's Next

Now that you understand why O(n²) is a problem, let's learn the mathematical trick that enables O(n) attention in `02_kernel_trick.md`.
