# The Sparse Attention Path

From pattern-based sparsity to learned compression—keeping exact attention while computing less.

## The Core Philosophy

Sparse attention takes a fundamentally different approach than linear attention:

```
LINEAR ATTENTION:           SPARSE ATTENTION:
Change the math             Keep the math, compute less

softmax(QK^T) → φ(Q)φ(K)^T  softmax(QK^T) → softmax(QK^T ⊙ M)
                                                       ↑
                                            Sparse mask M
```

**Key insight**: In practice, most attention weights are near-zero. We can skip computing them entirely.

## The Empirical Observation (2019)

OpenAI's Sparse Transformer paper made a key observation:

```
LEARNED ATTENTION PATTERNS
──────────────────────────

When you visualize attention weights in trained Transformers:

Head 1 (local):        Head 2 (strided):      Head 3 (global):
█░░░░░░░░░░░░░░░░     █░░░█░░░█░░░█░░░     █████████████████
██░░░░░░░░░░░░░░░     █░░░█░░░█░░░█░░░     █░░░░░░░░░░░░░░░░
███░░░░░░░░░░░░░░     █░░░█░░░█░░░█░░░     █░░░░░░░░░░░░░░░░
████░░░░░░░░░░░░░     █░░░█░░░█░░░█░░░     █░░░░░░░░░░░░░░░░
█████░░░░░░░░░░░░     ...                   ...

Most heads learn SPARSE patterns!
→ Why compute full n×n if most entries will be ~0?
```

This suggests we can **predetermine** sparse patterns without significant quality loss.

## Taxonomy of Sparse Patterns

### Local/Sliding Window

The most intuitive: each token attends to nearby tokens.

```
SLIDING WINDOW ATTENTION (width w=3)
────────────────────────────────────

Sequence:    The quick brown fox jumps over
Position:     1    2     3    4    5    6

Position 4 ("fox") attends to:
  positions 3, 4, 5 (brown, fox, jumps)

Attention matrix (█ = computed, ░ = zero):

         1   2   3   4   5   6
     ┌───────────────────────────┐
  1  │ ███░░░░░░░░░░░░░░░░░░░░░ │
  2  │ █████░░░░░░░░░░░░░░░░░░░ │
  3  │ ░█████░░░░░░░░░░░░░░░░░░ │
  4  │ ░░░█████░░░░░░░░░░░░░░░░ │
  5  │ ░░░░░█████░░░░░░░░░░░░░░ │
  6  │ ░░░░░░░█████░░░░░░░░░░░░ │
     └───────────────────────────┘

Complexity: O(n × w) = O(n) when w is constant
```

**Intuition**: Language is largely local. "The quick brown fox" - understanding "fox" mostly needs nearby words.

### Strided/Dilated

Attend to positions at regular intervals (like dilated convolutions).

```
STRIDED ATTENTION (stride s=4)
──────────────────────────────

Position 16 attends to: 4, 8, 12, 16

         1   4   8  12  16  20  24
     ┌───────────────────────────────┐
  4  │ ░░░█░░░░░░░░░░░░░░░░░░░░░░░░ │
  8  │ ░░░█░░░█░░░░░░░░░░░░░░░░░░░░ │
 12  │ ░░░█░░░█░░░█░░░░░░░░░░░░░░░░ │
 16  │ ░░░█░░░█░░░█░░░█░░░░░░░░░░░░ │
 20  │ ░░░█░░░█░░░█░░░█░░░█░░░░░░░░ │
     └───────────────────────────────┘

Often combined with local for FACTORIZED attention:
  - Odd layers: local window
  - Even layers: strided
  - Together: any position can reach any other in 2 hops
```

### Global Tokens

Designate certain positions as "global" - they attend to everything and are attended by everything.

```
GLOBAL TOKENS
─────────────

Sequence: [CLS] The capital of France is Paris [SEP]
          ↑                                     ↑
        Global                               Global

Attention pattern:
           CLS  The cap  of  Fra  is Par SEP
     ┌───────────────────────────────────────┐
 CLS │  █   █   █   █   █   █   █   █   █   │ ← attends all
 The │  █  ███ ░░░ ░░░ ░░░ ░░░ ░░░ ░░░  █   │
 cap │  █  ███ ███ ░░░ ░░░ ░░░ ░░░ ░░░  █   │
  of │  █  ░░░ ███ ███ ░░░ ░░░ ░░░ ░░░  █   │
 Fra │  █  ░░░ ░░░ ███ ███ ░░░ ░░░ ░░░  █   │
  is │  █  ░░░ ░░░ ░░░ ███ ███ ░░░ ░░░  █   │
 Par │  █  ░░░ ░░░ ░░░ ░░░ ███ ███ ░░░  █   │
 SEP │  █   █   █   █   █   █   █   █   █   │ ← attends all
     └───────────────────────────────────────┘
         ↑                               ↑
       attended                       attended
       by all                         by all
```

**Use case**: Global tokens aggregate information that can then be distributed to the whole sequence.

### Random

BigBird's innovation: add random attention connections.

```
RANDOM ATTENTION
────────────────

For each query position, randomly select r keys to attend to.

Position 10 might attend to: 3, 7, 15, 22, 31 (random)
Position 11 might attend to: 2, 9, 18, 25, 30 (different random)

WHY THIS HELPS (graph theory):
─────────────────────────────

Without random:    With random:
  Local only         Local + random edges

  ○─○─○─○─○         ○─○─○─○─○
                      ╲ ╱   ╲
  Max distance:      ○─○─○─○─○
  O(n) hops          ╱   ╲ ╱
                    ○─○─○─○─○

                    Max distance:
                    O(log n) hops!

Random edges create "shortcuts" through the graph.
Any token can reach any other in O(log n) hops in expectation.
```

**Theoretical result**: BigBird proved this random sparse attention is a **universal approximator** of full attention.

## Key Papers Deep Dive

### Sparse Transformers (OpenAI, 2019)

The paper that started it all.

```
SPARSE TRANSFORMER FACTORIZATION
────────────────────────────────

Instead of full n×n attention, factorize into two sparse patterns:

Layer l (local + summary):
  - Attend to previous √n positions (local)
  - Attend to positions 0, √n, 2√n, ... (summary positions)

Layer l+1 (strided):
  - Attend to positions i, i-√n, i-2√n, ... (stride √n)

Together over 2 layers:
  Position 100 in layer l+1 can reach:
    → 100, 90, 80, ... (strided from layer l+1)
    → Each of those reached 10 local positions (layer l)
    → Total coverage: all positions!

COMPLEXITY:
  Each position: O(√n) connections
  Total: O(n × √n) = O(n√n)
```

**Architecture innovations**:
- Gradient checkpointing for memory efficiency
- Custom CUDA kernels for sparse attention
- Deeper networks (up to 128 layers)

### Longformer (AllenAI, 2020)

Practical sparse attention for NLP tasks.

```
LONGFORMER PATTERNS
───────────────────

1. SLIDING WINDOW (all positions)
   Each token attends to w tokens on each side
   w = 256 or 512 typically

2. DILATED SLIDING WINDOW (optional)
   Like sliding window but with gaps
   Increases receptive field without more compute

3. GLOBAL ATTENTION (task-specific)
   - Classification: [CLS] token is global
   - QA: question tokens are global
   - Summarization: first tokens are global

THE KEY INSIGHT:
────────────────
Different tasks need different global tokens!
→ Make it configurable, not fixed

Code pattern (pseudocode):
  global_attention_mask[cls_position] = 1
  global_attention_mask[question_positions] = 1
  output = longformer(input, global_attention_mask)
```

**Practical details**:
- Can process 4,096 tokens (vs 512 for BERT)
- Drop-in replacement for BERT on many tasks
- Linear complexity O(n × w)

### BigBird (Google, 2020)

Added theoretical rigor and random attention.

```
BIGBIRD = LOCAL + GLOBAL + RANDOM
─────────────────────────────────

Configuration:
  w = window size (local)
  g = number of global tokens
  r = number of random connections per position

Total connections per position: w + g + r
Complexity: O(n × (w + g + r)) = O(n)

THEORETICAL GUARANTEE:
──────────────────────
BigBird's sparse attention can approximate any function
that full attention can compute!

Proof sketch:
1. Global tokens can aggregate info from anywhere
2. Random connections ensure short paths exist
3. Local attention captures fine-grained patterns
4. Together: universal approximation
```

**Configurations**:
```
BigBird-ITC (Internal Transformer Construction):
  - Global tokens are special [CLS]-like tokens
  - Good for classification

BigBird-ETC (Extended Transformer Construction):
  - First g tokens are global
  - Good for sequence-to-sequence

BigBird-Random:
  - Different random pattern each layer
  - Maximum expressiveness
```

### Mistral (Mistral AI, 2023)

Brought sparse attention to production LLMs.

```
MISTRAL 7B: SLIDING WINDOW ATTENTION
────────────────────────────────────

Simple but effective:
  - Window size: 4,096 tokens
  - Every token attends to previous 4,096 tokens
  - No global tokens (simpler)

THE ROLLING BUFFER:
───────────────────

Instead of KV cache growing with sequence:

Standard (full attention):
  Sequence length 1000: cache 1000 KV pairs
  Sequence length 10000: cache 10000 KV pairs
  → Memory grows linearly

Mistral (sliding window):
  Sequence length 1000: cache 1000 KV pairs
  Sequence length 10000: cache 4096 KV pairs only!
  → Memory capped at window size

Implementation (circular buffer):
  cache[position % window_size] = (K, V)

EFFECTIVE CONTEXT THROUGH LAYERS:
─────────────────────────────────

Layer 1: token sees 4,096 tokens
Layer 2: token sees what layer 1 saw → 8,192 effective
Layer 3: → 12,288 effective
...
Layer 32: → 131,072 effective context!

  Layer 1    Layer 2    Layer 3
    ▼          ▼          ▼
  ┌───┐      ┌───┐      ┌───┐
  │4K │  →   │ + │  →   │ + │  → ...
  └───┘      │4K │      │4K │
             └───┘      └───┘

Information propagates through layers,
creating much larger effective context.
```

## KV Cache Compression: A Different Approach

While pattern-based sparsity reduces **compute**, KV cache compression reduces **memory**.

### The KV Cache Problem

```
INFERENCE MEMORY BREAKDOWN
──────────────────────────

For a 7B parameter model generating 10K tokens:

Model weights:     ~14 GB (float16)
KV Cache:          ~20 GB (!!)
  = batch_size × seq_len × num_layers × 2 × num_heads × head_dim × bytes
  = 1 × 10000 × 32 × 2 × 32 × 128 × 2
  = 5.2 GB per batch item!

The KV cache can exceed model size for long sequences!
```

### Grouped-Query Attention (GQA)

First attempt: share keys and values across heads.

```
MULTI-HEAD ATTENTION (MHA)
──────────────────────────
32 query heads
32 key heads
32 value heads
→ KV cache: 32 × 2 = 64 vectors per position

GROUPED-QUERY ATTENTION (GQA)
─────────────────────────────
32 query heads
8 key heads (shared by 4 query heads each)
8 value heads
→ KV cache: 8 × 2 = 16 vectors per position
→ 4x reduction!

  Q heads: [q1 q2 q3 q4] [q5 q6 q7 q8] ...
              ↓    ↓         ↓    ↓
  K heads:    k1    k1       k2    k2      (shared)
  V heads:    v1    v1       v2    v2
```

**Trade-off**: Some quality loss due to reduced KV capacity.

### Multi-Query Attention (MQA)

Extreme version: single K, V for all heads.

```
MULTI-QUERY ATTENTION (MQA)
───────────────────────────
32 query heads
1 key head (shared by all)
1 value head
→ KV cache: 1 × 2 = 2 vectors per position
→ 32x reduction!

Trade-off: Significant quality degradation
→ Often not worth it for high-quality models
```

### Multi-head Latent Attention (MLA) - DeepSeek

The breakthrough: compress KV to a latent space.

```
MULTI-HEAD LATENT ATTENTION
───────────────────────────

Standard MHA:
  K = X @ W_K    # [seq, num_heads × head_dim]
  V = X @ W_V    # [seq, num_heads × head_dim]
  Cache: K, V    # Large!

MLA:
  c = X @ W_C    # [seq, latent_dim]  ← COMPRESS
  Cache: c       # Small (93% reduction)!

  # At attention time:
  K = c @ W_UK   # Decompress K
  V = c @ W_UV   # Decompress V

WHY THIS IS DIFFERENT FROM GQA:
───────────────────────────────

GQA: Fewer KV heads → fewer parameters → less expressive
MLA: Same parameters, just compressed for caching!

  GQA:     32 Q heads, 8 KV heads → 8 different K,V
  MLA:     32 Q heads, 32 KV heads, but stored as compressed c
           → 32 different K,V reconstructed from c

MLA maintains full expressiveness while reducing cache size!
```

**How compression works**:
```
LATENT COMPRESSION INTUITION
────────────────────────────

Observation: K, V across heads are often correlated
  K_head1 ≈ f₁(some latent)
  K_head2 ≈ f₂(some latent)
  ...

MLA learns:
  c = encode(X)        # Low-rank latent
  K_i = decode_i(c)    # Different decoder per head

The latent c captures shared structure
Decoders reconstruct head-specific details

Mathematically:
  [K; V] ≈ U @ c       # Low-rank factorization
  where U is the decompression matrix
```

**Results**:
- 93.3% KV cache reduction
- 5.76x inference throughput improvement
- Quality actually **improved** vs standard MHA (!)

## Comparison of Approaches

```
SPARSE ATTENTION APPROACHES
═══════════════════════════

                    Pattern-Based           KV Compression
────────────────────────────────────────────────────────────
Representative     Longformer, BigBird,    GQA, MQA, MLA
                   Mistral
────────────────────────────────────────────────────────────
Reduces            Compute O(n²→n)         Memory (KV cache)
────────────────────────────────────────────────────────────
Attention type     Exact (on computed)     Exact (full)
────────────────────────────────────────────────────────────
Quality impact     Depends on pattern      MLA: minimal
                   choice                  GQA/MQA: some loss
────────────────────────────────────────────────────────────
Best for           Very long sequences     All sequence lengths,
                   with local structure    especially inference
────────────────────────────────────────────────────────────
Can combine?       Yes! Mistral + MLA      Yes! Pattern + compress
────────────────────────────────────────────────────────────
```

## Hybrid Approaches (2024-2025)

### DeepSeek-V3: Sparse + MLA

```
DEEPSEEK V3 APPROACH
────────────────────

Combines multiple techniques:

1. MLA for KV compression (93% cache reduction)
2. Mixture-of-Experts (MoE) for sparse FFN
3. Fine-grained sparse attention selection

DeepSeek Sparse Attention (DSA):
  Stage 1: "Lightning indexer" quickly identifies relevant chunks
  Stage 2: Fine-grained token selection within chunks

  Sequence: [==========][==========][==========]
                ↓ Stage 1: chunk selection
            [==========]            [==========]
                ↓ Stage 2: token selection
            [=  = =    ]            [   ==   = ]

  Only compute attention on selected tokens!
```

### Flash Attention + Sparse

Flash Attention's tiling approach works naturally with sparse patterns:

```
SPARSE FLASH ATTENTION
──────────────────────

Standard Flash Attention:
  Tile Q, K, V into blocks
  Compute attention block by block
  Never materialize full n×n matrix

Sparse Flash Attention:
  Same tiling approach
  But SKIP blocks that are zero in sparse pattern!

  ┌─────────────────────┐
  │ ████ ░░░░ ░░░░ ░░░░ │  Block (1,1): compute
  │ ████ ████ ░░░░ ░░░░ │  Block (2,1): compute
  │ ░░░░ ████ ████ ░░░░ │  Block (2,2): skip (all zeros)
  │ ░░░░ ░░░░ ████ ████ │  ...
  └─────────────────────┘

  Combines memory efficiency of Flash
  with compute reduction of sparsity!
```

## When to Use What

```
DECISION GUIDE
══════════════

Sequence length < 4K:
  → Full attention with Flash Attention
  → Sparse adds complexity without much benefit

Sequence length 4K-32K:
  → Sliding window (Mistral-style) + Flash Attention
  → Consider MLA if memory constrained

Sequence length 32K-128K:
  → Sliding window + global tokens (Longformer-style)
  → MLA for inference efficiency
  → Consider hybrid sparse selection

Sequence length > 128K:
  → Linear attention / SSM might be better
  → Or very aggressive sparsity (DeepSeek DSA)

Memory constrained (large batch):
  → MLA >> GQA >> MQA
  → Pattern sparsity + compression

Quality critical:
  → MLA (maintains quality)
  → Carefully designed global tokens
  → Avoid aggressive MQA
```

## 2025: The Breakthroughs

### Native Sparse Attention (NSA) - February 2025
**Paper**: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)

DeepSeek introduced NSA as a **natively trainable** sparse attention mechanism that doesn't require post-hoc conversion from full attention.

```
NATIVE SPARSE ATTENTION (NSA)
─────────────────────────────

THREE-PATH ATTENTION:
  For each query, compute attention through:

  1. Compressed coarse-grained tokens (global view)
     - Group tokens into temporal blocks
     - Compress to summaries
     - Attend to all summaries

  2. Selected fine-grained tokens (precision)
     - Score all tokens quickly
     - Select top-k most relevant
     - Full attention on selected

  3. Sliding window (local context)
     - Always attend to nearby tokens
     - Ensures local info preserved

KEY INNOVATIONS:
  - Hardware-aligned: Balanced arithmetic intensity
  - End-to-end trainable: No full→sparse conversion
  - Reduces pretraining compute significantly
```

**Result**: NSA **outperforms Full Attention** on average, especially on multi-hop reasoning (+8.7%).

---

### DeepSeek Sparse Attention (DSA) - September-December 2025
**Paper**: [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

DSA streamlines NSA into a practical two-stage pipeline for production.

```
DSA: LIGHTNING INDEXER + FINE-GRAINED SELECTION
───────────────────────────────────────────────

STAGE 1: LIGHTNING INDEXER
──────────────────────────
  - Ultra-light FP8 scorer
  - Separate cache for indexer keys (not main KV)
  - Quickly identifies relevant chunks

  Implementation:
    - Query shape: (h, d) with h heads
    - For each query, compute relevance scores
    - Output: indices of top chunks

STAGE 2: FINE-GRAINED TOKEN SELECTION
─────────────────────────────────────
  - Select top-k=2048 tokens per query
  - Standard softmax attention on subset
  - Exact attention on selected tokens

  Per-query cost: O(k) instead of O(L)
  With k=2048, L=128K: 62x reduction!

TRAINING PIPELINE:
──────────────────

Phase 1: Dense Warm-up (2.1B tokens)
  - Freeze main model
  - Train indexer to predict full attention output
  - "Teaches" indexer what's important

Phase 2: Sparse Training (943.7B tokens)
  - Enable sparse selection
  - Train full model with top-2048 selection
  - Model learns to work with sparse attention
```

**Results**:
- 70% cost reduction for 128K context
- 3.5x inference speedup
- 70% memory reduction
- DeepSeek-V3.2-Speciale **surpasses GPT-5** on reasoning

---

### MiniMax M2: The Counterargument - October 2025
**Blog**: [Why Did M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)

Interestingly, MiniMax chose **full attention** for M2, against the trend.

```
MINIMAX'S LESSONS FROM EFFICIENT ATTENTION
──────────────────────────────────────────

What they tried (MiniMax-Text-01):
  - Hybrid Lightning Attention + Full Attention
  - Looked good on standard benchmarks (MMLU, BBH, MATH)

What they found at scale:
  - "Clear deficits in complex, multi-hop reasoning"
  - Critical attention patterns (retrieval heads, induction heads)
    established early in pretraining
  - Continued pretraining couldn't fix these patterns
  - "Nearly impossible to discover all important heads from human priors"

Their conclusion:
  "In a real-world, industrial-grade system, efficient attention
   still has some way to go before it can definitively beat full attention."

IMPORTANT CONTEXT:
  - This was BEFORE Kimi Linear and NSA results published
  - Their hybrid approach didn't have the innovations of KDA/DSA
  - The field has moved fast since their decision
```

**M2 specs**: 230B total / 10B active (MoE), 204K context, full attention.

---

## Summary

The sparse attention path has evolved from simple patterns to sophisticated learned selection:

```
2019: "Attention is often sparse" → use fixed patterns
         ↓
2020: "Task needs differ" → configurable global tokens
         ↓
2020: "Add randomness" → theoretical guarantees (BigBird)
         ↓
2023: "Keep it simple" → sliding window at scale (Mistral)
         ↓
2024: "Compress, don't skip" → MLA (DeepSeek-V2)
         ↓
2025: "Learn what to select" → NSA/DSA
         ↓
2025: "Hardware co-design" → Lightning indexer (FP8), sparse kernels

THE 2025 RESULT: Sparse attention CAN beat full attention!
```

The key innovations that made this possible:
1. **Learned selection**: Let the model decide what's important (not hand-designed patterns)
2. **Two-stage pipeline**: Cheap scorer → expensive attention on subset
3. **Native training**: Train with sparsity from the start (no conversion)
4. **Hardware alignment**: FP8 indexer, optimized sparse kernels

## References

1. [Generating Long Sequences with Sparse Transformers (Child et al., 2019)](https://arxiv.org/abs/1904.10509)
2. [Longformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
3. [BigBird (Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062)
4. [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
5. [Mistral 7B (Mistral AI, 2023)](https://arxiv.org/abs/2310.06825)
6. [DeepSeek-V2 (DeepSeek, 2024)](https://arxiv.org/abs/2405.04434) - MLA
7. [Understanding MLA](https://planetbanatt.net/articles/mla.html)
8. [Native Sparse Attention (DeepSeek, Feb 2025)](https://arxiv.org/abs/2502.11089) - **NSA outperforms full attention**
9. [DeepSeek-V3.2 (DeepSeek, Dec 2025)](https://arxiv.org/abs/2512.02556) - **DSA with lightning indexer**
10. [MiniMax M2 Blog (Oct 2025)](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) - Counterargument for full attention
