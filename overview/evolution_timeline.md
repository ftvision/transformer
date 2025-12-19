# Evolution Timeline: Efficient Attention Mechanisms

A chronological journey through the development of efficient attention, tracing how the field branched into linear and sparse approaches.

## 2017: The Beginning

### Attention Is All You Need (Vaswani et al., Google)
**Paper**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

The Transformer architecture introduced self-attention as the primary mechanism for sequence modeling, replacing recurrence entirely.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Key insight**: Attention allows every position to directly attend to every other position in O(1) hops (vs O(n) for RNNs).

```
WHY O(1) HOPS BUT O(n²) COMPUTE?
────────────────────────────────

"Hops" refers to information flow distance, not compute cost:

RNN: Token 1 → Token 2 → Token 3 → ... → Token n
     Information from token 1 must "hop" through n-1 tokens
     to reach token n. Path length = O(n).

Transformer: Token 1 ─────────────────────→ Token n
             Direct connection! Path length = O(1).

But to CREATE these direct connections for ALL pairs:
  - Token 1 computes attention with tokens 1,2,3,...,n  → n operations
  - Token 2 computes attention with tokens 1,2,3,...,n  → n operations
  - ...
  - Token n computes attention with tokens 1,2,3,...,n  → n operations
  ──────────────────────────────────────────────────────────────────
  Total: n × n = O(n²) operations

The BENEFIT (O(1) hops for gradient flow) comes at a COST (O(n²) compute).
```

**The problem it created**: The attention matrix is n×n, leading to O(n²) complexity in both time and memory.

**Practical impact**: Transformers were limited to ~512-1024 tokens in practice.

---

## 2019: The First Major Attempt

### Sparse Transformers (Child et al., OpenAI)
**Paper**: [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)

The first systematic attempt to address quadratic complexity. Key observation: **learned attention patterns are often naturally sparse**.

**Core idea**: Instead of full n×n attention, use factorized sparse patterns.

```
SPARSE ATTENTION PATTERNS
─────────────────────────

STRIDED (for 2D data like images):
Every token attends to:
  - Previous √n tokens (local)
  - Every √n-th token (strided)

    Position:  1  2  3  4  5  6  7  8  9
    Token 9:   ·  ·  ·  ·  ·  ✓  ✓  ✓  ✓  (local: 6,7,8,9)
               ✓  ·  ·  ✓  ·  ·  ✓  ·  ·  (strided: 1,4,7)

FIXED (alternating patterns):
  - Odd layers: local window
  - Even layers: strided/fixed positions
```

**Complexity**: Reduced from O(n²) to O(n√n)

**Why this matters**:
- First to show you don't need full attention
- Proved sparse patterns can match full attention quality
- Influenced all subsequent sparse attention work
- OpenAI used it to train on 64K+ token sequences

---

## 2020: The Great Divergence

2020 was the pivotal year where two distinct philosophies crystallized.

### LINEAR ATTENTION BRANCH

#### Transformers are RNNs (Katharopoulos et al., EPFL)
**Paper**: [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) - ICML 2020

The foundational paper for linear attention. **Key insight**: By changing the kernel function, we can reorder the computation.

```
WHAT IS A "KERNEL" IN THIS CONTEXT?
───────────────────────────────────

A kernel K(x, y) is a function that measures SIMILARITY between two vectors.

In standard attention, the similarity between query q and key k is:
  sim(q, k) = exp(q · k / √d)    ← this is the "softmax kernel"

The "kernel trick" from machine learning says:
  Any kernel K(x,y) can be written as: K(x, y) = φ(x) · φ(y)
  where φ is a "feature map" that transforms vectors.

Example:
  Polynomial kernel: K(x,y) = (x·y)²
  Can be rewritten as: K(x,y) = φ(x) · φ(y) where φ(x) = [x₁², x₂², √2·x₁x₂]

WHY does φ apply to BOTH Q and K?
─────────────────────────────────
Because similarity is symmetric! To measure how similar q is to k,
we transform both into the same feature space and take their dot product.

  Original: sim(q,k) = kernel(q, k)
  Rewritten: sim(q,k) = φ(q) · φ(k)

The feature map φ must be the SAME for both - that's what makes it a valid kernel.
```

```
STANDARD ATTENTION:
  Attention = softmax(QK^T) V
            = [n×n matrix] × V
            → Must compute n×n matrix first
            → O(n²)

LINEAR ATTENTION (kernel trick):
  Replace softmax with feature map φ:

  Attention = φ(Q) φ(K)^T V

  WHY CAN WE REORDER?
  ───────────────────
  Matrix multiplication is associative: (AB)C = A(BC)

  φ(Q) φ(K)^T V
  └─┬─┘└──┬──┘
    n×d   d×n  n×d_v

  Option 1: (φ(Q) φ(K)^T) V = [n×n] × V → O(n²) ✗
  Option 2: φ(Q) (φ(K)^T V) = [n×d] × [d×d_v] → O(n·d²) ✓

  Since d << n (typically d=64, n=thousands), Option 2 wins!

            = φ(Q) (φ(K)^T V)    ← associativity!
            = φ(Q) × [d×d matrix]
            → Compute K^T V first (d×d, independent of n)
            → O(n)
```

**The RNN revelation**: Linear attention can be written as an RNN!
```
s_i = s_{i-1} + φ(k_i) v_i^T    (hidden state accumulation)
z_i = z_{i-1} + φ(k_i)          (normalizer)
y_i = (φ(q_i)^T s_i) / (φ(q_i)^T z_i)
```

**Trade-off**: Quality depends heavily on feature map φ choice. Simple maps like ELU+1 work but don't match softmax quality.

**Impact**: 4000x speedup on very long sequences, but quality gap remained.

---

#### Performer (Choromanski et al., Google)
**Paper**: [arXiv:2009.14794](https://arxiv.org/abs/2009.14794) - ICLR 2021

Addressed the quality gap with **FAVOR+** (Fast Attention Via positive Orthogonal Random features).

**Key insight**: Use random Fourier features to better approximate softmax.

```
PERFORMER'S APPROACH:
  softmax(QK^T) ≈ φ(Q) φ(K)^T

  Where φ uses random features:
  φ(x) = exp(x^T ω_1), exp(x^T ω_2), ..., exp(x^T ω_r)

  With ω sampled from appropriate distribution
```

```
WHAT EXACTLY IS LOST FROM SOFTMAX?
──────────────────────────────────

Softmax attention has special properties that linear attention approximations struggle to match:

1. SHARP ATTENTION (spikiness)
   ────────────────────────────
   Softmax with large values → nearly one-hot distribution
   Example: softmax([10, 1, 1]) ≈ [0.9999, 0.00005, 0.00005]

   Linear attention can't do this! φ(Q)φ(K)^T produces smoother distributions.
   This hurts tasks requiring precise retrieval ("find the exact token that...").

2. NORMALIZATION IS APPROXIMATE
   ────────────────────────────
   Softmax guarantees: Σ weights = 1 (exact probability distribution)
   Linear attention: normalization is estimated, can be unstable

   WHY DOES EXACT NORMALIZATION MATTER?

   a) Interpretability: Attention weights should mean "fraction of focus"
      - Softmax: "I put 30% attention on word A, 70% on word B" ✓
      - Linear: weights might sum to 0.8 or 1.3 (what does that mean?)

   b) Numerical stability: During training, unnormalized values can explode
      - Softmax: bounded output, gradients well-behaved
      - Linear: may need extra LayerNorm, careful initialization

   c) Weighted average interpretation:
      output = Σ (attention_weight_i × value_i)

      If weights sum to 1: output is a proper weighted average of values
      If weights sum to ≠1: output magnitude depends on normalization error

      This can cause the model to behave inconsistently across sequences
      of different lengths or with different attention patterns.

3. NON-NEGATIVITY
   ───────────────
   Softmax: all weights ≥ 0 (true attention scores)
   Some linear φ: can produce negative "attention" (harder to interpret)
   Performer's FAVOR+ addresses this with positive random features.

4. INFINITE FEATURE DIMENSION
   ──────────────────────────
   Softmax kernel exp(q·k) theoretically needs INFINITE dimensional φ.
   Any finite φ is an approximation. More features → better but slower.

WHY DOES THIS MATTER?

   Task: "What color was the car mentioned in paragraph 3?"

   Full attention: Can put 99% weight on the exact token "red"
   Linear attention: Spreads weight more evenly, might miss precision

This is why early linear attention worked on language modeling (predicting
next token, where smooth distributions are fine) but struggled on retrieval
and reasoning tasks (where sharp focus is essential).

The 2025 breakthrough (KDA) solved this by using HYBRID architectures:
linear attention for most layers + periodic full attention for sharp focus.
```

**Trade-off**: Requires more random features for better approximation, increasing the hidden dimension d → r.

---

### SPARSE ATTENTION BRANCH

#### Longformer (Beltagy et al., AllenAI)
**Paper**: [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)

Practical sparse attention for NLP with **sliding window + global tokens**.

```
LONGFORMER ATTENTION PATTERN
────────────────────────────

Sequence: [CLS] The quick brown fox jumps over the lazy dog [SEP]

Global tokens (attend to all, attended by all):
  [CLS] ←→ everything
  [SEP] ←→ everything

Sliding window (local context, width w):
  "fox" attends to: quick, brown, FOX, jumps, over

┌─────────────────────────────────────────────┐
│  CLS  The quick brown fox jumps over lazy   │
│ ┌───┐                                       │
│ │ ■ │ ■   ■     ■     ■    ■     ■    ■     │ CLS (global)
│ └───┘                                       │
│   ■  ███                                    │ The (local)
│   ■   ███                                   │ quick (local)
│   ■    ███                                  │ brown (local)
│   ■     ███                                 │ fox (local)
│   ■      ███                                │ jumps (local)
│   ■       ███                               │ over (local)
│   ■        ███                              │ lazy (local)
└─────────────────────────────────────────────┘
  ■ = attention computed
```

**Complexity**: O(n × w) where w is window size, effectively O(n).

**Key innovation**: Task-specific global tokens (e.g., [CLS] for classification, question tokens for QA).

---

#### BigBird (Zaheer et al., Google)
**Paper**: [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) - NeurIPS 2020

Added **random attention** to the mix. Theoretically proven to be a **universal approximator** of full attention.

```
BIGBIRD = LOCAL + GLOBAL + RANDOM
──────────────────────────────────

┌─────────────────────────────────┐
│                                 │
│  ████·····█···█··               │ Random sparse
│  ·████····█·█····               │ connections
│  ··████···█··█···               │ ensure info
│  ···████··█····█·               │ can flow
│  ····████·█·█····               │ anywhere
│  █████████████████              │ Global row
│  ·····████·······               │
│  ······████······               │
│  ·······████·····               │
│                                 │
└─────────────────────────────────┘
  █ = local window (diagonal band)
  █ = global tokens (full row/column)
  █ = random connections
```

**Theoretical guarantee**: Random edges ensure any two tokens are connected within O(1) hops in expectation.

---

## 2021-2022: Refinement and New Directions

### DeltaNet (Schlag et al.)
**Paper**: [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) - ICML 2021

**Key insight**: Linear attention as "fast weight programming" - the hidden state is a fast-weight matrix that gets updated.

Uses the **delta rule** (error correction) for updates:
```
Δs = β (v - s^T k) k^T    (delta rule update)
s_{t+1} = s_t + Δs
```

This allows the model to **overwrite** old associations, not just accumulate.

---

### S4: Structured State Spaces (Gu et al.)
**Paper**: [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) - ICLR 2022

A parallel development: **state space models** (SSMs) that aren't attention at all, but achieve similar goals.

```
STATE SPACE MODEL
─────────────────
x'(t) = Ax(t) + Bu(t)    (continuous state evolution)
y(t)  = Cx(t) + Du(t)    (output)

Discretized:
x_k = Āx_{k-1} + B̄u_k
y_k = Cx_k + Du_k
```

**Connection to linear attention**: Both maintain a hidden state that accumulates information. S4's innovation was **structured matrices** (diagonal + low-rank) for efficient computation.

---

### Flash Attention (Dao et al.)
**Paper**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**Important clarification**: Flash Attention is NOT a new attention mechanism—it's a **hardware-aware implementation** of standard softmax attention.

```
FLASH ATTENTION INSIGHT
───────────────────────

GPU Memory Hierarchy:
  HBM (High Bandwidth Memory): Large but slow
  SRAM (Shared Memory): Small but fast

Standard attention:
  1. Compute QK^T → store n×n in HBM (slow)
  2. Apply softmax → read/write HBM (slow)
  3. Multiply by V → read/write HBM (slow)

Flash attention:
  1. Tile computation into blocks that fit in SRAM
  2. Compute softmax incrementally (online softmax)
  3. Never materialize full n×n matrix in HBM
```

**Complexity**: Still O(n²) compute, but O(n) memory and much faster in practice due to reduced HBM access.

**Impact**: Made standard attention practical for longer sequences, reducing pressure to use approximations.

---

## 2023: The Modern Era

### RWKV (Peng et al.)
**Paper**: [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)

"Reinventing RNNs for the Transformer Era" - a pure linear complexity model that scales to 14B parameters.

```
RWKV MECHANISM
──────────────
Combines:
  - Time-mixing (like attention, with decay)
  - Channel-mixing (like FFN)

Key formula:
  wkv_t = Σ_{i=1}^{t-1} e^{-(t-i-1)w+k_i} v_i + e^{u+k_t} v_t
                        └─────────────────┘   └──────────┘
                         Decayed history      Current token
```

**Trained at scale**: 14B parameters, competitive with Transformers.

---

### RetNet (Microsoft)
**Paper**: [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)

"Retentive Network: A Successor to Transformer" - aims to achieve the "impossible triangle":
1. Training parallelism (like Transformers)
2. Low-cost inference (like RNNs)
3. Good performance (like both)

```
RETENTION MECHANISM
───────────────────

Parallel form (training):
  Retention(Q, K, V) = (QK^T ⊙ D) V
  where D is a causal decay mask: D_{ij} = γ^{i-j}

Recurrent form (inference):
  s_n = γ s_{n-1} + k_n^T v_n
  y_n = q_n s_n
```

**Key insight**: Explicit decay γ provides forgetting mechanism missing from basic linear attention.

---

### Mistral 7B (Mistral AI)
**Paper**: [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

Adopted **sliding window attention** for production LLMs.

```
MISTRAL'S SLIDING WINDOW
────────────────────────
Window size: 4096 tokens

Layer 1:  Token sees tokens [t-4096, t]
Layer 2:  Token sees tokens [t-4096, t], which saw [t-8192, t]
...
Layer N:  Effective context = N × 4096

With 32 layers: theoretical 131K context from 4K window!
```

**Practical impact**: Showed sparse attention works in production-grade models.

---

### Mamba (Gu & Dao)
**Paper**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

The breakthrough that made state space models competitive with Transformers on language.

```
MAMBA'S KEY INNOVATION: Selective State Spaces
──────────────────────────────────────────────

Problem with S4: Fixed state dynamics (A, B, C) can't do content-based reasoning

Solution: Make parameters input-dependent!

Standard SSM:    x_t = A x_{t-1} + B u_t
Selective SSM:   x_t = A_t x_{t-1} + B_t u_t
                       └─────────────────┘
                       A_t, B_t depend on input!

This allows the model to:
  - Selectively remember relevant information
  - Selectively forget irrelevant information
  - Perform content-based reasoning
```

**Results**: Mamba-3B matches Transformer performance at 2x size.

---

## 2024: Convergence and Specialization

### Gated Linear Attention (Yang et al.)
**Paper**: [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

Hardware-efficient linear attention with gating, closing the gap with softmax attention.

```
GLA FORMULATION
───────────────
y_t = q_t^T (Σ_{i=1}^t G_t,i k_i v_i^T)

Where G_t,i = Π_{j=i+1}^t g_j  (cumulative gating)

Key: Data-dependent gating g_t allows forgetting!
```

**Connection to Mamba**: Both use data-dependent gating for selective state updates.

---

### DeepSeek-V2: Multi-head Latent Attention (MLA)
**Paper**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

A different approach: **compress the KV cache** rather than sparsify attention.

```
MULTI-HEAD LATENT ATTENTION
───────────────────────────

Standard MHA:
  Cache per layer: [batch, seq, num_heads, head_dim] for K and V

MLA:
  1. Project K, V to low-rank latent: c = W_c [K; V]  (compress)
  2. Cache only c: [batch, seq, latent_dim]           (much smaller!)
  3. At attention time: K, V = W_k c, W_v c           (decompress)

Result: 93.3% KV cache reduction!
```

**Trade-off**: Still O(n²) attention, but dramatically reduced memory for long-context inference.

---

### Mamba-2 (Dao & Gu)
**Paper**: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

"Transformers are SSMs" - unifying the theories.

**Key insight**: State space models and attention are duals!
```
                    Attention              SSM
Compute:            Quadratic              Linear
State size:         O(1)                   O(d × d)
Expressiveness:     Full                   Limited by state size
```

Mamba-2 finds the sweet spot with matrix-valued states and scalar gating.

---

## 2025: The Year of Maturation

2025 has been a pivotal year where both paths reached production maturity, and some surprising conclusions emerged.

### Native Sparse Attention (DeepSeek, February 2025)
**Paper**: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)

A hardware-aligned, **natively trainable** sparse attention mechanism.

```
NSA: THREE-PATH SPARSE ATTENTION
────────────────────────────────

For each query, NSA computes attention through three paths:

1. COMPRESSED COARSE-GRAINED TOKENS
   - Group tokens into temporal blocks
   - Compress each block to a summary
   - Attend to all summaries (global view)

2. SELECTIVELY RETAINED FINE-GRAINED TOKENS
   - Score all tokens quickly
   - Select top-k most relevant
   - Full attention on these (precision)

3. SLIDING WINDOW
   - Always attend to local context
   - Ensures no local information lost

                    Query
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │Compressed│  │Selected │  │ Local   │
   │ Global   │  │ Top-k   │  │ Window  │
   └─────────┘  └─────────┘  └─────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
                   Output
```

**Key result**: NSA **outperforms Full Attention** on average across benchmarks, especially on complex reasoning over long contexts (+8.7% on multi-hop QA).

---

### DeepSeek-V3.2 with DSA (September-December 2025)
**Paper**: [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)

DeepSeek Sparse Attention (DSA) builds on NSA with a streamlined two-stage pipeline.

```
DSA: LIGHTNING INDEXER + FINE-GRAINED SELECTION
───────────────────────────────────────────────

Stage 1: Lightning Indexer (ultra-fast scoring)
  - Lightweight FP8 scorer
  - Cached indexer keys (separate from main KV)
  - Quickly identifies relevant chunks

Stage 2: Fine-Grained Token Selection
  - Select top-k=2048 tokens per query
  - Standard attention only on selected subset

COMPLEXITY REDUCTION:
  Full attention:  O(L²) for context length L
  DSA:            O(L × k) where k=2048 << L

For 128K context:
  - 70% cost reduction
  - 3.5x speed increase
  - 70% memory reduction
```

**Training process**:
1. **Dense warm-up** (2.1B tokens): Train indexer to predict full attention output
2. **Sparse training** (943.7B tokens): Train entire model with sparse selection

**Result**: DeepSeek-V3.2 performs comparably to GPT-5, with DeepSeek-V3.2-Speciale **surpassing** GPT-5 on reasoning benchmarks.

---

### Kimi Linear with KDA (Moonshot AI, October 2025)
**Paper**: [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

A landmark achievement: **linear attention that outperforms full attention**.

```
KIMI DELTA ATTENTION (KDA)
──────────────────────────

KDA extends Gated DeltaNet with finer-grained gating:

DeltaNet:  S_t = G_t ⊙ S_{t-1} + β(v_t - S_{t-1}k_t)k_t^T

KDA adds:
  - Per-dimension gating (not just scalar)
  - Diagonal-Plus-Low-Rank (DPLR) transitions
  - Optimized chunkwise algorithm

HYBRID ARCHITECTURE:
  - 3:1 ratio of KDA to MLA layers
  - KDA for most computation (efficient)
  - MLA for global attention (precision)

Layer 1:  KDA
Layer 2:  KDA
Layer 3:  KDA
Layer 4:  MLA (global)  ← every 4th layer
Layer 5:  KDA
...
```

**Results**:
- **Outperforms full MLA** with identical training recipe
- 75% KV cache reduction
- 6x decoding throughput at 1M context length

**Why this matters**: First time linear attention **definitively beats** full attention under fair comparison.

---

### Kimi K2 Thinking (Moonshot AI, November 2025)
**Model**: [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

A 1T parameter MoE reasoning model with agentic capabilities.

```
KIMI K2 ARCHITECTURE
────────────────────

Total parameters:     1 Trillion
Active parameters:    32 Billion
Context length:       256K tokens

Architecture:
  - 61 layers (1 dense + 60 MoE)
  - 384 experts, 8 selected per token
  - 1 shared expert
  - 64 attention heads
  - Multi-head Latent Attention (MLA)
  - SwiGLU activation

Special capability:
  - 200-300 sequential tool calls
  - Interleaved thinking + tool use
  - Native INT4 quantization (lossless 2x speedup)
```

**Training cost**: $4.6 million

---

### MiniMax M2: The Contrarian (October 2025)
**Blog**: [Why Did M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)

MiniMax made the surprising decision to use **full attention** despite efficient alternatives.

```
MINIMAX'S REASONING
───────────────────

Previous model (MiniMax-Text-01):
  - Hybrid Lightning Attention + Full Attention
  - Seemed to match full attention on MMLU, BBH, MATH

At scale, they found:
  - "Clear deficits in complex, multi-hop reasoning"
  - Global attention patterns (retrieval heads, induction heads)
    were established early in pretraining
  - Continued pretraining couldn't fix these patterns
  - "Nearly impossible to discover all important heads from human priors"

Conclusion:
  "In a real-world, industrial-grade system, efficient attention
   still has some way to go before it can definitively beat full attention."

Note: This was BEFORE Kimi Linear's results showed linear can beat full!
```

**M2 Architecture**:
- 230B total / 10B active parameters (MoE)
- 204K input context, 131K output
- Full attention (not sparse or linear)
- Optimized for coding and agentic tasks

---

## 2025 Summary: The State of the Art

```
2025 LANDSCAPE
══════════════

LINEAR ATTENTION PATH:
  Kimi Linear (KDA)     → First to beat full attention fairly
  Kimi K2               → Production 1T model with MLA

SPARSE ATTENTION PATH:
  NSA                   → Hardware-aligned, outperforms full attention
  DSA (V3.2)           → 70% cost reduction, GPT-5 competitive

FULL ATTENTION PATH:
  MiniMax M2           → Argues efficient attention not ready
  (But predates Kimi Linear's results)

THE EMERGING CONSENSUS:
  1. Linear attention CAN beat full attention (Kimi Linear proved it)
  2. Sparse attention CAN beat full attention (NSA proved it)
  3. Hybrid approaches are winning (KDA+MLA, DSA two-stage)
  4. Hardware co-design is crucial (DPLR, FP8 indexer)
```

---

## Summary: The Two Paths

```
                        EFFICIENCY SPECTRUM

         Linear                                    Sparse
         Attention                                 Attention
            ◄──────────────────────────────────────────►

COMPUTE    O(n)                                    O(n) to O(n²)
           ════════════════════════════════════════════

MEMORY     O(1) per step                          O(n) KV cache
           (recurrent)                            (can compress)
           ════════════════════════════════════════════

QUALITY    Approximation                          Exact attention
           (gap closing)                          (on computed subset)
           ════════════════════════════════════════════

BEST FOR   Streaming,                             Documents,
           100K+ tokens                           structured data
           ════════════════════════════════════════════
```

**The frontier**: Hybrid models that use linear/SSM for most computation with sparse attention for critical positions.

---

## Further Reading

- [Linear Attention Path](linear_attention_path.md) - Deep dive into linear attention evolution
- [Sparse Attention Path](sparse_attention_path.md) - Deep dive into sparse attention evolution
