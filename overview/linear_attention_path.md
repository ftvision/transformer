# The Linear Attention Path

From kernel tricks to state space models—the quest for O(n) sequence modeling.

## The Core Problem

Standard softmax attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

The `softmax(QK^T)` term requires computing the full n×n matrix before normalization, because softmax needs the entire row to normalize.

**Linear attention asks**: Can we avoid computing the full n×n matrix?

## The Kernel Trick (2020)

### The Mathematical Insight

The key insight from Katharopoulos et al. (2020) is that softmax can be viewed as a **kernel function**:

```
softmax(q_i^T k_j / √d) ≈ κ(q_i, k_j) = φ(q_i)^T φ(k_j)
```

If we can find a feature map φ such that the kernel κ approximates softmax, then:

```
STANDARD (must compute n×n first):
  y = softmax(QK^T) V
    = [n×n matrix] × [n×d matrix]
    → O(n²d)

LINEAR (associativity trick):
  y = φ(Q) φ(K)^T V
    = φ(Q) (φ(K)^T V)       ← reorder multiplication!
    = [n×d_φ] × [d_φ×d]     ← no n×n matrix!
    → O(n d_φ d)  which is O(n) when d_φ, d are fixed
```

### Why This Works: Matrix Associativity

```
Consider computing AB C where:
  A is n×d
  B is d×n  (so AB is n×n)
  C is n×m

Option 1: (AB)C
  AB = n×n matrix        → O(n²d)
  (AB)C = n×m matrix     → O(n²m)
  Total: O(n²(d+m))

Option 2: A(BC)
  BC = d×m matrix        → O(ndm)
  A(BC) = n×m matrix     → O(ndm)
  Total: O(ndm)          → Linear in n!
```

**The catch**: This reordering only works when we don't have non-linearities (like softmax) that must see the whole row.

### Feature Maps: The Quality Problem

The challenge is finding φ(·) that approximates softmax well:

```
FEATURE MAP OPTIONS
───────────────────

1. ELU+1 (Katharopoulos et al.):
   φ(x) = ELU(x) + 1 = { x + 1       if x > 0
                       { e^x         if x ≤ 0

   Pro: Simple, fast
   Con: Doesn't approximate softmax well

2. ReLU (Kasai et al., 2021):
   φ(x) = max(0, x)

   Pro: Very simple
   Con: Information loss, poor approximation

3. Random Fourier Features (Performer):
   φ(x) = [exp(x^T ω_1), exp(x^T ω_2), ..., exp(x^T ω_r)]
   where ω_i ~ N(0, I/d)

   Pro: Better softmax approximation
   Con: Need many random features (large r)

4. Learned Feature Maps (2024):
   φ(x) = MLP(x) or similar

   Pro: Can learn task-specific features
   Con: More parameters, harder to train
```

## The RNN Formulation

A profound realization: linear attention IS an RNN.

### Standard Attention (No Recurrence)

```
For each query position i:
  y_i = Σ_j softmax(q_i^T k_j) v_j
      = Σ_j [α_ij] v_j

  This requires access to ALL k_j, v_j
  → No natural recurrent form
```

### Linear Attention (Recurrent Form)

```
With feature map φ:
  y_i = Σ_j φ(q_i)^T φ(k_j) v_j / Σ_j φ(q_i)^T φ(k_j)
      = φ(q_i)^T [Σ_j φ(k_j) v_j^T] / φ(q_i)^T [Σ_j φ(k_j)]
      = φ(q_i)^T S_i / φ(q_i)^T z_i

Where:
  S_i = Σ_{j≤i} φ(k_j) v_j^T    (accumulated "memory")
  z_i = Σ_{j≤i} φ(k_j)          (normalizer)

RECURRENCE:
  S_i = S_{i-1} + φ(k_i) v_i^T
  z_i = z_{i-1} + φ(k_i)
  y_i = φ(q_i)^T S_i / φ(q_i)^T z_i
```

### Visualization

```
LINEAR ATTENTION AS RNN
───────────────────────

Input:     x_1      x_2      x_3      x_4      ...
            │        │        │        │
            ▼        ▼        ▼        ▼
Keys:      k_1      k_2      k_3      k_4
Values:    v_1      v_2      v_3      v_4
            │        │        │        │
            ▼        ▼        ▼        ▼
         ┌────┐  ┌────┐  ┌────┐  ┌────┐
State:   │ S₁ │─▶│ S₂ │─▶│ S₃ │─▶│ S₄ │─▶ ...
         │ z₁ │  │ z₂ │  │ z₃ │  │ z₄ │
         └────┘  └────┘  └────┘  └────┘
            │        │        │        │
            ▼        ▼        ▼        ▼
Queries:   q_1      q_2      q_3      q_4
            │        │        │        │
            ▼        ▼        ▼        ▼
Output:    y_1      y_2      y_3      y_4

State update: S_i = S_{i-1} + φ(k_i) ⊗ v_i
Output:       y_i = (φ(q_i)^T S_i) / (φ(q_i)^T z_i)
```

### Inference Advantage

```
SOFTMAX ATTENTION INFERENCE:
  Token 1: compute attention over [1]           → store K₁, V₁
  Token 2: compute attention over [1,2]         → store K₁,K₂, V₁,V₂
  Token 3: compute attention over [1,2,3]       → store K₁,K₂,K₃, ...
  ...
  Token n: compute attention over [1,...,n]     → O(n) KV cache

  Per-token cost: O(n)
  Total inference: O(n²)

LINEAR ATTENTION INFERENCE:
  Token 1: y₁ = f(S₁, q₁)                       → store S₁ (fixed size)
  Token 2: S₂ = S₁ + k₂v₂^T, y₂ = f(S₂, q₂)   → store S₂ (fixed size)
  Token 3: S₃ = S₂ + k₃v₃^T, y₃ = f(S₃, q₃)   → store S₃ (fixed size)
  ...

  Per-token cost: O(1)
  Total inference: O(n)
  Memory: O(d²) constant regardless of sequence length!
```

## Evolution: The Quality Gap

### The Problem

Early linear attention (2020) had a significant quality gap vs softmax:

```
BENCHMARK: Language Modeling Perplexity (lower is better)
─────────────────────────────────────────────────────────

Model                   WikiText-103   Params
────────────────────────────────────────────
Transformer (baseline)  24.1          ~150M
Linear Transformer      26.5          ~150M   (+10% worse)
Performer               27.2          ~150M   (+13% worse)
```

**Why the gap?**
1. Feature maps don't perfectly approximate softmax
2. Loss of "sharpness" - softmax can focus on few tokens; linear is smoother
3. No position-dependent weighting decay

### The Solutions: Gating and Decay

The field discovered that RNNs' **gating mechanisms** were crucial:

```
EVOLUTION OF LINEAR ATTENTION
─────────────────────────────

Basic (2020):
  S_t = S_{t-1} + k_t v_t^T

  Problem: Can only accumulate, never forget
  → State gets "polluted" with old information

With Decay (RetNet, 2023):
  S_t = γ S_{t-1} + k_t v_t^T     where γ < 1

  Pro: Automatic exponential forgetting
  Con: γ is fixed, can't do content-based forgetting

With Gating (GLA, 2024):
  S_t = G_t ⊙ S_{t-1} + k_t v_t^T    where G_t = f(x_t)

  Pro: Data-dependent forgetting!
  Con: More complex, harder to parallelize

Selective SSM (Mamba, 2023):
  Similar idea but through state space formulation
  Δ_t, B_t, C_t all depend on input
```

## Key Papers Deep Dive

### Performer (2020): FAVOR+

**Problem**: Simple feature maps (ELU+1) don't approximate softmax well.

**Solution**: Use random Fourier features that provably approximate the exponential kernel.

```
FAVOR+ (Fast Attention Via positive Orthogonal Random features)
──────────────────────────────────────────────────────────────

Observation: softmax(q^T k) = exp(q^T k) / Z
           ∝ exp(q^T k)
           ≈ φ(q)^T φ(k)

Where φ is constructed from random features:
  φ(x) = exp(-||x||²/2) / √r × [exp(ω₁^T x), ..., exp(ω_r^T x)]

  With {ω_i} sampled orthogonally from N(0, I)
```

**Trade-off**: Need many random features (r >> d) for good approximation, increasing computation.

### DeltaNet (2021): Learning to Forget

**Insight**: Linear attention can be viewed as "fast weight programming" - the state S is a "fast weight" matrix that stores associations.

**Problem**: Basic accumulation can't **overwrite** old associations.

```
DELTA RULE (from neural network learning)
─────────────────────────────────────────

Basic update:  S_t = S_{t-1} + k_t v_t^T
               (just adds new association)

Delta update:  S_t = S_{t-1} + β (v_t - S_{t-1} k_t) k_t^T
               (corrects toward desired output)

Expanding:     S_t = S_{t-1} + β v_t k_t^T - β (S_{t-1} k_t) k_t^T
                            └─────────┘   └────────────────────┘
                            Add new       Remove old association
                            association   in k_t direction
```

This allows the model to **update** memories, not just accumulate them.

### RetNet (2023): The Impossible Triangle

RetNet aimed to achieve three goals simultaneously:
1. **Training parallelism** (like Transformers)
2. **O(1) inference cost** (like RNNs)
3. **Competitive performance** (like both)

```
RETENTION MECHANISM
───────────────────

Parallel form (for training):
  Retention(X) = (QK^T ⊙ D) V

  Where D_{nm} = { γ^{n-m}  if n ≥ m
                 { 0         otherwise

  D is a causal decay mask:
  ┌                               ┐
  │  1                            │
  │  γ      1                     │
  │  γ²     γ      1              │
  │  γ³     γ²     γ      1       │
  │  ...                          │
  └                               ┘

Recurrent form (for inference):
  S_n = γ S_{n-1} + k_n v_n^T
  y_n = q_n S_n

Chunkwise form (for long training):
  - Process in chunks of size c
  - Parallel within chunk
  - Recurrent across chunks
```

**Multi-scale retention**: Use different γ values for different heads to capture different timescales.

### Mamba (2023): Selective State Spaces

Mamba's key insight: The state dynamics should be **content-dependent**.

```
STANDARD STATE SPACE MODEL
──────────────────────────
x_t = A x_{t-1} + B u_t
y_t = C x_t

A, B, C are FIXED matrices
→ Same dynamics regardless of input
→ Can't do content-based reasoning

SELECTIVE STATE SPACE (Mamba)
─────────────────────────────
x_t = A_t x_{t-1} + B_t u_t
y_t = C_t x_t

A_t, B_t, C_t DEPEND ON INPUT x_t
→ Can selectively remember/forget
→ Content-based reasoning possible

Implementation:
  Δ_t = softplus(Linear(x_t))     # step size
  B_t = Linear(x_t)                # input projection
  C_t = Linear(x_t)                # output projection
  A_t = exp(-Δ_t A)               # discretized transition
```

**The selection mechanism**:
```
For sequence: "The capital of France is ___"

Non-selective (S4):
  Processes all tokens with same dynamics
  Can't emphasize "France" or "capital"

Selective (Mamba):
  Δ_t larger for informative tokens → stronger update
  Δ_t smaller for uninformative tokens → more forgetting
  Can learn to focus on "France" and "capital"
```

### GLA (2024): Hardware-Efficient Gated Linear Attention

**Problem**: Gating mechanisms break the simple recurrence, making parallel training hard.

**Solution**: Carefully designed gating that allows efficient "chunkwise" computation.

```
GLA FORMULATION
───────────────

Output:   y_t = q_t^T S_t / q_t^T z_t

State:    S_t = Σ_{i=1}^t (Π_{j=i+1}^t g_j) k_i v_i^T

Where the cumulative gate Π_{j=i+1}^t g_j represents
"how much of token i's contribution survives to token t"

CHUNKWISE PARALLEL ALGORITHM
────────────────────────────

Divide sequence into chunks: [1..c], [c+1..2c], ...

Within chunk (parallel):
  Use matrix operations to compute all outputs

Across chunks (recurrent):
  Pass compressed state S from chunk to chunk

This gives O(n) compute with good GPU utilization!
```

## The State Space Connection

Linear attention and state space models are deeply connected:

```
LINEAR ATTENTION AS STATE SPACE
───────────────────────────────

Linear attention recurrence:
  S_t = S_{t-1} + k_t v_t^T
  y_t = q_t^T S_t

Can be written as:
  S_t = I × S_{t-1} + k_t v_t^T    (A = I, B = k_t)
  y_t = q_t^T S_t                   (C = q_t)

This IS a state space model with:
  - State: S (a d×d matrix)
  - Transition: identity (no decay)
  - Input-dependent B, C
```

**Mamba-2's insight**: They're the same thing!

```
                    Linear Attention          State Space Model
─────────────────────────────────────────────────────────────────
State shape         d × d (or d × d_v)        d_state × 1
Transition          Usually I or γI            Structured A (diagonal)
Input projection    k_t                        B_t
Output projection   q_t                        C_t
Gating              Data-dependent             Selection (Δ_t)
```

## Trade-offs Summary

```
LINEAR ATTENTION TRADE-OFFS
═══════════════════════════

                    PRO                         CON
────────────────────────────────────────────────────────────────
Complexity          O(n) vs O(n²)              Constant factors higher
────────────────────────────────────────────────────────────────
Memory              O(d²) vs O(n×d)            State can be large
(inference)         No KV cache growth!        (but fixed)
────────────────────────────────────────────────────────────────
Quality             Closing the gap            Still some gap vs softmax
                    (Mamba, GLA ~= Transformer) on some tasks
────────────────────────────────────────────────────────────────
Training            Parallel possible          Chunkwise adds complexity
                    (with care)
────────────────────────────────────────────────────────────────
Expressiveness      Can't do sharp attention   May miss fine details
                    (inherent limitation)
────────────────────────────────────────────────────────────────
Best sequence       Very long (100K+)          Moderate lengths may not
lengths                                        benefit as much
```

## 2025 Breakthrough: Kimi Linear (KDA)

### The Milestone

In October 2025, Moonshot AI released **Kimi Linear** with **Kimi Delta Attention (KDA)**, achieving a historic milestone: **linear attention that outperforms full attention under fair comparison**.

**Paper**: [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

### What is KDA?

KDA extends Gated DeltaNet with three key innovations:

```
KIMI DELTA ATTENTION (KDA)
──────────────────────────

Building on DeltaNet's delta rule:
  S_t = G_t ⊙ S_{t-1} + β(v_t - S_{t-1}k_t)k_t^T

KDA adds:

1. FINER-GRAINED GATING
   - Per-dimension gates instead of scalar
   - More expressive control over what to remember/forget

2. DIAGONAL-PLUS-LOW-RANK (DPLR) TRANSITIONS
   - State transition: A = diag(a) + uv^T
   - Captures both element-wise decay and low-rank interactions
   - More expressive than pure diagonal (Mamba) or identity

3. OPTIMIZED CHUNKWISE ALGORITHM
   - Specialized variant for DPLR
   - ~2x faster than general DPLR computation
   - Hardware-efficient for modern GPUs
```

### The Hybrid Architecture

Kimi Linear uses a **3:1 hybrid** of KDA and MLA:

```
KIMI LINEAR ARCHITECTURE
────────────────────────

Layer 1:   KDA (linear)
Layer 2:   KDA (linear)
Layer 3:   KDA (linear)
Layer 4:   MLA (global)  ← Full attention every 4th layer
Layer 5:   KDA (linear)
Layer 6:   KDA (linear)
Layer 7:   KDA (linear)
Layer 8:   MLA (global)
...

WHY HYBRID?
  - KDA handles most computation (75% of layers)
  - MLA provides global view periodically
  - Best of both worlds

Model size: 48B total, 3B active (MoE)
```

### Results

```
KIMI LINEAR VS FULL MLA (IDENTICAL TRAINING)
────────────────────────────────────────────

                    Kimi Linear    Full MLA    Delta
────────────────────────────────────────────────────
Short-context        Better         ---        ✓
Long-context         Better         ---        ✓
RL scaling           Better         ---        ✓
────────────────────────────────────────────────────
KV cache             -75%           ---        ✓
Throughput (1M ctx)  6x faster      ---        ✓
────────────────────────────────────────────────────
```

**This is historic**: For years, linear attention was "almost as good" as full attention. Kimi Linear shows it can be **better**.

### Why KDA Succeeds Where Others Struggled

```
THE LINEAR ATTENTION QUALITY GAP - SOLVED
─────────────────────────────────────────

Problem 1: Can't do sharp attention
  Old solution: Accept some quality loss
  KDA solution: Hybrid with periodic global attention (MLA)

Problem 2: Poor forgetting mechanism
  Old solution: Fixed decay γ (RetNet)
  KDA solution: Per-dimension data-dependent gating

Problem 3: Limited state expressiveness
  Old solution: Larger state (more memory)
  KDA solution: DPLR transitions (diagonal + low-rank)

Problem 4: Hardware inefficiency
  Old solution: Accept slower training
  KDA solution: Specialized chunkwise algorithm

All four problems addressed → quality matches and exceeds full attention!
```

### Open Source Release

Moonshot released:
- KDA kernel implementation
- vLLM integration
- Pre-trained and instruction-tuned checkpoints

See: [GitHub - MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)

---

## Current Frontier (Late 2025)

The landscape has shifted dramatically:

1. **Linear can beat full attention** - Kimi Linear proved it
2. **Hybrid is the winning strategy** - Mix linear + sparse global attention
3. **Hardware co-design is essential** - DPLR, chunking, FP8
4. **The quality gap is closed** - No longer a trade-off for quality

### The Remaining Questions

1. **Scaling**: Does linear attention maintain advantages at 100B+ parameters?
2. **Tasks**: Which specific tasks still benefit from full attention?
3. **Training**: Can we train linear from scratch without hybrid?
4. **Hardware**: Purpose-built accelerators for linear attention?

## References

1. [Transformers are RNNs (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236)
2. [Performer (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794)
3. [DeltaNet (Schlag et al., 2021)](https://arxiv.org/abs/2102.11174)
4. [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396)
5. [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048)
6. [RetNet (Microsoft, 2023)](https://arxiv.org/abs/2307.08621)
7. [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
8. [GLA (Yang et al., 2024)](https://arxiv.org/abs/2312.06635)
9. [Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
10. [Kimi Linear / KDA (Moonshot AI, Oct 2025)](https://arxiv.org/abs/2510.26692) - **First linear attention to beat full attention**
