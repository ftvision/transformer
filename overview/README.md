# Attention Evolution: From Quadratic to Efficient

This directory explores the evolution of attention mechanisms from the original Transformer to modern efficient variants. Two major paths emerged to address the fundamental O(n²) complexity problem: **Linear Attention** and **Sparse Attention**.

**The 2025 verdict**: Both paths have now demonstrated they can **match or exceed** full attention quality, ending years of debate about whether efficient attention requires quality trade-offs.

## The Common Root: The Quadratic Bottleneck

The original Transformer ("Attention Is All You Need", Vaswani et al., 2017) revolutionized sequence modeling but introduced a fundamental scaling problem:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The `QK^T` matrix multiplication creates an **n × n attention matrix**, where n is sequence length. This leads to:
- **O(n²) time complexity** - doubling sequence length = 4x compute
- **O(n²) memory complexity** - must store the full attention matrix
- **Practical limit**: ~4K-8K tokens on typical hardware

For long sequences (documents, code, audio, DNA), this becomes prohibitive.

## The Divergence: Two Philosophical Approaches

Both linear and sparse attention address the quadratic bottleneck, but they make fundamentally different trade-offs:

| Aspect | Linear Attention | Sparse Attention |
|--------|------------------|------------------|
| **Core idea** | Change the math to avoid n×n matrix | Keep full attention, but only compute subset |
| **Key technique** | Kernel trick + associativity | Pattern masks + learned selection |
| **Complexity** | O(n) | O(n√n) to O(n) |
| **Inference** | RNN-like recurrent form (O(1) per token) | Still requires attention computation |
| **KV Cache** | Can be O(d²) constant | Reduced but still O(n), or compressed (MLA) |
| **2025 state** | KDA beats full attention | DSA/NSA beat full attention |

## Visual Evolution Map

```
                            ┌─────────────────────────────────────┐
                            │     Attention Is All You Need       │
                            │        (Vaswani et al., 2017)       │
                            │           O(n²) attention           │
                            └─────────────────┬───────────────────┘
                                              │
                                              │ Problem: Quadratic complexity
                                              │ limits sequence length
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
                    ▼                                                   ▼
    ┌───────────────────────────────┐             ┌───────────────────────────────┐
    │      SPARSE ATTENTION         │             │      LINEAR ATTENTION         │
    │   "Keep attention exact,      │             │   "Change the math to         │
    │    but compute less of it"    │             │    avoid n×n entirely"        │
    └───────────────┬───────────────┘             └───────────────┬───────────────┘
                    │                                             │
    ┌───────────────┴───────────────┐             ┌───────────────┴───────────────┐
    │                               │             │                               │
    ▼                               ▼             ▼                               ▼
┌─────────┐                   ┌─────────┐   ┌─────────┐                   ┌─────────┐
│ Pattern │                   │  KV     │   │ Kernel  │                   │  State  │
│  Based  │                   │ Cache   │   │  Trick  │                   │  Space  │
│ (2019)  │                   │ Compress│   │ (2020)  │                   │  Models │
└────┬────┘                   └────┬────┘   └────┬────┘                   └────┬────┘
     │                             │             │                             │
     ▼                             ▼             ▼                             ▼
┌──────────┐               ┌──────────┐   ┌──────────┐               ┌──────────┐
│Sparse    │               │  MLA     │   │Linear    │               │  S4      │
│Transform.│               │ (2024)   │   │Transform.│               │  Mamba   │
│Longformer│               │  GQA     │   │Performer │               │  RWKV    │
│BigBird   │               │          │   │RetNet    │               │  GLA     │
│Mistral   │               │          │   │          │               │          │
└────┬─────┘               └────┬─────┘   └────┬─────┘               └────┬─────┘
     │                          │              │                          │
     └────────────┬─────────────┘              └────────────┬─────────────┘
                  │                                         │
                  ▼                                         ▼
        ┌─────────────────┐                       ┌─────────────────┐
        │   2025 SPARSE   │                       │   2025 LINEAR   │
        │   ─────────────  │                       │   ─────────────  │
        │ NSA (Feb)       │                       │ Kimi KDA (Oct)  │
        │ DSA (Sep-Dec)   │                       │ Kimi Linear     │
        │ ✓ Beats full!   │                       │ ✓ Beats full!   │
        └────────┬────────┘                       └────────┬────────┘
                 │                                         │
                 └─────────────────┬───────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │  HYBRID (2025)  │
                         │  KDA + MLA      │
                         │  DSA + MLA      │
                         │  Best of both   │
                         └─────────────────┘
```

## The Journey: Key Milestones

### Phase 1: Exploration (2019-2020)

The field recognized the quadratic problem and began exploring solutions:

| Paper | Approach | Key Insight |
|-------|----------|-------------|
| **Sparse Transformer** (OpenAI, 2019) | Fixed sparse patterns | Learned attention is naturally sparse |
| **Linear Transformer** (EPFL, 2020) | Kernel trick | Reorder computation via associativity |
| **Performer** (Google, 2020) | Random features | Approximate softmax with random Fourier |
| **Longformer** (AllenAI, 2020) | Sliding window + global | Task-specific global tokens |
| **BigBird** (Google, 2020) | Local + global + random | Theoretically universal approximator |

**The challenge**: Early approaches showed promise but had noticeable quality gaps vs full attention.

### Phase 2: Scaling Up (2022-2023)

State space models emerged, and efficient attention reached production:

| Paper | Approach | Key Insight |
|-------|----------|-------------|
| **Flash Attention** (2022) | Hardware optimization | Tiling for memory efficiency (still O(n²) compute) |
| **S4** (2022) | Structured state spaces | Continuous-time view, structured matrices |
| **RWKV** (2023) | Linear attention at scale | 14B params, competitive with Transformers |
| **RetNet** (2023) | Retention mechanism | Explicit decay, parallel/recurrent/chunkwise forms |
| **Mistral** (2023) | Sliding window | Production LLM with sparse attention |
| **Mamba** (2023) | Selective SSM | **Input-dependent** state dynamics |

**The shift**: Models began matching Transformer quality, but the debate continued.

### Phase 3: Resolution (2024-2025)

The question "Can efficient attention match full attention?" was definitively answered: **Yes**.

| Paper | Approach | Key Result |
|-------|----------|------------|
| **DeepSeek-V2 MLA** (2024) | KV compression | 93% cache reduction, quality maintained |
| **GLA** (2024) | Gated linear attention | Hardware-efficient, closing the gap |
| **NSA** (DeepSeek, Feb 2025) | Native sparse attention | **Outperforms** full attention (+8.7% multi-hop QA) |
| **Kimi Linear/KDA** (Oct 2025) | Delta attention + DPLR | **First linear attention to beat full attention** |
| **DSA** (DeepSeek, Dec 2025) | Lightning indexer | 70% cost reduction, GPT-5 competitive |

**The counterpoint**: MiniMax M2 (Oct 2025) chose full attention, arguing efficient methods weren't ready. However, this decision predated Kimi Linear and NSA results.

## The 2025 Consensus

```
┌────────────────────────────────────────────────────────────────────────┐
│                        2025 STATE OF THE ART                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  LINEAR ATTENTION PATH                 SPARSE ATTENTION PATH           │
│  ─────────────────────                 ─────────────────────           │
│                                                                        │
│  Kimi Linear (KDA)                     DeepSeek NSA                    │
│  • First to beat full attention        • Three-path: compress+select  │
│  • 3:1 KDA:MLA hybrid                    +sliding window              │
│  • 75% KV cache reduction              • Outperforms full attention   │
│  • 6x throughput at 1M context         • Hardware-aligned design      │
│                                                                        │
│  Key: DPLR transitions +               DeepSeek DSA (V3.2)            │
│       per-dimension gating             • Lightning indexer (FP8)      │
│                                        • Top-2048 token selection     │
│                                        • 70% cost reduction           │
│                                        • Surpasses GPT-5              │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                         EMERGING PRINCIPLES                            │
│                                                                        │
│  1. HYBRID WINS: Pure linear or pure sparse < hybrid approaches       │
│     • KDA (linear) + MLA (global) every 4th layer                     │
│     • DSA two-stage: cheap indexer → expensive attention subset       │
│                                                                        │
│  2. HARDWARE CO-DESIGN IS ESSENTIAL                                   │
│     • DPLR chunkwise algorithms (Kimi)                                │
│     • FP8 lightning indexer (DeepSeek)                                │
│     • Sparse kernels in FlashMLA                                      │
│                                                                        │
│  3. LEARNED > FIXED                                                   │
│     • Learned token selection (DSA) > fixed patterns (Longformer)    │
│     • Data-dependent gating (KDA) > fixed decay (RetNet)             │
│                                                                        │
│  4. QUALITY GAP IS CLOSED                                             │
│     • No longer a trade-off: efficient attention CAN be better       │
│     • The question now is WHEN to use which, not IF                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## The Transparency Gap: Closed-Source Frontier Models

While open models like DeepSeek and Kimi publish detailed papers on their attention mechanisms, closed-source frontier models from Google, Anthropic, and OpenAI disclose minimal architectural details. Here's what we know (and don't know) as of December 2025:

### What's Publicly Known

| Model | Context Length | Attention Mechanism | Source |
|-------|---------------|---------------------|--------|
| **Gemini 3.0 Pro/Flash** | 1M tokens | Sparse MoE + Multi-Query Attention (MQA), cross-modal attention | Blog posts, limited technical details |
| **Claude Opus 4.5** | 200K (base) | "Adaptive gating mechanism", "code-optimized attention layers" | Marketing materials only |
| **Claude Sonnet 4.5** | 1M tokens | Unknown (2x pricing for >200K suggests computational step-change) | Pricing structure, no technical disclosure |
| **GPT-5 / GPT-5.2** | ~128K | Unknown | No architectural disclosure |

### Inferences from Pricing and Behavior

```
┌────────────────────────────────────────────────────────────────────────┐
│                    READING THE TEA LEAVES                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Claude Sonnet 4.5 1M Context                                         │
│  • 2x premium for requests >200K tokens                               │
│  • Suggests a computational discontinuity at 200K boundary            │
│  • Possible interpretation: hybrid approach that switches modes       │
│    (e.g., efficient attention kicks in beyond 200K)                   │
│                                                                        │
│  Gemini 1M Context (since Gemini 1.5)                                 │
│  • Sustained 1M context since early 2024                              │
│  • Multi-Query Attention confirmed (KV cache optimization)            │
│  • Cross-modal attention for multimodal inputs                        │
│  • Likely uses some form of sparse or hierarchical attention          │
│                                                                        │
│  GPT-5 Series                                                         │
│  • Minimal disclosure on attention architecture                       │
│  • DeepSeek claims DSA "surpasses GPT-5" - implies comparison exists │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### The Disclosure Spectrum

```
FULL DISCLOSURE ◄─────────────────────────────────────────► MINIMAL DISCLOSURE

DeepSeek        Kimi           MiniMax         Gemini         Claude/OpenAI
├───────────────┼──────────────┼───────────────┼──────────────┼──────────────┤
│               │              │               │              │              │
│ Full papers   │ Full papers  │ Technical     │ Blog posts   │ Marketing    │
│ Open weights  │ Open weights │ report        │ Some arch    │ only         │
│ Training code │ Training code│ Open weights  │ details      │ No arch      │
│               │              │               │              │ details      │
│               │              │               │              │              │
│ DSA paper     │ KDA paper    │ M2 report     │ "MQA"        │ "Adaptive    │
│ NSA paper     │ Linear paper │               │ "Cross-modal"│  gating"     │
│ MLA paper     │              │               │              │              │
```

### Why This Matters

1. **For Researchers**: Open models provide reproducible baselines and clear ablations. Closed models force reliance on benchmarks alone.

2. **For Practitioners**: Understanding attention mechanisms helps predict:
   - Cost scaling with context length
   - Quality on long-context tasks
   - Latency characteristics

3. **For the Field**: The 2025 breakthroughs (KDA beating full attention, DSA achieving 70% cost reduction) came from open research. Closed models benefit from but don't contribute to this shared knowledge.

### Practical Implications

| If you need... | Open models tell you | Closed models tell you |
|---------------|---------------------|----------------------|
| Cost at 500K context | Exact compute formula from papers | Pricing table (input) |
| Quality on needle-in-haystack | Ablation studies, attention maps | Benchmark numbers |
| Why performance degrades | Architecture analysis possible | Black box |
| Adaptation for your use case | Full weight access, fine-tuning | API-only access |

**Bottom line**: For understanding *how* modern attention works, study DeepSeek and Kimi. For using frontier capabilities regardless of mechanism, closed models remain competitive options.

## Quick Reference: When to Use What

### By Sequence Length

| Context Length | Recommended Approach | Examples |
|----------------|---------------------|----------|
| < 4K | Full attention + Flash Attention | Standard tasks |
| 4K - 32K | Sliding window or MLA | Mistral, DeepSeek-V2 |
| 32K - 128K | DSA or KDA+MLA hybrid | DeepSeek-V3.2, Kimi Linear |
| > 128K | Linear attention (KDA) or aggressive DSA | Kimi Linear (1M context) |

### By Use Case

| Use Case | Best Approach | Why |
|----------|---------------|-----|
| **Streaming/real-time** | Linear (KDA) | O(1) per-token inference |
| **Long document QA** | DSA | Learned selection finds relevant tokens |
| **Code generation** | Hybrid (KDA+MLA) | Needs both local and global patterns |
| **Multi-hop reasoning** | NSA/DSA | Proven to outperform full attention |
| **Memory constrained** | MLA | 93% KV cache reduction |
| **Maximum quality** | Hybrid (any) | No longer need to compromise |

## Document Structure

```
overview/
├── README.md                    # This file - high-level overview
├── evolution_timeline.md        # Chronological journey with paper details
├── linear_attention_path.md     # Deep dive: kernel trick → KDA
└── sparse_attention_path.md     # Deep dive: patterns → DSA
```

## Detailed Documents

- **[Evolution Timeline](evolution_timeline.md)** - Chronological journey from 2017 to 2025
- **[Linear Attention Path](linear_attention_path.md)** - From kernel trick to Kimi KDA
- **[Sparse Attention Path](sparse_attention_path.md)** - From patterns to DeepSeek DSA

## References

### Foundational Papers (2017-2020)
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - The original Transformer
- [Generating Long Sequences with Sparse Transformers (2019)](https://arxiv.org/abs/1904.10509) - OpenAI's Sparse Transformer
- [Transformers are RNNs (2020)](https://arxiv.org/abs/2006.16236) - Linear attention foundation
- [Performer (2020)](https://arxiv.org/abs/2009.14794) - Random Fourier features
- [Longformer (2020)](https://arxiv.org/abs/2004.05150) - Sliding window + global attention
- [BigBird (2020)](https://arxiv.org/abs/2007.14062) - Random + local + global patterns

### Scaling Era (2021-2023)
- [DeltaNet (2021)](https://arxiv.org/abs/2102.11174) - Delta rule for linear attention
- [S4 (2022)](https://arxiv.org/abs/2111.00396) - Structured state spaces
- [Flash Attention (2022)](https://arxiv.org/abs/2205.14135) - Hardware-efficient attention
- [RWKV (2023)](https://arxiv.org/abs/2305.13048) - Linear attention at 14B scale
- [RetNet (2023)](https://arxiv.org/abs/2307.08621) - Retention mechanism
- [Mistral 7B (2023)](https://arxiv.org/abs/2310.06825) - Sliding window in production
- [Mamba (2023)](https://arxiv.org/abs/2312.00752) - Selective state spaces

### Resolution Era (2024-2025)
- [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) - Multi-head Latent Attention (MLA)
- [GLA (2024)](https://arxiv.org/abs/2312.06635) - Gated Linear Attention
- [Mamba-2 (2024)](https://arxiv.org/abs/2405.21060) - State space duality
- [Native Sparse Attention (Feb 2025)](https://arxiv.org/abs/2502.11089) - NSA outperforms full attention
- [Kimi Linear (Oct 2025)](https://arxiv.org/abs/2510.26692) - KDA beats full attention
- [DeepSeek-V3.2 (Dec 2025)](https://arxiv.org/abs/2512.02556) - DSA with lightning indexer

### Production Models (2025)
- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2) - 1T MoE with MLA, 256K context, agentic
- [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear) - 48B MoE with KDA, 1M context
- [MiniMax M2](https://github.com/MiniMax-AI/MiniMax-M2) - 230B MoE, full attention (counterpoint)
- [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) - DSA, GPT-5 competitive
