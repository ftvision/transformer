# Flash Linear Attention: Memory-Efficient Training

## The Problem Flash Linear Attention Solves

We've seen that linear attention can be computed in O(n) time and O(d²) memory using the recurrent form. But during training, we need:

1. **Parallel computation** - GPUs need parallelism for efficiency
2. **Memory for backward pass** - Gradients require storing intermediate values

The naive parallel implementation has O(n²) memory for the attention matrix. Flash Linear Attention achieves:

- **O(n) memory** - Linear in sequence length
- **Parallel training** - Efficient on GPUs
- **Fused operations** - Fewer memory round-trips

## Core Ideas

Flash Linear Attention builds on three key techniques:

### 1. Chunkwise Computation (from 02_chunkwise_parallel.md)

Split sequence into chunks, compute within chunks in parallel, propagate state between chunks.

### 2. Tiling for Memory Efficiency

Don't materialize the full cumulative sum tensor. Instead, tile the computation:

```
Standard:
  KV = torch.einsum('bnd,bnv->bndv', K, V)  # O(n × d_k × d_v) memory!
  S = torch.cumsum(KV, dim=1)

Flash:
  Process one chunk at a time, accumulate state incrementally
  Only materialize one chunk's worth of data at a time
```

### 3. Recomputation in Backward Pass

Instead of storing all intermediate values for backward:
- Store only the state at chunk boundaries
- Recompute intermediate values during backward pass

Trade compute for memory - similar to gradient checkpointing.

## The Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│              Flash Linear Attention - Forward Pass              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Q, K, V ∈ ℝ^{n×d}, chunk size C                        │
│  Output: O ∈ ℝ^{n×d}, saved states for backward                │
│                                                                  │
│  Initialize:                                                     │
│    S = 0 ∈ ℝ^{d_k×d_v}  (running state)                        │
│    saved_states = []                                            │
│                                                                  │
│  For chunk c = 0, 1, ..., n/C - 1:                             │
│    1. Load Q_c, K_c, V_c from HBM to SRAM                       │
│                                                                  │
│    2. Compute intra-chunk attention (in SRAM):                  │
│       For i = 0 to C-1:                                         │
│         S_local[i] = Σ_{j≤i} φ(k_j)^T v_j  (cumsum)            │
│         O_intra[i] = φ(q_i) @ S_local[i]                       │
│                                                                  │
│    3. Compute inter-chunk contribution:                         │
│       O_inter = Q_c @ S  (broadcast state to all positions)     │
│                                                                  │
│    4. Combine: O_c = O_intra + O_inter                          │
│                                                                  │
│    5. Update state: S = S + Σ_i φ(k_i)^T v_i                   │
│                                                                  │
│    6. Save state: saved_states.append(S.clone())                │
│                                                                  │
│    7. Write O_c to HBM                                          │
│                                                                  │
│  Return: O, saved_states                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Analysis

| What | Size | Stored Where |
|------|------|--------------|
| Q, K, V (input) | O(nd) | HBM |
| Output O | O(nd) | HBM |
| Running state S | O(d²) | SRAM |
| Chunk data (Q_c, K_c, V_c) | O(Cd) | SRAM |
| Saved states (backward) | O(n/C × d²) | HBM |

**Total HBM**: O(nd + n/C × d²) = O(nd) when C = O(d)

Compare to naive: O(n²) for attention matrix!

## Implementation Sketch

```python
class FlashLinearAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, chunk_size=64):
        batch, seq_len, d = Q.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Apply feature map (e.g., elu + 1)
        Q = elu(Q) + 1
        K = elu(K) + 1

        # Outputs and saved states
        O = torch.zeros_like(V)
        saved_states = []
        state = torch.zeros(batch, d, V.shape[-1], device=Q.device)

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seq_len)

            Q_c = Q[:, start:end]
            K_c = K[:, start:end]
            V_c = V[:, start:end]

            # Intra-chunk: local causal attention
            KV_c = torch.einsum('bcd,bcv->bcdv', K_c, V_c)
            S_local = torch.cumsum(KV_c, dim=1)
            O_intra = torch.einsum('bcd,bcdv->bcv', Q_c, S_local)

            # Inter-chunk: contribution from past
            O_inter = torch.einsum('bcd,bdv->bcv', Q_c, state)

            # Combined output
            O[:, start:end] = O_intra + O_inter

            # Update state
            state = state + KV_c.sum(dim=1)
            saved_states.append(state.clone())

        # Save for backward
        ctx.save_for_backward(Q, K, V)
        ctx.saved_states = saved_states
        ctx.chunk_size = chunk_size

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V = ctx.saved_tensors
        saved_states = ctx.saved_states
        chunk_size = ctx.chunk_size

        # Gradient computation (simplified)
        # In practice, this involves careful recomputation
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Process chunks in reverse for gradients
        # ... (detailed backward implementation)

        return dQ, dK, dV, None
```

## The Backward Pass Challenge

The backward pass is more complex because gradients flow in reverse:

```
Forward:  chunk 0 → chunk 1 → chunk 2 → ... → chunk N
          state propagates forward →

Backward: chunk 0 ← chunk 1 ← chunk 2 ← ... ← chunk N
          ← gradient flows backward
```

**Key insight**: We need both forward state (for recomputation) and backward gradient state.

### Backward Pass Structure

```
┌─────────────────────────────────────────────────────────────────┐
│              Flash Linear Attention - Backward Pass             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For chunk c = n/C - 1, ..., 1, 0 (reverse order):             │
│                                                                  │
│    1. Load dO_c, Q_c, K_c, V_c                                  │
│                                                                  │
│    2. Retrieve saved state S_c from forward pass                │
│                                                                  │
│    3. Recompute forward values as needed:                       │
│       - Local cumsum S_local                                    │
│       - Attention outputs (for gradient computation)            │
│                                                                  │
│    4. Compute gradients:                                        │
│       dQ_c = dO_c @ S_c^T   (gradient w.r.t. query)            │
│       dKV_c = Q_c^T @ dO_c  (gradient for state)               │
│       ...propagate to dK_c, dV_c                                │
│                                                                  │
│    5. Accumulate gradients across chunks                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison: Standard vs Flash Linear Attention

| Aspect | Standard Linear Attn | Flash Linear Attn |
|--------|---------------------|-------------------|
| Forward memory | O(nd) for cumsum | O(Cd) per chunk |
| Backward memory | O(nd) activations | O(n/C × d²) states |
| Parallelism | Full | Within-chunk |
| HBM accesses | High | Reduced |
| Implementation | Simple | Complex |

## When to Use Flash Linear Attention

**Use it when:**
- Training on long sequences (1K+ tokens)
- Memory is the bottleneck
- Using linear attention variants (GLA, RetNet, etc.)

**Consider alternatives when:**
- Short sequences (standard attention fits in memory)
- Inference only (recurrent form is simpler)
- Hardware doesn't benefit from tiling (some accelerators)

## Practical Libraries

The `fla` (Flash Linear Attention) library provides optimized implementations:

```python
# Installation
pip install fla

# Usage
from fla.ops.linear_attn import fused_chunk_linear_attn

output = fused_chunk_linear_attn(
    q=queries,
    k=keys,
    v=values,
    chunk_size=64
)
```

This library is used by models like GLA, RetNet, and Mamba-2.

## Connection to Flash Attention (Standard)

Flash Linear Attention is inspired by Flash Attention (for standard softmax attention), but differs:

| Aspect | Flash Attention | Flash Linear Attention |
|--------|----------------|----------------------|
| Target | Softmax attention | Linear attention |
| Key insight | Online softmax | State accumulation |
| Memory | O(n) | O(n/C × d²) |
| Complexity | O(n²) time | O(n) time |

Both use tiling and recomputation, but the algorithms differ due to different attention mechanisms.

## What's Next

Flash Linear Attention enables efficient training of linear attention models. But pure linear attention (additive state updates) has limited expressiveness.

In `04_gated_linear_attention.md`, we'll see how **gating mechanisms** improve the model while still allowing efficient Flash-style training.
