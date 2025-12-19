"""
Lab 02: Fused Attention in Triton

Implement a fused attention kernel combining matmul, scaling, and softmax.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Requirements:
- PyTorch 2.0+ (includes Triton)
- NVIDIA GPU with CUDA support
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def fused_attention_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    # Matrix dimensions
    seq_len,        # Sequence length (same for Q, K, V)
    d_k,            # Head dimension
    # Strides (for memory access)
    stride_q_seq, stride_q_d,    # Q strides
    stride_k_seq, stride_k_d,    # K strides
    stride_v_seq, stride_v_d,    # V strides
    stride_o_seq, stride_o_d,    # Output strides
    # Scale factor
    scale,          # 1 / sqrt(d_k)
    # Causal flag
    IS_CAUSAL: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,   # Block size for queries
    BLOCK_N: tl.constexpr,   # Block size for keys
    BLOCK_D: tl.constexpr,   # Block size for head dimension (should be >= d_k)
):
    """
    Fused attention kernel.

    Computes: output = softmax(Q @ K.T / sqrt(d_k)) @ V

    Each program computes one block of output rows (BLOCK_M queries).

    Algorithm (online softmax):
    1. Initialize accumulators for output and softmax normalization
    2. For each block of keys:
       a. Load Q block, K block, V block
       b. Compute scores = Q @ K.T * scale
       c. Apply causal mask if needed
       d. Update running max and normalizer (online softmax)
       e. Accumulate: output += softmax_weights @ V
    3. Normalize final output

    Online Softmax Key Insight:
    - Standard softmax needs max over ALL keys
    - But we process keys in blocks
    - Solution: Track running max, rescale accumulated values when max changes

    softmax(x_i) = exp(x_i - m) / Σexp(x_j - m)  where m = max(x)

    When we see a new block with larger max m':
    - Old contributions: exp(x - m) need rescaling by exp(m - m')
    - This preserves the ratio: exp(x - m) * exp(m - m') = exp(x - m')
    """
    # YOUR CODE HERE
    #
    # Step-by-step guide:
    #
    # 1. Get program ID and compute which block of queries this program handles
    #    pid_m = tl.program_id(0)
    #    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Query indices
    #
    # 2. Initialize accumulators:
    #    - acc: accumulated output, shape (BLOCK_M, BLOCK_D)
    #    - m_i: running max per query, shape (BLOCK_M,), init to -inf
    #    - l_i: running sum of exp per query, shape (BLOCK_M,), init to 0
    #
    # 3. Load Q block once (stays in registers)
    #    offs_d = tl.arange(0, BLOCK_D)
    #    q = tl.load(Q_ptr + offs_m[:, None] * stride_q_seq + offs_d[None, :] * stride_q_d,
    #                mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_k),
    #                other=0.0)
    #
    # 4. Loop over key/value blocks:
    #    for block_n in range(0, num_key_blocks):
    #        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    #
    #        # Load K block, shape (BLOCK_N, BLOCK_D)
    #        k = tl.load(...)
    #
    #        # Compute scores = Q @ K.T, shape (BLOCK_M, BLOCK_N)
    #        scores = tl.dot(q, tl.trans(k)) * scale
    #
    #        # Apply causal mask if needed
    #        if IS_CAUSAL:
    #            causal_mask = offs_m[:, None] >= offs_n[None, :]
    #            scores = tl.where(causal_mask, scores, float('-inf'))
    #
    #        # Also mask out-of-bounds keys
    #        scores = tl.where(offs_n[None, :] < seq_len, scores, float('-inf'))
    #
    #        # Online softmax update:
    #        # 1. Find new max
    #        m_ij = tl.max(scores, axis=1)  # Max per query in this block
    #        m_new = tl.maximum(m_i, m_ij)
    #
    #        # 2. Rescale old accumulator
    #        alpha = tl.exp(m_i - m_new)
    #        acc = acc * alpha[:, None]
    #        l_i = l_i * alpha
    #
    #        # 3. Compute new contribution
    #        p = tl.exp(scores - m_new[:, None])  # Softmax numerator
    #        l_i = l_i + tl.sum(p, axis=1)  # Update denominator
    #
    #        # 4. Load V and accumulate
    #        v = tl.load(...)
    #        acc = acc + tl.dot(p.to(v.dtype), v)
    #
    #        # 5. Update max
    #        m_i = m_new
    #
    # 5. Final normalization
    #    acc = acc / l_i[:, None]
    #
    # 6. Store output
    #    tl.store(Output_ptr + offs_m[:, None] * stride_o_seq + offs_d[None, :] * stride_o_d,
    #             acc,
    #             mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_k))

    raise NotImplementedError("Implement fused_attention_kernel")


def fused_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    Compute attention using fused Triton kernel.

    Args:
        Q: Query tensor of shape (seq_len, d_k) on CUDA
        K: Key tensor of shape (seq_len, d_k) on CUDA
        V: Value tensor of shape (seq_len, d_v) on CUDA (d_v == d_k for this lab)
        causal: If True, apply causal mask

    Returns:
        Output tensor of shape (seq_len, d_k)

    The kernel computes:
        output = softmax(Q @ K.T / sqrt(d_k)) @ V

    Example:
        >>> Q = torch.randn(128, 64, device='cuda')
        >>> K = torch.randn(128, 64, device='cuda')
        >>> V = torch.randn(128, 64, device='cuda')
        >>> output = fused_attention(Q, K, V, causal=True)
    """
    # Validate inputs
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All inputs must be on CUDA"
    assert Q.dim() == 2 and K.dim() == 2 and V.dim() == 2, "Inputs must be 2D"
    assert Q.shape[0] == K.shape[0] == V.shape[0], "Sequence lengths must match"
    assert Q.shape[1] == K.shape[1], "Q and K dimensions must match"
    assert K.shape[1] == V.shape[1], "K and V dimensions must match for this simplified version"

    seq_len, d_k = Q.shape
    scale = 1.0 / math.sqrt(d_k)

    # Allocate output
    output = torch.empty_like(Q)

    # YOUR CODE HERE
    #
    # 1. Choose block sizes
    #    BLOCK_M = 64  # Queries per block
    #    BLOCK_N = 64  # Keys per block
    #    BLOCK_D = triton.next_power_of_2(d_k)  # Must be >= d_k
    #
    # 2. Compute grid size
    #    num_blocks_m = triton.cdiv(seq_len, BLOCK_M)
    #    grid = (num_blocks_m,)
    #
    # 3. Launch kernel
    #    fused_attention_kernel[grid](
    #        Q, K, V, output,
    #        seq_len, d_k,
    #        Q.stride(0), Q.stride(1),
    #        K.stride(0), K.stride(1),
    #        V.stride(0), V.stride(1),
    #        output.stride(0), output.stride(1),
    #        scale,
    #        IS_CAUSAL=causal,
    #        BLOCK_M=BLOCK_M,
    #        BLOCK_N=BLOCK_N,
    #        BLOCK_D=BLOCK_D,
    #    )

    raise NotImplementedError("Implement fused_attention")

    return output


def attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    Reference attention implementation in PyTorch.

    Used for testing correctness.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d_k)

    if causal:
        seq_len = Q.shape[0]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    return weights @ V


# =============================================================================
# BONUS: Batched Multi-Head Attention (Optional)
# =============================================================================

def fused_multihead_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    BONUS: Batched multi-head attention.

    Args:
        Q: Query tensor of shape (batch, num_heads, seq_len, d_k)
        K: Key tensor of shape (batch, num_heads, seq_len, d_k)
        V: Value tensor of shape (batch, num_heads, seq_len, d_k)
        causal: If True, apply causal mask

    Returns:
        Output tensor of shape (batch, num_heads, seq_len, d_k)

    Implementation hint:
    - Flatten batch and num_heads dimensions
    - Process each (head) independently
    - Or modify the kernel to handle the extra dimensions
    """
    # YOUR CODE HERE (optional bonus challenge)
    raise NotImplementedError("Bonus: Implement batched multi-head attention")
