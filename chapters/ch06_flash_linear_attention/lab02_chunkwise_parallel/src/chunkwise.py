"""
Lab 02: Chunkwise Parallel Linear Attention

Implement the chunkwise parallel algorithm for efficient training.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable, List


def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """ELU + 1 feature map (ensures positivity)."""
    return np.where(x > 0, x + 1, np.exp(x))


def chunk_sequence(
    x: np.ndarray,
    chunk_size: int,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Split a sequence into chunks of fixed size.

    If the sequence length is not divisible by chunk_size, pad with pad_value.

    Args:
        x: Input tensor of shape (seq_len, d) or (batch, seq_len, d)
        chunk_size: Size of each chunk
        pad_value: Value to use for padding

    Returns:
        chunks: Tensor of shape (num_chunks, chunk_size, d) or
                (batch, num_chunks, chunk_size, d)
        original_len: Original sequence length (for un-padding later)

    Example:
        >>> x = np.arange(10).reshape(10, 1)
        >>> chunks, orig_len = chunk_sequence(x, chunk_size=3)
        >>> chunks.shape
        (4, 3, 1)  # 10 -> 12 (padded), 12 / 3 = 4 chunks
        >>> orig_len
        10
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement chunk_sequence")


def unchunk_sequence(
    chunks: np.ndarray,
    original_len: int
) -> np.ndarray:
    """
    Reconstruct the original sequence from chunks.

    Args:
        chunks: Tensor of shape (num_chunks, chunk_size, d) or
                (batch, num_chunks, chunk_size, d)
        original_len: Original sequence length (to remove padding)

    Returns:
        x: Reconstructed tensor of shape (original_len, d) or
           (batch, original_len, d)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement unchunk_sequence")


def intra_chunk_attention(
    Q_chunk: np.ndarray,
    K_chunk: np.ndarray,
    V_chunk: np.ndarray,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute causal linear attention within a single chunk.

    This is the "parallel" part of chunkwise - we compute attention
    within the chunk using cumulative sums.

    Args:
        Q_chunk: Queries for this chunk, shape (chunk_size, d_k) or
                 (batch, chunk_size, d_k)
        K_chunk: Keys for this chunk, same shape as Q_chunk
        V_chunk: Values for this chunk, shape (..., chunk_size, d_v)
        feature_map_fn: Feature map to apply (default: elu_plus_one)

    Returns:
        output: Attention output, same shape as V_chunk
        chunk_state: State contribution from this chunk,
                     shape (d_k, d_v) or (batch, d_k, d_v)
                     This is sum of all φ(k)^T @ v in the chunk.

    Note:
        This computes causal attention WITHIN the chunk.
        Position i in the chunk attends to positions 0..i in the chunk.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement intra_chunk_attention")


def inter_chunk_contribution(
    Q_chunk: np.ndarray,
    state: np.ndarray,
    feature_map_fn: Callable = None
) -> np.ndarray:
    """
    Compute the contribution from previous chunks' accumulated state.

    Each position in the current chunk attends to ALL positions in
    previous chunks through the accumulated state.

    Args:
        Q_chunk: Queries for this chunk, shape (chunk_size, d_k) or
                 (batch, chunk_size, d_k)
        state: Accumulated state from previous chunks,
               shape (d_k, d_v) or (batch, d_k, d_v)
        feature_map_fn: Feature map to apply (default: elu_plus_one)

    Returns:
        output: Inter-chunk contribution, same shape as would be V_chunk
                Each position gets the same state (broadcast across chunk)

    Formula:
        output[i] = φ(q_i) @ state
        (Same state for all positions in chunk)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement inter_chunk_contribution")


def chunkwise_linear_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    chunk_size: int = 64,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute causal linear attention using chunkwise parallel algorithm.

    This combines:
    - Parallel computation within chunks (GPU-friendly)
    - Sequential state passing between chunks (minimal overhead)

    Args:
        Q: Query tensor, shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Key tensor, same shape as Q
        V: Value tensor, shape (..., seq_len, d_v)
        chunk_size: Number of positions per chunk
        feature_map_fn: Feature map to apply (default: elu_plus_one)

    Returns:
        output: Attention output, same shape as V
        final_state: Final accumulated state, shape (..., d_k, d_v)

    Algorithm:
        state = 0
        outputs = []
        for each chunk c:
            # Intra-chunk: attention within chunk
            intra_out, chunk_state = intra_chunk_attention(Q[c], K[c], V[c])

            # Inter-chunk: contribution from past
            inter_out = inter_chunk_contribution(Q[c], state)

            # Combine
            outputs.append(intra_out + inter_out)

            # Update state for next chunk
            state = state + chunk_state

        return concat(outputs), state
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement chunkwise_linear_attention")


def compare_chunkwise_to_full(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    chunk_size: int = 64,
    feature_map_fn: Callable = None,
    rtol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Compare chunkwise algorithm to full linear attention.

    The chunkwise algorithm should produce identical outputs to
    processing the entire sequence at once.

    Args:
        Q, K, V: Input tensors
        chunk_size: Chunk size to use
        feature_map_fn: Feature map
        rtol: Relative tolerance

    Returns:
        match: True if outputs match
        max_diff: Maximum absolute difference
    """
    # YOUR CODE HERE
    # Import or implement full linear attention and compare
    raise NotImplementedError("Implement compare_chunkwise_to_full")
