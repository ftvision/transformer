"""
Lab 03: Tiled Attention

Implement block-by-block attention computation.

Your task: Complete the TiledAttention class to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def standard_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard attention (for comparison).
    Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V

    return output, attn_weights


class TiledAttention:
    """
    Tiled (block-by-block) attention computation.

    This implementation processes attention in blocks, never storing
    the full N×N attention matrix. It uses online softmax to
    accumulate results correctly across blocks.

    This is the core algorithm used by Flash Attention!

    Attributes:
        block_size_q: Block size for queries (B_r)
        block_size_kv: Block size for keys/values (B_c)
    """

    def __init__(
        self,
        block_size_q: int = 32,
        block_size_kv: int = 32
    ):
        """
        Initialize tiled attention.

        Args:
            block_size_q: Block size for Q blocks (B_r)
            block_size_kv: Block size for K, V blocks (B_c)

        Example:
            >>> tiled = TiledAttention(block_size_q=64, block_size_kv=64)
        """
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv

    def _process_block(
        self,
        Q_block: np.ndarray,
        K_block: np.ndarray,
        V_block: np.ndarray,
        O_block: np.ndarray,
        l_block: np.ndarray,
        m_block: np.ndarray,
        d_k: int,
        mask_block: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process one Q block against one K, V block.

        Updates the running output accumulator using online softmax.

        Args:
            Q_block: Query block (B_r, d_k)
            K_block: Key block (B_c, d_k)
            V_block: Value block (B_c, d_v)
            O_block: Current output accumulator (B_r, d_v)
            l_block: Current softmax denominator (B_r,)
            m_block: Current max values (B_r,)
            d_k: Key dimension for scaling
            mask_block: Optional mask for this block (B_r, B_c)

        Returns:
            Updated (O_block, l_block, m_block)

        The algorithm:
            1. Compute scores: S = Q_block @ K_block.T / sqrt(d_k)
            2. Apply mask if provided
            3. Find block max: m_new_block = max(S, axis=-1)
            4. Update running max: m_new = max(m_block, m_new_block)
            5. Compute rescaling factor: scale = exp(m_block - m_new)
            6. Rescale previous values: l = l * scale, O = O * scale
            7. Compute block attention: P = exp(S - m_new)
            8. Update: l += sum(P), O += P @ V_block
            9. Return updated (O, l, m_new)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _process_block")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute attention using tiled algorithm.

        This is the main entry point. It processes Q, K, V in blocks
        and never stores the full N×N attention matrix.

        Args:
            Q: Query matrix (seq_len_q, d_k) or (batch, seq_len_q, d_k)
            K: Key matrix (seq_len_k, d_k) or (batch, seq_len_k, d_k)
            V: Value matrix (seq_len_k, d_v) or (batch, seq_len_k, d_v)
            mask: Optional attention mask

        Returns:
            output: Attention output, same as standard attention

        Example:
            >>> tiled = TiledAttention(block_size_q=32, block_size_kv=32)
            >>> output = tiled.forward(Q, K, V)
        """
        # YOUR CODE HERE
        #
        # For unbatched input (2D):
        #
        # Steps:
        # 1. Get dimensions
        # 2. Initialize O = zeros, l = zeros, m = -inf (per row)
        # 3. Outer loop over K, V blocks (j)
        # 4.   Inner loop over Q blocks (i)
        # 5.     Extract blocks: Q_i, K_j, V_j
        # 6.     Get mask block if mask provided
        # 7.     Call _process_block to update O, l, m for this Q block
        # 8. Final normalization: O = O / l[:, None]
        # 9. Return O
        #
        # For batched input (3D), add batch loop or use batched operations
        raise NotImplementedError("Implement forward")

    def forward_causal(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray
    ) -> np.ndarray:
        """
        Compute causal (autoregressive) attention with tiling optimization.

        For causal attention, position i can only attend to positions <= i.
        This means we can skip entire blocks where all positions are masked!

        Block (i, j) where (i+1)*B_r <= j*B_c can be skipped entirely.

        Args:
            Q: Query matrix (seq_len, d_k) or (batch, seq_len, d_k)
            K: Key matrix (seq_len, d_k)
            V: Value matrix (seq_len, d_v)

        Returns:
            output: Causal attention output
        """
        # YOUR CODE HERE
        #
        # Same as forward, but:
        # 1. Create causal mask on-the-fly for each block
        # 2. Skip blocks where all positions are masked
        #    (when i * B_r + B_r - 1 < j * B_c, all queries come before all keys)
        raise NotImplementedError("Implement forward_causal")

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(Q, K, V, mask)


def count_blocks(
    seq_len_q: int,
    seq_len_k: int,
    block_size_q: int,
    block_size_kv: int,
    causal: bool = False
) -> int:
    """
    Count the number of blocks that need to be computed.

    Useful for understanding the computational savings from causal masking.

    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        block_size_q: Q block size
        block_size_kv: K, V block size
        causal: Whether using causal attention

    Returns:
        Number of blocks to compute

    Example:
        >>> # Full attention
        >>> count_blocks(128, 128, 32, 32, causal=False)
        16  # 4 * 4 = 16 blocks

        >>> # Causal attention (skips upper triangle)
        >>> count_blocks(128, 128, 32, 32, causal=True)
        10  # Only lower triangle blocks
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_blocks")


def verify_memory_usage(
    seq_len: int,
    d_model: int,
    block_size: int = 32
) -> dict:
    """
    Verify that tiled attention uses less memory than standard.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        block_size: Block size for tiled attention

    Returns:
        Dictionary with memory comparison:
        {
            'standard_attention_matrix': bytes for N×N matrix,
            'tiled_max_block': bytes for largest block in memory,
            'memory_reduction': ratio of standard / tiled
        }
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement verify_memory_usage")
