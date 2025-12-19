"""
Lab 03: KV Compression

Implement Key-Value compression techniques for efficient attention.

Your task: Complete the functions and classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_kv_cache_size(
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> int:
    """
    Calculate the KV-cache size in bytes.

    The KV-cache stores Key and Value tensors for all past tokens
    across all layers, used during autoregressive generation.

    Args:
        seq_len: Maximum sequence length
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

    Returns:
        Total KV-cache size in bytes

    Formula:
        2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_bytes
        (2 for K and V)

    Example:
        >>> # Llama-2 7B style
        >>> compute_kv_cache_size(4096, 32, 32, 128, 2)
        2147483648  # ~2GB
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_kv_cache_size")


def compute_compression_ratio(
    original_num_kv_heads: int,
    original_head_dim: int,
    compressed_num_kv_heads: int = None,
    compressed_dim: int = None
) -> float:
    """
    Compute the compression ratio for KV-cache.

    Args:
        original_num_kv_heads: Original number of KV heads
        original_head_dim: Original head dimension
        compressed_num_kv_heads: Compressed number of KV heads (for GQA)
        compressed_dim: Compressed dimension (for low-rank)

    Returns:
        Compression ratio (original_size / compressed_size)

    Example:
        >>> # GQA: 32 heads -> 4 heads
        >>> compute_compression_ratio(32, 128, compressed_num_kv_heads=4)
        8.0

        >>> # Low-rank: 4096 -> 512
        >>> compute_compression_ratio(32, 128, compressed_dim=512)
        8.0
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_compression_ratio")


def grouped_query_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_kv_groups: int,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Grouped-Query Attention.

    In GQA, multiple query heads share the same KV head.
    This reduces KV-cache size while maintaining query expressiveness.

    Args:
        Q: Query tensor of shape (batch, num_q_heads, seq_len, head_dim)
           or (num_q_heads, seq_len, head_dim)
        K: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
           or (num_kv_heads, seq_len, head_dim)
        V: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim)
           or (num_kv_heads, seq_len, head_dim)
        num_kv_groups: Number of KV groups (num_q_heads // num_kv_heads)
        mask: Optional attention mask

    Returns:
        output: Attention output of shape (..., num_q_heads, seq_len, head_dim)
        weights: Attention weights of shape (..., num_q_heads, seq_len, seq_len)

    Example:
        8 Q heads, 2 KV heads, num_kv_groups = 4
        Q heads [0,1,2,3] share K head 0, V head 0
        Q heads [4,5,6,7] share K head 1, V head 1
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Expand K and V to match Q's number of heads
    #    - Use repeat_interleave or tile to expand num_kv_heads -> num_q_heads
    # 2. Compute standard scaled dot-product attention
    # 3. Return output and weights
    raise NotImplementedError("Implement grouped_query_attention")


def compress_kv(
    K: np.ndarray,
    V: np.ndarray,
    W_down: np.ndarray
) -> np.ndarray:
    """
    Compress K and V into a low-dimensional latent representation.

    Args:
        K: Key tensor of shape (..., seq_len, kv_dim)
        V: Value tensor of shape (..., seq_len, kv_dim)
        W_down: Down-projection matrix of shape (2 * kv_dim, d_latent)

    Returns:
        kv_latent: Compressed representation of shape (..., seq_len, d_latent)

    The compression concatenates K and V, then projects down:
        kv_concat = concat(K, V, axis=-1)  # shape: (..., seq_len, 2*kv_dim)
        kv_latent = kv_concat @ W_down     # shape: (..., seq_len, d_latent)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compress_kv")


def decompress_kv(
    kv_latent: np.ndarray,
    W_up_k: np.ndarray,
    W_up_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompress latent representation back to K and V.

    Args:
        kv_latent: Compressed representation of shape (..., seq_len, d_latent)
        W_up_k: Up-projection for K of shape (d_latent, kv_dim)
        W_up_v: Up-projection for V of shape (d_latent, kv_dim)

    Returns:
        K: Reconstructed keys of shape (..., seq_len, kv_dim)
        V: Reconstructed values of shape (..., seq_len, kv_dim)

    The decompression projects the latent back up:
        K = kv_latent @ W_up_k
        V = kv_latent @ W_up_v
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement decompress_kv")


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA).

    GQA uses fewer KV heads than query heads, reducing KV-cache size
    while maintaining model quality.

    Used in: Llama-2 70B, Mistral, Gemma

    Attributes:
        d_model: Model dimension
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        num_kv_groups: Query heads per KV head
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int
    ):
        """
        Initialize Grouped-Query Attention.

        Args:
            d_model: Model dimension
            num_q_heads: Number of query heads
            num_kv_heads: Number of KV heads (must divide num_q_heads)

        Raises:
            ValueError: If num_q_heads not divisible by num_kv_heads
            ValueError: If d_model not divisible by num_q_heads
        """
        # YOUR CODE HERE
        #
        # 1. Validate divisibility constraints
        # 2. Store dimensions
        # 3. Calculate num_kv_groups = num_q_heads // num_kv_heads
        # 4. Initialize projection matrices:
        #    - W_Q: (d_model, num_q_heads * head_dim)
        #    - W_K: (d_model, num_kv_heads * head_dim)  # Smaller!
        #    - W_V: (d_model, num_kv_heads * head_dim)  # Smaller!
        #    - W_O: (num_q_heads * head_dim, d_model)
        raise NotImplementedError("Implement __init__")

    def _split_heads(self, x: np.ndarray, num_heads: int) -> np.ndarray:
        """Split the last dimension into (num_heads, head_dim)."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back into a single dimension."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute grouped-query attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Project Q, K, V (note: K, V have fewer heads)
        # 2. Split into heads
        # 3. Apply grouped_query_attention
        # 4. Combine heads
        # 5. Output projection
        raise NotImplementedError("Implement forward")

    def get_kv_cache_size(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """Calculate KV-cache size for this layer."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_kv_cache_size")

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.forward(x, mask)


class LowRankKVAttention:
    """
    Attention with low-rank KV compression.

    Instead of storing full K and V, we store a compressed latent
    representation and decompress on-the-fly during attention.

    This is a simplified version of DeepSeek MLA.

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        d_latent: Dimension of compressed KV representation
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int
    ):
        """
        Initialize low-rank KV attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_latent: Dimension of compressed KV (should be < num_heads * head_dim)

        Raises:
            ValueError: If d_model not divisible by num_heads
        """
        # YOUR CODE HERE
        #
        # 1. Validate dimensions
        # 2. Store d_model, num_heads, head_dim, d_latent
        # 3. Initialize projection matrices:
        #    - W_Q: (d_model, num_heads * head_dim)
        #    - W_down: (d_model, d_latent)  # Compress input to latent
        #    - W_up_k: (d_latent, num_heads * head_dim)  # Decompress to K
        #    - W_up_v: (d_latent, num_heads * head_dim)  # Decompress to V
        #    - W_O: (num_heads * head_dim, d_model)
        raise NotImplementedError("Implement __init__")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple heads."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention with low-rank KV compression.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model)
            kv_cache: Optional cached latent from previous tokens
                      Shape: (batch, prev_seq_len, d_latent)
            mask: Optional attention mask

        Returns:
            output: Attention output of same shape as input
            new_kv_cache: Updated KV cache (latent representation)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Project Q normally
        # 2. Compress input to latent: kv_latent = x @ W_down
        # 3. Update cache (concatenate with previous cache if provided)
        # 4. Decompress to K and V: K = cache @ W_up_k, V = cache @ W_up_v
        # 5. Split into heads
        # 6. Compute attention
        # 7. Combine heads and output projection
        raise NotImplementedError("Implement forward")

    def get_cache_size(self, seq_len: int, dtype_bytes: int = 2) -> int:
        """Calculate compressed KV-cache size."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_cache_size")

    def __call__(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x, kv_cache, mask)


def compare_memory_usage(
    seq_len: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    dtype_bytes: int = 2,
    gqa_kv_heads: int = None,
    lowrank_d_latent: int = None
) -> dict:
    """
    Compare memory usage across different attention variants.

    Args:
        seq_len: Sequence length
        num_layers: Number of layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dtype_bytes: Bytes per element
        gqa_kv_heads: Number of KV heads for GQA (None = skip)
        lowrank_d_latent: Latent dimension for low-rank (None = skip)

    Returns:
        Dictionary with memory usage for each variant:
        {
            'mha': int,  # Standard MHA
            'gqa': int,  # GQA (if gqa_kv_heads provided)
            'lowrank': int,  # Low-rank (if lowrank_d_latent provided)
            'gqa_ratio': float,  # MHA / GQA
            'lowrank_ratio': float,  # MHA / low-rank
        }
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_memory_usage")
