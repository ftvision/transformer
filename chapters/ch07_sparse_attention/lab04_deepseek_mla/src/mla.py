"""
Lab 04: DeepSeek MLA (Multi-head Latent Attention)

Implement Multi-head Latent Attention for efficient KV caching.

Your task: Complete the class below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Optional
import math


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: (..., seq_len_q, d_k)
        K: (..., seq_len_k, d_k)
        V: (..., seq_len_k, d_v)
        mask: Optional boolean mask, True = blocked

    Returns:
        output: (..., seq_len_q, d_v)
        weights: (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, float('-inf'), scores)

    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights


class MultiHeadLatentAttention:
    """
    Multi-head Latent Attention (MLA) from DeepSeek-V2.

    MLA compresses Key-Value pairs into a compact latent representation,
    dramatically reducing KV-cache memory while maintaining quality.

    Architecture:
        Q = X @ W_Q                    (standard query projection)
        c_KV = X @ W_DKV               (compress to latent)
        K = c_KV @ W_UK                (decompress keys)
        V = c_KV @ W_UV                (decompress values)
        output = Attention(Q, K, V)

    During inference, we cache c_KV (small) instead of K, V (large).

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        d_latent: Dimension of compressed KV latent
        W_Q: Query projection (d_model, num_heads * head_dim)
        W_DKV: Down-projection to latent (d_model, d_latent)
        W_UK: Up-projection for keys (d_latent, num_heads * head_dim)
        W_UV: Up-projection for values (d_latent, num_heads * head_dim)
        W_O: Output projection (num_heads * head_dim, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        head_dim: Optional[int] = None
    ):
        """
        Initialize Multi-head Latent Attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_latent: Dimension of compressed KV representation
                     Should be < num_heads * head_dim for memory savings
            head_dim: Dimension per head (default: d_model // num_heads)

        Raises:
            ValueError: If d_model not divisible by num_heads when head_dim not specified
        """
        # YOUR CODE HERE
        #
        # 1. Set head_dim (default to d_model // num_heads)
        # 2. Validate dimensions
        # 3. Store d_model, num_heads, head_dim, d_latent
        # 4. Initialize weight matrices:
        #    - W_Q: (d_model, num_heads * head_dim)
        #    - W_DKV: (d_model, d_latent) - compress to latent
        #    - W_UK: (d_latent, num_heads * head_dim) - decompress K
        #    - W_UV: (d_latent, num_heads * head_dim) - decompress V
        #    - W_O: (num_heads * head_dim, d_model)
        #
        # Use np.random.randn(...).astype(np.float32) * 0.02 for initialization
        raise NotImplementedError("Implement __init__")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, head_dim).

        Args:
            x: (..., seq_len, num_heads * head_dim)

        Returns:
            (..., num_heads, seq_len, head_dim)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back into single dimension.

        Args:
            x: (..., num_heads, seq_len, head_dim)

        Returns:
            (..., seq_len, num_heads * head_dim)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def compress_kv(self, x: np.ndarray) -> np.ndarray:
        """
        Compress input to KV latent representation.

        Args:
            x: Input tensor (..., seq_len, d_model)

        Returns:
            c_KV: Compressed latent (..., seq_len, d_latent)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement compress_kv")

    def decompress_kv(
        self,
        c_kv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompress latent to K and V.

        Args:
            c_kv: Compressed latent (..., seq_len, d_latent)

        Returns:
            K: Keys (..., seq_len, num_heads * head_dim)
            V: Values (..., seq_len, num_heads * head_dim)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement decompress_kv")

    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Multi-head Latent Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model) for unbatched
            kv_cache: Optional cached latent from previous tokens
                     Shape: (batch, prev_seq_len, d_latent)
            mask: Optional attention mask

        Returns:
            output: Attention output, same shape as x
            new_kv_cache: Updated latent cache for next iteration
                         Shape: (batch, total_seq_len, d_latent)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Handle 2D vs 3D input (add batch dim if needed)
        # 2. Compute Q = x @ W_Q
        # 3. Compress new KV: c_kv_new = x @ W_DKV
        # 4. Update cache:
        #    - If kv_cache is None: c_kv = c_kv_new
        #    - Else: c_kv = concat(kv_cache, c_kv_new, axis=seq_dim)
        # 5. Decompress: K, V = decompress_kv(c_kv)
        # 6. Split Q, K, V into heads
        # 7. Compute attention
        # 8. Combine heads
        # 9. Output projection: output = combined @ W_O
        # 10. Return (output, c_kv)
        raise NotImplementedError("Implement forward")

    def get_attention_weights(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor
            kv_cache: Optional cached latent
            mask: Optional attention mask

        Returns:
            Attention weights (..., num_heads, seq_len_q, seq_len_k)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_attention_weights")

    def get_cache_size_bytes(
        self,
        seq_len: int,
        batch_size: int = 1,
        dtype_bytes: int = 2
    ) -> int:
        """
        Calculate KV cache size in bytes.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for fp16)

        Returns:
            Cache size in bytes
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_cache_size_bytes")

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio vs standard MHA.

        Returns:
            Ratio of standard MHA KV size to MLA cache size
        """
        # YOUR CODE HERE
        #
        # Standard MHA: 2 * num_heads * head_dim (for K and V)
        # MLA: d_latent
        # Ratio = (2 * num_heads * head_dim) / d_latent
        raise NotImplementedError("Implement get_compression_ratio")

    def __call__(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x, kv_cache, mask)


def compare_mha_vs_mla(
    d_model: int,
    num_heads: int,
    d_latent: int,
    seq_len: int,
    num_layers: int,
    dtype_bytes: int = 2
) -> dict:
    """
    Compare memory usage between standard MHA and MLA.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_latent: MLA latent dimension
        seq_len: Sequence length
        num_layers: Number of layers
        dtype_bytes: Bytes per element

    Returns:
        Dictionary with:
        - 'mha_cache_bytes': Standard MHA KV cache size
        - 'mla_cache_bytes': MLA latent cache size
        - 'compression_ratio': MHA / MLA
        - 'mha_cache_gb': MHA size in GB
        - 'mla_cache_gb': MLA size in GB
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_mha_vs_mla")


class MLAWithRoPE(MultiHeadLatentAttention):
    """
    MLA with Rotary Position Embedding (RoPE) support.

    DeepSeek-V2 uses decoupled RoPE where:
    - A portion of Q/K is used for positional encoding
    - The rest comes from the compressed latent

    This is a simplified version that applies RoPE after decompression.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_latent: int,
        head_dim: Optional[int] = None,
        rope_dim: Optional[int] = None
    ):
        """
        Initialize MLA with RoPE.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_latent: Latent dimension
            head_dim: Dimension per head
            rope_dim: Dimension for RoPE (default: head_dim // 2)
        """
        super().__init__(d_model, num_heads, d_latent, head_dim)
        # YOUR CODE HERE
        # Store rope_dim (default to head_dim // 2)
        raise NotImplementedError("Implement __init__ for MLAWithRoPE")

    def _apply_rope(
        self,
        x: np.ndarray,
        position_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply Rotary Position Embedding.

        Args:
            x: Tensor of shape (..., seq_len, dim)
            position_ids: Position indices (default: 0, 1, 2, ...)

        Returns:
            Tensor with RoPE applied
        """
        # YOUR CODE HERE
        # This is a simplified RoPE implementation
        # Full RoPE covered in Chapter 2
        raise NotImplementedError("Implement _apply_rope")

    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[np.ndarray] = None,
        position_offset: int = 0,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward with RoPE.

        Args:
            x: Input tensor
            kv_cache: Cached latent
            position_offset: Starting position for RoPE
            mask: Attention mask

        Returns:
            output, new_cache
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward for MLAWithRoPE")
