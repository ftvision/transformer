"""
Lab 02: Sliding Window Attention

Implement Longformer-style sliding window attention with global tokens.

Your task: Complete the functions and class below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, List


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    global_positions: Optional[List[int]] = None
) -> np.ndarray:
    """
    Create a sliding window attention mask with optional global tokens.

    The sliding window allows each position to attend to `window_size` previous
    positions (including itself). Global positions can attend to all positions
    and all positions can attend to global positions.

    Args:
        seq_len: Length of the sequence
        window_size: Number of positions in the local window
        global_positions: List of position indices that are global
                         If None, no global tokens

    Returns:
        Boolean mask of shape (seq_len, seq_len)
        mask[i, j] = True means position i CANNOT attend to position j

    Example (seq_len=6, window_size=2, global_positions=[0]):
        Position 0 (global): can attend to all [0,1,2,3,4,5]
        Position 1: can attend to [0,1] (global + local)
        Position 2: can attend to [0,1,2] (global + local)
        Position 3: can attend to [0,2,3] (global + local)
        Position 4: can attend to [0,3,4] (global + local)
        Position 5: can attend to [0,4,5] (global + local)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create base causal mask
    # 2. Add sliding window constraint
    # 3. If global_positions provided:
    #    - Global positions can attend to all (make their row all False, except future)
    #    - All positions can attend to global positions (make those columns False, except future)
    raise NotImplementedError("Implement create_sliding_window_mask")


def sliding_window_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    window_size: int,
    global_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding window attention with optional global tokens.

    Args:
        Q: Query tensor of shape (..., seq_len, d_k)
        K: Key tensor of shape (..., seq_len, d_k)
        V: Value tensor of shape (..., seq_len, d_v)
        window_size: Size of the local attention window
        global_mask: Optional boolean array of shape (seq_len,)
                    True indicates a global position

    Returns:
        output: Attention output of shape (..., seq_len, d_v)
        attention_weights: Weights of shape (..., seq_len, seq_len)

    Note:
        This is a simplified implementation using masking.
        Production implementations use more efficient algorithms.
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get seq_len from Q shape
    # 2. Convert global_mask to list of global positions (if provided)
    # 3. Create the sliding window mask
    # 4. Compute scaled dot-product attention with the mask
    raise NotImplementedError("Implement sliding_window_attention")


class SlidingWindowAttention:
    """
    Sliding Window Multi-Head Attention with optional global tokens.

    This implements Longformer-style attention where:
    - Most tokens use local sliding window attention
    - Designated global tokens can attend to/from all positions

    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_k: Dimension per head (d_model // num_heads)
        window_size: Local attention window size
        num_global_tokens: Number of initial positions that are global
        W_Q, W_K, W_V, W_O: Projection matrices
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        num_global_tokens: int = 0
    ):
        """
        Initialize sliding window attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            window_size: Number of positions in local window
            num_global_tokens: Number of initial tokens that are global
                              (positions 0 to num_global_tokens-1)

        Raises:
            ValueError: If d_model not divisible by num_heads
        """
        # YOUR CODE HERE
        #
        # 1. Validate d_model divisible by num_heads
        # 2. Store d_model, num_heads, d_k, window_size, num_global_tokens
        # 3. Initialize weight matrices W_Q, W_K, W_V, W_O
        raise NotImplementedError("Implement __init__")

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (..., seq_len, d_model)

        Returns:
            Tensor of shape (..., num_heads, seq_len, d_k)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _split_heads")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back into d_model dimension.

        Args:
            x: Tensor of shape (..., num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _combine_heads")

    def _create_global_mask(self, seq_len: int) -> Optional[np.ndarray]:
        """
        Create boolean mask indicating global positions.

        Args:
            seq_len: Sequence length

        Returns:
            Boolean array of shape (seq_len,) where True = global position
            Returns None if num_global_tokens == 0
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _create_global_mask")

    def forward(
        self,
        x: np.ndarray,
        global_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute sliding window multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               or (seq_len, d_model) for unbatched input
            global_mask: Optional custom global mask of shape (seq_len,)
                        If None, uses default based on num_global_tokens

        Returns:
            Output tensor of same shape as input
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Handle 2D vs 3D input
        # 2. Project to Q, K, V
        # 3. Split into heads
        # 4. Determine global mask (custom or default)
        # 5. Apply sliding window attention to each head
        # 6. Combine heads
        # 7. Output projection
        raise NotImplementedError("Implement forward")

    def get_attention_weights(
        self,
        x: np.ndarray,
        global_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor
            global_mask: Optional custom global mask

        Returns:
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
            or (num_heads, seq_len, seq_len) for unbatched input
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_attention_weights")

    def __call__(
        self,
        x: np.ndarray,
        global_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(x, global_mask)


def compute_effective_context(
    seq_len: int,
    window_size: int,
    num_layers: int
) -> int:
    """
    Compute the effective context size after multiple layers.

    With sliding window attention, information can propagate across
    the window at each layer. After L layers with window size W,
    a token can potentially access information from L*W positions away.

    Args:
        seq_len: Total sequence length
        window_size: Local window size
        num_layers: Number of transformer layers

    Returns:
        Effective context size (capped at seq_len)

    Example:
        >>> compute_effective_context(1000, window_size=128, num_layers=32)
        4096  # But capped at seq_len=1000, so returns 1000
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_effective_context")


def analyze_attention_pattern(
    attention_weights: np.ndarray,
    global_mask: Optional[np.ndarray] = None
) -> dict:
    """
    Analyze attention pattern statistics.

    Args:
        attention_weights: Attention matrix of shape (seq_len, seq_len)
        global_mask: Boolean mask of global positions

    Returns:
        Dictionary with:
        - 'local_attention_ratio': Fraction of attention to local positions
        - 'global_attention_ratio': Fraction of attention to global positions
        - 'avg_entropy': Average entropy of attention distributions
        - 'sparsity': Fraction of near-zero attention weights
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement analyze_attention_pattern")
