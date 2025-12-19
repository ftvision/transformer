"""
Lab 05: Gradient Checkpointing

Implement gradient checkpointing to trade compute for memory.

Your task: Implement the checkpointing classes and functions.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, List, Callable


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard attention forward pass.

    Args:
        Q: Queries (seq_len, d_k)
        K: Keys (seq_len, d_k)
        V: Values (seq_len, d_v)
        mask: Optional attention mask

    Returns:
        output: (seq_len, d_v)
        attention_weights: (seq_len, seq_len) - stored for backward
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V

    return output, attn_weights


def attention_backward(
    grad_output: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard attention backward pass.

    Uses stored attention weights from forward pass.

    Args:
        grad_output: Gradient w.r.t. output (seq_len, d_v)
        Q, K, V: Inputs from forward pass
        attn_weights: Stored attention weights (seq_len, seq_len)

    Returns:
        grad_Q, grad_K, grad_V
    """
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # Gradient of output = attn @ V
    grad_attn = grad_output @ V.T  # (seq_len, seq_len)
    grad_V = attn_weights.T @ grad_output  # (seq_len, d_v)

    # Gradient of softmax
    # d softmax / d x = softmax * (d_out - sum(softmax * d_out))
    sum_grad_attn = np.sum(attn_weights * grad_attn, axis=-1, keepdims=True)
    grad_scores = attn_weights * (grad_attn - sum_grad_attn)  # (seq_len, seq_len)

    # Gradient of scores = Q @ K.T * scale
    grad_scores_scaled = grad_scores * scale
    grad_Q = grad_scores_scaled @ K  # (seq_len, d_k)
    grad_K = grad_scores_scaled.T @ Q  # (seq_len, d_k)

    return grad_Q, grad_K, grad_V


class CheckpointedAttention:
    """
    Attention with gradient checkpointing.

    Instead of storing the N×N attention weights during forward pass,
    we only store Q, K, V and recompute attention during backward.

    This saves O(N²) memory but requires recomputing attention.

    Attributes:
        saved_inputs: Stored inputs for recomputation
    """

    def __init__(self):
        """
        Initialize checkpointed attention.

        Example:
            >>> attn = CheckpointedAttention()
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass with checkpointing.

        Computes attention output but does NOT store attention weights.
        Only stores Q, K, V for recomputation in backward.

        Args:
            Q: Queries (seq_len, d_k)
            K: Keys (seq_len, d_k)
            V: Values (seq_len, d_v)
            mask: Optional attention mask

        Returns:
            output: (seq_len, d_v)

        Example:
            >>> attn = CheckpointedAttention()
            >>> output = attn.forward(Q, K, V)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Store Q, K, V, mask (NOT attention weights!)
        # 2. Compute attention output
        # 3. Return output only
        raise NotImplementedError("Implement forward")

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass with recomputation.

        Recomputes attention weights from stored Q, K, V,
        then computes gradients.

        Args:
            grad_output: Gradient w.r.t. output (seq_len, d_v)

        Returns:
            grad_Q, grad_K, grad_V

        Example:
            >>> output = attn.forward(Q, K, V)
            >>> grad_Q, grad_K, grad_V = attn.backward(grad_output)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Retrieve stored Q, K, V, mask
        # 2. Recompute attention weights (this is the "extra" compute)
        # 3. Use attention_backward with recomputed weights
        raise NotImplementedError("Implement backward")

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(Q, K, V, mask)


class StandardAttention:
    """
    Standard attention (for comparison).

    Stores attention weights during forward for use in backward.
    """

    def __init__(self):
        """Initialize standard attention."""
        self.saved_inputs = None
        self.saved_weights = None

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass storing attention weights."""
        output, attn_weights = attention_forward(Q, K, V, mask)
        self.saved_inputs = (Q, K, V, mask)
        self.saved_weights = attn_weights  # O(N²) storage!
        return output

    def backward(
        self,
        grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass using stored weights."""
        Q, K, V, mask = self.saved_inputs
        return attention_backward(grad_output, Q, K, V, self.saved_weights)

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Allow calling instance like a function."""
        return self.forward(Q, K, V, mask)


def measure_memory_savings(
    seq_len: int,
    d_model: int,
    num_layers: int = 1
) -> dict:
    """
    Measure theoretical memory savings from checkpointing.

    Compares memory usage between:
    - Standard: Stores all attention weights
    - Checkpointed: Only stores inputs, recomputes weights

    Args:
        seq_len: Sequence length (N)
        d_model: Model dimension (d)
        num_layers: Number of attention layers

    Returns:
        Dictionary with:
        - standard_memory_bytes: Memory for standard attention
        - checkpointed_memory_bytes: Memory for checkpointed attention
        - memory_saved_bytes: Bytes saved
        - memory_saved_pct: Percentage saved
        - compute_overhead_pct: Extra compute (approx)

    Example:
        >>> result = measure_memory_savings(2048, 512, 12)
        >>> print(f"Saved: {result['memory_saved_pct']:.1f}%")
    """
    # YOUR CODE HERE
    #
    # Memory analysis:
    #
    # Standard attention stores per layer:
    # - Q, K, V: 3 × N × d × 4 bytes
    # - Attention weights: N × N × 4 bytes
    # - Output: N × d × 4 bytes
    #
    # Checkpointed attention stores per layer:
    # - Q, K, V: 3 × N × d × 4 bytes
    # - Output: N × d × 4 bytes
    # - NO attention weights!
    #
    # Compute overhead:
    # - One extra forward pass per layer (~33% overhead)
    raise NotImplementedError("Implement measure_memory_savings")


def estimate_memory_bytes(
    seq_len: int,
    d_model: int,
    num_layers: int,
    checkpointed: bool = False
) -> int:
    """
    Estimate memory usage in bytes.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        num_layers: Number of layers
        checkpointed: Whether using checkpointing

    Returns:
        Estimated memory in bytes

    Example:
        >>> standard = estimate_memory_bytes(2048, 512, 12, False)
        >>> checkpointed = estimate_memory_bytes(2048, 512, 12, True)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement estimate_memory_bytes")


def checkpoint_sequential(
    functions: List[Callable],
    x: np.ndarray,
    checkpoint_every: int = 1
) -> np.ndarray:
    """
    Apply functions sequentially with checkpointing.

    Simplified version of PyTorch's checkpoint_sequential.

    Args:
        functions: List of functions to apply
        x: Input tensor
        checkpoint_every: Checkpoint every N functions

    Returns:
        Output tensor

    Example:
        >>> layers = [layer1, layer2, layer3, layer4]
        >>> output = checkpoint_sequential(layers, x, checkpoint_every=2)
    """
    # YOUR CODE HERE
    #
    # For simplicity, this just applies functions sequentially.
    # In a real implementation, this would:
    # 1. Group functions into segments
    # 2. Save only segment boundaries
    # 3. Recompute within segments during backward
    raise NotImplementedError("Implement checkpoint_sequential")


def verify_gradient_correctness(
    seq_len: int = 32,
    d_model: int = 16,
    seed: int = 42
) -> dict:
    """
    Verify that checkpointed gradients match standard gradients.

    Args:
        seq_len: Sequence length for test
        d_model: Model dimension for test
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - max_diff_Q: Max difference in grad_Q
        - max_diff_K: Max difference in grad_K
        - max_diff_V: Max difference in grad_V
        - all_close: Whether all gradients are close

    Example:
        >>> result = verify_gradient_correctness()
        >>> assert result['all_close'], "Gradients should match!"
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create random Q, K, V, grad_output
    # 2. Run standard attention forward/backward
    # 3. Run checkpointed attention forward/backward
    # 4. Compare gradients
    raise NotImplementedError("Implement verify_gradient_correctness")
