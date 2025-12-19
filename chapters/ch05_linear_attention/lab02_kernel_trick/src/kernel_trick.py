"""
Lab 02: The Kernel Trick

Implement the associativity trick that enables O(n) attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Callable, Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def standard_attention_order(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention in standard order: (QK^T)V

    This explicitly forms the (seq_len, seq_len) attention matrix.
    Complexity: O(n²d) where n = seq_len, d = dimension

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        scale: Whether to scale by sqrt(d_k) and apply softmax

    Returns:
        Tuple of:
        - output: Attention output of shape (seq_len, d_v)
        - attention_matrix: The (seq_len, seq_len) attention weights

    Example:
        >>> Q = np.random.randn(100, 64)
        >>> K = np.random.randn(100, 64)
        >>> V = np.random.randn(100, 64)
        >>> output, attn = standard_attention_order(Q, K, V)
        >>> attn.shape
        (100, 100)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute attention scores: Q @ K.T -> (n, n)
    # 2. If scale=True: scale by sqrt(d_k) and apply softmax
    # 3. Compute output: attention_matrix @ V -> (n, d_v)
    # 4. Return (output, attention_matrix)
    raise NotImplementedError("Implement standard_attention_order")


def linear_attention_order(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute attention in linear order: Q(K^T V)

    This NEVER forms the (seq_len, seq_len) attention matrix.
    Complexity: O(nd²) where n = seq_len, d = dimension

    The key insight is that with a feature map φ:
        attention[i] = Σ_j (φ(q_i) · φ(k_j)) * v_j / Σ_j (φ(q_i) · φ(k_j))
                     = φ(q_i) · Σ_j (φ(k_j) ⊗ v_j) / φ(q_i) · Σ_j φ(k_j)

    We can precompute Σ_j (φ(k_j) ⊗ v_j) once and reuse for all queries.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: Function φ that transforms Q and K
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (seq_len, d_v)

    Example:
        >>> Q = np.random.randn(100, 64)
        >>> K = np.random.randn(100, 64)
        >>> V = np.random.randn(100, 64)
        >>> output = linear_attention_order(Q, K, V, lambda x: np.abs(x))
        >>> output.shape
        (100, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Apply feature map: Q_prime = feature_map(Q), K_prime = feature_map(K)
    # 2. Compute KV aggregate: KV = K_prime.T @ V  -> (d_k, d_v)
    # 3. Compute normalizer: Z = K_prime.sum(axis=0)  -> (d_k,)
    # 4. Compute numerator: Q_prime @ KV  -> (n, d_v)
    # 5. Compute denominator: Q_prime @ Z  -> (n,)
    # 6. Normalize: output = numerator / (denominator[:, None] + eps)
    raise NotImplementedError("Implement linear_attention_order")


def identity_feature_map(x: np.ndarray) -> np.ndarray:
    """
    Identity feature map: φ(x) = x

    This is the simplest feature map. It doesn't approximate softmax,
    but demonstrates the associativity principle.

    Args:
        x: Input tensor of any shape

    Returns:
        Same tensor unchanged
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement identity_feature_map")


def relu_feature_map(x: np.ndarray) -> np.ndarray:
    """
    ReLU feature map: φ(x) = max(0, x)

    Simple and fast, but doesn't guarantee positive attention weights
    when Q and K have different signs.

    Args:
        x: Input tensor of any shape

    Returns:
        ReLU applied elementwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement relu_feature_map")


def elu_feature_map(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    ELU+1 feature map: φ(x) = ELU(x) + 1

    ELU(x) = x if x > 0, else alpha * (exp(x) - 1)

    Adding 1 ensures the output is always positive (> 0).
    This is the feature map used in the original Linear Transformer paper.

    Args:
        x: Input tensor of any shape
        alpha: ELU alpha parameter (default 1.0)

    Returns:
        ELU(x) + 1 applied elementwise

    Properties:
        - Always positive: φ(x) > 0 for all x
        - Smooth and differentiable
        - Approximates identity for x > 0
    """
    # YOUR CODE HERE
    #
    # ELU(x) = x if x > 0
    #        = alpha * (exp(x) - 1) if x <= 0
    #
    # Then add 1 to make it positive
    raise NotImplementedError("Implement elu_feature_map")


def exp_feature_map(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Exponential feature map: φ(x) = exp(x * scale)

    This more closely approximates softmax attention, but can have
    numerical stability issues for large values.

    Args:
        x: Input tensor of any shape
        scale: Scaling factor (default 1.0)

    Returns:
        exp(x * scale) applied elementwise

    Warning:
        Can overflow for large x values! Use with caution.
    """
    # YOUR CODE HERE
    #
    # Hint: For stability, you might want to subtract the max
    # But for this lab, simple exp is fine
    raise NotImplementedError("Implement exp_feature_map")


def verify_associativity(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Verify that (QK^T)V = Q(K^T V) for simple matrix multiplication.

    This demonstrates that matrix multiplication is associative,
    which is the foundation of the kernel trick.

    Note: This is for raw matrix multiplication WITHOUT softmax.
    With softmax, the two orders are NOT equivalent.

    Args:
        Q: Query tensor of shape (n, d)
        K: Key tensor of shape (n, d)
        V: Value tensor of shape (n, d)

    Returns:
        Tuple of:
        - result1: (Q @ K.T) @ V
        - result2: Q @ (K.T @ V)
        - max_diff: Maximum absolute difference between results

    Example:
        >>> Q = np.random.randn(10, 5)
        >>> K = np.random.randn(10, 5)
        >>> V = np.random.randn(10, 5)
        >>> r1, r2, diff = verify_associativity(Q, K, V)
        >>> diff < 1e-10
        True
    """
    # YOUR CODE HERE
    #
    # Compute both orders and compare:
    # result1 = (Q @ K.T) @ V  # Standard order
    # result2 = Q @ (K.T @ V)  # Linear order
    # max_diff = np.max(np.abs(result1 - result2))
    raise NotImplementedError("Implement verify_associativity")


def compare_complexity(
    seq_len: int,
    d_model: int
) -> dict:
    """
    Compare theoretical complexity of standard vs linear attention.

    Args:
        seq_len: Sequence length (n)
        d_model: Model dimension (d)

    Returns:
        Dictionary with:
        - 'standard_ops': Number of operations for (QK^T)V
        - 'linear_ops': Number of operations for Q(K^T V)
        - 'speedup': Ratio of standard to linear ops
        - 'crossover_seq_len': Sequence length where linear becomes faster

    Example:
        >>> result = compare_complexity(4096, 64)
        >>> result['speedup']
        64.0
    """
    # YOUR CODE HERE
    #
    # Standard: (n×d) @ (d×n) = n²d ops, then (n×n) @ (n×d) = n²d ops
    #           Total: 2 × n² × d
    #
    # Linear: (d×n) @ (n×d) = nd² ops, then (n×d) @ (d×d) = nd² ops
    #         Total: 2 × n × d²
    #
    # Speedup: (2n²d) / (2nd²) = n/d
    #
    # Crossover: when n²d = nd², i.e., n = d
    raise NotImplementedError("Implement compare_complexity")
