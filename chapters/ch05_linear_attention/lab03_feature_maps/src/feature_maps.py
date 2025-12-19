"""
Lab 03: Feature Maps

Implement and compare different feature maps for linear attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Callable, Dict, Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# Feature Maps
# ============================================================================

def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """
    ELU+1 feature map: φ(x) = ELU(x) + 1

    This is the feature map from "Transformers are RNNs" (Katharopoulos et al.).

    ELU(x) = x if x > 0
           = exp(x) - 1 if x <= 0

    Adding 1 ensures φ(x) > 0 for all x:
    - For x > 0: φ(x) = x + 1 > 1
    - For x <= 0: φ(x) = exp(x) > 0

    Args:
        x: Input tensor of any shape

    Returns:
        ELU(x) + 1 applied elementwise

    Example:
        >>> elu_plus_one(np.array([-1.0, 0.0, 1.0]))
        array([0.368..., 1.0, 2.0])
    """
    # YOUR CODE HERE
    #
    # Use np.where for conditional:
    # np.where(condition, value_if_true, value_if_false)
    raise NotImplementedError("Implement elu_plus_one")


def relu_feature_map(x: np.ndarray) -> np.ndarray:
    """
    ReLU feature map: φ(x) = max(0, x)

    Simple but has drawbacks:
    - Can produce zero attention weights
    - Not positive everywhere

    Args:
        x: Input tensor of any shape

    Returns:
        max(0, x) applied elementwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement relu_feature_map")


def squared_relu_feature_map(x: np.ndarray) -> np.ndarray:
    """
    Squared ReLU feature map: φ(x) = max(0, x)²

    Advantages over plain ReLU:
    - Smoother gradients (continuous derivative at 0)
    - Always non-negative output
    - Used in some recent architectures

    Args:
        x: Input tensor of any shape

    Returns:
        max(0, x)² applied elementwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement squared_relu_feature_map")


def exp_feature_map(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Exponential feature map: φ(x) = exp(scale * x)

    Closest to softmax in spirit, but can overflow.

    Args:
        x: Input tensor of any shape
        scale: Scaling factor (use smaller values for stability)

    Returns:
        exp(scale * x) applied elementwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement exp_feature_map")


def random_feature_map(
    x: np.ndarray,
    projection_matrix: np.ndarray,
    use_cos_sin: bool = True
) -> np.ndarray:
    """
    Random Fourier Features for approximating softmax kernel.

    This is based on the FAVOR+ mechanism from the Performers paper.

    For the softmax kernel exp(q·k), random features give:
    E[φ(q)·φ(k)] ≈ exp(q·k)

    Args:
        x: Input tensor of shape (seq_len, d_model)
        projection_matrix: Random projection of shape (d_model, num_features)
        use_cos_sin: If True, use [cos(proj), sin(proj)] (2x features)
                     If False, use just cos(proj)

    Returns:
        Transformed features of shape:
        - (seq_len, 2*num_features) if use_cos_sin=True
        - (seq_len, num_features) if use_cos_sin=False

    Example:
        >>> x = np.random.randn(100, 64)
        >>> proj = np.random.randn(64, 128)
        >>> features = random_feature_map(x, proj)
        >>> features.shape
        (100, 256)  # 2 * 128 because of cos and sin
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Project: projected = x @ projection_matrix
    # 2. If use_cos_sin:
    #    - features = [cos(projected), sin(projected)]
    #    - Concatenate along last axis
    # 3. Scale by 1/sqrt(num_features) for proper variance
    raise NotImplementedError("Implement random_feature_map")


def create_random_projection(
    d_model: int,
    num_features: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create a random projection matrix for random feature maps.

    The projection should be orthonormal for best results.

    Args:
        d_model: Input dimension
        num_features: Number of random features
        seed: Random seed for reproducibility

    Returns:
        Projection matrix of shape (d_model, num_features)
    """
    # YOUR CODE HERE
    #
    # Use np.random.randn and scale appropriately
    # For better quality, you could use orthogonal initialization
    raise NotImplementedError("Implement create_random_projection")


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_implicit_attention(
    Q: np.ndarray,
    K: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute the implicit attention matrix from linear attention.

    In linear attention, we never explicitly form the attention matrix.
    But for analysis, we can compute what it would be:

    A[i,j] = φ(q_i)·φ(k_j) / Σ_l φ(q_i)·φ(k_l)

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        feature_map: The feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Implicit attention matrix of shape (seq_len, seq_len)
        Each row sums to 1 (normalized)

    Example:
        >>> Q = np.random.randn(10, 64)
        >>> K = np.random.randn(10, 64)
        >>> A = compute_implicit_attention(Q, K, elu_plus_one)
        >>> A.shape
        (10, 10)
        >>> np.allclose(A.sum(axis=-1), 1.0)
        True
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Apply feature map: Q_prime = feature_map(Q), K_prime = feature_map(K)
    # 2. Compute raw attention: A_raw = Q_prime @ K_prime.T
    # 3. Normalize each row: A = A_raw / (A_raw.sum(axis=-1, keepdims=True) + eps)
    raise NotImplementedError("Implement compute_implicit_attention")


def compute_softmax_attention(
    Q: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """
    Compute standard softmax attention matrix.

    A[i,j] = exp(q_i · k_j / √d_k) / Σ_l exp(q_i · k_l / √d_k)

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)

    Returns:
        Softmax attention matrix of shape (seq_len, seq_len)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute scores: scores = Q @ K.T
    # 2. Scale: scores = scores / sqrt(d_k)
    # 3. Softmax: attention = softmax(scores, axis=-1)
    raise NotImplementedError("Implement compute_softmax_attention")


def compare_to_softmax(
    Q: np.ndarray,
    K: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray]
) -> Dict[str, float]:
    """
    Compare linear attention weights to softmax attention.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        feature_map: The feature map function φ

    Returns:
        Dictionary with:
        - 'mse': Mean squared error between attention matrices
        - 'max_diff': Maximum absolute difference
        - 'correlation': Pearson correlation (flattened matrices)
        - 'row_correlation_mean': Mean correlation per row

    Example:
        >>> Q = np.random.randn(50, 64).astype(np.float32)
        >>> K = np.random.randn(50, 64).astype(np.float32)
        >>> metrics = compare_to_softmax(Q, K, elu_plus_one)
        >>> print(f"MSE: {metrics['mse']:.4f}")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute softmax attention: A_softmax = compute_softmax_attention(Q, K)
    # 2. Compute linear attention: A_linear = compute_implicit_attention(Q, K, feature_map)
    # 3. Compute metrics:
    #    - MSE: np.mean((A_softmax - A_linear) ** 2)
    #    - max_diff: np.max(np.abs(A_softmax - A_linear))
    #    - correlation: np.corrcoef(A_softmax.flatten(), A_linear.flatten())[0, 1]
    #    - row_correlation_mean: average of per-row correlations
    raise NotImplementedError("Implement compare_to_softmax")


def analyze_feature_map_quality(
    feature_map: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive analysis of a feature map's quality.

    Args:
        feature_map: The feature map function φ
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)

    Returns:
        Dictionary with:
        - 'attention_mse': MSE vs softmax attention
        - 'attention_correlation': Correlation with softmax attention
        - 'output_mse': MSE of output vs softmax output
        - 'output_correlation': Correlation of outputs
        - 'feature_sparsity': Fraction of zero features
        - 'feature_mean': Mean of feature values
        - 'feature_std': Std of feature values
    """
    # YOUR CODE HERE
    #
    # This combines attention comparison with output comparison
    # and feature statistics
    raise NotImplementedError("Implement analyze_feature_map_quality")


def linear_attention_with_feature_map(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute linear attention output using a feature map.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: The feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (seq_len, d_v)
    """
    # YOUR CODE HERE
    #
    # This is similar to Lab 02's linear_attention_order
    # 1. Q_prime = feature_map(Q)
    # 2. K_prime = feature_map(K)
    # 3. KV = K_prime.T @ V
    # 4. Z = K_prime.sum(axis=0)
    # 5. output = (Q_prime @ KV) / (Q_prime @ Z + eps)[:, None]
    raise NotImplementedError("Implement linear_attention_with_feature_map")


def softmax_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Compute standard softmax attention output.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)

    Returns:
        Output tensor of shape (seq_len, d_v)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement softmax_attention")
