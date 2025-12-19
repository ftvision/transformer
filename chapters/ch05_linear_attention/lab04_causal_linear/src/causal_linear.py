"""
Lab 04: Causal Linear Attention

Implement causal (autoregressive) linear attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
import time
from typing import Callable, Tuple, Dict, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """
    ELU+1 feature map (provided for convenience).

    φ(x) = ELU(x) + 1 where ELU(x) = x if x > 0, else exp(x) - 1
    """
    return np.where(x > 0, x + 1, np.exp(x))


# ============================================================================
# Parallel Form (Training)
# ============================================================================

def causal_linear_attention_parallel(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Causal linear attention using parallel cumsum formulation.

    This form is efficient for training because it can be parallelized.

    For position i, the output is:
        output_i = (φ(q_i) · KV_i) / (φ(q_i) · Z_i)

    where:
        KV_i = Σ_{j≤i} φ(k_j) ⊗ v_j   (cumsum of outer products)
        Z_i = Σ_{j≤i} φ(k_j)           (cumsum of keys)

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: Feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (seq_len, d_v)

    Example:
        >>> Q = np.random.randn(100, 64)
        >>> K = np.random.randn(100, 64)
        >>> V = np.random.randn(100, 64)
        >>> output = causal_linear_attention_parallel(Q, K, V, elu_plus_one)
        >>> output.shape
        (100, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Apply feature map: Q_prime = feature_map(Q), K_prime = feature_map(K)
    # 2. Compute outer products: kv[i] = φ(k_i) ⊗ v_i
    #    Shape: (seq_len, d_phi, d_v)
    #    Use: np.einsum('nd,nv->ndv', K_prime, V)
    # 3. Cumulative sum: kv_cumsum = np.cumsum(kv, axis=0)
    # 4. Cumulative sum of keys: z_cumsum = np.cumsum(K_prime, axis=0)
    # 5. Query each position:
    #    numerator[i] = φ(q_i) · KV_i
    #    denominator[i] = φ(q_i) · Z_i
    # 6. output = numerator / (denominator + eps)
    raise NotImplementedError("Implement causal_linear_attention_parallel")


def causal_linear_attention_parallel_batched(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Batched version of causal linear attention (parallel form).

    Args:
        Q: Query tensor of shape (batch, seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_v)
        feature_map: Feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (batch, seq_len, d_v)
    """
    # YOUR CODE HERE
    #
    # Same as non-batched, but handle batch dimension
    # cumsum should be along axis=1 (sequence dimension)
    raise NotImplementedError("Implement causal_linear_attention_parallel_batched")


# ============================================================================
# Recurrent Form (Inference)
# ============================================================================

def causal_linear_attention_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> np.ndarray:
    """
    Causal linear attention using recurrent state updates.

    This form is efficient for inference because it's O(1) per token.

    The state S accumulates key-value products:
        S_0 = 0
        S_i = S_{i-1} + φ(k_i) ⊗ v_i

    The output at each position:
        output_i = (φ(q_i) · S_i) / (φ(q_i) · Z_i)

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: Feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (seq_len, d_v)

    Note:
        This should produce IDENTICAL output to the parallel form.
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Apply feature map: Q_prime = feature_map(Q), K_prime = feature_map(K)
    # 2. Initialize state S = zeros(d_phi, d_v) and Z = zeros(d_phi)
    # 3. For each position i:
    #    a. Update state: S = S + outer(K_prime[i], V[i])
    #    b. Update normalizer: Z = Z + K_prime[i]
    #    c. Compute output: output[i] = (Q_prime[i] @ S) / (Q_prime[i] @ Z + eps)
    # 4. Return stacked outputs
    raise NotImplementedError("Implement causal_linear_attention_recurrent")


def create_initial_state(d_phi: int, d_v: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create initial state for recurrent linear attention.

    Args:
        d_phi: Feature dimension (after feature map)
        d_v: Value dimension

    Returns:
        Tuple of (S, Z) where:
        - S: State matrix of shape (d_phi, d_v), initialized to zeros
        - Z: Normalizer vector of shape (d_phi,), initialized to zeros
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_initial_state")


def causal_linear_attention_rnn_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    state: Tuple[np.ndarray, np.ndarray],
    feature_map: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Single step of recurrent linear attention.

    This is used for autoregressive generation where we process
    one token at a time.

    Args:
        q: Query vector of shape (d_k,)
        k: Key vector of shape (d_k,)
        v: Value vector of shape (d_v,)
        state: Tuple of (S, Z) from previous step
               S: shape (d_phi, d_v)
               Z: shape (d_phi,)
        feature_map: Feature map function φ
        eps: Small constant for numerical stability

    Returns:
        Tuple of:
        - output: Output vector of shape (d_v,)
        - new_state: Updated (S, Z) tuple

    Example:
        >>> d_model = 64
        >>> state = create_initial_state(d_model, d_model)
        >>> for i in range(100):
        ...     output, state = causal_linear_attention_rnn_step(
        ...         Q[i], K[i], V[i], state, elu_plus_one
        ...     )
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Unpack state: S, Z = state
    # 2. Apply feature map: q_prime = feature_map(q), k_prime = feature_map(k)
    # 3. Update state: S_new = S + outer(k_prime, v)
    # 4. Update normalizer: Z_new = Z + k_prime
    # 5. Compute output: output = (q_prime @ S_new) / (q_prime @ Z_new + eps)
    # 6. Return output, (S_new, Z_new)
    raise NotImplementedError("Implement causal_linear_attention_rnn_step")


# ============================================================================
# Comparison Functions
# ============================================================================

def causal_softmax_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    """
    Standard causal softmax attention for comparison.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)

    Returns:
        Output tensor of shape (seq_len, d_v)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute scores: scores = Q @ K.T / sqrt(d_k)
    # 2. Create causal mask (upper triangle is -inf)
    # 3. Apply mask: scores = scores + mask
    # 4. Apply softmax: attention = softmax(scores, axis=-1)
    # 5. Compute output: output = attention @ V
    raise NotImplementedError("Implement causal_softmax_attention")


def compare_parallel_vs_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray]
) -> Dict[str, float]:
    """
    Verify parallel and recurrent forms give identical outputs.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: Feature map function φ

    Returns:
        Dictionary with:
        - 'max_diff': Maximum absolute difference
        - 'mean_diff': Mean absolute difference
        - 'match': True if outputs match within tolerance
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_parallel_vs_recurrent")


def compare_to_causal_softmax(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map: Callable[[np.ndarray], np.ndarray]
) -> Dict[str, float]:
    """
    Compare linear attention output to softmax attention output.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)
        feature_map: Feature map function φ

    Returns:
        Dictionary with:
        - 'output_mse': MSE between outputs
        - 'output_correlation': Correlation between outputs (flattened)
        - 'output_max_diff': Maximum absolute difference
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_to_causal_softmax")


def benchmark_causal_attention(
    seq_lengths: list,
    d_model: int = 64,
    num_runs: int = 3
) -> Dict[str, list]:
    """
    Benchmark causal attention implementations.

    Args:
        seq_lengths: List of sequence lengths to test
        d_model: Model dimension
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with:
        - 'seq_lengths': Input sequence lengths
        - 'softmax_times': Time for softmax attention
        - 'linear_parallel_times': Time for parallel linear attention
        - 'linear_recurrent_times': Time for recurrent linear attention
        - 'speedup': Ratio of softmax time to linear parallel time
    """
    # YOUR CODE HERE
    #
    # For each seq_length:
    # 1. Create Q, K, V
    # 2. Time each implementation (average over num_runs)
    # 3. Record results
    raise NotImplementedError("Implement benchmark_causal_attention")
