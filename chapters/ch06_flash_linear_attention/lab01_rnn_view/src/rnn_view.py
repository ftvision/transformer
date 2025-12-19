"""
Lab 01: RNN View of Linear Attention

Implement linear attention in both parallel and recurrent forms.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation: x if x > 0 else alpha * (exp(x) - 1)."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def feature_map(x: np.ndarray, method: str = 'elu_plus_one') -> np.ndarray:
    """
    Apply a feature map φ(x) to the input.

    Feature maps transform Q and K before computing attention.
    For linear attention to work correctly, the feature map should
    produce non-negative outputs (so attention weights are non-negative).

    Args:
        x: Input tensor of any shape
        method: Which feature map to use:
            - 'elu_plus_one': ELU(x) + 1 (default, ensures positivity)
            - 'relu': ReLU(x) (simple, but can have dead neurons)
            - 'identity': No transformation (not recommended for attention)
            - 'softmax_kernel': exp(x) (used in Performers)

    Returns:
        Transformed tensor of same shape as x

    Example:
        >>> x = np.array([-1.0, 0.0, 1.0])
        >>> feature_map(x, 'elu_plus_one')
        array([0.632..., 1.0, 2.0])
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement feature_map")


def linear_attention_parallel(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute causal linear attention using parallel (cumsum) formulation.

    This is the training-friendly formulation:
        φ(Q) @ cumsum(φ(K)^T @ V)

    For causal attention, position i can only attend to positions <= i.
    We achieve this by using cumulative sum along the sequence dimension.

    Args:
        Q: Query tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_v) or (seq_len, d_v)
        feature_map_fn: Function to apply to Q and K (default: elu_plus_one)

    Returns:
        output: Attention output of shape matching V
        final_state: Final state S_n of shape (..., d_k, d_v)

    Steps:
        1. Apply feature map to Q and K
        2. Compute outer products: KV[i] = φ(k_i)^T @ v_i  (shape: d_k x d_v per position)
        3. Cumulative sum: S[i] = sum_{j<=i} KV[j]
        4. Query the state: output[i] = φ(q_i) @ S[i]

    Note:
        The normalizer (sum of attention weights) is omitted here for simplicity.
        In practice, you'd also compute cumsum(φ(K)) and divide.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement linear_attention_parallel")


def linear_attention_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute causal linear attention using recurrent formulation.

    This is the inference-friendly formulation that processes one token at a time:
        S_i = S_{i-1} + φ(k_i)^T @ v_i
        output_i = φ(q_i) @ S_i

    This function processes the entire sequence but uses the recurrent formula,
    which is mathematically equivalent to the parallel form.

    Args:
        Q: Query tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k) or (seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_v) or (seq_len, d_v)
        feature_map_fn: Function to apply to Q and K (default: elu_plus_one)

    Returns:
        output: Attention output of shape matching V
        final_state: Final state S_n of shape (..., d_k, d_v)

    Note:
        This should produce IDENTICAL output to linear_attention_parallel.
        The difference is in how we compute it (loop vs vectorized).
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement linear_attention_recurrent")


def linear_attention_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    state: Optional[np.ndarray] = None,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-step linear attention for autoregressive inference.

    This is what you'd use during generation:
    - Receive one new token
    - Update the state
    - Produce one output
    - State size is CONSTANT regardless of how many tokens generated

    Args:
        q: Query for single position, shape (batch, d_k) or (d_k,)
        k: Key for single position, shape (batch, d_k) or (d_k,)
        v: Value for single position, shape (batch, d_v) or (d_v,)
        state: Current state S, shape (batch, d_k, d_v) or (d_k, d_v)
               If None, initializes to zeros
        feature_map_fn: Function to apply to q and k (default: elu_plus_one)

    Returns:
        output: Output for this position, shape (batch, d_v) or (d_v,)
        new_state: Updated state, same shape as input state

    Example usage during generation:
        state = None
        for token in sequence:
            q, k, v = compute_qkv(token)
            output, state = linear_attention_step(q, k, v, state)
            # Use output for next token prediction
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement linear_attention_step")


def compare_parallel_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map_fn: Callable = None,
    rtol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Verify that parallel and recurrent forms give identical outputs.

    Args:
        Q, K, V: Input tensors
        feature_map_fn: Feature map to use
        rtol: Relative tolerance for comparison

    Returns:
        match: True if outputs match within tolerance
        max_diff: Maximum absolute difference
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_parallel_recurrent")
