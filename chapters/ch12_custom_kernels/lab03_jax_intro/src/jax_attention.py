"""
Lab 03: Introduction to JAX

Implement attention mechanisms using JAX's functional programming model.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Requirements:
- JAX
"""

import jax
import jax.numpy as jnp
from typing import Optional


def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Compute numerically stable softmax.

    Args:
        x: Input array of any shape
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities with same shape as x

    Example:
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> result = softmax(x)
        >>> jnp.allclose(jnp.sum(result), 1.0)
        True

    Implementation:
        For numerical stability, subtract the maximum before exponentiating:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Hint:
        Use jnp.max with keepdims=True for broadcasting
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Find the maximum value along axis (keep dims for broadcasting)
    #    x_max = jnp.max(x, axis=axis, keepdims=True)
    #
    # 2. Subtract max and exponentiate (for numerical stability)
    #    exp_x = jnp.exp(x - x_max)
    #
    # 3. Normalize by the sum
    #    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

    raise NotImplementedError("Implement softmax")


def scaled_dot_product_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)
        mask: Optional boolean mask of shape (seq_q, seq_k)
              True values are masked out (set to -inf before softmax)

    Returns:
        Output tensor of shape (seq_q, d_v)

    The attention formula:
        Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Example:
        >>> Q = jnp.ones((4, 8))
        >>> K = jnp.ones((6, 8))
        >>> V = jnp.ones((6, 16))
        >>> output = scaled_dot_product_attention(Q, K, V)
        >>> output.shape
        (4, 16)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get d_k from Q's last dimension
    #    d_k = Q.shape[-1]
    #
    # 2. Compute attention scores
    #    scores = Q @ K.T / jnp.sqrt(d_k)
    #    # scores shape: (seq_q, seq_k)
    #
    # 3. Apply mask if provided
    #    if mask is not None:
    #        scores = jnp.where(mask, float('-inf'), scores)
    #
    # 4. Apply softmax to get attention weights
    #    weights = softmax(scores, axis=-1)
    #
    # 5. Apply weights to values
    #    output = weights @ V
    #
    # 6. Return output

    raise NotImplementedError("Implement scaled_dot_product_attention")


def linear(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Apply a linear transformation.

    Args:
        x: Input tensor of shape (..., in_features)
        weight: Weight matrix of shape (in_features, out_features)
        bias: Optional bias vector of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)

    Formula:
        output = x @ weight + bias (if bias provided)
        output = x @ weight (if no bias)

    Example:
        >>> x = jnp.ones((4, 8))
        >>> weight = jnp.ones((8, 16))
        >>> bias = jnp.zeros((16,))
        >>> output = linear(x, weight, bias)
        >>> output.shape
        (4, 16)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Compute x @ weight
    # 2. Add bias if provided
    # 3. Return result

    raise NotImplementedError("Implement linear")


def multi_head_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    W_q: jnp.ndarray,
    W_k: jnp.ndarray,
    W_v: jnp.ndarray,
    W_o: jnp.ndarray,
    num_heads: int
) -> jnp.ndarray:
    """
    Compute multi-head attention.

    Args:
        Q: Query tensor of shape (seq_q, d_model)
        K: Key tensor of shape (seq_k, d_model)
        V: Value tensor of shape (seq_k, d_model)
        W_q: Query projection of shape (d_model, d_model)
        W_k: Key projection of shape (d_model, d_model)
        W_v: Value projection of shape (d_model, d_model)
        W_o: Output projection of shape (d_model, d_model)
        num_heads: Number of attention heads

    Returns:
        Output tensor of shape (seq_q, d_model)

    Algorithm:
        1. Project Q, K, V using weight matrices
        2. Reshape to separate heads: (seq, d_model) -> (seq, num_heads, d_head)
        3. Compute attention for each head
        4. Concatenate heads: (seq, num_heads, d_head) -> (seq, d_model)
        5. Final projection with W_o

    Example:
        >>> seq_q, seq_k, d_model, num_heads = 4, 6, 64, 8
        >>> Q = jnp.ones((seq_q, d_model))
        >>> K = jnp.ones((seq_k, d_model))
        >>> V = jnp.ones((seq_k, d_model))
        >>> W_q = jnp.ones((d_model, d_model)) * 0.01
        >>> W_k = jnp.ones((d_model, d_model)) * 0.01
        >>> W_v = jnp.ones((d_model, d_model)) * 0.01
        >>> W_o = jnp.ones((d_model, d_model)) * 0.01
        >>> output = multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads)
        >>> output.shape
        (4, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get dimensions
    #    seq_q = Q.shape[0]
    #    seq_k = K.shape[0]
    #    d_model = Q.shape[1]
    #    d_head = d_model // num_heads
    #
    # 2. Project Q, K, V
    #    Q_proj = Q @ W_q  # (seq_q, d_model)
    #    K_proj = K @ W_k  # (seq_k, d_model)
    #    V_proj = V @ W_v  # (seq_k, d_model)
    #
    # 3. Reshape to separate heads
    #    Q_heads = Q_proj.reshape(seq_q, num_heads, d_head)  # (seq_q, num_heads, d_head)
    #    K_heads = K_proj.reshape(seq_k, num_heads, d_head)  # (seq_k, num_heads, d_head)
    #    V_heads = V_proj.reshape(seq_k, num_heads, d_head)  # (seq_k, num_heads, d_head)
    #
    # 4. Compute attention for each head
    #    You can use a loop or einsum:
    #
    #    Option A: Loop over heads
    #    outputs = []
    #    for h in range(num_heads):
    #        q_h = Q_heads[:, h, :]  # (seq_q, d_head)
    #        k_h = K_heads[:, h, :]  # (seq_k, d_head)
    #        v_h = V_heads[:, h, :]  # (seq_k, d_head)
    #        out_h = scaled_dot_product_attention(q_h, k_h, v_h)
    #        outputs.append(out_h)
    #    heads_out = jnp.stack(outputs, axis=1)  # (seq_q, num_heads, d_head)
    #
    #    Option B: Use einsum for batched attention (more efficient)
    #    # Transpose to (num_heads, seq, d_head) for batched computation
    #    Q_t = jnp.transpose(Q_heads, (1, 0, 2))  # (num_heads, seq_q, d_head)
    #    K_t = jnp.transpose(K_heads, (1, 0, 2))  # (num_heads, seq_k, d_head)
    #    V_t = jnp.transpose(V_heads, (1, 0, 2))  # (num_heads, seq_k, d_head)
    #
    #    # Batched attention scores
    #    scores = jnp.einsum('hqd,hkd->hqk', Q_t, K_t) / jnp.sqrt(d_head)
    #    weights = jax.nn.softmax(scores, axis=-1)
    #    heads_out_t = jnp.einsum('hqk,hkd->hqd', weights, V_t)  # (num_heads, seq_q, d_head)
    #    heads_out = jnp.transpose(heads_out_t, (1, 0, 2))  # (seq_q, num_heads, d_head)
    #
    # 5. Concatenate heads
    #    concat = heads_out.reshape(seq_q, d_model)  # (seq_q, d_model)
    #
    # 6. Final projection
    #    output = concat @ W_o
    #
    # 7. Return output

    raise NotImplementedError("Implement multi_head_attention")


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute GELU activation.

    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
    of the standard normal distribution.

    Args:
        x: Input tensor

    Returns:
        GELU-activated tensor

    Note:
        You can use jax.nn.gelu directly, but implementing it helps understand
        the function.
    """
    # You can use JAX's built-in:
    return jax.nn.gelu(x)


def feedforward(
    x: jnp.ndarray,
    W1: jnp.ndarray,
    b1: jnp.ndarray,
    W2: jnp.ndarray,
    b2: jnp.ndarray
) -> jnp.ndarray:
    """
    Two-layer feedforward network with GELU activation.

    Args:
        x: Input tensor of shape (seq_len, d_model)
        W1: First weight matrix of shape (d_model, d_ff)
        b1: First bias vector of shape (d_ff,)
        W2: Second weight matrix of shape (d_ff, d_model)
        b2: Second bias vector of shape (d_model,)

    Returns:
        Output tensor of shape (seq_len, d_model)

    Formula:
        FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Example:
        >>> x = jnp.ones((4, 64))
        >>> W1 = jnp.ones((64, 256)) * 0.01
        >>> b1 = jnp.zeros((256,))
        >>> W2 = jnp.ones((256, 64)) * 0.01
        >>> b2 = jnp.zeros((64,))
        >>> output = feedforward(x, W1, b1, W2, b2)
        >>> output.shape
        (4, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. First linear layer
    #    hidden = x @ W1 + b1
    #
    # 2. Apply GELU activation
    #    hidden = gelu(hidden)
    #
    # 3. Second linear layer
    #    output = hidden @ W2 + b2
    #
    # 4. Return output

    raise NotImplementedError("Implement feedforward")


# =============================================================================
# Reference Implementation (for testing)
# =============================================================================

def attention_reference(Q, K, V, mask=None):
    """Reference attention implementation using JAX operations."""
    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.T) / jnp.sqrt(d_k)

    if mask is not None:
        scores = jnp.where(mask, float('-inf'), scores)

    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, V)


# =============================================================================
# BONUS: Causal Attention (Optional)
# =============================================================================

def causal_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """
    BONUS: Compute causal (autoregressive) attention.

    Each query can only attend to keys at the same or earlier positions.

    Args:
        Q: Query tensor of shape (seq_len, d_k)
        K: Key tensor of shape (seq_len, d_k)
        V: Value tensor of shape (seq_len, d_v)

    Returns:
        Output tensor of shape (seq_len, d_v)

    Hint:
        Create a causal mask where mask[i, j] = True if j > i
        Then pass this mask to scaled_dot_product_attention
    """
    # YOUR CODE HERE (optional bonus challenge)
    raise NotImplementedError("Bonus: Implement causal_attention")
