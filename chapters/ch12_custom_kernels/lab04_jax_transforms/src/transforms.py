"""
Lab 04: JAX Transformations

Master JAX's function transformations: jit, vmap, grad.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Requirements:
- JAX
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from functools import partial


# =============================================================================
# Helper Functions
# =============================================================================

def attention_single(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Single-example attention (no batching).

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)

    Returns:
        Output tensor of shape (seq_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.T) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, V)


# =============================================================================
# JIT Compilation
# =============================================================================

def jit_attention(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled attention function.

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)

    Returns:
        Output tensor of shape (seq_q, d_v)

    This should be a JIT-compiled version of attention_single.

    Hint:
        Use @jax.jit decorator or jax.jit(fn)

    Example:
        >>> Q = jnp.ones((4, 8))
        >>> K = jnp.ones((6, 8))
        >>> V = jnp.ones((6, 16))
        >>> output = jit_attention(Q, K, V)
        >>> output.shape
        (4, 16)
    """
    # YOUR CODE HERE
    #
    # Option 1: Decorate a function with @jax.jit
    # @jax.jit
    # def _attention(Q, K, V):
    #     return attention_single(Q, K, V)
    # return _attention(Q, K, V)
    #
    # Option 2: Use jax.jit inline
    # return jax.jit(attention_single)(Q, K, V)
    #
    # Note: For best performance, the JIT-compiled function should be
    # created once and reused. Here we demonstrate the concept.

    raise NotImplementedError("Implement jit_attention")


# Create a properly cached JIT version for performance testing
_jit_attention_cached = jax.jit(attention_single)


def jit_attention_cached(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Cached JIT attention for performance testing."""
    return _jit_attention_cached(Q, K, V)


# =============================================================================
# Vectorization with vmap
# =============================================================================

def batched_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """
    Batched attention using vmap.

    Args:
        Q: Query tensor of shape (batch, seq_q, d_k)
        K: Key tensor of shape (batch, seq_k, d_k)
        V: Value tensor of shape (batch, seq_k, d_v)

    Returns:
        Output tensor of shape (batch, seq_q, d_v)

    Use jax.vmap to automatically vectorize attention_single over the batch dimension.

    Example:
        >>> Q = jnp.ones((8, 4, 16))  # batch=8, seq_q=4, d_k=16
        >>> K = jnp.ones((8, 6, 16))  # batch=8, seq_k=6, d_k=16
        >>> V = jnp.ones((8, 6, 32))  # batch=8, seq_k=6, d_v=32
        >>> output = batched_attention(Q, K, V)
        >>> output.shape
        (8, 4, 32)
    """
    # YOUR CODE HERE
    #
    # vmap transforms a function that operates on single examples
    # to operate on batches.
    #
    # jax.vmap(fn) by default batches over axis 0 of all inputs
    #
    # return jax.vmap(attention_single)(Q, K, V)

    raise NotImplementedError("Implement batched_attention")


def batched_attention_shared_kv(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """
    Batched attention where K and V are shared across the batch.

    Args:
        Q: Query tensor of shape (batch, seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k) - NOT batched
        V: Value tensor of shape (seq_k, d_v) - NOT batched

    Returns:
        Output tensor of shape (batch, seq_q, d_v)

    This is useful when all batch elements attend to the same memory.

    Hint:
        Use in_axes parameter of vmap to specify which inputs to batch over.
        in_axes=(0, None, None) means: batch Q over axis 0, don't batch K and V

    Example:
        >>> Q = jnp.ones((8, 4, 16))  # batch=8
        >>> K = jnp.ones((6, 16))     # shared
        >>> V = jnp.ones((6, 32))     # shared
        >>> output = batched_attention_shared_kv(Q, K, V)
        >>> output.shape
        (8, 4, 32)
    """
    # YOUR CODE HERE
    #
    # Use in_axes to specify batching behavior:
    # in_axes=(0, None, None) means:
    #   - Batch Q over axis 0
    #   - Don't batch K (broadcast same K to all examples)
    #   - Don't batch V (broadcast same V to all examples)
    #
    # return jax.vmap(attention_single, in_axes=(0, None, None))(Q, K, V)

    raise NotImplementedError("Implement batched_attention_shared_kv")


# =============================================================================
# Automatic Differentiation
# =============================================================================

def attention_loss(Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a scalar loss from attention output.

    Used for gradient testing.
    """
    output = attention_single(Q, K, V)
    return jnp.sum(output ** 2)


def attention_gradient(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute gradients of attention loss with respect to Q, K, V.

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)

    Returns:
        Tuple of (dL_dQ, dL_dK, dL_dV) with same shapes as inputs

    Hint:
        Use jax.grad with argnums to specify which arguments to differentiate.
        argnums=(0, 1, 2) means differentiate with respect to all three inputs.

    Example:
        >>> Q = jnp.ones((4, 8))
        >>> K = jnp.ones((6, 8))
        >>> V = jnp.ones((6, 16))
        >>> dQ, dK, dV = attention_gradient(Q, K, V)
        >>> dQ.shape == Q.shape
        True
    """
    # YOUR CODE HERE
    #
    # jax.grad computes gradients of scalar functions.
    # By default, it computes gradient w.r.t. first argument.
    # Use argnums to specify multiple arguments.
    #
    # grad_fn = jax.grad(attention_loss, argnums=(0, 1, 2))
    # return grad_fn(Q, K, V)

    raise NotImplementedError("Implement attention_gradient")


def value_and_gradient(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Compute both the loss value and gradients in one pass.

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)

    Returns:
        Tuple of (loss_value, (dL_dQ, dL_dK, dL_dV))

    Hint:
        Use jax.value_and_grad instead of jax.grad

    Example:
        >>> Q = jnp.ones((4, 8))
        >>> K = jnp.ones((6, 8))
        >>> V = jnp.ones((6, 16))
        >>> loss, (dQ, dK, dV) = value_and_gradient(Q, K, V)
    """
    # YOUR CODE HERE
    #
    # jax.value_and_grad returns both the function value and gradients.
    # This is more efficient than computing them separately.
    #
    # val_grad_fn = jax.value_and_grad(attention_loss, argnums=(0, 1, 2))
    # return val_grad_fn(Q, K, V)

    raise NotImplementedError("Implement value_and_gradient")


# =============================================================================
# Training Step
# =============================================================================

def simple_model(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """
    Simple two-layer model for training demonstration.

    params contains:
        - 'W1': shape (in_features, hidden)
        - 'b1': shape (hidden,)
        - 'W2': shape (hidden, out_features)
        - 'b2': shape (out_features,)
    """
    h = x @ params['W1'] + params['b1']
    h = jax.nn.relu(h)
    return h @ params['W2'] + params['b2']


def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((pred - target) ** 2)


def train_step(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    lr: float = 0.01
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Single training step: forward, backward, update.

    Args:
        params: Dictionary of model parameters
        x: Input tensor of shape (batch, in_features)
        y: Target tensor of shape (batch, out_features)
        lr: Learning rate

    Returns:
        Tuple of (updated_params, loss_value)

    Algorithm:
        1. Compute loss and gradients w.r.t. params
        2. Update params with gradient descent: param -= lr * grad
        3. Return updated params and loss

    Hint:
        Define an inner loss function that only takes params,
        then use jax.value_and_grad on it.

    Example:
        >>> params = {'W1': jnp.ones((4, 8)), 'b1': jnp.zeros(8),
        ...           'W2': jnp.ones((8, 2)), 'b2': jnp.zeros(2)}
        >>> x = jnp.ones((16, 4))
        >>> y = jnp.ones((16, 2))
        >>> new_params, loss = train_step(params, x, y)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Define a loss function that only takes params
    #    def loss_fn(p):
    #        pred = simple_model(p, x)
    #        return mse_loss(pred, y)
    #
    # 2. Compute loss and gradients
    #    loss, grads = jax.value_and_grad(loss_fn)(params)
    #
    # 3. Update parameters (gradient descent)
    #    Use jax.tree.map to apply the same operation to all leaves
    #    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    #
    # 4. Return new params and loss

    raise NotImplementedError("Implement train_step")


# =============================================================================
# Composing Transformations
# =============================================================================

def jit_batched_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """
    JIT-compiled batched attention.

    Args:
        Q: Query tensor of shape (batch, seq_q, d_k)
        K: Key tensor of shape (batch, seq_k, d_k)
        V: Value tensor of shape (batch, seq_k, d_v)

    Returns:
        Output tensor of shape (batch, seq_q, d_v)

    Compose jit and vmap for maximum performance.

    Hint:
        Order matters! Usually you want vmap inside jit:
        jax.jit(jax.vmap(fn))

    Example:
        >>> Q = jnp.ones((8, 4, 16))
        >>> K = jnp.ones((8, 6, 16))
        >>> V = jnp.ones((8, 6, 32))
        >>> output = jit_batched_attention(Q, K, V)
        >>> output.shape
        (8, 4, 32)
    """
    # YOUR CODE HERE
    #
    # Compose vmap and jit:
    # return jax.jit(jax.vmap(attention_single))(Q, K, V)

    raise NotImplementedError("Implement jit_batched_attention")


# Create a properly cached version for performance testing
_jit_batched_attention_cached = jax.jit(jax.vmap(attention_single))


def jit_batched_attention_cached(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """Cached JIT batched attention for performance testing."""
    return _jit_batched_attention_cached(Q, K, V)


# =============================================================================
# Multi-Head Attention with Nested vmap
# =============================================================================

def attention_head(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    scale: float
) -> jnp.ndarray:
    """
    Single attention head.

    Args:
        q: Query of shape (seq_q, d_head)
        k: Key of shape (seq_k, d_head)
        v: Value of shape (seq_k, d_head)
        scale: Scale factor (1/sqrt(d_head))

    Returns:
        Output of shape (seq_q, d_head)
    """
    scores = jnp.einsum('qd,kd->qk', q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('qk,kd->qd', weights, v)


def multi_head_batched(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    num_heads: int
) -> jnp.ndarray:
    """
    Batched multi-head attention using nested vmap.

    Args:
        Q: Query tensor of shape (batch, seq_q, d_model)
        K: Key tensor of shape (batch, seq_k, d_model)
        V: Value tensor of shape (batch, seq_k, d_model)
        num_heads: Number of attention heads

    Returns:
        Output tensor of shape (batch, seq_q, d_model)

    Algorithm:
        1. Reshape to (batch, seq, num_heads, d_head)
        2. Apply attention per head using vmap
        3. Apply attention per batch using vmap
        4. Reshape back to (batch, seq_q, d_model)

    Hint:
        Use nested vmap: outer for batch, inner for heads
        Or transpose and use single vmap over combined (batch * heads) dimension

    Example:
        >>> Q = jnp.ones((4, 8, 64))  # batch=4, seq=8, d_model=64
        >>> K = jnp.ones((4, 8, 64))
        >>> V = jnp.ones((4, 8, 64))
        >>> output = multi_head_batched(Q, K, V, num_heads=8)
        >>> output.shape
        (4, 8, 64)
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get dimensions
    #    batch, seq_q, d_model = Q.shape
    #    seq_k = K.shape[1]
    #    d_head = d_model // num_heads
    #    scale = 1.0 / jnp.sqrt(d_head)
    #
    # 2. Reshape to separate heads
    #    Q_heads = Q.reshape(batch, seq_q, num_heads, d_head)
    #    K_heads = K.reshape(batch, seq_k, num_heads, d_head)
    #    V_heads = V.reshape(batch, seq_k, num_heads, d_head)
    #
    # 3. Transpose to (batch, num_heads, seq, d_head) for easier vmap
    #    Q_t = jnp.transpose(Q_heads, (0, 2, 1, 3))  # (batch, heads, seq_q, d_head)
    #    K_t = jnp.transpose(K_heads, (0, 2, 1, 3))  # (batch, heads, seq_k, d_head)
    #    V_t = jnp.transpose(V_heads, (0, 2, 1, 3))  # (batch, heads, seq_k, d_head)
    #
    # 4. vmap over heads (axis 1), then vmap over batch (axis 0)
    #    # Inner vmap: over heads
    #    head_fn = jax.vmap(attention_head, in_axes=(0, 0, 0, None))
    #    # Outer vmap: over batch
    #    batched_fn = jax.vmap(head_fn, in_axes=(0, 0, 0, None))
    #
    #    output_t = batched_fn(Q_t, K_t, V_t, scale)  # (batch, heads, seq_q, d_head)
    #
    # 5. Transpose back and reshape
    #    output = jnp.transpose(output_t, (0, 2, 1, 3))  # (batch, seq_q, heads, d_head)
    #    output = output.reshape(batch, seq_q, d_model)
    #
    # return output

    raise NotImplementedError("Implement multi_head_batched")


# =============================================================================
# BONUS: Higher-Order Gradients
# =============================================================================

def hessian_diagonal(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray
) -> jnp.ndarray:
    """
    BONUS: Compute diagonal of Hessian w.r.t. Q.

    The Hessian is the matrix of second derivatives.
    The diagonal contains d²L/dQ_i² for each element.

    Args:
        Q: Query tensor of shape (seq_q, d_k)
        K: Key tensor of shape (seq_k, d_k)
        V: Value tensor of shape (seq_k, d_v)

    Returns:
        Hessian diagonal with same shape as Q

    Hint:
        Use nested jax.grad to compute second derivatives
    """
    # YOUR CODE HERE (optional bonus challenge)
    raise NotImplementedError("Bonus: Implement hessian_diagonal")
