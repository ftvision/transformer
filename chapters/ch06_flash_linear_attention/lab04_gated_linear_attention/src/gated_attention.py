"""
Lab 04: Gated Linear Attention (GLA)

Implement linear attention with data-dependent gating.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """ELU + 1 feature map."""
    return np.where(x > 0, x + 1, np.exp(x))


def compute_gate(
    x: np.ndarray,
    W_g: np.ndarray,
    b_g: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute data-dependent gate from input.

    The gate controls how much of the previous state to retain:
    - High gate (→1): Keep previous state
    - Low gate (→0): Write new value

    Args:
        x: Input tensor of shape (..., d_in)
        W_g: Gate weight matrix of shape (d_in, d_gate)
        b_g: Optional gate bias of shape (d_gate,)

    Returns:
        Gate tensor of shape (..., d_gate) with values in (0, 1)

    Formula:
        gate = sigmoid(x @ W_g + b_g)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_gate")


def gated_state_update(
    state: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gate: np.ndarray,
    feature_map_fn: Callable = None
) -> np.ndarray:
    """
    Update state with gating mechanism.

    The gated update interpolates between keeping old state and writing new:
        new_state = gate * state + (1 - gate) * (φ(k)^T @ v)

    Args:
        state: Current state of shape (..., d_k, d_v)
        k: Key tensor of shape (..., d_k)
        v: Value tensor of shape (..., d_v)
        gate: Gate tensor of shape (..., d_k) - will be broadcast to state shape
        feature_map_fn: Feature map for k (default: elu_plus_one)

    Returns:
        new_state: Updated state of same shape as input state

    Note:
        The gate is applied element-wise. To match state shape (d_k, d_v),
        we broadcast gate from (d_k,) to (d_k, d_v) by expanding the last dim.
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement gated_state_update")


def gla_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gate: np.ndarray,
    state: Optional[np.ndarray] = None,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-step Gated Linear Attention for autoregressive inference.

    Args:
        q: Query for single position, shape (..., d_k)
        k: Key for single position, shape (..., d_k)
        v: Value for single position, shape (..., d_v)
        gate: Gate for this position, shape (..., d_k)
        state: Current state of shape (..., d_k, d_v)
               If None, initializes to zeros
        feature_map_fn: Feature map for q and k (default: elu_plus_one)

    Returns:
        output: Output for this position, shape (..., d_v)
        new_state: Updated state of shape (..., d_k, d_v)

    Steps:
        1. Update state: new_state = gate * state + (1-gate) * (φ(k)^T @ v)
        2. Query output: output = φ(q) @ new_state
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement gla_step")


def gla_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    gates: np.ndarray,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full sequence GLA using recurrent computation.

    Args:
        Q: Queries, shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Keys, same shape as Q
        V: Values, shape (..., seq_len, d_v)
        gates: Gates, shape matching Q
        feature_map_fn: Feature map (default: elu_plus_one)

    Returns:
        output: Attention output, shape matching V
        final_state: Final state, shape (..., d_k, d_v)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement gla_recurrent")


class GatedLinearAttention:
    """
    Gated Linear Attention (GLA) module.

    GLA adds data-dependent gating to linear attention, allowing the model
    to selectively forget old information.

    Architecture:
        q = W_q(x)
        k = W_k(x)
        v = W_v(x)
        g = sigmoid(W_g(x) + b_g)  # Data-dependent gate

        State update (recurrent view):
        S_t = g_t * S_{t-1} + (1 - g_t) * (φ(k_t)^T @ v_t)

        Output:
        o_t = φ(q_t) @ S_t
    """

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        gate_bias_init: float = 2.0,
        feature_map: str = 'elu_plus_one'
    ):
        """
        Initialize GLA module.

        Args:
            d_model: Input/output dimension
            d_k: Key/query dimension (default: d_model)
            d_v: Value dimension (default: d_model)
            gate_bias_init: Initial bias for gate (positive = remember more)
            feature_map: Name of feature map to use
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of GLA.

        Args:
            x: Input tensor of shape (seq_len, d_model) or
               (batch, seq_len, d_model)
            state: Optional initial state

        Returns:
            output: Attention output of same shape as x
            final_state: Final state
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward")

    def step(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step forward for autoregressive generation.

        Args:
            x: Input for single position, shape (..., d_model)
            state: Current state

        Returns:
            output: Output for this position
            new_state: Updated state
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement step")

    def __call__(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allow calling as function."""
        return self.forward(x, state)


def compare_gla_to_vanilla(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    gates: np.ndarray,
    feature_map_fn: Callable = None
) -> Dict[str, np.ndarray]:
    """
    Compare GLA to vanilla linear attention.

    Args:
        Q, K, V: Input tensors
        gates: Gate values (0 = all write, 1 = all retain)
        feature_map_fn: Feature map

    Returns:
        Dictionary with:
        - 'vanilla_output': Output from vanilla linear attention
        - 'gla_output': Output from GLA
        - 'vanilla_state': Final state from vanilla
        - 'gla_state': Final state from GLA
        - 'state_norm_ratio': norm(gla_state) / norm(vanilla_state)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_gla_to_vanilla")
