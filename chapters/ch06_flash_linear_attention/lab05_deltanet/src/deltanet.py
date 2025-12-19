"""
Lab 05: DeltaNet

Implement DeltaNet using the delta rule for linear attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict


def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """ELU + 1 feature map."""
    return np.where(x > 0, x + 1, np.exp(x))


def delta_rule_update(
    state: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    feature_map_fn: Callable = None,
    beta: float = 1.0
) -> np.ndarray:
    """
    Apply the delta rule update to the state.

    The delta rule comes from Hopfield networks and implements
    error-correction learning:
        1. Retrieve current value at key k: retrieved = S^T @ φ(k)
        2. Compute error: error = v - beta * retrieved
        3. Update state: S_new = S + φ(k)^T @ error

    Expanded form:
        S_new = S + φ(k)^T @ v - beta * φ(k)^T @ (S^T @ φ(k))
              = S + φ(k)^T @ v - beta * (φ(k)^T @ φ(k)) @ S

    The second term removes the old value at key k (implicit forgetting).

    Args:
        state: Current state of shape (..., d_k, d_v)
        k: Key tensor of shape (..., d_k)
        v: Value tensor of shape (..., d_v)
        feature_map_fn: Feature map for k (default: elu_plus_one)
        beta: Forgetting strength. 0 = no forgetting (vanilla linear attn),
              1 = full delta rule (complete overwriting)

    Returns:
        new_state: Updated state of same shape as input state
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement delta_rule_update")


def deltanet_step(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    state: Optional[np.ndarray] = None,
    feature_map_fn: Callable = None,
    beta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-step DeltaNet for autoregressive inference.

    Args:
        q: Query for single position, shape (..., d_k)
        k: Key for single position, shape (..., d_k)
        v: Value for single position, shape (..., d_v)
        state: Current state of shape (..., d_k, d_v)
               If None, initializes to zeros
        feature_map_fn: Feature map for q and k (default: elu_plus_one)
        beta: Forgetting strength

    Returns:
        output: Output for this position, shape (..., d_v)
        new_state: Updated state

    Steps:
        1. Apply delta rule update to state
        2. Query the updated state: output = φ(q) @ new_state
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement deltanet_step")


def deltanet_recurrent(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map_fn: Callable = None,
    beta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full sequence DeltaNet using recurrent computation.

    Args:
        Q: Queries, shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Keys, same shape as Q
        V: Values, shape (..., seq_len, d_v)
        feature_map_fn: Feature map (default: elu_plus_one)
        beta: Forgetting strength

    Returns:
        output: Attention output, shape matching V
        final_state: Final state
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement deltanet_recurrent")


class DeltaNet:
    """
    DeltaNet module using the delta rule for linear attention.

    DeltaNet uses error-correction instead of explicit gating:
        S_new = S + φ(k)^T @ (v - beta * S^T @ φ(k))

    This achieves implicit forgetting: before writing a new value at key k,
    it first removes the old value that was stored at k.

    Architecture:
        q = W_q(x)
        k = W_k(x)
        v = W_v(x)

        State update (delta rule):
        retrieved = S^T @ φ(k)
        error = v - beta * retrieved
        S = S + φ(k)^T @ error

        Output:
        o = φ(q) @ S
    """

    def __init__(
        self,
        d_model: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        beta: float = 1.0,
        learnable_beta: bool = False,
        feature_map: str = 'elu_plus_one'
    ):
        """
        Initialize DeltaNet module.

        Args:
            d_model: Input/output dimension
            d_k: Key/query dimension (default: d_model)
            d_v: Value dimension (default: d_model)
            beta: Initial forgetting strength
            learnable_beta: If True, beta can be learned (not implemented in NumPy)
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
        Forward pass of DeltaNet.

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


def compare_deltanet_beta(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    betas: list = [0.0, 0.5, 1.0],
    feature_map_fn: Callable = None
) -> Dict[str, np.ndarray]:
    """
    Compare DeltaNet outputs with different beta values.

    Args:
        Q, K, V: Input tensors
        betas: List of beta values to compare
        feature_map_fn: Feature map

    Returns:
        Dictionary with:
        - 'beta_{x}_output': Output for each beta value
        - 'beta_{x}_state': Final state for each beta value
        - 'beta_0_matches_vanilla': Whether beta=0 matches vanilla linear attn
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_deltanet_beta")


def analyze_memory_capacity(
    d_k: int,
    d_v: int,
    num_patterns: int,
    beta: float = 1.0
) -> Dict[str, float]:
    """
    Analyze the memory capacity of DeltaNet.

    Store num_patterns key-value pairs, then try to retrieve them.
    Measure retrieval accuracy.

    Args:
        d_k: Key dimension
        d_v: Value dimension
        num_patterns: Number of patterns to store
        beta: Forgetting strength

    Returns:
        Dictionary with:
        - 'retrieval_accuracy': Fraction of patterns correctly retrieved
        - 'avg_retrieval_error': Average error in retrieval
        - 'capacity_ratio': num_patterns / min(d_k, d_v)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement analyze_memory_capacity")
