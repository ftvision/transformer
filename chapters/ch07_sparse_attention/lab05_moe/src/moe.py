"""
Lab 05: Mixture of Experts (MoE)

Implement a Mixture of Experts layer with routing and load balancing.

Your task: Complete the classes and functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple, Optional, List
import math


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * (1 / (1 + np.exp(-x)))


class Expert:
    """
    A single expert (feed-forward network).

    Each expert is a standard FFN:
        hidden = silu(x @ W1)
        output = hidden @ W2

    Attributes:
        d_model: Input/output dimension
        d_ff: Hidden dimension
        W1: Up-projection (d_model, d_ff)
        W2: Down-projection (d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize expert FFN.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden (intermediate) dimension
        """
        # YOUR CODE HERE
        #
        # 1. Store d_model, d_ff
        # 2. Initialize W1: (d_model, d_ff)
        # 3. Initialize W2: (d_ff, d_model)
        # Use np.random.randn(...).astype(np.float32) * 0.02
        raise NotImplementedError("Implement Expert.__init__")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through expert.

        Args:
            x: Input tensor (..., d_model)

        Returns:
            Output tensor (..., d_model)
        """
        # YOUR CODE HERE
        #
        # hidden = silu(x @ W1)
        # output = hidden @ W2
        raise NotImplementedError("Implement Expert.forward")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Router:
    """
    Router that determines which experts process each token.

    The router computes a probability distribution over experts
    for each token, then selects the top-k experts.

    Attributes:
        d_model: Input dimension
        num_experts: Number of experts to route to
        W_router: Routing weight matrix (d_model, num_experts)
    """

    def __init__(self, d_model: int, num_experts: int):
        """
        Initialize router.

        Args:
            d_model: Input dimension
            num_experts: Number of experts
        """
        # YOUR CODE HERE
        #
        # 1. Store d_model, num_experts
        # 2. Initialize W_router: (d_model, num_experts)
        raise NotImplementedError("Implement Router.__init__")

    def forward(
        self,
        x: np.ndarray,
        top_k: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute routing decisions.

        Args:
            x: Input tensor (..., d_model)
            top_k: Number of experts to select per token

        Returns:
            top_k_indices: Expert indices (..., top_k)
            top_k_weights: Normalized weights for selected experts (..., top_k)
            router_probs: Full probability distribution (..., num_experts)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Compute router logits: logits = x @ W_router
        # 2. Compute full probabilities: router_probs = softmax(logits)
        # 3. Select top-k experts: use np.argsort or similar
        # 4. Get top-k probabilities
        # 5. Normalize top-k weights to sum to 1
        # 6. Return (top_k_indices, top_k_weights, router_probs)
        raise NotImplementedError("Implement Router.forward")

    def __call__(
        self,
        x: np.ndarray,
        top_k: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x, top_k)


def compute_aux_loss(
    router_probs: np.ndarray,
    top_k_indices: np.ndarray,
    num_experts: int
) -> float:
    """
    Compute auxiliary load balancing loss.

    This loss encourages uniform expert usage by penalizing
    when some experts receive more tokens than others.

    Loss = num_experts * Σ_i (f_i * P_i)

    Where:
    - f_i = fraction of tokens routed to expert i
    - P_i = mean routing probability assigned to expert i

    Args:
        router_probs: Full routing probabilities (..., num_experts)
        top_k_indices: Selected expert indices (..., top_k)
        num_experts: Total number of experts

    Returns:
        Auxiliary loss value (scalar)

    Note:
        When experts are used uniformly (f_i = P_i = 1/num_experts),
        the loss equals 1.0. Higher values indicate imbalance.
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Flatten router_probs and top_k_indices
    # 2. Compute f_i: fraction of tokens assigned to each expert
    #    - Count how many times each expert appears in top_k_indices
    #    - Divide by total number of (token, expert) assignments
    # 3. Compute P_i: mean routing probability for each expert
    #    - Average of router_probs[:, i] across all tokens
    # 4. Return num_experts * sum(f_i * P_i)
    raise NotImplementedError("Implement compute_aux_loss")


def compute_expert_usage(
    top_k_indices: np.ndarray,
    num_experts: int
) -> np.ndarray:
    """
    Compute the fraction of tokens routed to each expert.

    Args:
        top_k_indices: Selected expert indices (..., top_k)
        num_experts: Total number of experts

    Returns:
        Array of shape (num_experts,) with usage fraction for each expert
        Sums to 1.0 (if top_k=1) or top_k (if top_k>1, normalized per assignment)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_expert_usage")


class MixtureOfExperts:
    """
    Mixture of Experts layer.

    Each token is routed to top-k experts, and the output is a
    weighted sum of the selected experts' outputs.

    Attributes:
        d_model: Input/output dimension
        d_ff: Expert hidden dimension
        num_experts: Number of expert FFNs
        top_k: Number of experts per token
        aux_loss_coef: Coefficient for auxiliary loss
        experts: List of Expert instances
        router: Router instance
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coef: float = 0.01
    ):
        """
        Initialize MoE layer.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension for each expert
            num_experts: Number of experts
            top_k: Number of experts to use per token
            aux_loss_coef: Weight for load balancing loss

        Raises:
            ValueError: If top_k > num_experts
        """
        # YOUR CODE HERE
        #
        # 1. Validate top_k <= num_experts
        # 2. Store all parameters
        # 3. Create router
        # 4. Create list of experts
        # 5. Initialize tracking variables (aux_loss, expert_usage)
        raise NotImplementedError("Implement MixtureOfExperts.__init__")

    def forward(
        self,
        x: np.ndarray,
        return_aux_loss: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch, seq_len, d_model) or (seq_len, d_model)
            return_aux_loss: Whether to compute and return auxiliary loss

        Returns:
            output: Output tensor, same shape as x
            aux_loss: Auxiliary load balancing loss (if return_aux_loss=True)
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Handle 2D vs 3D input
        # 2. Get routing decisions from router
        # 3. For each token, compute weighted sum of selected experts' outputs
        # 4. Compute auxiliary loss (if requested)
        # 5. Update expert usage tracking
        # 6. Return (output, aux_loss)
        #
        # Implementation note:
        # The naive way loops over tokens and experts.
        # A more efficient way batches tokens going to the same expert.
        # For this lab, the naive way is fine.
        raise NotImplementedError("Implement MixtureOfExperts.forward")

    def get_expert_usage(self) -> np.ndarray:
        """
        Get the most recent expert usage statistics.

        Returns:
            Array of shape (num_experts,) with fraction of tokens
            routed to each expert in the last forward pass
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_expert_usage")

    def get_aux_loss(self) -> float:
        """
        Get the most recent auxiliary loss value.

        Returns:
            Auxiliary loss from the last forward pass
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_aux_loss")

    def __call__(
        self,
        x: np.ndarray,
        return_aux_loss: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        return self.forward(x, return_aux_loss)


class SparseMoE(MixtureOfExperts):
    """
    Sparse MoE with expert capacity limiting.

    This version limits how many tokens each expert can process,
    dropping tokens that exceed capacity. This prevents any single
    expert from being overwhelmed.

    Attributes:
        capacity_factor: Multiplier for expected tokens per expert
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        capacity_factor: float = 1.25
    ):
        """
        Initialize Sparse MoE with capacity limiting.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension for each expert
            num_experts: Number of experts
            top_k: Number of experts per token
            aux_loss_coef: Weight for load balancing loss
            capacity_factor: Multiplier for expert capacity
                            capacity = (num_tokens / num_experts) * capacity_factor
        """
        super().__init__(d_model, d_ff, num_experts, top_k, aux_loss_coef)
        # YOUR CODE HERE
        raise NotImplementedError("Implement SparseMoE.__init__")

    def compute_capacity(self, num_tokens: int) -> int:
        """
        Compute the maximum tokens per expert.

        Args:
            num_tokens: Total number of tokens to process

        Returns:
            Maximum tokens any single expert should process
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement compute_capacity")

    def forward(
        self,
        x: np.ndarray,
        return_aux_loss: bool = True
    ) -> Tuple[np.ndarray, Optional[float], float]:
        """
        Forward with capacity limiting.

        Returns:
            output: Output tensor
            aux_loss: Load balancing loss
            drop_rate: Fraction of tokens that exceeded capacity
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement SparseMoE.forward")


def analyze_moe_efficiency(
    d_model: int,
    d_ff: int,
    num_experts: int,
    top_k: int,
    seq_len: int,
    batch_size: int
) -> dict:
    """
    Analyze MoE computational efficiency.

    Args:
        d_model: Model dimension
        d_ff: Expert hidden dimension
        num_experts: Number of experts
        top_k: Experts per token
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Dictionary with:
        - 'total_params': Total parameters in MoE
        - 'active_params': Parameters active per token
        - 'dense_flops': FLOPs for equivalent dense FFN
        - 'moe_flops': FLOPs for MoE
        - 'param_efficiency': total_params / active_params
        - 'compute_efficiency': dense_flops / moe_flops
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement analyze_moe_efficiency")
