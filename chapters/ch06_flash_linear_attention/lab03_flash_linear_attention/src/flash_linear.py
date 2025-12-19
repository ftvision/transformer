"""
Lab 03: Flash Linear Attention

Implement memory-efficient linear attention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional, Tuple, Callable, List, Dict
import sys


def elu_plus_one(x: np.ndarray) -> np.ndarray:
    """ELU + 1 feature map (ensures positivity)."""
    return np.where(x > 0, x + 1, np.exp(x))


def naive_linear_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_map_fn: Callable = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Naive linear attention implementation (memory-inefficient).

    This materializes the full cumulative sum tensor.
    Used as a reference to verify correctness.

    Args:
        Q, K, V: Input tensors
        feature_map_fn: Feature map (default: elu_plus_one)

    Returns:
        output: Attention output
        cumsum_tensor: The full cumsum tensor (for memory comparison)
    """
    if feature_map_fn is None:
        feature_map_fn = elu_plus_one

    Q = feature_map_fn(Q)
    K = feature_map_fn(K)

    # Compute outer products at each position
    # This is the memory-expensive part!
    if Q.ndim == 2:
        # (seq_len, d_k), (seq_len, d_v) -> (seq_len, d_k, d_v)
        KV = np.einsum('nd,nv->ndv', K, V)
    else:
        # (batch, seq_len, d_k) -> (batch, seq_len, d_k, d_v)
        KV = np.einsum('bnd,bnv->bndv', K, V)

    # Cumulative sum - the main memory consumer
    S = np.cumsum(KV, axis=-3)

    # Query the accumulated states
    if Q.ndim == 2:
        output = np.einsum('nd,ndv->nv', Q, S)
    else:
        output = np.einsum('bnd,bndv->bnv', Q, S)

    return output, S


def tiled_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    chunk_size: int = 64,
    feature_map_fn: Callable = None,
    save_states: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Memory-efficient forward pass using tiled computation.

    Instead of materializing the full cumsum tensor, we:
    1. Process one chunk at a time
    2. Keep a running state
    3. Only save states at chunk boundaries (for backward pass)

    Args:
        Q: Queries, shape (seq_len, d_k) or (batch, seq_len, d_k)
        K: Keys, same shape as Q
        V: Values, shape (..., seq_len, d_v)
        chunk_size: Number of positions per chunk
        feature_map_fn: Feature map (default: elu_plus_one)
        save_states: Whether to save boundary states

    Returns:
        output: Attention output, same shape as V
        saved_states: List of states at chunk boundaries (for backward)
                      Length = num_chunks
        final_state: Final accumulated state

    Memory savings:
        - Naive: O(n × d_k × d_v) for cumsum tensor
        - Tiled: O(n/C × d_k × d_v) for saved states + O(C × d) working memory
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement tiled_forward")


def compute_memory_footprint(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    method: str = 'naive'
) -> Dict[str, int]:
    """
    Compute the memory footprint of different implementations.

    Args:
        Q, K, V: Input tensors
        method: 'naive' or 'tiled'

    Returns:
        Dictionary with memory statistics:
        - 'input_bytes': Memory for inputs
        - 'intermediate_bytes': Memory for intermediate tensors
        - 'output_bytes': Memory for outputs
        - 'total_bytes': Total memory
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_memory_footprint")


class FlashLinearAttention:
    """
    Flash Linear Attention module.

    This class implements memory-efficient linear attention with:
    - Tiled forward pass
    - State checkpointing for backward pass
    - Memory usage tracking
    """

    def __init__(
        self,
        chunk_size: int = 64,
        feature_map: str = 'elu_plus_one'
    ):
        """
        Initialize Flash Linear Attention.

        Args:
            chunk_size: Size of each processing chunk
            feature_map: Name of feature map to use
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray
    ) -> np.ndarray:
        """
        Memory-efficient forward pass.

        Args:
            Q, K, V: Input tensors

        Returns:
            Attention output

        Side effects:
            Saves necessary states for backward pass
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward")

    def get_saved_states(self) -> List[np.ndarray]:
        """
        Get the saved states from the last forward pass.

        Returns:
            List of states at chunk boundaries
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_saved_states")

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get memory statistics from the last forward pass.

        Returns:
            Dictionary with:
            - 'peak_memory': Peak memory during forward
            - 'saved_states_memory': Memory for saved states
            - 'num_chunks': Number of chunks processed
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_memory_stats")

    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Allow calling as function."""
        return self.forward(Q, K, V)


def flash_vs_naive_comparison(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    chunk_size: int = 64
) -> Dict[str, any]:
    """
    Compare Flash Linear Attention to naive implementation.

    Args:
        Q, K, V: Input tensors
        chunk_size: Chunk size for flash implementation

    Returns:
        Dictionary with:
        - 'outputs_match': Whether outputs match within tolerance
        - 'max_diff': Maximum difference in outputs
        - 'naive_memory': Estimated memory for naive
        - 'flash_memory': Estimated memory for flash
        - 'memory_ratio': naive_memory / flash_memory (savings factor)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement flash_vs_naive_comparison")


def optimal_chunk_size(
    seq_len: int,
    d_k: int,
    d_v: int,
    available_memory: int
) -> int:
    """
    Compute optimal chunk size given memory constraints.

    The chunk size should:
    1. Fit working memory in available_memory
    2. Balance compute and memory transfer overhead
    3. Be a power of 2 (optional, for efficiency)

    Args:
        seq_len: Sequence length
        d_k: Key/query dimension
        d_v: Value dimension
        available_memory: Available memory in bytes

    Returns:
        Optimal chunk size
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement optimal_chunk_size")
