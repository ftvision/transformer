"""
Lab 02: Attention Visualization

Build tools to visualize and analyze attention patterns.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def create_attention_heatmap(
    attention_weights: np.ndarray,
    query_labels: List[str],
    key_labels: List[str]
) -> Dict[str, Any]:
    """
    Create a heatmap data structure for attention weights.

    This function prepares attention weights for visualization by packaging
    them with their labels into a structured format.

    Args:
        attention_weights: Attention matrix of shape (seq_len_q, seq_len_k)
                          Values should be between 0 and 1, rows sum to 1
        query_labels: List of labels for query positions (length seq_len_q)
        key_labels: List of labels for key positions (length seq_len_k)

    Returns:
        Dictionary containing:
            - 'weights': The attention weights array
            - 'query_labels': List of query labels
            - 'key_labels': List of key labels
            - 'shape': Tuple of (num_queries, num_keys)

    Raises:
        ValueError: If dimensions don't match

    Example:
        >>> weights = np.array([[0.7, 0.3], [0.4, 0.6]])
        >>> heatmap = create_attention_heatmap(weights, ['Q1', 'Q2'], ['K1', 'K2'])
        >>> heatmap['shape']
        (2, 2)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_attention_heatmap")


def compute_attention_entropy(attention_weights: np.ndarray) -> np.ndarray:
    """
    Compute the entropy of attention distributions.

    Entropy measures how "spread out" the attention is:
    - High entropy: attention distributed across many positions (diffuse)
    - Low entropy: attention concentrated on few positions (peaked)

    Formula: H = -Σ p * log(p)

    For numerical stability, use a small epsilon when computing log
    to avoid log(0).

    Args:
        attention_weights: Attention matrix of shape (seq_len_q, seq_len_k)
                          or (batch, seq_len_q, seq_len_k)
                          Rows should sum to 1

    Returns:
        Entropy for each query position:
            - Shape (seq_len_q,) for 2D input
            - Shape (batch, seq_len_q) for 3D input

    Example:
        >>> # Peaked attention (low entropy)
        >>> peaked = np.array([[1.0, 0.0, 0.0]])
        >>> compute_attention_entropy(peaked)
        array([0.])

        >>> # Uniform attention (high entropy)
        >>> uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        >>> compute_attention_entropy(uniform)
        array([1.386...])  # log(4) ≈ 1.386
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_attention_entropy")


def find_top_k_attended(
    attention_weights: np.ndarray,
    query_idx: int,
    k: int = 3
) -> np.ndarray:
    """
    Find the top-k most attended key positions for a given query.

    This answers: "What positions does query at index `query_idx` attend to most?"

    Args:
        attention_weights: Attention matrix of shape (seq_len_q, seq_len_k)
        query_idx: Index of the query to analyze
        k: Number of top positions to return

    Returns:
        Array of k key indices, sorted by attention weight (highest first)

    Example:
        >>> weights = np.array([
        ...     [0.1, 0.7, 0.1, 0.1],
        ...     [0.4, 0.3, 0.2, 0.1]
        ... ])
        >>> find_top_k_attended(weights, query_idx=0, k=2)
        array([1, 0])  # or [1, 2] or [1, 3] since 0, 2, 3 are tied
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement find_top_k_attended")


def compute_attention_sparsity(
    attention_weights: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Compute the sparsity of attention weights.

    Sparsity is the fraction of attention weights below a threshold.
    High sparsity means attention is concentrated on few positions.

    Args:
        attention_weights: Attention matrix of shape (seq_len_q, seq_len_k)
                          or any shape
        threshold: Weights below this value are considered "sparse"

    Returns:
        Fraction of weights below threshold (between 0 and 1)

    Example:
        >>> # Very sparse attention (mostly zeros)
        >>> sparse = np.array([[0.9, 0.05, 0.05], [0.95, 0.025, 0.025]])
        >>> compute_attention_sparsity(sparse, threshold=0.1)
        0.666...  # 4 out of 6 weights are below 0.1

        >>> # Uniform attention (not sparse)
        >>> uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        >>> compute_attention_sparsity(uniform, threshold=0.1)
        0.0  # No weights below 0.1
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_attention_sparsity")


def compare_attention_patterns(
    weights1: np.ndarray,
    weights2: np.ndarray
) -> Dict[str, float]:
    """
    Compare two attention patterns using various metrics.

    Useful for comparing attention across different heads or layers.

    Args:
        weights1: First attention matrix (seq_len_q, seq_len_k)
        weights2: Second attention matrix (same shape as weights1)

    Returns:
        Dictionary containing:
            - 'cosine_similarity': Average cosine similarity between
                                   corresponding rows
            - 'mse': Mean squared error between the matrices
            - 'max_diff': Maximum absolute difference

    Example:
        >>> w1 = np.array([[0.5, 0.5], [0.3, 0.7]])
        >>> w2 = np.array([[0.5, 0.5], [0.3, 0.7]])
        >>> compare_attention_patterns(w1, w2)
        {'cosine_similarity': 1.0, 'mse': 0.0, 'max_diff': 0.0}
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_attention_patterns")
