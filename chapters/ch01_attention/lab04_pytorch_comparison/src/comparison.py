"""
Lab 04: PyTorch Comparison

Verify your implementation matches PyTorch's nn.MultiheadAttention.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Tuple

# Import PyTorch for comparison
import torch
import torch.nn as nn

# Import your implementation from Lab 03
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lab03_multihead" / "src"))

# Note: Students need to complete Lab 03 first
# from multihead import MultiHeadAttention


class MultiHeadAttention:
    """
    Multi-Head Attention (stub for Lab 04).

    Students should either:
    1. Import their implementation from Lab 03
    2. Or copy their implementation here

    This stub is provided so tests can run even if Lab 03 is incomplete.
    """

    def __init__(self, d_model: int, num_heads: int):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Initialize weight matrices
        # Shape: (d_model, d_model) - input features to output features
        scale = 0.02
        self.W_Q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_K = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_V = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_O = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, d_k)."""
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement _split_heads or copy from Lab 03")

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine heads back into d_model dimension."""
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement _combine_heads or copy from Lab 03")

    def forward(self, x: np.ndarray, mask=None) -> np.ndarray:
        """Compute multi-head self-attention."""
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement forward or copy from Lab 03")

    def __call__(self, x: np.ndarray, mask=None) -> np.ndarray:
        return self.forward(x, mask)


def load_weights_from_pytorch(
    numpy_mha: MultiHeadAttention,
    pytorch_mha: nn.MultiheadAttention
) -> None:
    """
    Load weights from PyTorch MHA into NumPy MHA.

    PyTorch's nn.MultiheadAttention stores weights as:
    - in_proj_weight: shape (3 * embed_dim, embed_dim)
      Contains [W_Q, W_K, W_V] stacked vertically
    - out_proj.weight: shape (embed_dim, embed_dim)

    PyTorch convention: weight shape is (out_features, in_features)
    Our convention: weight shape is (in_features, out_features)

    So we need to transpose when copying.

    Args:
        numpy_mha: Your NumPy implementation (will be modified in-place)
        pytorch_mha: PyTorch's nn.MultiheadAttention with weights to copy

    Example:
        >>> pytorch_mha = nn.MultiheadAttention(64, 8, batch_first=True, bias=False)
        >>> numpy_mha = MultiHeadAttention(64, 8)
        >>> load_weights_from_pytorch(numpy_mha, pytorch_mha)
        >>> # numpy_mha now has same weights as pytorch_mha
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get in_proj_weight from pytorch_mha (shape: 3*d_model, d_model)
    # 2. Split into W_Q, W_K, W_V (each shape: d_model, d_model)
    # 3. Transpose each (PyTorch: out x in, NumPy: in x out)
    # 4. Copy to numpy_mha.W_Q, W_K, W_V
    # 5. Do the same for out_proj.weight -> W_O
    #
    # Hint: Use .detach().numpy() to convert PyTorch tensors
    raise NotImplementedError("Implement load_weights_from_pytorch")


def compare_outputs(
    numpy_mha: MultiHeadAttention,
    pytorch_mha: nn.MultiheadAttention,
    x: np.ndarray,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Compare outputs of NumPy and PyTorch implementations.

    Args:
        numpy_mha: Your NumPy implementation
        pytorch_mha: PyTorch's nn.MultiheadAttention
        x: Input array of shape (batch, seq_len, d_model)
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of:
        - match: True if outputs match within tolerance
        - max_diff: Maximum absolute difference between outputs

    Example:
        >>> match, diff = compare_outputs(numpy_mha, pytorch_mha, x)
        >>> print(f"Match: {match}, Max diff: {diff:.2e}")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Run input through numpy_mha
    # 2. Convert input to PyTorch tensor
    # 3. Run through pytorch_mha (use torch.no_grad(), and handle the tuple return)
    # 4. Convert PyTorch output back to NumPy
    # 5. Compare and return (match, max_diff)
    raise NotImplementedError("Implement compare_outputs")


def create_matching_mha(
    d_model: int,
    num_heads: int
) -> Tuple[MultiHeadAttention, nn.MultiheadAttention]:
    """
    Create a NumPy MHA with weights copied from a PyTorch MHA.

    This is a convenience function for testing.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads

    Returns:
        Tuple of (numpy_mha, pytorch_mha) with matching weights
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Create PyTorch MHA (use batch_first=True, bias=False)
    # 2. Create NumPy MHA
    # 3. Load weights from PyTorch to NumPy
    # 4. Return both
    raise NotImplementedError("Implement create_matching_mha")


def analyze_weight_differences(
    numpy_mha: MultiHeadAttention,
    pytorch_mha: nn.MultiheadAttention
) -> dict:
    """
    Analyze the differences between weight matrices.

    Useful for debugging when outputs don't match.

    Args:
        numpy_mha: Your NumPy implementation
        pytorch_mha: PyTorch's nn.MultiheadAttention

    Returns:
        Dictionary with analysis for each weight matrix:
        {
            'W_Q': {'max_diff': float, 'mean_diff': float},
            'W_K': {...},
            'W_V': {...},
            'W_O': {...}
        }
    """
    # YOUR CODE HERE
    #
    # Compare each weight matrix and compute statistics
    raise NotImplementedError("Implement analyze_weight_differences")
