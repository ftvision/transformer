"""Tests for Lab 04: PyTorch Comparison."""

import numpy as np
import pytest
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comparison import (
    MultiHeadAttention,
    load_weights_from_pytorch,
    compare_outputs,
    create_matching_mha,
    analyze_weight_differences,
)


class TestLoadWeights:
    """Tests for weight loading from PyTorch."""

    def test_load_weights_shapes(self):
        """Loaded weights should have correct shapes."""
        d_model, num_heads = 64, 8

        pytorch_mha = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, bias=False
        )
        numpy_mha = MultiHeadAttention(d_model, num_heads)

        load_weights_from_pytorch(numpy_mha, pytorch_mha)

        assert numpy_mha.W_Q.shape == (d_model, d_model)
        assert numpy_mha.W_K.shape == (d_model, d_model)
        assert numpy_mha.W_V.shape == (d_model, d_model)
        assert numpy_mha.W_O.shape == (d_model, d_model)

    def test_load_weights_not_zero(self):
        """Loaded weights should not be all zeros."""
        d_model, num_heads = 64, 8

        pytorch_mha = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, bias=False
        )
        numpy_mha = MultiHeadAttention(d_model, num_heads)

        load_weights_from_pytorch(numpy_mha, pytorch_mha)

        assert not np.allclose(numpy_mha.W_Q, 0)
        assert not np.allclose(numpy_mha.W_K, 0)
        assert not np.allclose(numpy_mha.W_V, 0)
        assert not np.allclose(numpy_mha.W_O, 0)

    def test_load_weights_matches_pytorch(self):
        """Loaded W_Q should match first chunk of in_proj_weight."""
        d_model, num_heads = 64, 8

        pytorch_mha = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, bias=False
        )
        numpy_mha = MultiHeadAttention(d_model, num_heads)

        load_weights_from_pytorch(numpy_mha, pytorch_mha)

        # Get PyTorch weights
        in_proj = pytorch_mha.in_proj_weight.detach().numpy()
        pytorch_W_Q = in_proj[:d_model, :].T  # Transpose: (out, in) -> (in, out)

        np.testing.assert_allclose(numpy_mha.W_Q, pytorch_W_Q, rtol=1e-5)


class TestCompareOutputs:
    """Tests for output comparison."""

    def test_compare_matching_outputs(self):
        """Matching implementations should return True."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        x = np.random.randn(2, 10, d_model).astype(np.float32)

        match, max_diff = compare_outputs(numpy_mha, pytorch_mha, x)

        assert match, f"Outputs should match, but max_diff={max_diff}"
        assert max_diff < 1e-5, f"Max diff too large: {max_diff}"

    def test_compare_returns_max_diff(self):
        """Should return the maximum difference."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        x = np.random.randn(2, 10, d_model).astype(np.float32)

        _, max_diff = compare_outputs(numpy_mha, pytorch_mha, x)

        assert isinstance(max_diff, float)
        assert max_diff >= 0

    def test_compare_different_inputs(self):
        """Different inputs should give consistent results."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        for _ in range(3):
            x = np.random.randn(2, 10, d_model).astype(np.float32)
            match, _ = compare_outputs(numpy_mha, pytorch_mha, x)
            assert match, "Should match for any input"


class TestCreateMatchingMHA:
    """Tests for create_matching_mha."""

    def test_creates_both(self):
        """Should create both NumPy and PyTorch MHA."""
        numpy_mha, pytorch_mha = create_matching_mha(64, 8)

        assert isinstance(numpy_mha, MultiHeadAttention)
        assert isinstance(pytorch_mha, nn.MultiheadAttention)

    def test_correct_dimensions(self):
        """Created MHAs should have correct dimensions."""
        d_model, num_heads = 128, 4

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        assert numpy_mha.d_model == d_model
        assert numpy_mha.num_heads == num_heads
        assert pytorch_mha.embed_dim == d_model
        assert pytorch_mha.num_heads == num_heads

    def test_weights_match(self):
        """Created MHAs should have matching weights."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        # Check W_Q matches
        in_proj = pytorch_mha.in_proj_weight.detach().numpy()
        pytorch_W_Q = in_proj[:d_model, :].T

        np.testing.assert_allclose(numpy_mha.W_Q, pytorch_W_Q, rtol=1e-5)


class TestAnalyzeWeights:
    """Tests for analyze_weight_differences."""

    def test_analyze_matching_weights(self):
        """Matching weights should have zero difference."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        analysis = analyze_weight_differences(numpy_mha, pytorch_mha)

        assert 'W_Q' in analysis
        assert 'W_K' in analysis
        assert 'W_V' in analysis
        assert 'W_O' in analysis

        for key in ['W_Q', 'W_K', 'W_V', 'W_O']:
            assert analysis[key]['max_diff'] < 1e-5

    def test_analyze_returns_stats(self):
        """Should return max_diff and mean_diff for each weight."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        analysis = analyze_weight_differences(numpy_mha, pytorch_mha)

        for key in ['W_Q', 'W_K', 'W_V', 'W_O']:
            assert 'max_diff' in analysis[key]
            assert 'mean_diff' in analysis[key]


class TestEndToEnd:
    """End-to-end tests verifying full implementation."""

    def test_self_attention_matches(self):
        """Self-attention output should match PyTorch exactly."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        # Test input
        x_np = np.random.randn(2, 10, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # NumPy output
        out_np = numpy_mha(x_np)

        # PyTorch output
        with torch.no_grad():
            out_torch, _ = pytorch_mha(x_torch, x_torch, x_torch)

        np.testing.assert_allclose(
            out_np, out_torch.numpy(),
            atol=1e-5,
            err_msg="NumPy and PyTorch outputs don't match!"
        )

    def test_different_batch_sizes(self):
        """Should work with various batch sizes."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        for batch_size in [1, 2, 4, 8]:
            x = np.random.randn(batch_size, 10, d_model).astype(np.float32)
            match, max_diff = compare_outputs(numpy_mha, pytorch_mha, x)
            assert match, f"Failed for batch_size={batch_size}, diff={max_diff}"

    def test_different_seq_lengths(self):
        """Should work with various sequence lengths."""
        d_model, num_heads = 64, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        for seq_len in [5, 10, 20, 50]:
            x = np.random.randn(2, seq_len, d_model).astype(np.float32)
            match, max_diff = compare_outputs(numpy_mha, pytorch_mha, x)
            assert match, f"Failed for seq_len={seq_len}, diff={max_diff}"

    def test_larger_model(self):
        """Should work with larger model dimensions."""
        d_model, num_heads = 256, 8

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        x = np.random.randn(2, 10, d_model).astype(np.float32)
        match, max_diff = compare_outputs(numpy_mha, pytorch_mha, x)

        assert match, f"Failed for d_model={d_model}, diff={max_diff}"


class TestMilestone:
    """
    Chapter 1 Milestone Test

    Your multi-head attention should match PyTorch's nn.MultiheadAttention
    output within 1e-5 tolerance.
    """

    def test_chapter_1_milestone(self):
        """
        MILESTONE: Match PyTorch within 1e-5 tolerance.

        This is the final test for Chapter 1.
        Passing this means you've correctly implemented multi-head attention!
        """
        # Standard GPT-2 small dimensions
        d_model = 768
        num_heads = 12

        numpy_mha, pytorch_mha = create_matching_mha(d_model, num_heads)

        # Test with realistic input
        batch_size = 4
        seq_len = 128
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

        match, max_diff = compare_outputs(numpy_mha, pytorch_mha, x, atol=1e-5)

        assert match, (
            f"MILESTONE NOT MET: Max difference is {max_diff:.2e}, "
            f"expected < 1e-5"
        )

        print(f"\n{'='*60}")
        print("CONGRATULATIONS! Chapter 1 Milestone Achieved!")
        print(f"Max difference: {max_diff:.2e} (< 1e-5)")
        print(f"{'='*60}\n")
