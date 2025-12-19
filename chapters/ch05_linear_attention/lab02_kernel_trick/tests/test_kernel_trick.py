"""Tests for Lab 02: The Kernel Trick."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_trick import (
    standard_attention_order,
    linear_attention_order,
    identity_feature_map,
    relu_feature_map,
    elu_feature_map,
    exp_feature_map,
    verify_associativity,
    compare_complexity,
)


class TestStandardAttentionOrder:
    """Tests for standard_attention_order function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        output, attn = standard_attention_order(Q, K, V)

        assert output.shape == (10, 64)
        assert attn.shape == (10, 10)

    def test_attention_matrix_formed(self):
        """Should explicitly form the (n, n) attention matrix."""
        seq_len = 20
        Q = np.random.randn(seq_len, 64)
        K = np.random.randn(seq_len, 64)
        V = np.random.randn(seq_len, 64)

        _, attn = standard_attention_order(Q, K, V)

        # Attention matrix should be (seq_len, seq_len)
        assert attn.shape == (seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """With scaling, attention weights should sum to 1."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        _, attn = standard_attention_order(Q, K, V, scale=True)

        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(10), rtol=1e-5)

    def test_no_scale_raw_matmul(self):
        """Without scaling, should be raw matrix multiplication."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        output, attn = standard_attention_order(Q, K, V, scale=False)

        # Without scaling, attn should just be Q @ K.T
        expected_attn = Q @ K.T
        np.testing.assert_allclose(attn, expected_attn, rtol=1e-5)

        # And output should be attn @ V
        expected_output = expected_attn @ V
        np.testing.assert_allclose(output, expected_output, rtol=1e-5)


class TestLinearAttentionOrder:
    """Tests for linear_attention_order function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        output = linear_attention_order(Q, K, V, identity_feature_map)

        assert output.shape == (10, 64)

    def test_different_seq_lengths(self):
        """Should work for various sequence lengths."""
        d_model = 64

        for seq_len in [10, 100, 500]:
            Q = np.random.randn(seq_len, d_model)
            K = np.random.randn(seq_len, d_model)
            V = np.random.randn(seq_len, d_model)

            output = linear_attention_order(Q, K, V, elu_feature_map)
            assert output.shape == (seq_len, d_model)

    def test_no_nan_output(self):
        """Output should not contain NaN values."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        output = linear_attention_order(Q, K, V, elu_feature_map)

        assert not np.any(np.isnan(output))

    def test_uses_feature_map(self):
        """Different feature maps should give different outputs."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        out_relu = linear_attention_order(Q, K, V, relu_feature_map)
        out_elu = linear_attention_order(Q, K, V, elu_feature_map)

        # Outputs should be different
        assert not np.allclose(out_relu, out_elu)


class TestFeatureMaps:
    """Tests for feature map functions."""

    def test_identity_feature_map(self):
        """Identity should return input unchanged."""
        x = np.random.randn(10, 64)
        result = identity_feature_map(x)

        np.testing.assert_array_equal(result, x)

    def test_relu_feature_map(self):
        """ReLU should zero out negative values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu_feature_map(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_relu_preserves_positive(self):
        """ReLU should preserve positive values."""
        x = np.abs(np.random.randn(10, 64)) + 0.1  # All positive
        result = relu_feature_map(x)

        np.testing.assert_allclose(result, x)

    def test_elu_feature_map_positive(self):
        """ELU+1 should always be positive."""
        x = np.random.randn(100, 64)  # Mix of positive and negative
        result = elu_feature_map(x)

        assert np.all(result > 0), "ELU+1 should always be positive"

    def test_elu_feature_map_positive_input(self):
        """For positive input, ELU+1(x) = x + 1."""
        x = np.abs(np.random.randn(10, 64)) + 0.1  # All positive
        result = elu_feature_map(x)

        np.testing.assert_allclose(result, x + 1)

    def test_elu_feature_map_negative_input(self):
        """For negative input, ELU+1(x) = exp(x)."""
        x = -np.abs(np.random.randn(10, 64)) - 0.1  # All negative
        result = elu_feature_map(x)

        # ELU(x) = exp(x) - 1 for x < 0
        # ELU(x) + 1 = exp(x)
        np.testing.assert_allclose(result, np.exp(x), rtol=1e-5)

    def test_exp_feature_map(self):
        """Exp should compute elementwise exponential."""
        x = np.array([0.0, 1.0, 2.0])
        result = exp_feature_map(x)

        expected = np.exp(x)
        np.testing.assert_allclose(result, expected)

    def test_exp_feature_map_scale(self):
        """Exp should respect scale parameter."""
        x = np.array([1.0, 2.0])
        result = exp_feature_map(x, scale=0.5)

        expected = np.exp(x * 0.5)
        np.testing.assert_allclose(result, expected)


class TestAssociativity:
    """Tests for verify_associativity function."""

    def test_associativity_holds(self):
        """(QK^T)V should equal Q(K^T V)."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        result1, result2, max_diff = verify_associativity(Q, K, V)

        # Should be essentially identical (up to floating point error)
        assert max_diff < 1e-10

    def test_results_have_same_shape(self):
        """Both computation orders should give same shape."""
        Q = np.random.randn(20, 32)
        K = np.random.randn(20, 32)
        V = np.random.randn(20, 32)

        result1, result2, _ = verify_associativity(Q, K, V)

        assert result1.shape == result2.shape

    def test_returns_max_diff(self):
        """Should return the maximum difference."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        _, _, max_diff = verify_associativity(Q, K, V)

        assert isinstance(max_diff, (float, np.floating))
        assert max_diff >= 0

    def test_different_sizes(self):
        """Associativity should hold for different sizes."""
        for n in [5, 10, 50, 100]:
            for d in [16, 32, 64]:
                Q = np.random.randn(n, d)
                K = np.random.randn(n, d)
                V = np.random.randn(n, d)

                _, _, max_diff = verify_associativity(Q, K, V)
                assert max_diff < 1e-9, f"Failed for n={n}, d={d}"


class TestComplexityComparison:
    """Tests for compare_complexity function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        result = compare_complexity(1024, 64)

        assert isinstance(result, dict)
        assert 'standard_ops' in result
        assert 'linear_ops' in result
        assert 'speedup' in result
        assert 'crossover_seq_len' in result

    def test_standard_is_quadratic(self):
        """Standard ops should be O(n²)."""
        result1 = compare_complexity(1000, 64)
        result2 = compare_complexity(2000, 64)

        # Doubling n should roughly quadruple standard ops
        ratio = result2['standard_ops'] / result1['standard_ops']
        np.testing.assert_allclose(ratio, 4.0, rtol=0.1)

    def test_linear_is_linear(self):
        """Linear ops should be O(n)."""
        result1 = compare_complexity(1000, 64)
        result2 = compare_complexity(2000, 64)

        # Doubling n should roughly double linear ops
        ratio = result2['linear_ops'] / result1['linear_ops']
        np.testing.assert_allclose(ratio, 2.0, rtol=0.1)

    def test_speedup_calculation(self):
        """Speedup should be n/d."""
        result = compare_complexity(4096, 64)

        # Speedup = n/d = 4096/64 = 64
        np.testing.assert_allclose(result['speedup'], 64.0, rtol=0.1)

    def test_crossover_point(self):
        """Crossover should be at n = d."""
        result = compare_complexity(1000, 64)

        # Linear becomes faster when n > d
        assert result['crossover_seq_len'] == 64


class TestKernelTrickMilestone:
    """
    Milestone test: Verify understanding of the kernel trick.
    """

    def test_kernel_trick_understood(self):
        """
        MILESTONE: Demonstrate the kernel trick.

        This test verifies that:
        1. You understand matrix associativity
        2. You can implement linear attention order
        3. The complexity advantage is clear
        """
        # Test 1: Associativity
        Q = np.random.randn(100, 64)
        K = np.random.randn(100, 64)
        V = np.random.randn(100, 64)

        _, _, max_diff = verify_associativity(Q, K, V)
        assert max_diff < 1e-9, "Associativity not verified"

        # Test 2: Linear attention works
        output = linear_attention_order(Q, K, V, elu_feature_map)
        assert output.shape == (100, 64)
        assert not np.any(np.isnan(output))

        # Test 3: Complexity advantage
        complexity = compare_complexity(4096, 64)
        assert complexity['speedup'] > 50, "Speedup should be significant"

        print(f"\n{'='*60}")
        print("MILESTONE: Kernel Trick Understood!")
        print(f"Associativity verified (diff={max_diff:.2e})")
        print(f"Linear attention output shape: {output.shape}")
        print(f"Theoretical speedup for n=4096, d=64: {complexity['speedup']:.1f}x")
        print(f"{'='*60}\n")
