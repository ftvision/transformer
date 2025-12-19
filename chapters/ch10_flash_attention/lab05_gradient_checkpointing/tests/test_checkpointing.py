"""Tests for Lab 05: Gradient Checkpointing."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checkpointing import (
    CheckpointedAttention,
    StandardAttention,
    measure_memory_savings,
    estimate_memory_bytes,
    checkpoint_sequential,
    verify_gradient_correctness,
    attention_forward,
    attention_backward,
)


class TestAttentionForwardBackward:
    """Tests for basic attention forward/backward."""

    def test_forward_shape(self):
        """Forward should return correct shapes."""
        seq_len, d_k, d_v = 32, 16, 16
        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)

        output, weights = attention_forward(Q, K, V)

        assert output.shape == (seq_len, d_v)
        assert weights.shape == (seq_len, seq_len)

    def test_backward_shape(self):
        """Backward should return correct gradient shapes."""
        seq_len, d_k, d_v = 32, 16, 16
        Q = np.random.randn(seq_len, d_k).astype(np.float32)
        K = np.random.randn(seq_len, d_k).astype(np.float32)
        V = np.random.randn(seq_len, d_v).astype(np.float32)
        grad_output = np.random.randn(seq_len, d_v).astype(np.float32)

        output, weights = attention_forward(Q, K, V)
        grad_Q, grad_K, grad_V = attention_backward(grad_output, Q, K, V, weights)

        assert grad_Q.shape == Q.shape
        assert grad_K.shape == K.shape
        assert grad_V.shape == V.shape


class TestStandardAttention:
    """Tests for standard attention class."""

    def test_forward(self):
        """Standard attention forward should work."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = StandardAttention()
        output = attn.forward(Q, K, V)

        assert output.shape == (seq_len, d_model)

    def test_stores_weights(self):
        """Standard attention should store weights."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = StandardAttention()
        attn.forward(Q, K, V)

        assert attn.saved_weights is not None
        assert attn.saved_weights.shape == (seq_len, seq_len)

    def test_backward(self):
        """Standard attention backward should work."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)
        grad_output = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = StandardAttention()
        attn.forward(Q, K, V)
        grad_Q, grad_K, grad_V = attn.backward(grad_output)

        assert grad_Q.shape == Q.shape
        assert grad_K.shape == K.shape
        assert grad_V.shape == V.shape


class TestCheckpointedAttention:
    """Tests for checkpointed attention class."""

    def test_init(self):
        """Should initialize without error."""
        attn = CheckpointedAttention()
        assert attn is not None

    def test_forward_shape(self):
        """Forward should return correct shape."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = CheckpointedAttention()
        output = attn.forward(Q, K, V)

        assert output.shape == (seq_len, d_model)

    def test_does_not_store_weights(self):
        """Checkpointed attention should NOT store attention weights."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = CheckpointedAttention()
        attn.forward(Q, K, V)

        # Should NOT have saved_weights attribute or it should be None
        assert not hasattr(attn, 'saved_weights') or attn.saved_weights is None

    def test_stores_inputs(self):
        """Checkpointed attention should store inputs for recomputation."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = CheckpointedAttention()
        attn.forward(Q, K, V)

        assert attn.saved_inputs is not None

    def test_backward(self):
        """Backward should return correct gradient shapes."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)
        grad_output = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = CheckpointedAttention()
        attn.forward(Q, K, V)
        grad_Q, grad_K, grad_V = attn.backward(grad_output)

        assert grad_Q.shape == Q.shape
        assert grad_K.shape == K.shape
        assert grad_V.shape == V.shape

    def test_callable(self):
        """Should be callable like a function."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        attn = CheckpointedAttention()
        out1 = attn.forward(Q, K, V)
        out2 = attn(Q, K, V)

        np.testing.assert_array_equal(out1, out2)


class TestGradientCorrectness:
    """Tests verifying checkpointed gradients match standard."""

    def test_output_matches(self):
        """Checkpointed output should match standard."""
        np.random.seed(42)
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        standard = StandardAttention()
        checkpointed = CheckpointedAttention()

        out_standard = standard.forward(Q, K, V)
        out_checkpointed = checkpointed.forward(Q, K, V)

        np.testing.assert_allclose(out_checkpointed, out_standard, rtol=1e-5, atol=1e-6)

    def test_gradients_match(self):
        """Checkpointed gradients should match standard."""
        np.random.seed(42)
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)
        grad_output = np.random.randn(seq_len, d_model).astype(np.float32)

        standard = StandardAttention()
        checkpointed = CheckpointedAttention()

        standard.forward(Q, K, V)
        checkpointed.forward(Q, K, V)

        grad_Q_std, grad_K_std, grad_V_std = standard.backward(grad_output)
        grad_Q_ckpt, grad_K_ckpt, grad_V_ckpt = checkpointed.backward(grad_output)

        np.testing.assert_allclose(grad_Q_ckpt, grad_Q_std, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_K_ckpt, grad_K_std, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_V_ckpt, grad_V_std, rtol=1e-5, atol=1e-6)

    def test_verify_function(self):
        """verify_gradient_correctness should pass."""
        result = verify_gradient_correctness(seq_len=32, d_model=16)

        assert 'all_close' in result
        assert result['all_close'], "Gradients should match"


class TestMemoryMeasurement:
    """Tests for memory measurement functions."""

    def test_measure_memory_savings_returns_dict(self):
        """Should return a dictionary with expected keys."""
        result = measure_memory_savings(seq_len=1024, d_model=512, num_layers=1)

        assert isinstance(result, dict)
        assert 'standard_memory_bytes' in result
        assert 'checkpointed_memory_bytes' in result
        assert 'memory_saved_bytes' in result
        assert 'memory_saved_pct' in result

    def test_checkpointed_uses_less_memory(self):
        """Checkpointed should use less memory than standard."""
        result = measure_memory_savings(seq_len=1024, d_model=512, num_layers=1)

        assert result['checkpointed_memory_bytes'] < result['standard_memory_bytes']
        assert result['memory_saved_bytes'] > 0
        assert result['memory_saved_pct'] > 0

    def test_memory_scales_with_layers(self):
        """Memory savings should scale with number of layers."""
        result_1 = measure_memory_savings(seq_len=1024, d_model=512, num_layers=1)
        result_4 = measure_memory_savings(seq_len=1024, d_model=512, num_layers=4)

        # Absolute savings should increase with more layers
        assert result_4['memory_saved_bytes'] > result_1['memory_saved_bytes']

    def test_estimate_memory_bytes(self):
        """estimate_memory_bytes should return positive values."""
        standard = estimate_memory_bytes(1024, 512, 4, checkpointed=False)
        checkpointed = estimate_memory_bytes(1024, 512, 4, checkpointed=True)

        assert standard > 0
        assert checkpointed > 0
        assert checkpointed < standard


class TestCheckpointSequential:
    """Tests for checkpoint_sequential function."""

    def test_applies_functions(self):
        """Should apply functions in sequence."""
        def add_one(x):
            return x + 1

        def double(x):
            return x * 2

        x = np.array([1.0, 2.0, 3.0])
        result = checkpoint_sequential([add_one, double], x)

        # (x + 1) * 2
        expected = np.array([4.0, 6.0, 8.0])
        np.testing.assert_array_equal(result, expected)

    def test_empty_functions(self):
        """Should handle empty function list."""
        x = np.array([1.0, 2.0, 3.0])
        result = checkpoint_sequential([], x)

        np.testing.assert_array_equal(result, x)

    def test_single_function(self):
        """Should handle single function."""
        def square(x):
            return x ** 2

        x = np.array([2.0, 3.0, 4.0])
        result = checkpoint_sequential([square], x)

        expected = np.array([4.0, 9.0, 16.0])
        np.testing.assert_array_equal(result, expected)


class TestWithMask:
    """Tests for attention with masking."""

    def test_checkpointed_with_mask(self):
        """Checkpointed attention should handle masks."""
        np.random.seed(42)
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        standard = StandardAttention()
        checkpointed = CheckpointedAttention()

        out_standard = standard.forward(Q, K, V, mask=mask)
        out_checkpointed = checkpointed.forward(Q, K, V, mask=mask)

        np.testing.assert_allclose(out_checkpointed, out_standard, rtol=1e-5, atol=1e-6)

    def test_mask_gradients_match(self):
        """Gradients should match with masking."""
        np.random.seed(123)
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)
        grad_output = np.random.randn(seq_len, d_model).astype(np.float32)

        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        standard = StandardAttention()
        checkpointed = CheckpointedAttention()

        standard.forward(Q, K, V, mask=mask)
        checkpointed.forward(Q, K, V, mask=mask)

        grad_Q_std, grad_K_std, grad_V_std = standard.backward(grad_output)
        grad_Q_ckpt, grad_K_ckpt, grad_V_ckpt = checkpointed.backward(grad_output)

        np.testing.assert_allclose(grad_Q_ckpt, grad_Q_std, rtol=1e-5, atol=1e-6)
