"""Tests for Lab 03: Multi-Head Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multihead import MultiHeadAttention


class TestMultiHeadInit:
    """Tests for MultiHeadAttention initialization."""

    def test_basic_init(self):
        """Basic initialization with valid parameters."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        assert mha.d_model == 64
        assert mha.num_heads == 8
        assert mha.d_k == 8  # 64 // 8

    def test_d_k_calculation(self):
        """d_k should be d_model // num_heads."""
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        assert mha.d_k == 64

        mha2 = MultiHeadAttention(d_model=256, num_heads=4)
        assert mha2.d_k == 64

    def test_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        d_model, num_heads = 64, 8
        mha = MultiHeadAttention(d_model, num_heads)

        assert mha.W_Q.shape == (d_model, d_model)
        assert mha.W_K.shape == (d_model, d_model)
        assert mha.W_V.shape == (d_model, d_model)
        assert mha.W_O.shape == (d_model, d_model)

    def test_invalid_d_model(self):
        """Should raise error if d_model not divisible by num_heads."""
        with pytest.raises(ValueError):
            MultiHeadAttention(d_model=100, num_heads=8)  # 100 not divisible by 8

    def test_weights_initialized(self):
        """Weights should be initialized (not zeros)."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        assert not np.allclose(mha.W_Q, 0)
        assert not np.allclose(mha.W_K, 0)
        assert not np.allclose(mha.W_V, 0)
        assert not np.allclose(mha.W_O, 0)


class TestSplitCombineHeads:
    """Tests for _split_heads and _combine_heads methods."""

    def test_split_heads_shape_batched(self):
        """_split_heads should reshape correctly for batched input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)  # (batch, seq_len, d_model)

        split = mha._split_heads(x)

        # Should be (batch, num_heads, seq_len, d_k)
        assert split.shape == (2, 8, 10, 8)

    def test_split_heads_shape_unbatched(self):
        """_split_heads should handle unbatched input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(10, 64)  # (seq_len, d_model)

        split = mha._split_heads(x)

        # Should be (num_heads, seq_len, d_k)
        assert split.shape == (8, 10, 8)

    def test_combine_heads_shape_batched(self):
        """_combine_heads should reshape correctly for batched input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 8, 10, 8)  # (batch, heads, seq_len, d_k)

        combined = mha._combine_heads(x)

        # Should be (batch, seq_len, d_model)
        assert combined.shape == (2, 10, 64)

    def test_combine_heads_shape_unbatched(self):
        """_combine_heads should handle unbatched input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(8, 10, 8)  # (heads, seq_len, d_k)

        combined = mha._combine_heads(x)

        # Should be (seq_len, d_model)
        assert combined.shape == (10, 64)

    def test_split_combine_roundtrip(self):
        """Split then combine should give back original shape."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        split = mha._split_heads(x)
        combined = mha._combine_heads(split)

        assert combined.shape == x.shape


class TestMultiHeadForward:
    """Tests for forward pass."""

    def test_output_shape_batched(self):
        """Output should have same shape as input (batched)."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        output = mha.forward(x)

        assert output.shape == x.shape

    def test_output_shape_unbatched(self):
        """Output should have same shape as input (unbatched)."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(10, 64)

        output = mha.forward(x)

        assert output.shape == x.shape

    def test_callable(self):
        """MHA should be callable like a function."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        output1 = mha.forward(x)
        output2 = mha(x)

        np.testing.assert_array_equal(output1, output2)

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        x1 = np.random.randn(2, 10, 64)
        x2 = np.random.randn(2, 10, 64)

        out1 = mha(x1)
        out2 = mha(x2)

        assert not np.allclose(out1, out2)

    def test_deterministic(self):
        """Same input should always give same output."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out1 = mha(x)
        out2 = mha(x)

        np.testing.assert_array_equal(out1, out2)

    def test_with_mask(self):
        """Forward pass should work with attention mask."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        # Causal mask
        seq_len = 10
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        output = mha.forward(x, mask=mask)

        assert output.shape == x.shape


class TestAttentionWeights:
    """Tests for attention weight extraction."""

    def test_attention_weights_shape_batched(self):
        """Attention weights should have correct shape."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        weights = mha.get_attention_weights(x)

        # (batch, num_heads, seq_len, seq_len)
        assert weights.shape == (2, 8, 10, 10)

    def test_attention_weights_shape_unbatched(self):
        """Attention weights should work for unbatched input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(10, 64)

        weights = mha.get_attention_weights(x)

        # (num_heads, seq_len, seq_len)
        assert weights.shape == (8, 10, 10)

    def test_attention_weights_sum_to_one(self):
        """Each row of attention weights should sum to 1."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        weights = mha.get_attention_weights(x)

        # Sum over last dimension (keys)
        row_sums = weights.sum(axis=-1)
        expected = np.ones((2, 8, 10))

        np.testing.assert_allclose(row_sums, expected, rtol=1e-5)

    def test_attention_weights_positive(self):
        """All attention weights should be non-negative."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        weights = mha.get_attention_weights(x)

        assert np.all(weights >= 0)

    def test_masked_attention_weights(self):
        """Masked positions should have zero attention weight."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(10, 64)

        # Causal mask
        seq_len = 10
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        weights = mha.get_attention_weights(x, mask=mask)

        # Upper triangle should be ~0
        for h in range(8):
            upper = np.triu(weights[h], k=1)
            np.testing.assert_allclose(upper, 0, atol=1e-6)


class TestMultiHeadProperties:
    """Tests for multi-head attention properties."""

    def test_different_heads_different_patterns(self):
        """Different heads should learn different attention patterns."""
        np.random.seed(42)
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        # Use a structured input
        x = np.random.randn(10, 64)

        weights = mha.get_attention_weights(x)

        # Check that heads are not identical
        # Compare first head with others
        head_0 = weights[0]
        different_heads = 0
        for h in range(1, 8):
            if not np.allclose(weights[h], head_0, rtol=0.1):
                different_heads += 1

        # At least some heads should be different
        assert different_heads > 0, "All heads have identical patterns"

    def test_num_parameters(self):
        """Verify total number of parameters."""
        d_model = 512
        mha = MultiHeadAttention(d_model=d_model, num_heads=8)

        # 4 weight matrices of shape (d_model, d_model)
        expected_params = 4 * d_model * d_model

        total_params = (
            mha.W_Q.size +
            mha.W_K.size +
            mha.W_V.size +
            mha.W_O.size
        )

        assert total_params == expected_params

    def test_varying_seq_lengths(self):
        """Should handle different sequence lengths."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        for seq_len in [5, 10, 20, 50]:
            x = np.random.randn(2, seq_len, 64)
            output = mha(x)
            assert output.shape == (2, seq_len, 64)

    def test_single_head(self):
        """Should work with a single head."""
        mha = MultiHeadAttention(d_model=64, num_heads=1)
        x = np.random.randn(2, 10, 64)

        output = mha(x)

        assert output.shape == x.shape
        assert mha.d_k == 64
