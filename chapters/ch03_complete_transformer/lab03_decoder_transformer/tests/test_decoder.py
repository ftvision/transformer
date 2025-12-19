"""Tests for Lab 03: Decoder-Only Transformer."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decoder import (
    LayerNorm,
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
    GPTModel,
    gelu,
    create_causal_mask,
)


class TestLayerNorm:
    """Tests for LayerNorm class."""

    def test_init(self):
        """Should initialize gamma and beta."""
        ln = LayerNorm(d_model=64)

        assert hasattr(ln, 'gamma')
        assert hasattr(ln, 'beta')
        assert ln.gamma.shape == (64,)
        assert ln.beta.shape == (64,)

    def test_gamma_init_ones(self):
        """Gamma should be initialized to ones."""
        ln = LayerNorm(d_model=64)
        np.testing.assert_array_equal(ln.gamma, np.ones(64))

    def test_beta_init_zeros(self):
        """Beta should be initialized to zeros."""
        ln = LayerNorm(d_model=64)
        np.testing.assert_array_equal(ln.beta, np.zeros(64))

    def test_forward_shape(self):
        """Output should have same shape as input."""
        ln = LayerNorm(d_model=64)
        x = np.random.randn(2, 10, 64)

        output = ln.forward(x)

        assert output.shape == x.shape

    def test_forward_normalized_mean(self):
        """Output should have approximately zero mean."""
        ln = LayerNorm(d_model=64)
        x = np.random.randn(2, 10, 64) * 10 + 5  # Non-zero mean input

        output = ln.forward(x)

        # Mean over last dimension should be ~0 (beta is 0)
        mean = output.mean(axis=-1)
        np.testing.assert_allclose(mean, 0, atol=1e-5)

    def test_forward_normalized_std(self):
        """Output should have approximately unit variance."""
        ln = LayerNorm(d_model=64)
        x = np.random.randn(2, 10, 64) * 10  # High variance input

        output = ln.forward(x)

        # Std over last dimension should be ~1 (gamma is 1)
        std = output.std(axis=-1)
        np.testing.assert_allclose(std, 1, atol=1e-2)

    def test_forward_with_custom_params(self):
        """Should apply gamma and beta correctly."""
        ln = LayerNorm(d_model=4)
        ln.gamma = np.array([2.0, 2.0, 2.0, 2.0])
        ln.beta = np.array([1.0, 1.0, 1.0, 1.0])

        x = np.array([[0.0, 0.0, 0.0, 0.0]])  # Zero input
        output = ln.forward(x)

        # With zero input, normalized is 0, so output = gamma*0 + beta = beta
        # Actually, with constant input, all values become 0 after normalization
        np.testing.assert_allclose(output, [[1.0, 1.0, 1.0, 1.0]], atol=1e-5)


class TestGelu:
    """Tests for GELU activation."""

    def test_gelu_zero(self):
        """GELU(0) should be 0."""
        assert np.isclose(gelu(np.array([0.0]))[0], 0.0, atol=1e-5)

    def test_gelu_positive(self):
        """GELU should be approximately linear for large positive values."""
        x = np.array([10.0])
        assert np.isclose(gelu(x)[0], x[0], rtol=0.01)

    def test_gelu_negative(self):
        """GELU should be small for large negative values."""
        x = np.array([-10.0])
        assert gelu(x)[0] < 0.01

    def test_gelu_shape(self):
        """GELU should preserve shape."""
        x = np.random.randn(2, 3, 4)
        output = gelu(x)
        assert output.shape == x.shape


class TestFeedForward:
    """Tests for FeedForward network."""

    def test_init_weights(self):
        """Should initialize weights with correct shapes."""
        ffn = FeedForward(d_model=64, d_ff=256)

        assert ffn.W1.shape == (64, 256)
        assert ffn.b1.shape == (256,)
        assert ffn.W2.shape == (256, 64)
        assert ffn.b2.shape == (64,)

    def test_forward_shape(self):
        """Output should have same shape as input."""
        ffn = FeedForward(d_model=64, d_ff=256)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output = ffn.forward(x, training=False)

        assert output.shape == x.shape

    def test_forward_unbatched(self):
        """Should work with unbatched input."""
        ffn = FeedForward(d_model=64, d_ff=256)
        x = np.random.randn(10, 64).astype(np.float32)

        output = ffn.forward(x, training=False)

        assert output.shape == (10, 64)

    def test_forward_different_inputs(self):
        """Different inputs should give different outputs."""
        ffn = FeedForward(d_model=64, d_ff=256)
        x1 = np.random.randn(2, 10, 64).astype(np.float32)
        x2 = np.random.randn(2, 10, 64).astype(np.float32)

        out1 = ffn.forward(x1, training=False)
        out2 = ffn.forward(x2, training=False)

        assert not np.allclose(out1, out2)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_init(self):
        """Should initialize weight matrices."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        assert hasattr(mha, 'W_Q')
        assert hasattr(mha, 'W_K')
        assert hasattr(mha, 'W_V')
        assert hasattr(mha, 'W_O')

    def test_init_shapes(self):
        """Weight matrices should have correct shapes."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)

        assert mha.W_Q.shape == (64, 64)
        assert mha.W_K.shape == (64, 64)
        assert mha.W_V.shape == (64, 64)
        assert mha.W_O.shape == (64, 64)

    def test_forward_shape(self):
        """Output should have same shape as input."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output = mha.forward(x, training=False)

        assert output.shape == x.shape

    def test_forward_with_mask(self):
        """Should work with attention mask."""
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        mask = create_causal_mask(10)

        output = mha.forward(x, mask=mask, training=False)

        assert output.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_init(self):
        """Should initialize all components."""
        block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)

        assert hasattr(block, 'ln1')
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'ffn')

    def test_forward_shape(self):
        """Output should have same shape as input."""
        block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output = block.forward(x, training=False)

        assert output.shape == x.shape

    def test_forward_with_mask(self):
        """Should work with causal mask."""
        block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
        x = np.random.randn(2, 10, 64).astype(np.float32)
        mask = create_causal_mask(10)

        output = block.forward(x, mask=mask, training=False)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Output should be different from zero (residual adds to input)."""
        block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
        x = np.random.randn(2, 10, 64).astype(np.float32)

        output = block.forward(x, training=False)

        # Output should not be zero (residual connections)
        assert not np.allclose(output, 0)


class TestGPTModel:
    """Tests for GPTModel."""

    def test_init(self):
        """Should initialize all components."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )

        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'pos_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'ln_f')
        assert len(model.blocks) == 2

    def test_forward_shape(self):
        """Output should be (batch, seq_len, vocab_size)."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )
        token_ids = np.array([[1, 2, 3, 4]])

        output = model.forward(token_ids, training=False)

        assert output.shape == (1, 4, 1000)

    def test_forward_batched(self):
        """Should work with batched input."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )
        token_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        output = model.forward(token_ids, training=False)

        assert output.shape == (2, 4, 1000)

    def test_forward_different_seq_lengths(self):
        """Should work with various sequence lengths."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )

        for seq_len in [1, 10, 50, 100]:
            token_ids = np.random.randint(0, 1000, size=(2, seq_len))
            output = model.forward(token_ids, training=False)
            assert output.shape == (2, seq_len, 1000)

    def test_forward_output_finite(self):
        """Output should not contain NaN or Inf."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=4,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )
        token_ids = np.random.randint(0, 1000, size=(2, 20))

        output = model.forward(token_ids, training=False)

        assert np.all(np.isfinite(output))

    def test_callable(self):
        """Model should be callable."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )
        token_ids = np.array([[1, 2, 3]])

        output1 = model.forward(token_ids, training=False)
        output2 = model(token_ids, training=False)

        np.testing.assert_array_equal(output1, output2)

    def test_tied_weights(self):
        """With tie_weights=True, input and output should share weights."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512,
            tie_weights=True
        )

        # Output projection weight should be same as token embedding weight
        assert model.lm_head.weight is model.token_embedding.weight


class TestGPTModelCausalMasking:
    """Tests for causal masking in GPT model."""

    def test_causal_mask_applied(self):
        """Output at position i should not depend on positions j > i."""
        model = GPTModel(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=8,
            d_ff=256,
            max_seq_len=512
        )

        # Two sequences that differ only in the last token
        tokens1 = np.array([[1, 2, 3, 4]])
        tokens2 = np.array([[1, 2, 3, 999]])  # Different last token

        out1 = model.forward(tokens1, training=False)
        out2 = model.forward(tokens2, training=False)

        # First 3 positions should have identical outputs (causal masking)
        np.testing.assert_allclose(out1[0, :3, :], out2[0, :3, :], rtol=1e-5)

        # Last position should be different
        assert not np.allclose(out1[0, 3, :], out2[0, 3, :])


class TestGPTModelIntegration:
    """Integration tests for GPT model."""

    def test_small_model_forward(self):
        """Small model should complete forward pass."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            max_seq_len=64
        )
        tokens = np.random.randint(0, 100, size=(1, 10))

        output = model(tokens, training=False)

        assert output.shape == (1, 10, 100)
        assert np.all(np.isfinite(output))

    def test_deterministic_inference(self):
        """Same input should give same output in inference mode."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            max_seq_len=64
        )
        tokens = np.array([[1, 2, 3, 4, 5]])

        out1 = model(tokens, training=False)
        out2 = model(tokens, training=False)

        np.testing.assert_array_equal(out1, out2)

    def test_next_token_prediction(self):
        """Should produce valid probability distribution."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            max_seq_len=64
        )
        tokens = np.array([[1, 2, 3]])

        logits = model(tokens, training=False)

        # Convert logits to probabilities
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        # Probabilities should sum to 1
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5)

        # All probabilities should be positive
        assert np.all(probs > 0)


class TestCreateCausalMask:
    """Tests for create_causal_mask helper."""

    def test_causal_mask_shape(self):
        """Mask should have correct shape."""
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_causal_mask_values(self):
        """Mask should have correct values."""
        mask = create_causal_mask(4)

        # Lower triangle (including diagonal) should be False
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == False

        # Upper triangle should be True
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == True
