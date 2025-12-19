"""Tests for Lab 02: Token Embeddings."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    SinusoidalPositionalEncoding,
    TransformerEmbedding,
    OutputProjection,
    dropout,
)


class TestTokenEmbedding:
    """Tests for TokenEmbedding class."""

    def test_init_shapes(self):
        """Weight matrix should have correct shape."""
        embed = TokenEmbedding(vocab_size=1000, d_model=64)

        assert hasattr(embed, 'weight')
        assert embed.weight.shape == (1000, 64)

    def test_init_values(self):
        """Weight should be initialized (not zeros)."""
        embed = TokenEmbedding(vocab_size=100, d_model=32)

        assert not np.allclose(embed.weight, 0)

    def test_forward_1d(self):
        """Forward should work with 1D input (unbatched)."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([1, 2, 3, 4])

        output = embed.forward(token_ids)

        assert output.shape == (4, 64)

    def test_forward_2d(self):
        """Forward should work with 2D input (batched)."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([[1, 2, 3], [4, 5, 6]])

        output = embed.forward(token_ids)

        assert output.shape == (2, 3, 64)

    def test_forward_lookup_correct(self):
        """Forward should return correct embeddings for each token."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([5, 10, 15])

        output = embed.forward(token_ids)

        # Check that we got the right rows from the weight matrix
        np.testing.assert_array_equal(output[0], embed.weight[5])
        np.testing.assert_array_equal(output[1], embed.weight[10])
        np.testing.assert_array_equal(output[2], embed.weight[15])

    def test_callable(self):
        """Should be callable like a function."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([1, 2, 3])

        output1 = embed.forward(token_ids)
        output2 = embed(token_ids)

        np.testing.assert_array_equal(output1, output2)

    def test_different_tokens_different_embeddings(self):
        """Different tokens should have different embeddings (with high probability)."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([0, 1])

        output = embed.forward(token_ids)

        assert not np.allclose(output[0], output[1])

    def test_same_token_same_embedding(self):
        """Same token should always get same embedding."""
        embed = TokenEmbedding(vocab_size=100, d_model=64)
        token_ids = np.array([5, 5, 5])

        output = embed.forward(token_ids)

        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(output[1], output[2])


class TestPositionalEmbedding:
    """Tests for PositionalEmbedding class."""

    def test_init_shapes(self):
        """Weight matrix should have correct shape."""
        pos_embed = PositionalEmbedding(max_seq_len=512, d_model=64)

        assert hasattr(pos_embed, 'weight')
        assert pos_embed.weight.shape == (512, 64)

    def test_forward_shape(self):
        """Forward should return correct shape."""
        pos_embed = PositionalEmbedding(max_seq_len=512, d_model=64)

        output = pos_embed.forward(10)

        assert output.shape == (10, 64)

    def test_forward_correct_rows(self):
        """Forward should return first seq_len rows."""
        pos_embed = PositionalEmbedding(max_seq_len=512, d_model=64)

        output = pos_embed.forward(5)

        np.testing.assert_array_equal(output, pos_embed.weight[:5])

    def test_forward_different_lengths(self):
        """Should work with different sequence lengths."""
        pos_embed = PositionalEmbedding(max_seq_len=100, d_model=32)

        for seq_len in [1, 10, 50, 100]:
            output = pos_embed.forward(seq_len)
            assert output.shape == (seq_len, 32)

    def test_forward_exceeds_max(self):
        """Should raise error if seq_len > max_seq_len."""
        pos_embed = PositionalEmbedding(max_seq_len=100, d_model=32)

        with pytest.raises(ValueError):
            pos_embed.forward(101)

    def test_callable(self):
        """Should be callable like a function."""
        pos_embed = PositionalEmbedding(max_seq_len=100, d_model=32)

        output1 = pos_embed.forward(10)
        output2 = pos_embed(10)

        np.testing.assert_array_equal(output1, output2)


class TestSinusoidalPositionalEncoding:
    """Tests for SinusoidalPositionalEncoding class."""

    def test_init_shapes(self):
        """Encoding should have correct shape."""
        pe = SinusoidalPositionalEncoding(max_seq_len=512, d_model=64)

        assert hasattr(pe, 'encoding')
        assert pe.encoding.shape == (512, 64)

    def test_forward_shape(self):
        """Forward should return correct shape."""
        pe = SinusoidalPositionalEncoding(max_seq_len=512, d_model=64)

        output = pe.forward(10)

        assert output.shape == (10, 64)

    def test_different_positions_different_encodings(self):
        """Different positions should have different encodings."""
        pe = SinusoidalPositionalEncoding(max_seq_len=100, d_model=64)

        encodings = pe.forward(5)

        # Each position should be unique
        for i in range(5):
            for j in range(i + 1, 5):
                assert not np.allclose(encodings[i], encodings[j])

    def test_sinusoidal_values_range(self):
        """Values should be in [-1, 1] range (sin and cos)."""
        pe = SinusoidalPositionalEncoding(max_seq_len=100, d_model=64)

        encodings = pe.forward(100)

        assert np.all(encodings >= -1.0)
        assert np.all(encodings <= 1.0)

    def test_even_indices_sin_odd_indices_cos(self):
        """Even indices should use sin, odd should use cos."""
        pe = SinusoidalPositionalEncoding(max_seq_len=100, d_model=64)

        # At position 0, sin(0) = 0 for all frequencies
        # So even indices at position 0 should be 0
        encoding_pos0 = pe.forward(1)[0]

        # Even indices: sin(0) = 0
        np.testing.assert_allclose(encoding_pos0[0::2], 0, atol=1e-6)

        # Odd indices: cos(0) = 1
        np.testing.assert_allclose(encoding_pos0[1::2], 1, atol=1e-6)

    def test_deterministic(self):
        """Sinusoidal encoding should be deterministic."""
        pe1 = SinusoidalPositionalEncoding(max_seq_len=100, d_model=64)
        pe2 = SinusoidalPositionalEncoding(max_seq_len=100, d_model=64)

        np.testing.assert_array_equal(pe1.forward(10), pe2.forward(10))


class TestTransformerEmbedding:
    """Tests for TransformerEmbedding class."""

    def test_init(self):
        """Should initialize token and positional embeddings."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512
        )

        assert hasattr(embed, 'token_embedding')
        assert hasattr(embed, 'positional_embedding')

    def test_forward_shape(self):
        """Forward should return correct shape."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512
        )
        token_ids = np.array([[1, 2, 3, 4]])

        output = embed.forward(token_ids)

        assert output.shape == (1, 4, 64)

    def test_forward_adds_position(self):
        """Output should be token embedding + positional embedding."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            dropout=0.0  # Disable dropout for this test
        )
        token_ids = np.array([5, 10, 15])

        output = embed.forward(token_ids, training=False)

        # Manual computation
        token_emb = embed.token_embedding(token_ids)
        pos_emb = embed.positional_embedding(3)
        expected = token_emb + pos_emb

        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_forward_batched(self):
        """Should work with batched input."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            dropout=0.0
        )
        token_ids = np.array([[1, 2, 3], [4, 5, 6]])

        output = embed.forward(token_ids, training=False)

        assert output.shape == (2, 3, 64)

    def test_sinusoidal_option(self):
        """Should use sinusoidal encoding when specified."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            use_sinusoidal=True
        )

        assert isinstance(embed.positional_embedding, SinusoidalPositionalEncoding)

    def test_learned_positional_default(self):
        """Should use learned positional embedding by default."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            use_sinusoidal=False
        )

        assert isinstance(embed.positional_embedding, PositionalEmbedding)

    def test_callable(self):
        """Should be callable like a function."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            dropout=0.0
        )
        token_ids = np.array([[1, 2, 3]])

        output1 = embed.forward(token_ids, training=False)
        output2 = embed(token_ids, training=False)

        np.testing.assert_array_equal(output1, output2)


class TestDropout:
    """Tests for dropout function."""

    def test_dropout_training_off(self):
        """With training=False, should return input unchanged."""
        x = np.random.randn(10, 10)

        result = dropout(x, rate=0.5, training=False)

        np.testing.assert_array_equal(result, x)

    def test_dropout_zero_rate(self):
        """With rate=0, should return input unchanged."""
        x = np.random.randn(10, 10)

        result = dropout(x, rate=0.0, training=True)

        np.testing.assert_array_equal(result, x)

    def test_dropout_creates_zeros(self):
        """Dropout should set some values to zero."""
        np.random.seed(42)
        x = np.ones((100, 100))

        result = dropout(x, rate=0.5, training=True)

        # Should have some zeros
        assert np.sum(result == 0) > 0

    def test_dropout_scaling(self):
        """Non-zero values should be scaled by 1/(1-rate)."""
        np.random.seed(42)
        x = np.ones((1000, 1000))

        result = dropout(x, rate=0.5, training=True)

        # Non-zero values should be scaled to 2.0 (1 / (1-0.5))
        non_zero_values = result[result != 0]
        np.testing.assert_allclose(non_zero_values, 2.0, rtol=1e-5)

    def test_dropout_preserves_mean(self):
        """Expected value should be approximately preserved."""
        np.random.seed(42)
        x = np.random.randn(1000, 1000)

        result = dropout(x, rate=0.3, training=True)

        # Mean should be approximately preserved
        np.testing.assert_allclose(result.mean(), x.mean(), atol=0.1)


class TestOutputProjection:
    """Tests for OutputProjection class."""

    def test_init_shapes(self):
        """Weight should have correct shape."""
        proj = OutputProjection(d_model=64, vocab_size=1000)

        assert hasattr(proj, 'weight')
        assert proj.weight.shape == (1000, 64)

    def test_init_tied_weight(self):
        """Should use tied weight when provided."""
        tied_weight = np.random.randn(1000, 64).astype(np.float32)
        proj = OutputProjection(d_model=64, vocab_size=1000, tied_weight=tied_weight)

        # Should use the same array (not a copy)
        assert proj.weight is tied_weight

    def test_forward_shape(self):
        """Forward should return correct shape."""
        proj = OutputProjection(d_model=64, vocab_size=1000)
        hidden = np.random.randn(2, 10, 64)

        output = proj.forward(hidden)

        assert output.shape == (2, 10, 1000)

    def test_forward_unbatched(self):
        """Should work with unbatched input."""
        proj = OutputProjection(d_model=64, vocab_size=1000)
        hidden = np.random.randn(10, 64)

        output = proj.forward(hidden)

        assert output.shape == (10, 1000)

    def test_forward_computation(self):
        """Forward should compute hidden @ weight.T."""
        proj = OutputProjection(d_model=64, vocab_size=1000)
        hidden = np.random.randn(5, 64).astype(np.float32)

        output = proj.forward(hidden)

        expected = hidden @ proj.weight.T
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_callable(self):
        """Should be callable like a function."""
        proj = OutputProjection(d_model=64, vocab_size=1000)
        hidden = np.random.randn(5, 64)

        output1 = proj.forward(hidden)
        output2 = proj(hidden)

        np.testing.assert_array_equal(output1, output2)


class TestTiedEmbeddings:
    """Tests for tied embeddings (input and output share weights)."""

    def test_tied_embedding_flow(self):
        """Token embedding weight should work as output projection."""
        vocab_size = 1000
        d_model = 64

        # Create token embedding
        token_embed = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)

        # Create output projection with tied weights
        output_proj = OutputProjection(
            d_model=d_model,
            vocab_size=vocab_size,
            tied_weight=token_embed.weight
        )

        # Verify weights are shared
        assert output_proj.weight is token_embed.weight

        # Forward pass should work
        hidden = np.random.randn(2, 10, d_model).astype(np.float32)
        logits = output_proj(hidden)

        assert logits.shape == (2, 10, vocab_size)


class TestEmbeddingIntegration:
    """Integration tests for embedding components."""

    def test_full_embedding_pipeline(self):
        """Test complete embedding pipeline."""
        vocab_size = 1000
        d_model = 64
        max_seq_len = 512
        batch_size = 2
        seq_len = 10

        # Create embedding layer
        embed = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=0.0
        )

        # Create output projection with tied weights
        output_proj = OutputProjection(
            d_model=d_model,
            vocab_size=vocab_size,
            tied_weight=embed.token_embedding.weight
        )

        # Generate random token IDs
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

        # Forward through embedding
        embeddings = embed(token_ids, training=False)
        assert embeddings.shape == (batch_size, seq_len, d_model)

        # Forward through output projection
        logits = output_proj(embeddings)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_different_sequence_lengths(self):
        """Embedding should handle various sequence lengths."""
        embed = TransformerEmbedding(
            vocab_size=1000,
            d_model=64,
            max_seq_len=512,
            dropout=0.0
        )

        for seq_len in [1, 10, 100, 500]:
            token_ids = np.random.randint(0, 1000, size=(2, seq_len))
            output = embed(token_ids, training=False)
            assert output.shape == (2, seq_len, 64)
