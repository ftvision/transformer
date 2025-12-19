"""Tests for Lab 04: Train Tiny Model."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import (
    CharTokenizer,
    TinyGPT,
    load_tiny_shakespeare,
    create_dataset,
    compute_loss_and_gradients,
    train_model,
    evaluate_model,
)


class TestCharTokenizer:
    """Tests for character-level tokenizer."""

    def test_initialization(self):
        """Should build vocabulary from text."""
        text = "hello world"
        tokenizer = CharTokenizer(text)

        # 8 unique characters: 'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'
        assert tokenizer.vocab_size == 8

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return original text."""
        text = "hello world"
        tokenizer = CharTokenizer(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_encode_returns_integers(self):
        """Encoded output should be list of integers."""
        text = "hello"
        tokenizer = CharTokenizer(text)

        encoded = tokenizer.encode(text)

        assert all(isinstance(i, (int, np.integer)) for i in encoded)
        assert len(encoded) == len(text)

    def test_vocab_sorted(self):
        """Vocabulary should be sorted for consistency."""
        text = "zyx abc"
        tokenizer = CharTokenizer(text)

        # First character in sorted order should have ID 0
        sorted_chars = sorted(set(text))
        assert tokenizer.char_to_id[sorted_chars[0]] == 0

    def test_shakespeare_sample(self):
        """Should work with Shakespeare-style text."""
        text = """ROMEO:
But, soft! what light through yonder window breaks?
"""
        tokenizer = CharTokenizer(text)

        # Should have letters, punctuation, whitespace
        assert '\n' in tokenizer.vocab
        assert ':' in tokenizer.vocab
        assert 'R' in tokenizer.vocab

        # Roundtrip should work
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text


class TestTinyGPT:
    """Tests for TinyGPT model."""

    def test_initialization(self):
        """Should initialize with correct attributes."""
        model = TinyGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)

        assert model.vocab_size == 65
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2

    def test_parameters_exist(self):
        """Should have trainable parameters."""
        model = TinyGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)

        params = model.parameters()

        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_forward_shape(self):
        """Forward pass should return correct shape."""
        vocab_size = 65
        model = TinyGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        x = np.array([[1, 2, 3, 4, 5]])  # (batch=1, seq_len=5)
        logits = model.forward(x)

        assert logits.shape == (1, 5, vocab_size)

    def test_forward_batched(self):
        """Forward pass should work with batched input."""
        vocab_size = 65
        model = TinyGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        x = np.random.randint(0, vocab_size, (4, 10))  # (batch=4, seq_len=10)
        logits = model.forward(x)

        assert logits.shape == (4, 10, vocab_size)

    def test_forward_different_seq_lengths(self):
        """Should handle different sequence lengths."""
        vocab_size = 65
        model = TinyGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2, max_seq_len=256)

        for seq_len in [5, 10, 50, 100]:
            x = np.random.randint(0, vocab_size, (2, seq_len))
            logits = model.forward(x)
            assert logits.shape == (2, seq_len, vocab_size)

    def test_generate_length(self):
        """Generate should produce correct length."""
        model = TinyGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)

        prompt = np.array([1, 2, 3])  # 3 tokens
        generated = model.generate(prompt, max_new_tokens=10)

        assert len(generated) == 13  # 3 + 10

    def test_generate_starts_with_prompt(self):
        """Generated sequence should start with prompt."""
        model = TinyGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)

        prompt = np.array([1, 2, 3])
        generated = model.generate(prompt, max_new_tokens=5)

        np.testing.assert_array_equal(generated[:3], prompt)

    def test_generate_valid_tokens(self):
        """Generated tokens should be valid (in vocab range)."""
        vocab_size = 65
        model = TinyGPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2)

        prompt = np.array([1, 2, 3])
        generated = model.generate(prompt, max_new_tokens=20)

        assert all(0 <= t < vocab_size for t in generated)

    def test_generate_temperature(self):
        """Temperature should affect generation diversity."""
        np.random.seed(42)
        model = TinyGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)

        prompt = np.array([1, 2, 3])

        # Low temperature should be more deterministic
        # (hard to test randomness, but shouldn't crash)
        gen_low = model.generate(prompt, max_new_tokens=10, temperature=0.1)
        gen_high = model.generate(prompt, max_new_tokens=10, temperature=2.0)

        assert len(gen_low) == len(gen_high)


class TestDataset:
    """Tests for dataset creation."""

    def test_create_dataset_shapes(self):
        """Dataset should have correct shapes."""
        text = "hello world! this is a test."
        tokenizer = CharTokenizer(text)

        inputs, labels = create_dataset(text, tokenizer, seq_len=8)

        assert inputs.shape[1] == 8
        assert labels.shape[1] == 8
        assert inputs.shape[0] == labels.shape[0]

    def test_labels_shifted(self):
        """Labels should be inputs shifted by 1."""
        text = "hello world! this is a test."
        tokenizer = CharTokenizer(text)

        inputs, labels = create_dataset(text, tokenizer, seq_len=8)

        # For each example, labels should be shifted by 1
        # Check first example
        full_tokens = tokenizer.encode(text)
        np.testing.assert_array_equal(inputs[0], full_tokens[0:8])
        np.testing.assert_array_equal(labels[0], full_tokens[1:9])

    def test_load_tiny_shakespeare(self):
        """Should load sample text."""
        text = load_tiny_shakespeare()

        assert len(text) > 100
        assert "ROMEO" in text or "romeo" in text.lower()


class TestTraining:
    """Tests for training functions."""

    def test_compute_loss_and_gradients(self):
        """Should compute loss and gradients."""
        model = TinyGPT(vocab_size=65, d_model=32, n_heads=2, n_layers=1)

        inputs = np.random.randint(0, 65, (2, 10))
        labels = np.random.randint(0, 65, (2, 10))

        loss, gradients = compute_loss_and_gradients(model, inputs, labels)

        assert isinstance(loss, (float, np.floating))
        assert loss > 0  # Loss should be positive
        assert len(gradients) == len(model.parameters())

    def test_loss_decreases_with_training(self):
        """Loss should decrease during training."""
        np.random.seed(42)

        # Small model and dataset for quick test
        model = TinyGPT(vocab_size=30, d_model=32, n_heads=2, n_layers=1)

        text = "hello world " * 100
        tokenizer = CharTokenizer(text)
        train_data = create_dataset(text, tokenizer, seq_len=16)

        history = train_model(
            model,
            train_data,
            epochs=3,
            batch_size=8,
            lr=1e-2,
            warmup_steps=10,
            log_interval=100  # Don't print during test
        )

        # Loss should decrease (may not always due to randomness, so use relaxed check)
        assert len(history['loss']) > 0
        # Check that at least some decrease happened
        initial_losses = history['loss'][:10]
        final_losses = history['loss'][-10:]
        assert np.mean(final_losses) < np.mean(initial_losses)

    def test_evaluate_model(self):
        """Should compute evaluation metrics."""
        model = TinyGPT(vocab_size=30, d_model=32, n_heads=2, n_layers=1)

        eval_inputs = np.random.randint(0, 30, (20, 16))
        eval_labels = np.random.randint(0, 30, (20, 16))

        metrics = evaluate_model(model, (eval_inputs, eval_labels), batch_size=8)

        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] >= 1


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline(self):
        """Test complete training pipeline."""
        np.random.seed(42)

        # Load data
        text = load_tiny_shakespeare()

        # Tokenize
        tokenizer = CharTokenizer(text)

        # Create dataset
        inputs, labels = create_dataset(text, tokenizer, seq_len=32)

        # Split
        split = int(len(inputs) * 0.9)
        train_data = (inputs[:split], labels[:split])
        eval_data = (inputs[split:], labels[split:])

        # Create model
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            max_seq_len=32
        )

        # Train briefly
        history = train_model(
            model,
            train_data,
            epochs=2,
            batch_size=16,
            lr=1e-2,
            warmup_steps=10,
            log_interval=1000
        )

        # Evaluate
        metrics = evaluate_model(model, eval_data)

        # Generate
        prompt = tokenizer.encode("ROMEO:")
        generated = model.generate(np.array(prompt), max_new_tokens=20, temperature=0.8)
        generated_text = tokenizer.decode(generated.tolist())

        # Basic sanity checks
        assert len(history['loss']) > 0
        assert metrics['perplexity'] > 1
        assert len(generated_text) > len("ROMEO:")
        assert generated_text.startswith("ROMEO:")

    def test_model_param_count(self):
        """Model should have approximately 1M parameters."""
        model = TinyGPT(
            vocab_size=65,
            d_model=128,
            n_heads=4,
            n_layers=4,
            max_seq_len=256
        )

        num_params = sum(p.size for p in model.parameters())

        # Should be roughly 1M (between 500K and 2M)
        assert 500_000 < num_params < 2_000_000, f"Got {num_params} params"


class TestMilestone:
    """
    Chapter 4 Milestone Tests

    Train a ~1M param model that generates coherent Shakespeare-like text.
    """

    def test_chapter_4_milestone(self):
        """
        MILESTONE: Train model that reduces perplexity significantly.

        This tests that the training infrastructure works.
        Full Shakespeare-like generation requires longer training
        than we want in a test.
        """
        np.random.seed(42)

        # Use sample text
        text = load_tiny_shakespeare()
        tokenizer = CharTokenizer(text)

        # Create dataset
        inputs, labels = create_dataset(text, tokenizer, seq_len=64)

        # Small model for testing
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=64
        )

        # Compute initial perplexity (before training)
        initial_metrics = evaluate_model(model, (inputs[:100], labels[:100]))
        initial_ppl = initial_metrics['perplexity']

        # Train
        history = train_model(
            model,
            (inputs, labels),
            epochs=3,
            batch_size=32,
            lr=1e-2,
            warmup_steps=50,
            log_interval=1000
        )

        # Compute final perplexity
        final_metrics = evaluate_model(model, (inputs[:100], labels[:100]))
        final_ppl = final_metrics['perplexity']

        # Perplexity should improve significantly
        assert final_ppl < initial_ppl, (
            f"Training didn't improve perplexity: {initial_ppl:.2f} -> {final_ppl:.2f}"
        )

        # Final perplexity should be reasonable (not random)
        # Random would be ~vocab_size (~65)
        assert final_ppl < tokenizer.vocab_size, (
            f"Perplexity {final_ppl:.2f} is no better than random ({tokenizer.vocab_size})"
        )

        print(f"\n{'='*60}")
        print("Chapter 4 Milestone Progress:")
        print(f"  Initial perplexity: {initial_ppl:.2f}")
        print(f"  Final perplexity: {final_ppl:.2f}")
        print(f"  Improvement: {(initial_ppl - final_ppl) / initial_ppl * 100:.1f}%")
        print(f"{'='*60}\n")
