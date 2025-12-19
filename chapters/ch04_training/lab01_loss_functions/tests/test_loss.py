"""Tests for Lab 01: Loss Functions."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loss import (
    log_softmax,
    cross_entropy_loss,
    cross_entropy_loss_masked,
    compute_perplexity,
    label_smoothing_loss,
    compute_token_accuracies,
)


class TestLogSoftmax:
    """Tests for the log_softmax function."""

    def test_log_softmax_sums_to_one(self):
        """exp(log_softmax) should sum to 1."""
        logits = np.array([1.0, 2.0, 3.0])
        log_probs = log_softmax(logits)
        probs = np.exp(log_probs)

        np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-5)

    def test_log_softmax_negative_values(self):
        """Log probabilities should be <= 0."""
        logits = np.array([1.0, 2.0, 3.0])
        log_probs = log_softmax(logits)

        assert np.all(log_probs <= 0)

    def test_log_softmax_numerical_stability(self):
        """Should handle large values without overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        log_probs = log_softmax(logits)

        assert not np.any(np.isnan(log_probs)), "Should not produce NaN"
        assert not np.any(np.isinf(log_probs)), "Should not produce Inf"
        assert np.all(log_probs <= 0), "Log probs should be <= 0"

    def test_log_softmax_2d(self):
        """Should work along specified axis for 2D input."""
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])
        log_probs = log_softmax(logits, axis=-1)

        # Each row should sum to 1 after exp
        row_sums = np.exp(log_probs).sum(axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-5)

    def test_log_softmax_3d(self):
        """Should work for 3D inputs (batch, seq, vocab)."""
        logits = np.random.randn(2, 3, 100)
        log_probs = log_softmax(logits, axis=-1)

        assert log_probs.shape == logits.shape
        # Each position should sum to 1 after exp
        sums = np.exp(log_probs).sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones((2, 3)), rtol=1e-5)

    def test_log_softmax_preserves_order(self):
        """Higher logits should give higher log probs."""
        logits = np.array([1.0, 2.0, 3.0])
        log_probs = log_softmax(logits)

        assert log_probs[2] > log_probs[1] > log_probs[0]


class TestCrossEntropyLoss:
    """Tests for the cross_entropy_loss function."""

    def test_perfect_prediction_low_loss(self):
        """Perfect prediction should have very low loss."""
        # Logit for correct class is much higher
        logits = np.array([[[10.0, -10.0, -10.0]]])
        targets = np.array([[0]])

        loss = cross_entropy_loss(logits, targets)

        assert loss < 0.01, f"Expected very low loss, got {loss}"

    def test_uniform_prediction(self):
        """Uniform logits should give loss = log(vocab_size)."""
        vocab_size = 100
        logits = np.zeros((1, 1, vocab_size))
        targets = np.array([[42]])

        loss = cross_entropy_loss(logits, targets)
        expected = np.log(vocab_size)

        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_loss_non_negative(self):
        """Loss should always be non-negative."""
        logits = np.random.randn(4, 10, 50)
        targets = np.random.randint(0, 50, (4, 10))

        loss = cross_entropy_loss(logits, targets)

        assert loss >= 0, f"Loss should be >= 0, got {loss}"

    def test_loss_shape(self):
        """Loss should be a scalar."""
        logits = np.random.randn(4, 10, 50)
        targets = np.random.randint(0, 50, (4, 10))

        loss = cross_entropy_loss(logits, targets)

        assert np.isscalar(loss) or loss.shape == (), f"Expected scalar, got shape {np.shape(loss)}"

    def test_loss_increases_with_wrong_predictions(self):
        """Loss should be higher when predictions are wrong."""
        vocab_size = 10

        # Good prediction: high logit for correct class
        logits_good = np.zeros((1, 1, vocab_size))
        logits_good[0, 0, 0] = 10.0  # Correct class is 0

        # Bad prediction: high logit for wrong class
        logits_bad = np.zeros((1, 1, vocab_size))
        logits_bad[0, 0, 5] = 10.0  # But correct class is 0

        targets = np.array([[0]])

        loss_good = cross_entropy_loss(logits_good, targets)
        loss_bad = cross_entropy_loss(logits_bad, targets)

        assert loss_bad > loss_good, "Wrong prediction should have higher loss"

    def test_batch_independence(self):
        """Each batch element should contribute equally."""
        vocab_size = 50

        # Create two identical sequences
        logits = np.random.randn(1, 10, vocab_size)
        targets = np.random.randint(0, vocab_size, (1, 10))

        loss_single = cross_entropy_loss(logits, targets)

        # Stack same sequence twice
        logits_double = np.concatenate([logits, logits], axis=0)
        targets_double = np.concatenate([targets, targets], axis=0)

        loss_double = cross_entropy_loss(logits_double, targets_double)

        np.testing.assert_allclose(loss_single, loss_double, rtol=1e-5)


class TestCrossEntropyLossMasked:
    """Tests for the cross_entropy_loss_masked function."""

    def test_all_masked_equals_unmasked(self):
        """All True mask should equal unmasked loss."""
        logits = np.random.randn(2, 10, 50)
        targets = np.random.randint(0, 50, (2, 10))
        mask = np.ones((2, 10), dtype=bool)

        loss_masked = cross_entropy_loss_masked(logits, targets, mask)
        loss_unmasked = cross_entropy_loss(logits, targets)

        np.testing.assert_allclose(loss_masked, loss_unmasked, rtol=1e-5)

    def test_partial_mask(self):
        """Should only consider masked positions."""
        vocab_size = 10

        # First position has high loss, second has low loss
        logits = np.zeros((1, 2, vocab_size))
        logits[0, 0, 5] = 10.0  # Wrong prediction for position 0
        logits[0, 1, 1] = 10.0  # Correct prediction for position 1

        targets = np.array([[0, 1]])  # Correct classes

        # Mask only position 0 (high loss)
        mask_first = np.array([[True, False]])
        loss_first = cross_entropy_loss_masked(logits, targets, mask_first)

        # Mask only position 1 (low loss)
        mask_second = np.array([[False, True]])
        loss_second = cross_entropy_loss_masked(logits, targets, mask_second)

        assert loss_first > loss_second, "Position 0 should have higher loss"

    def test_empty_mask(self):
        """All False mask should handle gracefully."""
        logits = np.random.randn(2, 10, 50)
        targets = np.random.randint(0, 50, (2, 10))
        mask = np.zeros((2, 10), dtype=bool)

        # Should return 0 or handle gracefully (not NaN)
        loss = cross_entropy_loss_masked(logits, targets, mask)

        assert not np.isnan(loss), "Should not return NaN for empty mask"

    def test_variable_length_sequences(self):
        """Common use case: different sequence lengths."""
        vocab_size = 50
        logits = np.random.randn(2, 10, vocab_size)
        targets = np.random.randint(0, vocab_size, (2, 10))

        # First sequence has 8 real tokens, second has 5
        mask = np.array([
            [True] * 8 + [False] * 2,
            [True] * 5 + [False] * 5
        ])

        loss = cross_entropy_loss_masked(logits, targets, mask)

        assert not np.isnan(loss)
        assert loss >= 0


class TestPerplexity:
    """Tests for the compute_perplexity function."""

    def test_zero_loss_gives_one(self):
        """Loss of 0 should give perplexity of 1."""
        perplexity = compute_perplexity(0.0)

        np.testing.assert_allclose(perplexity, 1.0, rtol=1e-5)

    def test_log_vocab_loss(self):
        """Loss of log(N) should give perplexity of N."""
        vocab_size = 100
        loss = np.log(vocab_size)

        perplexity = compute_perplexity(loss)

        np.testing.assert_allclose(perplexity, vocab_size, rtol=1e-5)

    def test_perplexity_at_least_one(self):
        """Perplexity should always be >= 1."""
        for loss in [0.0, 0.5, 1.0, 2.0, 5.0]:
            perplexity = compute_perplexity(loss)
            assert perplexity >= 1.0, f"Perplexity should be >= 1, got {perplexity}"

    def test_perplexity_increases_with_loss(self):
        """Higher loss should give higher perplexity."""
        perp_low = compute_perplexity(1.0)
        perp_high = compute_perplexity(2.0)

        assert perp_high > perp_low


class TestLabelSmoothing:
    """Tests for label_smoothing_loss function."""

    def test_no_smoothing_equals_regular(self):
        """Smoothing=0 should equal regular cross-entropy."""
        logits = np.random.randn(2, 10, 50)
        targets = np.random.randint(0, 50, (2, 10))

        loss_smooth = label_smoothing_loss(logits, targets, smoothing=0.0)
        loss_regular = cross_entropy_loss(logits, targets)

        np.testing.assert_allclose(loss_smooth, loss_regular, rtol=1e-5)

    def test_smoothing_increases_loss(self):
        """Label smoothing should generally increase loss for confident predictions."""
        vocab_size = 10

        # Very confident prediction
        logits = np.zeros((1, 1, vocab_size))
        logits[0, 0, 0] = 100.0
        targets = np.array([[0]])

        loss_no_smooth = label_smoothing_loss(logits, targets, smoothing=0.0)
        loss_smooth = label_smoothing_loss(logits, targets, smoothing=0.1)

        assert loss_smooth > loss_no_smooth, "Smoothing should increase loss for confident predictions"

    def test_smoothing_valid_range(self):
        """Loss with smoothing should still be reasonable."""
        logits = np.random.randn(2, 10, 50)
        targets = np.random.randint(0, 50, (2, 10))

        loss = label_smoothing_loss(logits, targets, smoothing=0.1)

        assert loss >= 0, "Loss should be non-negative"
        assert not np.isnan(loss), "Loss should not be NaN"


class TestTokenAccuracies:
    """Tests for compute_token_accuracies function."""

    def test_perfect_accuracy(self):
        """All correct predictions should give accuracy 1.0."""
        vocab_size = 10
        logits = np.zeros((1, 5, vocab_size))

        # Make each position's correct class have highest logit
        targets = np.array([[0, 1, 2, 3, 4]])
        for i in range(5):
            logits[0, i, targets[0, i]] = 10.0

        metrics = compute_token_accuracies(logits, targets)

        assert metrics['accuracy'] == 1.0
        assert metrics['top5_accuracy'] == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions should give accuracy 0.0."""
        vocab_size = 10
        logits = np.zeros((1, 5, vocab_size))

        targets = np.array([[0, 0, 0, 0, 0]])  # All target class 0
        # Make class 5 highest (wrong)
        logits[0, :, 5] = 10.0

        metrics = compute_token_accuracies(logits, targets)

        assert metrics['accuracy'] == 0.0

    def test_accuracy_range(self):
        """Accuracy should be between 0 and 1."""
        logits = np.random.randn(4, 10, 50)
        targets = np.random.randint(0, 50, (4, 10))

        metrics = compute_token_accuracies(logits, targets)

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['top5_accuracy'] <= 1

    def test_top5_at_least_top1(self):
        """Top-5 accuracy should be >= top-1 accuracy."""
        logits = np.random.randn(4, 10, 50)
        targets = np.random.randint(0, 50, (4, 10))

        metrics = compute_token_accuracies(logits, targets)

        assert metrics['top5_accuracy'] >= metrics['accuracy']

    def test_masked_accuracy(self):
        """Should only count masked positions."""
        vocab_size = 10
        logits = np.zeros((1, 4, vocab_size))

        # First 2 positions correct, last 2 wrong
        targets = np.array([[0, 1, 0, 0]])
        logits[0, 0, 0] = 10.0  # Correct
        logits[0, 1, 1] = 10.0  # Correct
        logits[0, 2, 5] = 10.0  # Wrong
        logits[0, 3, 5] = 10.0  # Wrong

        # Only count first 2 positions
        mask = np.array([[True, True, False, False]])

        metrics = compute_token_accuracies(logits, targets, mask)

        assert metrics['accuracy'] == 1.0, "Should only count correct masked positions"
        assert metrics['num_tokens'] == 2

    def test_num_tokens(self):
        """Should correctly count number of tokens."""
        logits = np.random.randn(2, 10, 50)
        targets = np.random.randint(0, 50, (2, 10))

        metrics = compute_token_accuracies(logits, targets)

        assert metrics['num_tokens'] == 20  # 2 * 10


class TestLossCorrectness:
    """Tests comparing against known correct values."""

    def test_cross_entropy_manual_calculation(self):
        """Test against manually calculated value."""
        # Simple case: 3 classes, batch=1, seq=1
        logits = np.array([[[1.0, 2.0, 3.0]]])
        targets = np.array([[2]])  # Target is class 2

        # Manual calculation:
        # softmax([1,2,3]) = exp([1,2,3]) / sum(exp([1,2,3]))
        # log_softmax([1,2,3]) = [1,2,3] - log(sum(exp([1,2,3])))
        # sum(exp([1,2,3])) = e + e^2 + e^3 ≈ 2.718 + 7.389 + 20.086 = 30.193
        # log_softmax = [1,2,3] - log(30.193) = [1,2,3] - 3.408
        #             = [-2.408, -1.408, -0.408]
        # CE loss for target=2: -(-0.408) = 0.408

        loss = cross_entropy_loss(logits, targets)
        expected = -np.log(np.exp(3.0) / (np.exp(1.0) + np.exp(2.0) + np.exp(3.0)))

        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_random_baseline(self):
        """Random model should have perplexity ≈ vocab_size."""
        vocab_size = 1000
        batch_size = 10
        seq_len = 100

        # Uniform logits (random guessing)
        logits = np.zeros((batch_size, seq_len, vocab_size))
        targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

        loss = cross_entropy_loss(logits, targets)
        perplexity = compute_perplexity(loss)

        # Should be approximately vocab_size
        np.testing.assert_allclose(perplexity, vocab_size, rtol=0.01)
