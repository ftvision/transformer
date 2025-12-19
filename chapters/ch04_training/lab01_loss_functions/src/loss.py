"""
Lab 01: Loss Functions

Implement cross-entropy loss and perplexity from scratch.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Optional


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log-softmax in a numerically stable way.

    The naive computation log(softmax(x)) can be unstable:
    - softmax can produce very small numbers that underflow to 0
    - log(0) = -inf

    Instead, use the identity:
    log_softmax(x) = x - log(sum(exp(x)))
                   = x - max(x) - log(sum(exp(x - max(x))))

    The second form is numerically stable because:
    - Subtracting max ensures exp() doesn't overflow
    - At least one exp() term equals 1, so sum >= 1 and log >= 0

    Args:
        logits: Input array of any shape
        axis: Axis along which to compute log_softmax (default: -1)

    Returns:
        Array of same shape as logits with log_softmax applied along axis

    Examples:
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> log_probs = log_softmax(logits)
        >>> np.exp(log_probs).sum()  # Should be ~1.0
        1.0

        >>> logits = np.array([1000.0, 1001.0, 1002.0])
        >>> log_probs = log_softmax(logits)  # Should not overflow
        >>> np.all(np.isfinite(log_probs))
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement log_softmax")


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Cross-entropy loss measures how well the predicted probability
    distribution matches the true distribution (one-hot encoded targets).

    For each position, the loss is: -log(P(correct_token))

    Steps:
    1. Compute log_softmax of logits
    2. For each position, get the log-prob of the correct token
    3. Negate and average

    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
                These are NOT probabilities - they can be any real numbers
        targets: Correct token indices of shape (batch_size, seq_len)
                 Each value is an integer in [0, vocab_size)

    Returns:
        Scalar loss value (mean across all positions)

    Example:
        >>> # Perfect prediction: logit for correct class is much higher
        >>> logits = np.array([[[10.0, -10.0, -10.0]]])  # (1, 1, 3)
        >>> targets = np.array([[0]])  # Correct class is 0
        >>> loss = cross_entropy_loss(logits, targets)
        >>> loss < 0.01  # Should be very small
        True

        >>> # Uniform prediction: loss should be log(vocab_size)
        >>> logits = np.zeros((1, 1, 100))  # (1, 1, 100) all equal
        >>> targets = np.array([[42]])
        >>> loss = cross_entropy_loss(logits, targets)
        >>> np.isclose(loss, np.log(100), rtol=1e-5)
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement cross_entropy_loss")


def cross_entropy_loss_masked(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute cross-entropy loss with masking.

    In practice, we often want to ignore certain positions:
    - Padding tokens (variable-length sequences)
    - Prompt tokens (only compute loss on completions)
    - Special tokens

    The mask indicates which positions to INCLUDE in loss computation.

    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
        targets: Correct token indices of shape (batch_size, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len)
              True = include in loss, False = ignore

    Returns:
        Scalar loss value (mean across masked positions only)

    Example:
        >>> logits = np.random.randn(2, 5, 100)
        >>> targets = np.random.randint(0, 100, (2, 5))
        >>> # Only compute loss on last 3 positions of each sequence
        >>> mask = np.array([[False, False, True, True, True],
        ...                  [False, False, True, True, True]])
        >>> loss = cross_entropy_loss_masked(logits, targets, mask)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement cross_entropy_loss_masked")


def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    Perplexity is a more interpretable metric than loss:
    - It represents the "effective vocabulary size" the model is choosing from
    - Lower is better (1.0 would be perfect prediction)
    - Random guessing gives perplexity = vocab_size

    Formula: perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value (should be non-negative)

    Returns:
        Perplexity value (always >= 1.0)

    Example:
        >>> # Loss of 0 -> perplexity of 1 (perfect)
        >>> compute_perplexity(0.0)
        1.0

        >>> # Loss of log(100) -> perplexity of 100
        >>> compute_perplexity(np.log(100))
        100.0
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_perplexity")


def label_smoothing_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1
) -> float:
    """
    Compute cross-entropy loss with label smoothing.

    Label smoothing prevents overconfidence by softening the target distribution:
    - Instead of [0, 0, 1, 0, 0] (one-hot)
    - Use [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K] where ε = smoothing, K = vocab_size

    This is equivalent to:
    loss = (1 - smoothing) * cross_entropy(targets)
         + smoothing * cross_entropy(uniform)

    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
        targets: Correct token indices of shape (batch_size, seq_len)
        smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 typical)

    Returns:
        Scalar loss value with label smoothing applied

    Note:
        The cross-entropy with uniform distribution is just the negative mean
        of log_softmax (since uniform probability is 1/vocab_size for all classes).
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement label_smoothing_loss")


def compute_token_accuracies(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> dict:
    """
    Compute accuracy metrics for token prediction.

    Useful for understanding model performance beyond just loss.

    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
        targets: Correct token indices of shape (batch_size, seq_len)
        mask: Optional boolean mask of shape (batch_size, seq_len)

    Returns:
        Dictionary containing:
        - 'accuracy': Fraction of correct predictions
        - 'top5_accuracy': Fraction where correct token is in top-5 predictions
        - 'num_tokens': Total number of tokens evaluated

    Example:
        >>> logits = np.random.randn(2, 10, 100)
        >>> targets = np.random.randint(0, 100, (2, 10))
        >>> metrics = compute_token_accuracies(logits, targets)
        >>> 0 <= metrics['accuracy'] <= 1
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_token_accuracies")
