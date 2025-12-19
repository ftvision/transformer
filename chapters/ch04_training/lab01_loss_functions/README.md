# Lab 01: Loss Functions

## Objective

Implement cross-entropy loss and perplexity from scratch using NumPy.

## What You'll Build

Functions that compute:
1. Cross-entropy loss for language modeling
2. Perplexity from loss
3. Masked loss (ignoring padding tokens)

## Prerequisites

Read these docs first:
- `../docs/01_loss_and_perplexity.md`

## Instructions

1. Open `src/loss.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `log_softmax(logits, axis=-1)`
Compute log-softmax in a numerically stable way.
- Use the log-sum-exp trick for stability
- `log_softmax(x) = x - log(sum(exp(x)))`
- But compute it as: `x - max(x) - log(sum(exp(x - max(x))))`

### `cross_entropy_loss(logits, targets)`
Compute cross-entropy loss for classification/language modeling.
- `logits`: shape `(batch, seq_len, vocab_size)` - raw model outputs
- `targets`: shape `(batch, seq_len)` - correct token indices
- Returns: scalar loss (mean over all positions)

### `cross_entropy_loss_masked(logits, targets, mask)`
Same as above, but with masking support.
- `mask`: shape `(batch, seq_len)` - True for positions to include in loss
- Only compute loss on masked positions

### `compute_perplexity(loss)`
Convert loss to perplexity.
- `perplexity = exp(loss)`

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_loss.py::TestLogSoftmax

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- For `log_softmax`, remember to subtract max for numerical stability
- Use `np.take_along_axis` or advanced indexing to get the log-prob of correct tokens
- For masked loss, compute mean only over positions where mask is True
- Perplexity should never be less than 1 (loss is always >= 0)

## Expected Shapes

```
logits: (batch_size, seq_len, vocab_size)
targets: (batch_size, seq_len)
mask: (batch_size, seq_len)

loss: scalar
perplexity: scalar
```

## Example

```python
import numpy as np

# Vocabulary of 100 tokens
vocab_size = 100
batch_size = 2
seq_len = 10

# Random logits from model
logits = np.random.randn(batch_size, seq_len, vocab_size)

# Target tokens (indices into vocabulary)
targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

# Compute loss
loss = cross_entropy_loss(logits, targets)
perplexity = compute_perplexity(loss)

print(f"Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
# For random predictions, perplexity should be ~vocab_size
```

## Verification

All tests pass = you've correctly implemented loss functions!

If stuck, check `solutions/loss.py` for the reference implementation.
