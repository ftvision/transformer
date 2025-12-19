# Lab 01: Causal Masking

## Objective

Implement causal (autoregressive) masking for decoder-style transformers.

## What You'll Build

Functions to:
1. Create causal masks (lower triangular)
2. Apply masks to attention scores using the -inf trick
3. Combine causal and padding masks for batched sequences

## Prerequisites

Read these docs first:
- `../docs/02_causal_masking.md`

Also helpful:
- Chapter 1, Lab 01 (scaled dot-product attention)

## Why Causal Masking?

In autoregressive models (like GPT), we generate tokens one at a time. Each token should only attend to previous tokens, not future ones.

```
Without masking:     With causal masking:
[✓ ✓ ✓ ✓]           [✓ ✗ ✗ ✗]
[✓ ✓ ✓ ✓]           [✓ ✓ ✗ ✗]
[✓ ✓ ✓ ✓]           [✓ ✓ ✓ ✗]
[✓ ✓ ✓ ✓]           [✓ ✓ ✓ ✓]
```

## Instructions

1. Open `src/masking.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `create_causal_mask(seq_len)`
Create a causal (lower triangular) boolean mask.
- Returns a mask where `True` means "should be masked out"
- Shape: `(seq_len, seq_len)`
- Upper triangle (above diagonal) is `True`

### `apply_mask_to_scores(scores, mask)`
Apply a mask to attention scores before softmax.
- Set masked positions to `-inf`
- Masked positions become 0 after softmax

### `create_padding_mask(seq_lengths, max_len)`
Create a mask for padding tokens in batched sequences.
- Different sequences have different lengths
- Padding tokens should not be attended to

### `combine_masks(causal_mask, padding_mask)`
Combine causal and padding masks.
- A position is masked if EITHER mask says to mask it

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_masking.py::TestCausalMask

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.triu` to create upper triangular matrices
- Remember: `True` in the mask means "mask this out"
- `-np.inf` in scores becomes `0` after softmax
- For padding mask, positions beyond `seq_length` should be masked

## Expected Shapes

```
causal_mask:  (seq_len, seq_len)          # True = masked
padding_mask: (batch, seq_len)             # True = masked
combined:     (batch, seq_len, seq_len)    # Broadcastable for attention
```

## Example Usage

```python
import numpy as np
from masking import create_causal_mask, apply_mask_to_scores

# Create causal mask for sequence of 4 tokens
mask = create_causal_mask(4)
# array([[False,  True,  True,  True],
#        [False, False,  True,  True],
#        [False, False, False,  True],
#        [False, False, False, False]])

# Apply to attention scores
scores = np.random.randn(4, 4)
masked_scores = apply_mask_to_scores(scores, mask)
# Upper triangle is now -inf

# After softmax, upper triangle becomes 0
from scipy.special import softmax
attention_weights = softmax(masked_scores, axis=-1)
# Upper triangle is ~0
```

## Verification

All tests pass = you've correctly implemented causal masking!

This is a key component for building GPT-style decoders in Lab 03.
