# Lab 04: Transformer Block

## Objective

Assemble a complete transformer block using all the components you've built, and verify it matches HuggingFace GPT-2's output.

## What You'll Build

A `TransformerBlock` class that combines:
1. Multi-head attention (from Chapter 1)
2. Layer normalization (from Lab 01)
3. Feed-forward network (from Lab 03)
4. Residual connections

## Prerequisites

- Complete Chapter 1 (multi-head attention)
- Complete Labs 01-03 (layer norm, positional encoding, FFN)
- Read `../docs/04_transformer_block.md`

## The Architecture

```
    Input
      │
      ├──────────────────────────────┐
      ↓                              │
  LayerNorm                          │
      │                              │
      ↓                              │
  Multi-Head Attention               │  Residual
      │                              │
      ↓                              │
      + ←────────────────────────────┘
      │
      ├──────────────────────────────┐
      ↓                              │
  LayerNorm                          │
      │                              │
      ↓                              │
  Feed-Forward Network               │  Residual
      │                              │
      ↓                              │
      + ←────────────────────────────┘
      │
      ↓
   Output
```

## Instructions

1. Open `src/transformer_block.py`
2. Implement the `TransformerBlock` class
3. Run tests: `uv run pytest tests/`

## Class to Implement

### `TransformerBlock`

```python
class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.0,
        pre_norm: bool = True
    ):
        """
        Initialize a transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension (default: 4 * d_model)
            dropout: Dropout probability (for future use)
            pre_norm: If True, use pre-norm; if False, use post-norm
        """

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass through the transformer block.

        Args:
            x: Input of shape (batch, seq_len, d_model) or (seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output of same shape as x
        """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_transformer_block.py::TestTransformerBlockInit

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Pre-Norm vs Post-Norm

**Pre-norm** (modern, recommended):
```python
# Attention sub-layer
x = x + attention(norm1(x))

# FFN sub-layer
x = x + ffn(norm2(x))
```

**Post-norm** (original transformer):
```python
# Attention sub-layer
x = norm1(x + attention(x))

# FFN sub-layer
x = norm2(x + ffn(x))
```

### Residual Connections

The key is: always add the input back to the output.
```python
residual = x
x = sublayer(x)
x = residual + x  # or x = x + sublayer(x) in one line
```

### Using Components from Previous Labs

You can either:
1. Import from previous labs
2. Copy the implementations into this file

The tests don't require PyTorch matching for the transformer block itself,
but there's a bonus test for matching GPT-2 if you want to verify.

## Expected Shapes

```
TransformerBlock(d_model=512, num_heads=8, d_ff=2048):

Input:  (batch=2, seq_len=10, d_model=512)
Output: (batch=2, seq_len=10, d_model=512)

Internal shapes:
  norm1(x):     (2, 10, 512)
  attention():  (2, 10, 512)
  + residual:   (2, 10, 512)
  norm2(x):     (2, 10, 512)
  ffn():        (2, 10, 512)
  + residual:   (2, 10, 512)
```

## Chapter 2 Milestone

The milestone for Chapter 2:
> Your transformer block forward pass matches HuggingFace GPT-2 block output.

This requires careful attention to:
- GPT-2 uses post-norm style with a specific ordering
- Weight shapes and transpositions
- The exact attention implementation

There's a bonus test that compares with GPT-2 if you want to achieve this milestone.

## Verification

All tests pass = you've correctly assembled a transformer block!

Congratulations on completing Chapter 2! You now have all the building blocks
to construct a complete transformer model in Chapter 3.
