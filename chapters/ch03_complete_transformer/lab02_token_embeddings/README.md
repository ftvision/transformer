# Lab 02: Token Embeddings

## Objective

Implement token embeddings and positional embeddings for transformer models.

## What You'll Build

Classes to:
1. Convert token IDs to dense vectors (token embeddings)
2. Add positional information (positional embeddings)
3. Combine them into a complete embedding layer

## Prerequisites

Read these docs first:
- `../docs/03_embeddings_and_vocabulary.md`

Also helpful:
- Chapter 2 docs on positional encoding (if available)

## Why Embeddings?

Neural networks work with continuous numbers, not discrete tokens. Embeddings provide:
- A learnable representation for each token
- Position information (transformers have no inherent sense of order)
- The input to the transformer stack

```
Token IDs:    [42,    156,    78,     12]
                ↓       ↓       ↓       ↓
Embeddings:  [0.12,  [0.45,  [-0.23, [0.89,
              -0.34,  0.12,   0.67,   -0.12,
              ...]    ...]    ...]    ...]

Positions:   [pos_0, pos_1,  pos_2,  pos_3]
                ↓       ↓       ↓       ↓
             [sin/cos patterns or learned vectors]

Final:       embed + position
```

## Instructions

1. Open `src/embeddings.py`
2. Implement the classes marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Classes to Implement

### `TokenEmbedding`

A lookup table that maps token IDs to dense vectors.

```python
class TokenEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Number of unique tokens
            d_model: Embedding dimension
        """

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Integer array of shape (batch, seq_len) or (seq_len,)

        Returns:
            Embeddings of shape (batch, seq_len, d_model) or (seq_len, d_model)
        """
```

### `PositionalEmbedding`

Learned positional embeddings (like GPT-2).

```python
class PositionalEmbedding:
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Args:
            max_seq_len: Maximum sequence length
            d_model: Embedding dimension
        """

    def forward(self, seq_len: int) -> np.ndarray:
        """
        Get positional embeddings for a sequence.

        Args:
            seq_len: Length of sequence

        Returns:
            Positional embeddings of shape (seq_len, d_model)
        """
```

### `SinusoidalPositionalEncoding`

Fixed sinusoidal positional encodings (like original Transformer).

```python
class SinusoidalPositionalEncoding:
    def __init__(self, max_seq_len: int, d_model: int):
        """Precompute sinusoidal patterns."""

    def forward(self, seq_len: int) -> np.ndarray:
        """Return positional encodings for given sequence length."""
```

### `TransformerEmbedding`

Complete embedding layer combining token and positional embeddings.

```python
class TransformerEmbedding:
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        """Initialize token and positional embeddings."""

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings with position information.

        Args:
            token_ids: Integer array of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_embeddings.py::TestTokenEmbedding

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Token embedding is just a lookup: `embeddings[token_ids]`
- Positional embedding indexes by position: `pos_embeddings[0:seq_len]`
- Sinusoidal: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- Sinusoidal: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

## Expected Shapes

```
TokenEmbedding:
    weight: (vocab_size, d_model)
    input:  (batch, seq_len) integers
    output: (batch, seq_len, d_model)

PositionalEmbedding:
    weight: (max_seq_len, d_model)
    input:  seq_len integer
    output: (seq_len, d_model)

TransformerEmbedding:
    input:  (batch, seq_len) integers
    output: (batch, seq_len, d_model)
```

## Example Usage

```python
import numpy as np
from embeddings import TransformerEmbedding

# Create embedding layer
embedding = TransformerEmbedding(
    vocab_size=50000,
    d_model=768,
    max_seq_len=1024
)

# Sample token IDs (batch of 2, sequence length 4)
token_ids = np.array([
    [101, 2054, 2003, 102],
    [101, 7592, 1010, 102]
])

# Get embeddings
embeddings = embedding(token_ids)
print(embeddings.shape)  # (2, 4, 768)
```

## Sinusoidal Encoding Formula

The sinusoidal positional encoding uses:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position (0, 1, 2, ...)
- `i` is the dimension index (0, 1, 2, ..., d_model/2 - 1)
- This creates unique patterns for each position

## Verification

All tests pass = you've correctly implemented embeddings!

These embeddings will be the input to your transformer in Lab 03.
