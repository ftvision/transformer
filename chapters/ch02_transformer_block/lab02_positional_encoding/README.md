# Lab 02: Positional Encodings

## Objective

Implement sinusoidal positional encodings and Rotary Position Embeddings (RoPE).

## What You'll Build

1. **Sinusoidal positional encoding** (original Transformer)
2. **Learned positional embeddings** (GPT-style)
3. **Rotary Position Embeddings (RoPE)** (LLaMA, Mistral)

## Prerequisites

Read these docs first:
- `../docs/02_positional_encoding.md`

## Why Position Matters

Transformers process all tokens in parallel - they have no inherent notion of order. Without positional information:
- "The cat sat" and "sat cat The" look identical
- The model can't distinguish first word from last

Positional encodings inject order information into the token representations.

## Instructions

1. Open `src/positional_encoding.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `sinusoidal_encoding(seq_len, d_model)`
Create the original transformer's positional encoding.

Formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### `class LearnedPositionalEmbedding`
A simple embedding table where position embeddings are learnable.

### `precompute_freqs_cis(d_model, max_seq_len, base=10000)`
Precompute the frequency values for RoPE.

Returns complex exponentials: `exp(i * pos * theta)` for each position and dimension.

### `apply_rotary_emb(x, freqs_cis)`
Apply rotary embeddings to input tensor.

This rotates pairs of dimensions based on position.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_positional_encoding.py::TestSinusoidal

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Sinusoidal Encoding
- Create a position array: `[0, 1, 2, ..., seq_len-1]`
- Create a dimension array for frequencies
- Use broadcasting to compute all values at once
- Even indices get sin, odd indices get cos

### RoPE
- Think of it as rotating 2D vectors
- Group consecutive pairs of dimensions
- Each pair is rotated by `pos * theta_i`
- Use complex numbers for easy rotation: `z * exp(i * angle)` rotates z by angle

## Expected Shapes

```
sinusoidal_encoding(seq_len=100, d_model=512):
    output: (100, 512)

LearnedPositionalEmbedding(max_seq_len=1024, d_model=512):
    embedding.shape: (1024, 512)

precompute_freqs_cis(d_model=64, max_seq_len=100):
    output: (100, 32) complex  # d_model/2 pairs

apply_rotary_emb(x, freqs_cis):
    x: (batch, seq_len, num_heads, d_k)
    freqs_cis: (seq_len, d_k/2)
    output: same shape as x
```

## Key Insights

### Sinusoidal: Different Frequencies
Each dimension has a different frequency:
- Low dimensions: high frequency (changes fast)
- High dimensions: low frequency (changes slow)

Together, they create a unique "fingerprint" for each position.

### RoPE: Relative Position
The magic of RoPE is that when computing attention:
```
q_m · k_n = f(q, k, m-n)
```
The dot product only depends on the *relative* position (m-n), not absolute positions.

## Verification

All tests pass = you've correctly implemented positional encodings!

Next up: Lab 03 where you'll implement the feed-forward network.
