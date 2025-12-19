# Lab 03: Decoder-Only Transformer

## Objective

Build a complete GPT-style decoder-only transformer by stacking transformer blocks.

## What You'll Build

A complete transformer model that:
1. Takes token IDs as input
2. Processes through embedding layer
3. Passes through multiple transformer blocks
4. Outputs logits for next-token prediction

## Prerequisites

- Complete Lab 01 (causal masking)
- Complete Lab 02 (token embeddings)
- Chapter 1 Lab 03 (multi-head attention)
- Chapter 2 concepts (transformer block components)

Read these docs:
- `../docs/01_encoder_decoder_architectures.md`
- `../docs/02_causal_masking.md`

## Architecture Overview

```
Token IDs: [101, 2054, 2003, 102]
              ↓
┌─────────────────────────────────┐
│     Token Embedding             │
│     + Positional Embedding      │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│     Transformer Block 1         │
│  ┌───────────────────────────┐  │
│  │  LayerNorm                │  │
│  │  Multi-Head Self-Attention│  │
│  │  (with causal mask)       │  │
│  │  + Residual               │  │
│  ├───────────────────────────┤  │
│  │  LayerNorm                │  │
│  │  Feed-Forward Network     │  │
│  │  + Residual               │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
              ↓
        ... (N blocks) ...
              ↓
┌─────────────────────────────────┐
│     Final LayerNorm             │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│     Output Projection (LM Head) │
│     → Logits over vocabulary    │
└─────────────────────────────────┘
```

## Instructions

1. Open `src/decoder.py`
2. Implement the classes marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Classes to Implement

### `LayerNorm`

Layer normalization for transformer blocks.

```python
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        """Layer normalization with learnable parameters."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize x to have mean=0, var=1, then scale and shift."""
```

### `FeedForward`

Position-wise feed-forward network.

```python
class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Two linear layers with activation.
        d_ff is typically 4 * d_model.
        """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x → Linear → GELU → Linear → output"""
```

### `TransformerBlock`

A single transformer decoder block.

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Initialize attention, feed-forward, and layer norms."""

    def forward(self, x, mask=None):
        """
        Pre-norm architecture:
        x → LayerNorm → Attention → + x → LayerNorm → FFN → + x
        """
```

### `GPTModel`

Complete GPT-style decoder transformer.

```python
class GPTModel:
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        """Initialize embeddings, blocks, final norm, and output projection."""

    def forward(self, token_ids, mask=None):
        """
        Full forward pass: tokens → embeddings → blocks → norm → logits
        """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_decoder.py::TestLayerNorm

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### LayerNorm
- Normalize over last dimension: `mean = x.mean(axis=-1, keepdims=True)`
- `gamma` (scale) initialized to 1, `beta` (shift) to 0
- Formula: `output = gamma * (x - mean) / sqrt(var + eps) + beta`

### FeedForward
- First linear: d_model → d_ff
- GELU activation
- Second linear: d_ff → d_model
- Dropout after second linear

### TransformerBlock (Pre-norm)
```python
# Attention sub-layer
residual = x
x = layer_norm_1(x)
x = attention(x, mask)
x = dropout(x)
x = residual + x

# FFN sub-layer
residual = x
x = layer_norm_2(x)
x = feed_forward(x)
x = residual + x
```

### GPTModel
- Create N transformer blocks in a list
- Apply causal mask automatically
- Final layer norm before output projection

## Expected Shapes

For batch_size=2, seq_len=10, d_model=64, vocab_size=1000:

```
Input token_ids:  (2, 10)          # Integer token IDs
After embedding:  (2, 10, 64)       # + positional
After blocks:     (2, 10, 64)       # Same shape throughout
After final norm: (2, 10, 64)
Output logits:    (2, 10, 1000)     # Logits for each position
```

## GELU Activation

GELU (Gaussian Error Linear Unit) is used in GPT-2:

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

Or the simpler approximation:
```python
def gelu(x):
    return x * 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
```

## Example Usage

```python
from decoder import GPTModel

# Create a small GPT model
model = GPTModel(
    vocab_size=50257,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    max_seq_len=1024
)

# Forward pass
token_ids = np.array([[101, 2054, 2003, 102]])
logits = model(token_ids)
print(logits.shape)  # (1, 4, 50257)

# Get next token prediction for last position
next_token_logits = logits[0, -1, :]
next_token = np.argmax(next_token_logits)
```

## GPT-2 Small Configuration

For reference, GPT-2 small uses:
- vocab_size: 50257
- d_model: 768
- num_layers: 12
- num_heads: 12
- d_ff: 3072 (4 * 768)
- max_seq_len: 1024

## Verification

All tests pass = you've built a complete decoder transformer!

In Lab 04, you'll load pretrained GPT-2 weights and verify your implementation matches HuggingFace.
