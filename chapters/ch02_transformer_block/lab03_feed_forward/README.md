# Lab 03: Feed-Forward Network

## Objective

Implement the feed-forward network (FFN) component of transformers, including both standard and gated variants.

## What You'll Build

1. **Standard FFN** with GELU activation (GPT-style)
2. **SwiGLU FFN** (LLaMA, Mistral-style)
3. Various activation functions (ReLU, GELU, SiLU/Swish)

## Prerequisites

Read these docs first:
- `../docs/03_feed_forward_network.md`

## The FFN Architecture

The FFN follows an expand-contract pattern:
```
d_model → d_ff (expand) → d_model (contract)
  512   →   2048        →    512
```

Standard FFN:
```
output = W2 @ activation(W1 @ x + b1) + b2
```

SwiGLU FFN:
```
output = W3 @ (silu(W1 @ x) * (W2 @ x))
```

## Instructions

1. Open `src/feed_forward.py`
2. Implement the functions and classes marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### Activation Functions

#### `gelu(x)`
Gaussian Error Linear Unit.
```
GELU(x) = x * Φ(x)  where Φ is the Gaussian CDF
```
Use the approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

#### `silu(x)` (also called Swish)
```
SiLU(x) = x * sigmoid(x)
```

### FFN Classes

#### `class FeedForward`
Standard FFN with configurable activation.
- Parameters: W1, b1, W2, b2
- Forward: `W2 @ activation(W1 @ x + b1) + b2`

#### `class SwiGLUFeedForward`
Gated FFN used in modern LLMs.
- Parameters: W1 (gate), W2 (value), W3 (output)
- No biases
- Forward: `W3 @ (silu(W1 @ x) * (W2 @ x))`

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_feed_forward.py::TestGELU

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

### Activation Functions
- `sigmoid(x) = 1 / (1 + exp(-x))`
- `tanh(x)` is available in NumPy
- For GELU, use the approximation formula (exact formula requires error function)

### FFN Dimensions
- Standard: d_ff = 4 * d_model typically
- SwiGLU: d_ff = (2/3) * 4 * d_model to match parameter count
- Round d_ff to multiple of 256 for efficiency (optional in this lab)

### Weight Initialization
- Initialize weights with small random values: `np.random.randn(...) * 0.02`
- Initialize biases to zeros

## Expected Shapes

```
# Standard FFN
FeedForward(d_model=512, d_ff=2048):
    W1: (512, 2048)
    b1: (2048,)
    W2: (2048, 512)
    b2: (512,)

# SwiGLU FFN
SwiGLUFeedForward(d_model=512, d_ff=1365):
    W1: (512, 1365)  # gate
    W2: (512, 1365)  # value
    W3: (1365, 512)  # output
```

## Key Insights

### Why Expand-Contract?
- The expanded hidden layer (d_ff) gives the network more capacity
- The contraction back to d_model extracts the most useful features
- Think of it as: expand to compute, compress to communicate

### Why Gating (SwiGLU)?
- The gate can "shut off" irrelevant features
- This provides a form of conditional computation
- Empirically improves performance in language models

## Verification

All tests pass = you've correctly implemented the FFN!

Next up: Lab 04 where you'll assemble the complete transformer block.
