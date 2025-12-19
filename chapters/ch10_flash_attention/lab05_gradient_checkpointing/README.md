# Lab 05: Gradient Checkpointing

## Objective

Implement gradient checkpointing to trade compute for memory.

## What You'll Build

A `CheckpointedAttention` class that:
1. Saves only essential activations during forward pass
2. Recomputes activations during backward pass
3. Reduces memory usage for training

## Prerequisites

- Complete Labs 01-04
- Read `../docs/06_gradient_checkpointing.md`

## Why Gradient Checkpointing?

During training, standard backpropagation stores all intermediate activations:
- For attention: O(N² + Nd) memory per layer
- With L layers: O(L × (N² + Nd)) total

Gradient checkpointing reduces this by:
- Not storing intermediate activations
- Recomputing them during backward pass
- Trading ~30% more compute for ~50% less memory

## Instructions

1. Open `src/checkpointing.py`
2. Implement the checkpointing functions
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `CheckpointedFunction`

Base class for checkpointed operations.

```python
class CheckpointedFunction:
    def __init__(self):
        """Initialize checkpointed function."""

    def forward(self, *args):
        """Forward pass - save inputs, not intermediates."""

    def backward(self, grad_output):
        """Backward pass - recompute forward, then compute gradients."""
```

### `CheckpointedAttention`

```python
class CheckpointedAttention:
    def __init__(self):
        """Initialize checkpointed attention."""

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass with checkpointing.

        Only stores Q, K, V (inputs), not attention weights!

        Returns:
            output: Attention output
        """

    def backward(self, grad_output):
        """
        Backward pass with recomputation.

        1. Recompute attention weights from stored Q, K, V
        2. Compute gradients using recomputed weights

        Returns:
            grad_Q, grad_K, grad_V
        """
```

### `measure_memory_savings()`

```python
def measure_memory_savings(
    seq_len: int,
    d_model: int,
    num_layers: int
) -> dict:
    """
    Measure memory savings from checkpointing.

    Returns:
        Dictionary with:
        - standard_memory: Memory without checkpointing
        - checkpointed_memory: Memory with checkpointing
        - memory_saved: Percentage saved
        - compute_overhead: Extra compute percentage
    """
```

### `checkpoint_sequential()`

```python
def checkpoint_sequential(
    functions: list,
    *inputs
) -> tuple:
    """
    Apply checkpointing to a sequence of functions.

    Like PyTorch's checkpoint_sequential but simplified.

    Args:
        functions: List of functions to apply
        inputs: Input tensors

    Returns:
        Output tensors
    """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_checkpointing.py::TestCheckpointedAttention

# Run with verbose output
uv run pytest tests/ -v
```

## The Trade-off

```
Standard Training:
- Forward: Compute + Store all activations
- Backward: Use stored activations
- Memory: O(L × N²) for L layers

Checkpointed Training:
- Forward: Compute, store only inputs
- Backward: Recompute activations, then compute gradients
- Memory: O(N²) per checkpoint segment
- Compute: ~1.3-1.5x (one extra forward pass)
```

## Expected Results

```python
# Memory comparison
results = measure_memory_savings(seq_len=2048, d_model=512, num_layers=12)
print(f"Standard: {results['standard_memory_mb']:.1f} MB")
print(f"Checkpointed: {results['checkpointed_memory_mb']:.1f} MB")
print(f"Memory saved: {results['memory_saved_pct']:.1f}%")
print(f"Compute overhead: {results['compute_overhead_pct']:.1f}%")

# Output should be identical
output_standard = standard_forward(Q, K, V)
output_checkpointed = checkpointed_forward(Q, K, V)
np.testing.assert_allclose(output_standard, output_checkpointed)
```

## Checkpointing Strategies

### Every Layer

```python
# Checkpoint every layer
for layer in layers:
    x = checkpoint(layer, x)
```

### Every N Layers

```python
# Checkpoint every 2 layers
for i, layer in enumerate(layers):
    if i % 2 == 0:
        x = checkpoint(layer, x)
    else:
        x = layer(x)
```

### Selective Checkpointing

```python
# Only checkpoint attention (most memory-intensive)
for layer in layers:
    x = checkpoint(layer.attention, x)  # Checkpointed
    x = layer.ffn(x)                     # Not checkpointed
```

## Connection to Flash Attention

Flash Attention and gradient checkpointing are complementary:
- Flash Attention: Reduces forward pass memory (no N² storage)
- Checkpointing: Reduces backward pass memory (recompute instead of store)
- Together: Maximum memory efficiency for training long sequences
