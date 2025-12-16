# Lab 02: Attention Visualization

## Objective

Build tools to visualize attention patterns, helping you understand what the model "looks at."

## What You'll Build

Functions to:
1. Create attention heatmaps showing which tokens attend to which
2. Visualize multiple attention patterns (useful for multi-head analysis later)
3. Analyze attention statistics (entropy, sparsity)

## Prerequisites

- Complete Lab 01 (dot-product attention)
- Read `../docs/01_attention_intuition.md` (especially the "it" → "animal" example)

## Why Visualization Matters

Attention weights are one of the few interpretable parts of transformers. Visualizing them helps you:
- Debug attention implementations
- Understand what patterns the model learns
- Build intuition for how attention solves different tasks

## Instructions

1. Open `src/visualization.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `create_attention_heatmap(attention_weights, query_labels, key_labels)`

Create a heatmap data structure for attention weights.

- Input: attention weights `(seq_len_q, seq_len_k)`, labels for queries and keys
- Output: A dict with data needed for visualization

### `compute_attention_entropy(attention_weights)`

Compute the entropy of attention distributions.

- High entropy = attention spread across many positions (diffuse)
- Low entropy = attention focused on few positions (peaked)
- Formula: `H = -Σ p * log(p)` where p is the attention weights

### `find_top_k_attended(attention_weights, query_idx, k=3)`

Find the top-k most attended positions for a given query.

Useful for answering "what does token X look at?"

### `compute_attention_sparsity(attention_weights, threshold=0.1)`

Compute what fraction of attention weights are below a threshold.

High sparsity = attention is focused on few positions.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test class
uv run pytest tests/test_visualization.py::TestAttentionHeatmap
```

## Example Usage

```python
import numpy as np
from visualization import (
    create_attention_heatmap,
    compute_attention_entropy,
    find_top_k_attended
)

# Example attention weights (4 queries attending to 4 keys)
weights = np.array([
    [0.7, 0.1, 0.1, 0.1],  # Query 0 focuses on position 0
    [0.1, 0.6, 0.2, 0.1],  # Query 1 focuses on position 1
    [0.1, 0.1, 0.7, 0.1],  # Query 2 focuses on position 2
    [0.2, 0.2, 0.2, 0.4],  # Query 3 is more diffuse
])

tokens = ["The", "cat", "sat", "down"]

# Create heatmap
heatmap = create_attention_heatmap(weights, tokens, tokens)

# Analyze entropy
entropy = compute_attention_entropy(weights)
# Query 3 will have higher entropy (more diffuse attention)

# Find what "sat" (position 2) attends to
top_attended = find_top_k_attended(weights, query_idx=2, k=2)
# Returns indices [2, 0] or similar (top attended positions)
```

## Visualization Tips

The tests don't require actual plotting (that would need matplotlib), but the data structures you create should be ready for visualization.

When you're done, you can optionally create plots:

```python
import matplotlib.pyplot as plt

def plot_heatmap(heatmap_data):
    """Optional: Actually plot the heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data['weights'], cmap='Blues')
    plt.xticks(range(len(heatmap_data['key_labels'])),
               heatmap_data['key_labels'], rotation=45)
    plt.yticks(range(len(heatmap_data['query_labels'])),
               heatmap_data['query_labels'])
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.title('Attention Heatmap')
    plt.tight_layout()
    plt.show()
```

## Verification

All tests pass = you've built the visualization toolkit!

These tools will be essential when analyzing multi-head attention in Lab 03.
