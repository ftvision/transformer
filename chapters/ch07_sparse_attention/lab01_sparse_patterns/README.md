# Lab 01: Sparse Attention Patterns

## Objective

Implement various sparse attention mask patterns that restrict which positions can attend to which.

## What You'll Build

Functions to create:
1. Local (sliding window) attention masks
2. Strided (dilated) attention masks
3. Block-sparse attention masks
4. Combined patterns

## Prerequisites

Read these docs first:
- `../docs/01_sparse_patterns.md`

## Why Sparse Patterns?

Standard attention has O(n²) complexity. For long sequences (4K+ tokens), this becomes prohibitive. Sparse patterns restrict attention to a subset of positions, reducing complexity while preserving most of the modeling power.

## Instructions

1. Open `src/sparse_patterns.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `create_local_mask(seq_len, window_size)`

Create a local (sliding window) attention mask.

- Each position attends to `window_size` neighbors on each side
- Combined with causal masking (can't see future)
- Returns a boolean mask where `True` = masked (blocked)

```
Example (seq_len=6, window_size=2):
Position 4 attends to: [2, 3, 4] (not 0, 1 or future)

Mask (False=attend, True=blocked):
[[F T T T T T]    pos 0: sees [0]
 [F F T T T T]    pos 1: sees [0,1]
 [F F F T T T]    pos 2: sees [0,1,2]
 [T F F F T T]    pos 3: sees [1,2,3]
 [T T F F F T]    pos 4: sees [2,3,4]
 [T T T F F F]]   pos 5: sees [3,4,5]
```

### `create_strided_mask(seq_len, stride)`

Create a strided (dilated) attention mask.

- Each position attends to every `stride`-th position
- Combined with causal masking

```
Example (seq_len=8, stride=2):
Position 6 attends to: [0, 2, 4, 6]
```

### `create_block_mask(seq_len, block_size)`

Create a block-sparse attention mask.

- Positions only attend within their block
- Combined with causal masking

```
Example (seq_len=8, block_size=4):
Positions 0-3 attend to positions 0-3
Positions 4-7 attend to positions 4-7
```

### `create_combined_mask(seq_len, window_size, stride)`

Combine local and strided patterns.

- Attend locally (within window)
- Also attend to strided positions (for long-range)
- Combined with causal masking

### `apply_sparse_mask(attention_scores, mask)`

Apply a sparse mask to attention scores.

- Set masked positions to `-inf` before softmax
- Preserves the sparsity pattern

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_sparse_patterns.py::TestLocalMask

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Use `np.triu` for the base causal mask
- For local mask, positions can only attend within `[i - window_size, i]`
- For strided mask, position i attends to positions where `(i - j) % stride == 0`
- Remember: `True` in the mask means BLOCKED (will be set to -inf)

## Expected Shapes

```
All masks: (seq_len, seq_len) boolean array
  - mask[i, j] = True means position i CANNOT attend to position j
  - mask[i, j] = False means position i CAN attend to position j
```

## Visualization

After implementing, you can visualize patterns:

```python
import matplotlib.pyplot as plt

mask = create_local_mask(16, window_size=4)
plt.imshow(~mask, cmap='Blues')  # Invert for visibility
plt.title('Local Attention Pattern')
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.colorbar(label='Can attend')
plt.show()
```

## Verification

All tests pass = you understand sparse attention patterns!

These patterns are the foundation for efficient attention in models like Longformer and Mistral.
