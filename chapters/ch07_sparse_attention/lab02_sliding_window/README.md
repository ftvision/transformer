# Lab 02: Sliding Window Attention

## Objective

Implement Longformer-style sliding window attention with global tokens.

## What You'll Build

A `SlidingWindowAttention` class that:
1. Applies local sliding window attention for most tokens
2. Allows designated "global" tokens to attend to/from all positions
3. Uses efficient masking for the combined pattern

## Prerequisites

- Complete Lab 01 (sparse patterns)
- Read `../docs/02_sliding_window.md`

## Why Sliding Window + Global?

Pure sliding window attention limits long-range dependencies. Global tokens solve this:
- Special tokens (like [CLS]) can aggregate information from the entire sequence
- Any token can access global context through these tokens
- Maintains O(n) complexity while enabling long-range communication

## Instructions

1. Open `src/sliding_window.py`
2. Implement the class and functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Class to Implement

### `SlidingWindowAttention`

```python
class SlidingWindowAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        num_global_tokens: int = 0
    ):
        """
        Initialize sliding window attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            window_size: Local window size (positions each token attends to)
            num_global_tokens: Number of initial tokens that are global
        """

    def forward(
        self,
        x: np.ndarray,
        global_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute sliding window attention.

        Args:
            x: Input of shape (batch, seq_len, d_model) or (seq_len, d_model)
            global_mask: Optional (seq_len,) boolean mask marking global positions

        Returns:
            Output of same shape as input
        """
```

## Functions to Implement

### `create_sliding_window_mask(seq_len, window_size, global_positions=None)`

Create the combined sliding window + global attention mask.

```
Example (seq_len=8, window_size=3, global_positions=[0]):
Position 0 is global: attends to all, all attend to it

Pattern (False=attend, True=blocked):
[[F F F F F F F F]    Global: attends to all
 [F F T T T T T T]    pos 1: [0, 1]
 [F F F T T T T T]    pos 2: [0, 1, 2]
 [F T F F T T T T]    pos 3: [0, 2, 3]
 [F T T F F T T T]    pos 4: [0, 3, 4]
 [F T T T F F T T]    pos 5: [0, 4, 5]
 [F T T T T F F T]    pos 6: [0, 5, 6]
 [F T T T T T F F]]   pos 7: [0, 6, 7]
  ↑
  All attend to global position 0
```

### `sliding_window_attention(Q, K, V, window_size, global_mask=None)`

Compute attention with sliding window + global tokens.

## The Algorithm

For each position i:
1. **Local attention**: Attend to positions `[max(0, i - window_size + 1), i]`
2. **Global attention**:
   - If i is global: attend to all positions
   - Otherwise: also attend to all global positions
3. Combine with causal masking

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_sliding_window.py::TestSlidingWindowMask

# Run with verbose output
uv run pytest tests/ -v
```

## Hints

- Start with the pure sliding window (no global tokens)
- Add global tokens as an extension
- Global tokens need separate Q, K, V projections in Longformer (but we simplify here)
- The mask should be: `local_mask AND NOT(is_global_query OR is_global_key)`

## Expected Behavior

```python
# Without global tokens
swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16)
x = np.random.randn(2, 100, 64)  # batch=2, seq=100
output = swa(x)  # Only local attention

# With global tokens (first 2 positions are global)
swa = SlidingWindowAttention(d_model=64, num_heads=8, window_size=16, num_global_tokens=2)
output = swa(x)  # Positions 0,1 attend to all; all attend to 0,1
```

## Memory Efficiency Note

In production implementations (like Mistral), the sliding window enables a **rolling KV-cache**:
- Only store the last `window_size` K, V vectors
- Memory is O(window_size) instead of O(seq_len)

We implement the mask-based version here for clarity. See Chapter 8 for KV-cache optimization.

## Verification

All tests pass = you've implemented Longformer-style attention!

This pattern is used in production models like Mistral and Longformer for long-context processing.
