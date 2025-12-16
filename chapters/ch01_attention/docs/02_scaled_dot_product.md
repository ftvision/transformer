# Scaled Dot-Product Attention: The Math

## The Formula

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Let's break this down step by step.

## Step 1: The Dot Product (QK^T)

Given:
- Q (Query): shape `(seq_len, d_k)`
- K (Key): shape `(seq_len, d_k)`

The dot product `QK^T` computes similarity between every query and every key:

```
QK^T has shape (seq_len, seq_len)

Entry [i, j] = dot(Q[i], K[j]) = Σ Q[i,k] * K[j,k]
```

This gives us a matrix of "raw attention scores":
- High score → query i is similar to key j
- Low score → query i is dissimilar to key j

**Example**:
```python
Q = [[1, 0],    # Query for token 0
     [0, 1]]    # Query for token 1

K = [[1, 0],    # Key for token 0
     [0, 1]]    # Key for token 1

QK^T = [[1, 0],    # Token 0 attends to token 0
        [0, 1]]    # Token 1 attends to token 1
```

## Step 2: Scaling by √d_k

Why divide by √d_k?

**The variance problem**: If Q and K have entries with mean 0 and variance 1, then:
- Each element of QK^T is a sum of d_k products
- The variance of QK^T entries is approximately d_k

When d_k is large (e.g., 64), QK^T entries become large in magnitude.

**Why this matters for softmax**: softmax(x) = exp(x) / Σexp(x)
- Large inputs → exp() values become huge or tiny
- This pushes softmax toward one-hot vectors
- Gradients become very small (saturation)

**The fix**: Divide by √d_k to keep variance ≈ 1
```
Var(QK^T / √d_k) ≈ Var(QK^T) / d_k = d_k / d_k = 1
```

## Step 3: Softmax

Softmax converts raw scores to a probability distribution:

```python
def softmax(x):
    exp_x = exp(x - max(x))  # Subtract max for numerical stability
    return exp_x / sum(exp_x)
```

Properties:
- All outputs are positive
- Outputs sum to 1
- Preserves relative ordering (highest score → highest weight)

**Example**:
```
scores = [2.0, 1.0, 0.1]
softmax(scores) ≈ [0.659, 0.242, 0.099]
```

After softmax, we have attention weights: `(seq_len, seq_len)` where each row sums to 1.

## Step 4: Weighted Sum (Attention × V)

Given:
- Attention weights: shape `(seq_len, seq_len)`
- V (Value): shape `(seq_len, d_v)`

The matrix multiplication `Attention @ V`:
- Each output position i gets a weighted combination of all values
- Weights come from attention[i, :]

```
Output[i] = Σ_j attention[i,j] * V[j]
```

**Result**: shape `(seq_len, d_v)` - same sequence length, value dimension.

## Putting It All Together

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    scores = Q @ K.T  # (seq_len, seq_len)

    # Step 2: Scale
    scores = scores / np.sqrt(d_k)

    # Step 3: Softmax (row-wise)
    attention_weights = softmax(scores, axis=-1)

    # Step 4: Weighted sum of values
    output = attention_weights @ V  # (seq_len, d_v)

    return output, attention_weights
```

## Batched Attention

In practice, we process batches of sequences:

```
Q: (batch_size, seq_len, d_k)
K: (batch_size, seq_len, d_k)
V: (batch_size, seq_len, d_v)
```

The formula is the same, applied independently to each batch element.

With PyTorch/NumPy broadcasting, `Q @ K.transpose(-2, -1)` handles this automatically.

## Computational Complexity

- **Time**: O(n² × d) where n = seq_len, d = dimension
  - QK^T: (n, d) × (d, n) = O(n²d)
  - Attention × V: (n, n) × (n, d) = O(n²d)

- **Memory**: O(n²) for the attention matrix
  - This is the bottleneck for long sequences!
  - 4096 tokens → 16M attention weights per layer per head

This O(n²) complexity is why linear attention (Chapter 5) and Flash Attention (Chapter 10) matter.

## What's Next

Now you understand the single-head case. But transformers use **multi-head attention** - running multiple attention operations in parallel. See `03_multihead_attention.md`.
