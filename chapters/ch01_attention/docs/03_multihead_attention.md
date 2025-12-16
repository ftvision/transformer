# Multi-Head Attention: Why Multiple Heads?

## The Limitation of Single-Head Attention

With single-head attention, each position can only attend to other positions in one way. But language has many types of relationships:

- Syntactic: subject-verb agreement ("The cats **are** sleeping")
- Semantic: word meanings ("bank" near "river" vs "money")
- Positional: nearby words, sentence structure
- Coreference: pronouns and their referents

One attention pattern can't capture all these simultaneously.

## The Multi-Head Solution

Multi-head attention runs several attention operations in parallel, each learning different patterns:

```
              Input
                |
        ┌───────┼───────┐
        ↓       ↓       ↓
      Head 1  Head 2  Head 3  ... Head H
        |       |       |           |
        ↓       ↓       ↓           ↓
    [pattern] [pattern] [pattern] [pattern]
        |       |       |           |
        └───────┼───────┘
                ↓
            Concat
                ↓
          Linear projection
                ↓
             Output
```

## The Math

Given input X of shape `(seq_len, d_model)`:

1. **Project to Q, K, V for each head**:
   ```
   Q_i = X @ W_Q_i    # shape: (seq_len, d_k)
   K_i = X @ W_K_i    # shape: (seq_len, d_k)
   V_i = X @ W_V_i    # shape: (seq_len, d_v)
   ```

2. **Compute attention for each head**:
   ```
   head_i = Attention(Q_i, K_i, V_i)    # shape: (seq_len, d_v)
   ```

3. **Concatenate all heads**:
   ```
   concat = [head_1, head_2, ..., head_H]    # shape: (seq_len, H * d_v)
   ```

4. **Final linear projection**:
   ```
   output = concat @ W_O    # shape: (seq_len, d_model)
   ```

## Typical Dimensions

For a model with d_model = 512 and 8 heads:
- d_k = d_v = d_model / H = 512 / 8 = 64
- Each head operates in a 64-dimensional subspace
- Total parameters are roughly the same as single-head with d_k = 512

## What Different Heads Learn

Research has shown that different heads often specialize:

- **Head A**: Attends to previous token (positional)
- **Head B**: Attends to syntactically related words
- **Head C**: Attends to semantically similar words
- **Head D**: Attends to specific token patterns

This specialization emerges from training - we don't explicitly program it.

## Efficient Implementation

Instead of computing H separate attention operations, we can batch them:

```python
# Naive: H separate operations
outputs = []
for i in range(num_heads):
    Q_i = X @ W_Q[i]
    K_i = X @ W_K[i]
    V_i = X @ W_V[i]
    outputs.append(attention(Q_i, K_i, V_i))

# Efficient: Single batched operation
# Combine all W_Q into one matrix, reshape to separate heads
Q = X @ W_Q  # (batch, seq_len, d_model)
Q = Q.reshape(batch, seq_len, num_heads, d_k)
Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
# Same for K, V

# Now attention is computed in parallel across heads
```

## Why Not Just Use a Bigger Single Head?

Two reasons multi-head is preferred:

1. **Different subspaces**: Each head can learn independent patterns in different representation subspaces

2. **Computational efficiency**:
   - Single head: d_model × d_model attention → O(d²)
   - Multi-head: H × (d_model/H)² = d²/H attention → O(d²/H) per head
   - Total is similar, but patterns are more diverse

## The Output Projection

The final W_O projection serves two purposes:

1. **Dimensionality**: Maps from `H * d_v` back to `d_model`
2. **Mixing**: Allows information from different heads to interact

Without W_O, heads would be completely independent.

## Code Structure

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Project and reshape to (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention for all heads in parallel
        attn_output = scaled_dot_product_attention(Q, K, V)

        # Reshape back and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_O(attn_output)

        return output
```

## What's Next

You now understand the theory! Time to implement it yourself in the labs:

1. **Lab 01**: Implement single-head scaled dot-product attention
2. **Lab 02**: Visualize attention patterns to build intuition
3. **Lab 03**: Build full multi-head attention
4. **Lab 04**: Verify your implementation matches PyTorch
