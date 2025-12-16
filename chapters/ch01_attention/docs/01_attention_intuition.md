# Attention Intuition: What Problem Does It Solve?

## The Bottleneck Problem

Before transformers, sequence-to-sequence models (like for translation) used encoder-decoder architectures with RNNs. The encoder would process the input sequence and compress it into a single "context vector." The decoder would then use this vector to generate the output.

```
Input: "The cat sat on the mat"
         ↓ (RNN encoder)
    [context vector]  ← Everything crammed in here!
         ↓ (RNN decoder)
Output: "Le chat était assis sur le tapis"
```

**The problem**: That single context vector has to contain *all* information about the input. For long sequences, important details get lost.

## The Attention Solution

Attention lets the decoder "look back" at all encoder states, not just the final one:

```
Input tokens:  [The] [cat] [sat] [on] [the] [mat]
                 ↓     ↓     ↓     ↓     ↓     ↓
Encoder states: [h1]  [h2]  [h3]  [h4]  [h5]  [h6]
                  \    |     |     |     |    /
                   \___|_____|_____|_____|___/
                              ↓
                   Attention weights
                              ↓
                      Weighted sum
                              ↓
                   Context for this step
```

When generating "chat" (French for "cat"), the model can pay high attention to "cat" and less to other words.

## Attention as Database Lookup

A useful analogy: attention is like a "soft" database lookup.

**Hard lookup (exact match)**:
```python
database = {"cat": "animal", "dog": "animal", "car": "vehicle"}
query = "cat"
result = database[query]  # Returns "animal"
```

**Soft lookup (attention)**:
```python
# Query: What am I looking for?
# Keys: What does each item contain?
# Values: What do I retrieve?

# Instead of exact match, compute similarity to ALL keys
similarities = [similarity(query, key) for key in keys]

# Convert to weights that sum to 1
weights = softmax(similarities)

# Weighted sum of ALL values
result = sum(weight * value for weight, value in zip(weights, values))
```

The "soft" part means we retrieve a little bit of everything, weighted by relevance.

## Self-Attention: Attending to Yourself

In transformers, we use **self-attention**: each token attends to all tokens in the same sequence (including itself).

Consider: "The animal didn't cross the street because it was too tired"

What does "it" refer to? A human instantly knows "it" = "animal" (not "street").

Self-attention learns these relationships:
- When processing "it", the model can attend strongly to "animal"
- The attention weights encode that "it" is related to "animal"

```
Tokens:    [The] [animal] [didn't] [cross] [the] [street] [because] [it] [was] [too] [tired]
                                                                      ↑
                                                              Processing this token
Attention:  0.02   0.45     0.03    0.02   0.01   0.08      0.03    0.30  0.02  0.02  0.02
                    ↑                                                  ↑
            High attention to "animal"                      Self-attention
```

## Why Attention Works Well

1. **No information bottleneck**: Can access any part of the input directly
2. **Parallel computation**: Unlike RNNs, all attention computations can happen simultaneously
3. **Long-range dependencies**: Token 1 can directly attend to token 1000 (no vanishing gradients through 999 steps)
4. **Interpretable**: Attention weights show what the model is "looking at"

## Query, Key, Value: The Three Projections

In practice, we don't use the raw token representations. We project them:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What do I return if matched?"

Why separate projections?
- Query and Key live in a "matching space" optimized for similarity computation
- Value lives in an "output space" optimized for information retrieval
- This gives the model flexibility to learn different representations for matching vs. retrieving

## What's Next

Now that you understand *what* attention does intuitively, let's look at the exact mathematical formulation in `02_scaled_dot_product.md`.
