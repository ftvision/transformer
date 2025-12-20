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

### A Deeper Intuition: The Library Analogy

Think of attention like searching a library:

1. **Query (Q)** - Your question/need: "I want to know about cooking Italian pasta"
2. **Keys (K)** - The labels on each book: "Italian Cuisine", "French Cooking", "Car Repair"
3. **Values (V)** - The actual content inside each book

The process:
- Compare your **query** against all **keys** (book labels)
- "Italian Cuisine" matches well → high score; "Car Repair" doesn't → low score
- Retrieve **values** (actual content) weighted by those match scores

**The key insight**: The query transforms "what I am" into "what I need."

For the word "it" in "The animal didn't cross the street because **it** was tired":
- The word "it" by itself just means "pronoun"
- But its **query projection** learns to ask "what's the antecedent noun?"
- The word "animal" has a **key** that advertises "I'm a noun, a subject, animate"
- When the query matches this key, "it" retrieves "animal"'s **value** (rich semantic content)

### Why Separate Keys and Values?

- **Key** is optimized for *matching* - like a label or index on a filing cabinet
- **Value** is optimized for *content* - the actual documents inside

If they were the same, the model would have to compromise between being a good "label" and providing good "content."

## Q, K, V: The Matrix Mechanics

Let's see how this works computationally with actual matrix operations.

### Starting Point: Token Embeddings

Say we have a sentence with 4 tokens, each represented as a 512-dimensional vector:

```
Input X: shape (4, 512)
         ┌─────────────────────────┐
Token 1  │  [0.2, -0.1, ..., 0.5]  │  ← 512 dimensions
Token 2  │  [0.8,  0.3, ..., 0.1]  │
Token 3  │  [-0.4, 0.7, ..., 0.2]  │
Token 4  │  [0.1, -0.5, ..., 0.9]  │
         └─────────────────────────┘
```

### The Three Projection Matrices

We have three **learned** weight matrices:

```
W_Q: shape (512, 64)   # Projects to query space
W_K: shape (512, 64)   # Projects to key space
W_V: shape (512, 64)   # Projects to value space
```

(64 is `d_k`, the dimension of each attention head)

### Computing Q, K, V

Simple matrix multiplication:

```python
Q = X @ W_Q   # (4, 512) @ (512, 64) → (4, 64)
K = X @ W_K   # (4, 512) @ (512, 64) → (4, 64)
V = X @ W_V   # (4, 512) @ (512, 64) → (4, 64)
```

Now each token has:
- A 64-dim **query vector**: "what am I looking for?"
- A 64-dim **key vector**: "what do I contain?"
- A 64-dim **value vector**: "what information do I provide?"

```
Q: shape (4, 64)           K: shape (4, 64)           V: shape (4, 64)
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  q₁ (64 dims)    │       │  k₁ (64 dims)    │       │  v₁ (64 dims)    │
│  q₂ (64 dims)    │       │  k₂ (64 dims)    │       │  v₂ (64 dims)    │
│  q₃ (64 dims)    │       │  k₃ (64 dims)    │       │  v₃ (64 dims)    │
│  q₄ (64 dims)    │       │  k₄ (64 dims)    │       │  v₄ (64 dims)    │
└──────────────────┘       └──────────────────┘       └──────────────────┘
"What do I need?"          "What do I offer?"         "My actual content"
```

### Step 1: Compute All Pairwise Similarities

```python
scores = Q @ K.T   # (4, 64) @ (64, 4) → (4, 4)
```

This gives us a **4×4 attention score matrix**:

```
scores: shape (4, 4)

              Key₁   Key₂   Key₃   Key₄
           ┌─────────────────────────────┐
Query₁     │  q₁·k₁  q₁·k₂  q₁·k₃  q₁·k₄ │  ← How much should Token 1 attend to each?
Query₂     │  q₂·k₁  q₂·k₂  q₂·k₃  q₂·k₄ │
Query₃     │  q₃·k₁  q₃·k₂  q₃·k₃  q₃·k₄ │
Query₄     │  q₄·k₁  q₄·k₂  q₄·k₃  q₄·k₄ │
           └─────────────────────────────┘

Each entry (i, j) = "How relevant is token j to token i's query?"
```

### Step 2: Scale and Softmax

```python
scores = scores / sqrt(64)        # Scale to prevent extreme values
weights = softmax(scores, dim=-1) # Each ROW sums to 1
```

```
weights: shape (4, 4)

              Token₁  Token₂  Token₃  Token₄
           ┌────────────────────────────────┐
Token 1    │  0.1     0.6     0.2     0.1   │  → sums to 1.0
Token 2    │  0.3     0.3     0.3     0.1   │  → sums to 1.0
Token 3    │  0.05    0.8     0.1     0.05  │  → sums to 1.0
Token 4    │  0.2     0.2     0.2     0.4   │  → sums to 1.0
           └────────────────────────────────┘
```

> **Why scale by √d_k?** When `d_k` is large (e.g., 64), dot products can become very large, pushing softmax into regions with tiny gradients. Dividing by √d_k keeps values in a reasonable range for stable training.

### Step 3: Weighted Sum of Values

```python
output = weights @ V   # (4, 4) @ (4, 64) → (4, 64)
```

For each token, we compute a weighted combination of ALL value vectors:

```
output[0] = 0.1*v₁ + 0.6*v₂ + 0.2*v₃ + 0.1*v₄   ← Token 1's new representation
output[1] = 0.3*v₁ + 0.3*v₂ + 0.3*v₃ + 0.1*v₄   ← Token 2's new representation
...
```

### The Full Computation Flow

```
X (4, 512)
    │
    ├──→ × W_Q (512, 64) ──→ Q (4, 64) ──┐
    │                                     │
    ├──→ × W_K (512, 64) ──→ K (4, 64) ──┼──→ Q @ K.T ──→ scores (4, 4)
    │                                     │                    │
    └──→ × W_V (512, 64) ──→ V (4, 64) ──┘              scale & softmax
                                  │                           │
                                  │                    weights (4, 4)
                                  │                           │
                                  └───────── weights @ V ─────┘
                                                   │
                                            output (4, 64)
```

### Why This Matrix Design?

1. **Asymmetric matching**: `Q @ K.T` lets token i's query match against token j's key—the matching function is learned (via W_Q and W_K), not just raw cosine similarity.

2. **Separate retrieval**: V is separate from K, so what you match on ≠ what you retrieve. A token can advertise "I'm a noun" (key) while providing rich semantic content (value).

3. **Parallelism**: All Q, K, V computations are independent matrix multiplies—massively parallel on GPUs.

## Multi-Head Attention: Why One Head Isn't Enough

Single-head attention has a limitation: each token can only create **one query**. But language has multiple simultaneous relationships!

Consider: "The **cat** that **I** saw yesterday **sat** on the mat"

The word "sat" needs to attend to multiple things for different reasons:
- **Syntactic**: "cat" is the subject (who is sitting?)
- **Semantic**: "mat" is the location (sitting where?)
- **Temporal**: "yesterday" provides time context

One attention head can only compute one weighted average. It might focus on the subject and lose the location, or vice versa.

### The Multi-Head Solution

Instead of one set of Q, K, V projections, we use **multiple heads in parallel**:

```
              ┌─── Head 1: W_Q1, W_K1, W_V1 ─── attention₁ ───┐
              │                                                │
X (4, 512) ───┼─── Head 2: W_Q2, W_K2, W_V2 ─── attention₂ ───┼─── Concat ─── W_O ─── output
              │                                                │
              └─── Head 8: W_Q8, W_K8, W_V8 ─── attention₈ ───┘
```

Each head has its own learned projections, so each can specialize:
- **Head 1** might learn to find syntactic subjects
- **Head 2** might learn to find locations/objects
- **Head 3** might learn to track coreference ("it" → "cat")
- **Head 4** might learn positional patterns (adjacent words)

### Matrix Dimensions with Multi-Head

With 8 heads and `d_model = 512`:

```
d_k = d_v = d_model / num_heads = 512 / 8 = 64

Per head:
  W_Q: (512, 64), W_K: (512, 64), W_V: (512, 64)

Each head produces: output_h with shape (4, 64)

Concatenate all heads: (4, 64×8) = (4, 512)

Final projection W_O: (512, 512) → output (4, 512)
```

The total parameters are similar to a single large head, but we get **8 different attention patterns**.

### What Heads Actually Learn

Research has shown that different heads learn interpretable patterns:

| Head | Learned Pattern |
|------|-----------------|
| Head 3, Layer 2 | Subject-verb agreement |
| Head 7, Layer 5 | Coreference resolution |
| Head 1, Layer 1 | Previous token attention |
| Head 4, Layer 3 | Punctuation/structure |

This specialization emerges naturally from training—we don't program it explicitly.

### The Key Insight

Multi-head attention lets the model ask **multiple different questions simultaneously**:

```
Token "sat" in "The cat that I saw yesterday sat on the mat"

Head 1 query: "Who is doing the action?"        → attends to "cat"
Head 2 query: "Where is this happening?"        → attends to "mat"
Head 3 query: "What's the grammatical context?" → attends to "The", structure
Head 4 query: "What came just before me?"       → attends to "yesterday"
```

The final output combines all these perspectives into a rich representation.

## What's Next

Now that you understand *what* attention does intuitively, let's look at the exact mathematical formulation in `02_scaled_dot_product.md`.
