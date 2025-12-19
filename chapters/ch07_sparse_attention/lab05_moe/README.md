# Lab 05: Mixture of Experts (MoE)

## Objective

Implement a Mixture of Experts layer with routing, enabling sparse feed-forward computation.

## What You'll Build

A `MixtureOfExperts` class that:
1. Routes each token to a subset of expert networks
2. Computes weighted expert outputs
3. Implements load balancing to prevent expert collapse
4. Achieves more model capacity without proportional compute increase

## Prerequisites

- Complete Labs 01-04
- Read `../docs/05_mixture_of_experts.md`

## Why MoE?

The feed-forward network (FFN) dominates transformer compute (~75%). MoE makes it sparse:

```
Standard FFN:
  Every token → Same FFN (all parameters active)

MoE with 8 experts, top-2 routing:
  Token → Router → Select 2 experts → Weighted output
  Only 2/8 = 25% of FFN parameters active per token
  But 8x more total parameters for model capacity!
```

Used in: Mixtral 8x7B, Switch Transformer, DeepSeek-MoE

## Instructions

1. Open `src/moe.py`
2. Implement the `MixtureOfExperts` class and helper functions
3. Run tests: `uv run pytest tests/`

## The MoE Architecture

```
Input X: (batch, seq_len, d_model)
                │
                ▼
┌───────────────────────────────────┐
│           Router                   │
│   logits = X @ W_router           │
│   probs = softmax(logits)         │
│   top_k_idx, top_k_probs = top_k  │
└───────────────────────────────────┘
                │
        ┌───────┼───────┬───────┐
        ▼       ▼       ▼       ▼
    ┌───────┐ ┌───────┐ ... ┌───────┐
    │Expert │ │Expert │     │Expert │
    │  0    │ │  1    │     │  N-1  │
    └───────┘ └───────┘     └───────┘
        │       │               │
        └───────┴───────┬───────┘
                        ▼
              Weighted Sum (by router probs)
                        │
                        ▼
                    Output
```

## Classes to Implement

### `Expert`

Simple feed-forward network (one expert):

```python
class Expert:
    def __init__(self, d_model: int, d_ff: int):
        """Single expert FFN: Linear -> Activation -> Linear"""
```

### `Router`

Determines which experts process each token:

```python
class Router:
    def __init__(self, d_model: int, num_experts: int):
        """Compute routing probabilities."""

    def forward(self, x: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            top_k_indices: Which experts to use
            top_k_weights: Normalized weights for each expert
        """
```

### `MixtureOfExperts`

Full MoE layer:

```python
class MixtureOfExperts:
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_coef: float = 0.01
    ):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension of each expert
            num_experts: Total number of experts
            top_k: Number of experts per token
            aux_loss_coef: Weight for load balancing loss
        """
```

## Functions to Implement

### `compute_aux_loss(router_probs, expert_assignments, num_experts)`

Compute the auxiliary load balancing loss:

```python
def compute_aux_loss(router_probs, expert_assignments, num_experts):
    """
    Encourage uniform expert usage to prevent collapse.

    Loss = num_experts * Σ(fraction_tokens_i * mean_prob_i)

    Where:
    - fraction_tokens_i: fraction of tokens assigned to expert i
    - mean_prob_i: mean routing probability for expert i
    """
```

### `compute_expert_usage(expert_assignments, num_experts)`

Compute how many tokens went to each expert.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_moe.py::TestRouter

# Run with verbose output
uv run pytest tests/ -v
```

## The Load Balancing Problem

Without intervention, routers collapse to using few experts:

```
Iteration 1: Expert 0 used 80%, others 20%
Iteration 2: Expert 0 improves more (more gradients)
...
Final: Expert 0 used 99%, others unused
```

The auxiliary loss prevents this by penalizing imbalanced routing.

## Hints

- Router output is softmax over experts, then select top-k
- Normalize top-k weights to sum to 1
- Each token's output = weighted sum of its selected experts' outputs
- The auxiliary loss encourages: `usage[i] ≈ 1/num_experts` for all i

## Expected Behavior

```python
moe = MixtureOfExperts(
    d_model=512,
    d_ff=2048,
    num_experts=8,
    top_k=2
)

x = np.random.randn(2, 10, 512)  # batch=2, seq=10
output = moe(x)
# Each token routed to 2 of 8 experts
# Output shape: (2, 10, 512)

# Check expert usage
usage = moe.get_expert_usage()
# Ideally close to [0.125, 0.125, ..., 0.125] (uniform)
```

## Complexity Analysis

```
Standard FFN:
  Compute: O(seq × d_model × d_ff)
  Parameters: d_model × d_ff × 2

MoE with N experts, top-k routing:
  Compute: O(seq × d_model × d_ff × k)  # k << N
  Parameters: d_model × d_ff × 2 × N     # N× more params

Example (Mixtral 8x7B):
  - 8 experts, top-2 routing
  - 8× parameters, ~2× compute vs single expert
  - Quality approaches 8× larger dense model
```

## Milestone

**Chapter 7 Final Milestone**: Your MoE implementation should:
1. Route tokens to top-k experts correctly
2. Produce valid outputs (weighted sum of expert outputs)
3. Track expert usage statistics
4. Compute auxiliary load balancing loss

## Verification

All tests pass = you've completed Chapter 7!

You now understand:
- Sparse attention patterns (local, strided, block, global)
- Sliding window attention (Longformer, Mistral)
- KV compression (GQA, low-rank)
- Multi-head Latent Attention (DeepSeek MLA)
- Mixture of Experts (Mixtral-style)

These techniques power efficient long-context and large-scale language models.
