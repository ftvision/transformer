# Lab 04: Load Pretrained Weights

## Objective

Load GPT-2 weights from HuggingFace into your custom implementation and verify outputs match.

## What You'll Build

Functions to:
1. Load GPT-2 weights from HuggingFace
2. Map HuggingFace weight names to your implementation
3. Handle weight transformations (transpose, split)
4. Verify your implementation produces identical outputs

## Prerequisites

- Complete Lab 03 (decoder transformer)
- Read `../docs/04_pretrained_models.md`

You'll need to install:
```bash
pip install transformers torch
```

## Why This Matters

Loading pretrained weights:
- Validates your architecture implementation
- Catches subtle bugs (wrong transpose, incorrect shapes)
- Lets you use powerful pretrained models
- Is a key skill for production ML

## Instructions

1. Open `src/pretrained.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## The Challenge: Weight Mapping

HuggingFace GPT-2 uses different naming conventions:

```
HuggingFace                          Your Implementation
-----------                          -------------------
transformer.wte.weight          →    token_embedding.weight
transformer.wpe.weight          →    pos_embedding.weight
transformer.h.0.ln_1.weight     →    blocks[0].ln1.gamma
transformer.h.0.ln_1.bias       →    blocks[0].ln1.beta
transformer.h.0.attn.c_attn.weight → blocks[0].attn.W_Q, W_K, W_V (combined!)
transformer.h.0.attn.c_proj.weight → blocks[0].attn.W_O
transformer.h.0.ln_2.weight     →    blocks[0].ln2.gamma
transformer.h.0.mlp.c_fc.weight →    blocks[0].ffn.W1
transformer.h.0.mlp.c_proj.weight →  blocks[0].ffn.W2
...
transformer.ln_f.weight         →    ln_f.gamma
transformer.ln_f.bias           →    ln_f.beta
```

## Key Challenges

### 1. Combined QKV Projection

GPT-2 combines Q, K, V into one matrix:
```python
# HuggingFace: c_attn is (d_model, 3*d_model)
c_attn = hf_state_dict['transformer.h.0.attn.c_attn.weight']

# Split into Q, K, V (each d_model x d_model)
W_Q = c_attn[:, :d_model]
W_K = c_attn[:, d_model:2*d_model]
W_V = c_attn[:, 2*d_model:]
```

### 2. Weight Transpose

HuggingFace uses Conv1D (which transposes weights):
```python
# HuggingFace: (in_features, out_features) for Conv1D
# Standard: (in_features, out_features) for Linear

# For c_attn: (d_model, 3*d_model) - NO transpose needed
# For c_proj: (d_model, d_model) - NO transpose needed
# etc.
```

### 3. Layer Norm Naming

```python
# HuggingFace uses weight/bias
ln_weight = hf_state_dict['transformer.ln_f.weight']
ln_bias = hf_state_dict['transformer.ln_f.bias']

# Your implementation might use gamma/beta
your_model.ln_f.gamma = ln_weight
your_model.ln_f.beta = ln_bias
```

## Functions to Implement

### `load_gpt2_weights(your_model, model_name='gpt2')`

Load GPT-2 weights from HuggingFace into your model.

```python
def load_gpt2_weights(your_model, model_name='gpt2'):
    """
    Load pretrained GPT-2 weights into your custom model.

    Args:
        your_model: Your GPTModel instance
        model_name: HuggingFace model name ('gpt2', 'gpt2-medium', etc.)
    """
```

### `compare_outputs(your_model, hf_model, tokenizer, text)`

Compare outputs of your model vs HuggingFace model.

```python
def compare_outputs(your_model, hf_model, tokenizer, text):
    """
    Compare outputs and return max difference.

    Returns:
        Tuple of (match: bool, max_diff: float)
    """
```

### `get_weight_mapping(num_layers)`

Get mapping from HuggingFace names to your implementation.

```python
def get_weight_mapping(num_layers):
    """
    Returns dict mapping HuggingFace keys to (your_key, transform_fn).
    """
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_pretrained.py::TestLoadWeights
```

## Example Usage

```python
from decoder import GPTModel
from pretrained import load_gpt2_weights, compare_outputs
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Create your model with GPT-2 config
your_model = GPTModel(
    vocab_size=50257,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    max_seq_len=1024
)

# Load pretrained weights
load_gpt2_weights(your_model, 'gpt2')

# Load HuggingFace model for comparison
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Compare outputs
text = "Hello, my name is"
match, max_diff = compare_outputs(your_model, hf_model, tokenizer, text)
print(f"Match: {match}, Max diff: {max_diff:.2e}")
```

## GPT-2 Configurations

| Model | Layers | d_model | Heads | d_ff | Vocab |
|-------|--------|---------|-------|------|-------|
| gpt2 | 12 | 768 | 12 | 3072 | 50257 |
| gpt2-medium | 24 | 1024 | 16 | 4096 | 50257 |
| gpt2-large | 36 | 1280 | 20 | 5120 | 50257 |
| gpt2-xl | 48 | 1600 | 25 | 6400 | 50257 |

## Debugging Tips

### 1. Check Weight Shapes

```python
for name, param in hf_model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 2. Compare Layer by Layer

```python
# After loading weights, compare intermediate outputs
your_emb = your_model.token_embedding(tokens)
hf_emb = hf_model.transformer.wte(torch.tensor(tokens))
print(f"Embedding diff: {np.abs(your_emb - hf_emb.numpy()).max()}")
```

### 3. Check Individual Weights

```python
# Verify a specific weight was loaded correctly
hf_weight = hf_model.transformer.h[0].ln_1.weight.detach().numpy()
your_weight = your_model.blocks[0].ln1.gamma
print(f"LN1 weight match: {np.allclose(hf_weight, your_weight)}")
```

## Common Pitfalls

1. **Forgetting tied weights**: GPT-2 ties input and output embeddings
2. **Wrong transpose**: Conv1D stores weights differently than Linear
3. **Missing biases**: Some layers have biases, some don't
4. **Float precision**: Use float32, compare with atol=1e-5

## Verification

All tests pass = your implementation matches GPT-2 exactly!

**Chapter 3 Milestone:**
> Your implementation generates same logits as HuggingFace GPT-2 for same input.

Congratulations on completing Chapter 3!
