# Lab 01: HuggingFace Basics

## Objective

Master the HuggingFace Transformers library for model loading, text generation, and basic fine-tuning.

## What You'll Build

Functions and utilities for:
- Loading models and tokenizers
- Text generation with various parameters
- Batched inference
- Simple fine-tuning setup

## Prerequisites

Read these docs first:
- `../docs/05_framework_comparison.md` (HuggingFace section)

## Instructions

1. Open `src/hf_basics.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Functions to Implement

### `load_model_and_tokenizer(model_name, device="auto")`
Load a causal LM model and its tokenizer from HuggingFace.
- Handle device placement (CPU, CUDA, MPS)
- Set up padding token if not present
- Return model and tokenizer

### `generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_p=1.0, do_sample=True)`
Generate text from a prompt.
- Tokenize the prompt
- Generate with the specified parameters
- Decode and return the generated text (excluding prompt)

### `generate_batch(model, tokenizer, prompts, max_new_tokens=50, **kwargs)`
Generate text for multiple prompts efficiently.
- Pad inputs properly
- Handle attention masks
- Return list of generated texts

### `calculate_perplexity(model, tokenizer, text)`
Calculate the perplexity of a text.
- Tokenize the text
- Get model logits
- Calculate cross-entropy loss
- Return perplexity (exp of loss)

### `get_model_info(model)`
Extract information about a model.

Returns dict with:
- `num_parameters`: Total parameters
- `num_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension
- `vocab_size`: Vocabulary size
- `model_type`: Architecture type

### `setup_for_inference(model)`
Prepare a model for efficient inference.
- Set eval mode
- Disable gradient computation
- Enable inference optimizations if available

### `simple_chat(model, tokenizer, messages, system_prompt=None)`
Simple chat interface (for instruction-tuned models).
- Format messages properly
- Handle system prompts
- Generate response

## Testing

```bash
# Run all tests (uses small models for speed)
uv run pytest tests/

# Run specific test
uv run pytest tests/test_hf_basics.py::TestLoadModel

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from hf_basics import (
    load_model_and_tokenizer,
    generate_text,
    generate_batch,
    calculate_perplexity
)

# Load a small model for testing
model, tokenizer = load_model_and_tokenizer("gpt2")

# Generate text
text = generate_text(model, tokenizer, "The meaning of life is", max_new_tokens=50)
print(text)

# Batch generation
prompts = ["Hello,", "The weather is", "In the future,"]
results = generate_batch(model, tokenizer, prompts, max_new_tokens=30)
for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result}")

# Calculate perplexity
ppl = calculate_perplexity(model, tokenizer, "The quick brown fox jumps over the lazy dog.")
print(f"Perplexity: {ppl:.2f}")
```

## Hints

- Use `model.config` to access model configuration
- Set `tokenizer.pad_token = tokenizer.eos_token` if no pad token
- Use `padding_side='left'` for generation with batching
- `torch.no_grad()` for inference
- `model.eval()` disables dropout

## Key Concepts

1. **Tokenization**: Converting text to token IDs
2. **Attention Mask**: Indicating which tokens to attend to
3. **Generation Parameters**: temperature, top_p, top_k, etc.
4. **Perplexity**: Measure of how well model predicts text (lower = better)

## Verification

All tests pass = you can use HuggingFace for inference!

The milestone for this lab is successfully loading a model and generating coherent text.
