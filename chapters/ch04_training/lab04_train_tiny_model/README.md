# Lab 04: Train Tiny Model

## Objective

Put everything together to train a small transformer language model on the tiny_shakespeare dataset.

## What You'll Build

A complete training pipeline that:
1. Loads and tokenizes the tiny_shakespeare dataset
2. Creates a small GPT-style model
3. Trains it using your Lab 03 trainer
4. Generates coherent Shakespeare-like text

## Prerequisites

- Complete Lab 01 (loss functions)
- Complete Lab 02 (gradient visualization)
- Complete Lab 03 (training loop)
- Read all Chapter 4 docs

## The Milestone

> Train a ~1M parameter model that generates coherent Shakespeare-like text.

This is the capstone for Chapter 4. By the end, you'll have trained your first language model from scratch!

## Instructions

1. Open `src/train.py`
2. Implement the components marked with `# YOUR CODE HERE`
3. Run training: `uv run python src/train.py`
4. Run tests: `uv run pytest tests/`

## Components to Implement

### `CharTokenizer`
Simple character-level tokenizer for tiny_shakespeare.

```python
class CharTokenizer:
    def __init__(self, text: str):
        """Build vocabulary from text."""

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
```

### `TinyGPT`
A small GPT-style model.

```python
class TinyGPT:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256
    ):
        """Initialize tiny GPT model."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning logits."""

    def generate(self, prompt: np.ndarray, max_new_tokens: int) -> np.ndarray:
        """Generate text autoregressively."""
```

### `train_model`
Main training function.

```python
def train_model(
    model: TinyGPT,
    train_data: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    seq_len: int = 64,
    lr: float = 1e-3
) -> Dict[str, List[float]]:
    """Train the model and return training history."""
```

## The Dataset

**tiny_shakespeare**: ~1MB of Shakespeare's works concatenated together.

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

The dataset is small enough to fit in memory and trains in minutes on a laptop.

## Model Architecture

Our TinyGPT follows GPT-2 architecture:

```
Token Embeddings + Position Embeddings
            ↓
    ┌───────────────────┐
    │  Transformer      │ ×n_layers
    │  Block:           │
    │  - LayerNorm      │
    │  - Multi-Head Attn│
    │  - Residual       │
    │  - LayerNorm      │
    │  - FFN            │
    │  - Residual       │
    └───────────────────┘
            ↓
        LayerNorm
            ↓
        LM Head (to vocab)
```

Suggested configuration (~1M params):
- `d_model = 128`
- `n_heads = 4`
- `n_layers = 4`
- `vocab_size ≈ 65` (unique characters in Shakespeare)

## Training Parameters

Suggested starting point:
```python
config = {
    'epochs': 10,
    'batch_size': 64,
    'seq_len': 128,
    'lr': 1e-3,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
}
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run training
uv run python src/train.py

# Run with specific config
uv run python src/train.py --epochs 5 --batch_size 32
```

## Expected Results

After training:

1. **Loss curve**: Should decrease from ~4.0 (random) to ~1.5 or lower
2. **Sample generation**: Should produce Shakespeare-like text

Example output after training:
```
Prompt: "ROMEO:"

Generated:
ROMEO:
What say'st thou? dost thou hear them, my lord?
I have no more to say. I pray thee, stay.

JULIET:
Good night, good night! parting is such sweet sorrow,
That I shall say good night till it be morrow.
```

(Note: Your results will vary - this is a tiny model!)

## Evaluation Metrics

### Perplexity
- Random model: ~65 (vocab_size)
- After training: Should be ~10-20

### Sample Quality
Ask yourself:
- Does it look like English?
- Does it use Shakespeare-like words?
- Does it follow play formatting (character names, colons)?

## Tips

1. **Start small**: Train for 1 epoch first to verify everything works
2. **Monitor loss**: If loss doesn't decrease, check learning rate
3. **Check gradients**: Use Lab 02 tools to verify gradients flow
4. **Temperature sampling**: Use temperature ~0.8 for generation
5. **Seed prompts**: Try prompts like "HAMLET:", "To be or", "The king"

## Debugging

**Loss doesn't decrease:**
- Learning rate too high or too low
- Gradients vanishing or exploding
- Bug in forward pass

**Generated text is garbage:**
- Model hasn't trained enough
- Temperature too high
- Bug in generation loop

**Out of memory:**
- Reduce batch_size
- Reduce seq_len
- Reduce model size

## Verification

All tests pass + model generates coherent text = Chapter 4 complete!

Sample output should:
1. Have proper words (not random characters)
2. Show some Shakespeare-like patterns
3. Follow play script formatting

## What's Next

Congratulations on training your first language model!

In later chapters, you'll learn:
- Chapter 5-7: More efficient attention mechanisms
- Chapter 8-9: Inference optimization
- Chapter 10-12: Scaling up training
