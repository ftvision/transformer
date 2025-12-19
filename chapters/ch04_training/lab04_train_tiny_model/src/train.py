"""
Lab 04: Train Tiny Model

Put everything together to train a small transformer on tiny_shakespeare.

Your task: Complete the classes and functions below to make all tests pass.
Run training: uv run python src/train.py
Run tests: uv run pytest tests/
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
import sys
from pathlib import Path

# Import from previous labs (or implement here)
# from trainer import AdamW, LRScheduler, Trainer


class CharTokenizer:
    """
    Simple character-level tokenizer.

    For tiny_shakespeare, we use character-level tokenization:
    - Each unique character gets an ID
    - Vocabulary is typically ~65 characters

    This is simpler than BPE/WordPiece but works well for small datasets.
    """

    def __init__(self, text: str):
        """
        Build vocabulary from text.

        Args:
            text: Training text to build vocabulary from

        Attributes after init:
            vocab: List of unique characters
            char_to_id: Dict mapping character to ID
            id_to_char: Dict mapping ID to character
            vocab_size: Number of unique characters

        Example:
            >>> tokenizer = CharTokenizer("hello world")
            >>> tokenizer.vocab_size
            8  # 'h', 'e', 'l', 'o', ' ', 'w', 'r', 'd'
        """
        # YOUR CODE HERE
        # 1. Get sorted list of unique characters
        # 2. Create char_to_id and id_to_char mappings
        # 3. Store vocab_size
        raise NotImplementedError("Implement CharTokenizer.__init__")

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: String to encode

        Returns:
            List of integer token IDs

        Example:
            >>> tokenizer = CharTokenizer("hello")
            >>> tokenizer.encode("hello")
            [0, 1, 2, 2, 3]  # IDs depend on vocab ordering
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement CharTokenizer.encode")

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string

        Example:
            >>> tokenizer = CharTokenizer("hello")
            >>> ids = tokenizer.encode("hello")
            >>> tokenizer.decode(ids)
            'hello'
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement CharTokenizer.decode")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class TinyGPT:
    """
    A tiny GPT-style language model.

    Architecture:
    - Token embeddings + positional embeddings
    - N transformer blocks (attention + FFN with residuals)
    - Final layer norm + linear head

    This is a simplified version for learning purposes.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0  # Ignored in numpy implementation
    ):
        """
        Initialize TinyGPT model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout probability (not used in numpy)

        Example:
            >>> model = TinyGPT(vocab_size=65, d_model=128, n_heads=4, n_layers=4)
            >>> x = np.array([[1, 2, 3, 4]])  # (batch=1, seq_len=4)
            >>> logits = model.forward(x)
            >>> logits.shape
            (1, 4, 65)
        """
        # YOUR CODE HERE
        # Initialize:
        # 1. token_embedding: (vocab_size, d_model)
        # 2. position_embedding: (max_seq_len, d_model)
        # 3. transformer_blocks: list of block parameters
        #    Each block needs: attention weights, FFN weights, layer norms
        # 4. final_ln: layer norm parameters
        # 5. lm_head: (d_model, vocab_size)
        raise NotImplementedError("Implement TinyGPT.__init__")

    def parameters(self) -> List[np.ndarray]:
        """
        Return list of all trainable parameters.

        Returns:
            List of numpy arrays (all weight matrices)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TinyGPT.parameters")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the model.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)

        Steps:
            1. Token embedding lookup
            2. Add positional embeddings
            3. Pass through transformer blocks
            4. Final layer norm
            5. Project to vocabulary

        Example:
            >>> model = TinyGPT(vocab_size=65)
            >>> x = np.array([[1, 2, 3]])
            >>> logits = model.forward(x)
            >>> logits.shape
            (1, 3, 65)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TinyGPT.forward")

    def generate(
        self,
        prompt: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively.

        Args:
            prompt: Starting token IDs of shape (seq_len,) or (1, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens

        Returns:
            Generated token IDs including prompt

        Generation loop:
        1. Forward pass to get logits for last position
        2. Apply temperature scaling
        3. Optionally apply top-k filtering
        4. Sample from distribution
        5. Append to sequence
        6. Repeat

        Example:
            >>> model = TinyGPT(vocab_size=65)
            >>> prompt = np.array([1, 2, 3])  # "abc" tokens
            >>> generated = model.generate(prompt, max_new_tokens=10)
            >>> len(generated)
            13  # 3 prompt + 10 new tokens
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement TinyGPT.generate")


def load_tiny_shakespeare(data_path: Optional[str] = None) -> str:
    """
    Load the tiny_shakespeare dataset.

    If no path provided, downloads or uses a small sample.

    Args:
        data_path: Path to data file (optional)

    Returns:
        String containing the dataset

    Note:
        For testing, returns a small sample if file not found.
    """
    # YOUR CODE HERE
    # If data_path exists, load from file
    # Otherwise, return a sample text for testing
    sample_text = """ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

ROMEO:
Shall I hear more, or shall I speak at this?

JULIET:
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet.
"""
    if data_path and Path(data_path).exists():
        with open(data_path, 'r') as f:
            return f.read()
    return sample_text


def create_dataset(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from text.

    For language modeling:
    - inputs: tokens[i:i+seq_len]
    - labels: tokens[i+1:i+seq_len+1]

    Args:
        text: Training text
        tokenizer: Tokenizer to encode text
        seq_len: Sequence length

    Returns:
        Tuple of (inputs, labels) arrays
        Each has shape (num_examples, seq_len)

    Example:
        >>> text = "hello world"
        >>> tokenizer = CharTokenizer(text)
        >>> inputs, labels = create_dataset(text, tokenizer, seq_len=4)
        >>> inputs.shape[1]
        4
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_dataset")


def compute_loss_and_gradients(
    model: TinyGPT,
    inputs: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, List[np.ndarray]]:
    """
    Compute loss and gradients for a batch.

    This is a simplified version that uses numerical gradients for learning.
    In practice, you'd use automatic differentiation (PyTorch, JAX).

    Args:
        model: TinyGPT model
        inputs: Input token IDs (batch_size, seq_len)
        labels: Target token IDs (batch_size, seq_len)

    Returns:
        Tuple of (loss, gradients)
        - loss: scalar cross-entropy loss
        - gradients: list of gradient arrays (same structure as model.parameters())

    Note:
        For this lab, we'll use numerical differentiation to keep things simple.
        This is slow but correct. Real training uses automatic differentiation.
    """
    # YOUR CODE HERE
    # 1. Forward pass to get logits
    # 2. Compute cross-entropy loss
    # 3. Compute gradients (numerically or analytically)
    raise NotImplementedError("Implement compute_loss_and_gradients")


def train_model(
    model: TinyGPT,
    train_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    log_interval: int = 10
) -> Dict[str, List[float]]:
    """
    Train the model and return training history.

    Args:
        model: TinyGPT model to train
        train_data: Tuple of (inputs, labels) arrays
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Peak learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay coefficient
        max_grad_norm: Maximum gradient norm for clipping
        log_interval: How often to print progress

    Returns:
        Dictionary containing training history:
        - 'loss': List of loss values
        - 'lr': List of learning rates
        - 'grad_norm': List of gradient norms

    Example:
        >>> model = TinyGPT(vocab_size=65)
        >>> inputs, labels = create_dataset(text, tokenizer, seq_len=64)
        >>> history = train_model(model, (inputs, labels), epochs=5)
        >>> history['loss'][-1] < history['loss'][0]  # Loss decreased
        True
    """
    # YOUR CODE HERE
    # 1. Setup optimizer and scheduler
    # 2. Training loop:
    #    - Sample batches
    #    - Forward pass
    #    - Compute loss and gradients
    #    - Clip gradients
    #    - Update parameters
    #    - Update learning rate
    #    - Log progress
    raise NotImplementedError("Implement train_model")


def evaluate_model(
    model: TinyGPT,
    eval_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate model on held-out data.

    Args:
        model: Trained model
        eval_data: Tuple of (inputs, labels) arrays
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with:
        - 'loss': Average loss
        - 'perplexity': exp(loss)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement evaluate_model")


def main():
    """Main training script."""
    print("=" * 60)
    print("Training TinyGPT on tiny_shakespeare")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    text = load_tiny_shakespeare()
    print(f"Data size: {len(text)} characters")

    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create dataset
    seq_len = 128
    inputs, labels = create_dataset(text, tokenizer, seq_len)
    print(f"Dataset: {len(inputs)} sequences of length {seq_len}")

    # Split into train/eval
    split_idx = int(len(inputs) * 0.9)
    train_data = (inputs[:split_idx], labels[:split_idx])
    eval_data = (inputs[split_idx:], labels[split_idx:])
    print(f"Train: {len(train_data[0])}, Eval: {len(eval_data[0])}")

    # Create model
    print("\nCreating model...")
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=seq_len
    )
    num_params = sum(p.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nTraining...")
    history = train_model(
        model,
        train_data,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        warmup_steps=100,
        log_interval=50
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, eval_data)
    print(f"Eval loss: {metrics['loss']:.4f}")
    print(f"Eval perplexity: {metrics['perplexity']:.2f}")

    # Generate samples
    print("\nGenerating samples...")
    prompts = ["ROMEO:", "To be", "The king"]

    for prompt in prompts:
        prompt_ids = np.array(tokenizer.encode(prompt))
        generated_ids = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)
        generated_text = tokenizer.decode(generated_ids.tolist())

        print(f"\nPrompt: {prompt}")
        print(f"Generated:\n{generated_text}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
