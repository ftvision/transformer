"""
Lab 04: Load Pretrained Weights

Load GPT-2 weights from HuggingFace into your custom implementation.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any

# Import PyTorch and HuggingFace
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Import your implementation from Lab 03
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lab03_decoder_transformer" / "src"))

# Students should have completed Lab 03 first
# from decoder import GPTModel, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock


# Stub GPTModel for Lab 04 (students should use their Lab 03 implementation)
class GPTModel:
    """
    GPT Model stub for Lab 04.

    Students should either:
    1. Import their implementation from Lab 03
    2. Or ensure Lab 03 is complete and importable

    This stub provides the expected interface.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        tie_weights: bool = True
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        # Token and position embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = LayerNorm(d_model)

        # Output projection (tied with token embedding)
        if tie_weights:
            self.lm_head = OutputProjection(d_model, vocab_size, self.token_embedding.weight)
        else:
            self.lm_head = OutputProjection(d_model, vocab_size)

    def forward(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement forward or copy from Lab 03")

    def __call__(self, token_ids: np.ndarray, training: bool = False) -> np.ndarray:
        return self.forward(token_ids, training)


# Supporting classes (stubs - students should have these from earlier labs)
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    def forward(self, token_ids):
        return self.weight[token_ids]

    def __call__(self, token_ids):
        return self.forward(token_ids)


class PositionalEmbedding:
    def __init__(self, max_seq_len, d_model):
        self.weight = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02
        self.max_seq_len = max_seq_len

    def forward(self, seq_len):
        return self.weight[:seq_len]

    def __call__(self, seq_len):
        return self.forward(seq_len)


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

    def __call__(self, x):
        return self.forward(x)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout

        # Weight matrices
        self.W_Q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_K = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_V = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_O = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        # Biases
        self.b_Q = np.zeros(d_model, dtype=np.float32)
        self.b_K = np.zeros(d_model, dtype=np.float32)
        self.b_V = np.zeros(d_model, dtype=np.float32)
        self.b_O = np.zeros(d_model, dtype=np.float32)

    def forward(self, x, mask=None, training=False):
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement forward")

    def __call__(self, x, mask=None, training=False):
        return self.forward(x, mask, training)


class FeedForward:
    def __init__(self, d_model, d_ff, dropout=0.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout

        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x, training=False):
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement forward")

    def __call__(self, x, training=False):
        return self.forward(x, training)


class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout_rate = dropout

    def forward(self, x, mask=None, training=False):
        # YOUR CODE HERE (or copy from Lab 03)
        raise NotImplementedError("Implement forward")

    def __call__(self, x, mask=None, training=False):
        return self.forward(x, mask, training)


class OutputProjection:
    def __init__(self, d_model, vocab_size, weight=None):
        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    def forward(self, x):
        return x @ self.weight.T

    def __call__(self, x):
        return self.forward(x)


# =============================================================================
# Functions to implement
# =============================================================================

def get_gpt2_config(model_name: str = 'gpt2') -> Dict[str, int]:
    """
    Get GPT-2 configuration for a given model name.

    Args:
        model_name: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

    Returns:
        Dictionary with: vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len
    """
    configs = {
        'gpt2': {
            'vocab_size': 50257,
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072,
            'max_seq_len': 1024,
        },
        'gpt2-medium': {
            'vocab_size': 50257,
            'd_model': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'd_ff': 4096,
            'max_seq_len': 1024,
        },
        'gpt2-large': {
            'vocab_size': 50257,
            'd_model': 1280,
            'num_layers': 36,
            'num_heads': 20,
            'd_ff': 5120,
            'max_seq_len': 1024,
        },
        'gpt2-xl': {
            'vocab_size': 50257,
            'd_model': 1600,
            'num_layers': 48,
            'num_heads': 25,
            'd_ff': 6400,
            'max_seq_len': 1024,
        },
    }
    return configs[model_name]


def create_gpt2_model(model_name: str = 'gpt2') -> GPTModel:
    """
    Create a GPTModel with GPT-2 configuration.

    Args:
        model_name: HuggingFace model name

    Returns:
        GPTModel instance with correct architecture
    """
    config = get_gpt2_config(model_name)
    return GPTModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0,
        tie_weights=True,
    )


def load_gpt2_weights(your_model: GPTModel, model_name: str = 'gpt2') -> None:
    """
    Load pretrained GPT-2 weights from HuggingFace into your model.

    This function downloads the GPT-2 weights and maps them to your
    custom implementation.

    Args:
        your_model: Your GPTModel instance (will be modified in-place)
        model_name: HuggingFace model name ('gpt2', 'gpt2-medium', etc.)

    Key mappings:
        transformer.wte.weight          → token_embedding.weight
        transformer.wpe.weight          → pos_embedding.weight
        transformer.h.{i}.ln_1.weight   → blocks[i].ln1.gamma
        transformer.h.{i}.ln_1.bias     → blocks[i].ln1.beta
        transformer.h.{i}.attn.c_attn.weight → blocks[i].attn.W_Q, W_K, W_V
        transformer.h.{i}.attn.c_attn.bias   → blocks[i].attn.b_Q, b_K, b_V
        transformer.h.{i}.attn.c_proj.weight → blocks[i].attn.W_O
        transformer.h.{i}.attn.c_proj.bias   → blocks[i].attn.b_O
        transformer.h.{i}.ln_2.weight   → blocks[i].ln2.gamma
        transformer.h.{i}.ln_2.bias     → blocks[i].ln2.beta
        transformer.h.{i}.mlp.c_fc.weight    → blocks[i].ffn.W1
        transformer.h.{i}.mlp.c_fc.bias      → blocks[i].ffn.b1
        transformer.h.{i}.mlp.c_proj.weight  → blocks[i].ffn.W2
        transformer.h.{i}.mlp.c_proj.bias    → blocks[i].ffn.b2
        transformer.ln_f.weight         → ln_f.gamma
        transformer.ln_f.bias           → ln_f.beta

    Note on Conv1D weights:
        GPT-2 uses Conv1D which stores weights as (in_features, out_features).
        This is the same as our convention, so NO transpose is needed.

    Note on c_attn:
        c_attn combines Q, K, V projections. Shape: (d_model, 3*d_model)
        Split along last dimension: [:, :d_model], [:, d_model:2*d_model], [:, 2*d_model:]

    Example:
        >>> your_model = create_gpt2_model('gpt2')
        >>> load_gpt2_weights(your_model, 'gpt2')
        >>> # your_model now has pretrained weights
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Load HuggingFace model: hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    # 2. Get state dict: hf_state_dict = hf_model.state_dict()
    # 3. Map and copy each weight:
    #    - Token embedding: transformer.wte.weight → token_embedding.weight
    #    - Position embedding: transformer.wpe.weight → pos_embedding.weight
    #    - For each layer i:
    #      - Layer norm 1: transformer.h.{i}.ln_1.weight/bias → blocks[i].ln1.gamma/beta
    #      - Attention: split c_attn into Q, K, V
    #      - Layer norm 2: transformer.h.{i}.ln_2.weight/bias → blocks[i].ln2.gamma/beta
    #      - FFN: mlp.c_fc → ffn.W1/b1, mlp.c_proj → ffn.W2/b2
    #    - Final layer norm: transformer.ln_f.weight/bias → ln_f.gamma/beta
    #
    # Important: Convert torch tensors to numpy with .detach().numpy()
    raise NotImplementedError("Implement load_gpt2_weights")


def compare_outputs(
    your_model: GPTModel,
    hf_model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    atol: float = 1e-4
) -> Tuple[bool, float]:
    """
    Compare outputs of your model vs HuggingFace model on the same input.

    Args:
        your_model: Your GPTModel with loaded weights
        hf_model: HuggingFace GPT2LMHeadModel
        tokenizer: HuggingFace tokenizer
        text: Input text to test
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (match: bool, max_diff: float)

    Example:
        >>> match, diff = compare_outputs(your_model, hf_model, tokenizer, "Hello world")
        >>> print(f"Match: {match}, Max diff: {diff:.2e}")
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Tokenize text: tokens = tokenizer.encode(text, return_tensors='pt')
    # 2. Get HuggingFace output (with torch.no_grad())
    # 3. Convert tokens to numpy for your model
    # 4. Get your model output
    # 5. Compare and return (match, max_diff)
    raise NotImplementedError("Implement compare_outputs")


def compare_intermediate_outputs(
    your_model: GPTModel,
    hf_model: GPT2LMHeadModel,
    token_ids: np.ndarray
) -> Dict[str, float]:
    """
    Compare intermediate outputs layer by layer for debugging.

    Useful when final outputs don't match - helps identify which
    layer has the bug.

    Args:
        your_model: Your GPTModel
        hf_model: HuggingFace model
        token_ids: Input token IDs (numpy array)

    Returns:
        Dictionary mapping layer name to max difference
        Example: {'embedding': 0.0001, 'block_0': 0.0002, ...}
    """
    # YOUR CODE HERE
    #
    # Compare at each stage:
    # 1. After token embedding
    # 2. After adding positional embedding
    # 3. After each transformer block
    # 4. After final layer norm
    # 5. After output projection
    raise NotImplementedError("Implement compare_intermediate_outputs")


def analyze_weight_differences(
    your_model: GPTModel,
    hf_model: GPT2LMHeadModel
) -> Dict[str, Dict[str, float]]:
    """
    Analyze differences between loaded weights and HuggingFace weights.

    Useful for verifying weight loading was correct.

    Args:
        your_model: Your GPTModel with loaded weights
        hf_model: HuggingFace model

    Returns:
        Nested dictionary with max_diff and mean_diff for each weight
        Example: {'token_embedding': {'max_diff': 0.0, 'mean_diff': 0.0}, ...}
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement analyze_weight_differences")


def generate_text(
    model: GPTModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0
) -> str:
    """
    Generate text using your model (bonus function).

    Uses simple greedy decoding (argmax) or sampling with temperature.

    Args:
        model: Your GPTModel with loaded weights
        tokenizer: HuggingFace tokenizer
        prompt: Starting text
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = normal, lower = more deterministic)

    Returns:
        Generated text including the prompt
    """
    # YOUR CODE HERE (optional bonus)
    #
    # Steps:
    # 1. Tokenize prompt
    # 2. For each position up to max_length:
    #    a. Forward pass to get logits
    #    b. Get logits for last position
    #    c. Apply temperature: logits = logits / temperature
    #    d. Convert to probabilities with softmax
    #    e. Sample next token (or argmax for greedy)
    #    f. Append to sequence
    # 3. Decode and return
    raise NotImplementedError("Implement generate_text")
