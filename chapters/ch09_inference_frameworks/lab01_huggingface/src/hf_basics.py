"""
Lab 01: HuggingFace Basics

Master the HuggingFace Transformers library for model loading and generation.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load a causal language model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
        device: Device to load model on ("auto", "cpu", "cuda", "mps")

    Returns:
        model: The loaded model
        tokenizer: The loaded tokenizer

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2")
        >>> model.device
        device(type='cpu')  # or cuda if available

        >>> model, tokenizer = load_model_and_tokenizer("gpt2", device="cpu")
        >>> model.device
        device(type='cpu')

    Note:
        - Use AutoModelForCausalLM and AutoTokenizer
        - Set tokenizer.pad_token = tokenizer.eos_token if pad_token is None
        - Handle device_map="auto" for automatic device placement
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement load_model_and_tokenizer")


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> Union[str, List[str]]:
    """
    Generate text continuation from a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text to continue
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        do_sample: Whether to sample (True) or greedy decode (False)
        num_return_sequences: Number of sequences to generate

    Returns:
        Generated text (excluding the prompt)
        If num_return_sequences > 1, returns list of strings

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2")
        >>> text = generate_text(model, tokenizer, "Hello, world!", max_new_tokens=20)
        >>> isinstance(text, str)
        True

        >>> texts = generate_text(model, tokenizer, "Hello", num_return_sequences=3)
        >>> len(texts)
        3

    Note:
        - Tokenize input with return_tensors="pt"
        - Move inputs to model device
        - Use model.generate() with the specified parameters
        - Decode output, skipping special tokens
        - Return only the NEW tokens (exclude prompt)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_text")


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 50,
    **kwargs
) -> List[str]:
    """
    Generate text for multiple prompts efficiently in a single batch.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate per prompt
        **kwargs: Additional generation parameters

    Returns:
        List of generated texts (one per prompt, excluding prompts)

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2")
        >>> prompts = ["Hello,", "The weather is", "AI will"]
        >>> results = generate_batch(model, tokenizer, prompts)
        >>> len(results)
        3
        >>> all(isinstance(r, str) for r in results)
        True

    Note:
        - Set tokenizer.padding_side = 'left' for batched generation
        - Use tokenizer with padding=True, return_tensors="pt"
        - Pass attention_mask to model.generate()
        - Decode each output separately
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_batch")


def calculate_perplexity(
    model: Any,
    tokenizer: Any,
    text: str
) -> float:
    """
    Calculate the perplexity of a text given a model.

    Perplexity = exp(average negative log likelihood)

    Lower perplexity = model assigns higher probability to the text.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to evaluate

    Returns:
        Perplexity score (float)

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2")
        >>> ppl = calculate_perplexity(model, tokenizer, "The quick brown fox")
        >>> isinstance(ppl, float)
        True
        >>> ppl > 0
        True

        >>> # Common text should have lower perplexity than gibberish
        >>> ppl_normal = calculate_perplexity(model, tokenizer, "Hello, how are you?")
        >>> ppl_gibberish = calculate_perplexity(model, tokenizer, "xyzzy plugh foo")
        >>> ppl_normal < ppl_gibberish  # Usually true for trained models
        True

    Note:
        - Tokenize text
        - Get model outputs (logits)
        - Calculate cross-entropy loss between predictions and targets
        - Perplexity = exp(loss)
        - Use torch.nn.functional.cross_entropy
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_perplexity")


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Extract information about a model's architecture.

    Args:
        model: The language model

    Returns:
        Dictionary with model information:
            - num_parameters: Total number of parameters
            - num_layers: Number of transformer layers
            - hidden_size: Hidden dimension size
            - vocab_size: Vocabulary size
            - model_type: Model architecture type (e.g., "gpt2", "llama")
            - num_attention_heads: Number of attention heads

    Examples:
        >>> model, _ = load_model_and_tokenizer("gpt2")
        >>> info = get_model_info(model)
        >>> info['model_type']
        'gpt2'
        >>> info['num_layers']
        12
        >>> info['hidden_size']
        768
        >>> info['vocab_size']
        50257

    Note:
        - Access model.config for architecture details
        - Count parameters with model.num_parameters() or sum of p.numel()
        - Different models may use different config attribute names
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_model_info")


def setup_for_inference(model: Any) -> Any:
    """
    Prepare a model for efficient inference.

    Args:
        model: The language model

    Returns:
        The model configured for inference

    This function should:
        - Set model to evaluation mode (model.eval())
        - Disable gradient computation context (torch.no_grad recommended in calling code)
        - Enable any available inference optimizations

    Examples:
        >>> model, _ = load_model_and_tokenizer("gpt2")
        >>> model = setup_for_inference(model)
        >>> model.training
        False

    Note:
        - model.eval() disables dropout
        - For production, consider torch.compile() on PyTorch 2.0+
        - Return the model (for chaining)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement setup_for_inference")


def simple_chat(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 100
) -> str:
    """
    Simple chat interface for instruction-tuned models.

    Args:
        model: The language model (should be instruction-tuned)
        tokenizer: The tokenizer
        messages: List of message dicts with "role" and "content"
                  Roles: "user", "assistant"
        system_prompt: Optional system prompt to prepend
        max_new_tokens: Maximum tokens to generate

    Returns:
        The assistant's response

    Examples:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2")  # Demo only
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = simple_chat(model, tokenizer, messages)
        >>> isinstance(response, str)
        True

    Note:
        - Use tokenizer.apply_chat_template if available
        - Otherwise, format messages manually
        - Handle models without chat templates gracefully
        - Generate and extract just the assistant's response
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement simple_chat")


def count_tokens(tokenizer: Any, text: str) -> int:
    """
    Count the number of tokens in a text.

    Args:
        tokenizer: The tokenizer
        text: Text to tokenize

    Returns:
        Number of tokens

    Examples:
        >>> _, tokenizer = load_model_and_tokenizer("gpt2")
        >>> count_tokens(tokenizer, "Hello, world!")
        4  # Approximate, depends on tokenizer
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_tokens")


def truncate_to_max_length(
    tokenizer: Any,
    text: str,
    max_length: int
) -> str:
    """
    Truncate text to fit within a maximum token length.

    Args:
        tokenizer: The tokenizer
        text: Text to truncate
        max_length: Maximum number of tokens

    Returns:
        Truncated text (decoded back to string)

    Examples:
        >>> _, tokenizer = load_model_and_tokenizer("gpt2")
        >>> long_text = "word " * 1000
        >>> short_text = truncate_to_max_length(tokenizer, long_text, 50)
        >>> count_tokens(tokenizer, short_text) <= 50
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement truncate_to_max_length")
