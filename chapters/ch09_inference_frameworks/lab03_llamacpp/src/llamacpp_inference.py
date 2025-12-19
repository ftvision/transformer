"""
Lab 03: llama.cpp Inference

Run efficient CPU inference with llama.cpp.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/

Note: Requires llama-cpp-python (pip install llama-cpp-python)
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkMetrics:
    """Metrics from benchmarking."""
    total_tokens: int
    total_time: float
    tokens_per_second: float
    time_to_first_token: float
    prompts_processed: int


def load_model(
    model_path: str,
    n_ctx: int = 2048,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
    **kwargs
) -> Any:
    """
    Load a GGUF model for inference.

    Args:
        model_path: Path to .gguf model file
        n_ctx: Context window size (max tokens)
        n_threads: Number of CPU threads (None for auto)
        n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
        verbose: Print loading information
        **kwargs: Additional llama_cpp.Llama arguments

    Returns:
        Loaded model instance

    Examples:
        >>> model = load_model("model.Q4_K_M.gguf", n_ctx=2048)
        >>> model = load_model("model.gguf", n_gpu_layers=-1)  # Full GPU

    Note:
        - Import llama_cpp.Llama
        - n_threads=None uses all available cores
        - n_gpu_layers=-1 offloads all layers to GPU
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement load_model")


def generate_text(
    model: Any,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    stop: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Generate text completion from a prompt.

    Args:
        model: Loaded llama.cpp model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        stop: Stop sequences
        **kwargs: Additional generation parameters

    Returns:
        Generated text (excluding prompt)

    Examples:
        >>> model = load_model("model.gguf")
        >>> text = generate_text(model, "Hello, world!", max_tokens=50)
        >>> isinstance(text, str)
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_text")


def generate_chat(
    model: Any,
    messages: List[Dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Generate chat completion.

    Args:
        model: Loaded llama.cpp model
        messages: List of message dicts with "role" and "content"
                  Roles: "system", "user", "assistant"
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional generation parameters

    Returns:
        Assistant's response

    Examples:
        >>> model = load_model("model.gguf")
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> response = generate_chat(model, messages)
        >>> isinstance(response, str)
        True

    Note:
        - Use model.create_chat_completion() if available
        - Otherwise, format messages manually and use generate_text
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_chat")


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Extract GGUF model metadata.

    Args:
        model_path: Path to .gguf model file

    Returns:
        Dictionary with model information:
            - architecture: Model architecture (llama, mistral, etc.)
            - quantization: Quantization type (Q4_K_M, etc.)
            - parameters: Estimated parameter count
            - context_length: Maximum context length
            - file_size_gb: Model file size in GB
            - vocab_size: Vocabulary size

    Examples:
        >>> info = get_model_info("llama-2-7b.Q4_K_M.gguf")
        >>> info['quantization']
        'Q4_K_M'
        >>> info['file_size_gb']
        4.0

    Note:
        - Parse GGUF header for metadata
        - File size can be obtained from os.path.getsize
        - Quantization often in filename
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_model_info")


def estimate_memory_usage(
    model_path: str,
    n_ctx: int = 2048,
    n_batch: int = 512
) -> Dict[str, float]:
    """
    Estimate memory requirements for loading a model.

    Args:
        model_path: Path to .gguf model file
        n_ctx: Context window size
        n_batch: Batch size for prompt processing

    Returns:
        Dictionary with memory estimates (in GB):
            - model_size: Size of model weights
            - kv_cache: Estimated KV-cache size
            - total_estimate: Total estimated RAM needed

    Examples:
        >>> mem = estimate_memory_usage("model.Q4_K_M.gguf", n_ctx=4096)
        >>> mem['model_size']
        4.0
        >>> mem['total_estimate']
        5.5
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement estimate_memory_usage")


def benchmark_inference(
    model: Any,
    prompts: List[str],
    max_tokens: int = 50,
    **kwargs
) -> BenchmarkMetrics:
    """
    Benchmark inference performance.

    Args:
        model: Loaded llama.cpp model
        prompts: List of test prompts
        max_tokens: Tokens to generate per prompt
        **kwargs: Additional generation parameters

    Returns:
        BenchmarkMetrics with:
            - total_tokens: Total tokens generated
            - total_time: Total wall clock time
            - tokens_per_second: Generation throughput
            - time_to_first_token: Average TTFT
            - prompts_processed: Number of prompts

    Examples:
        >>> model = load_model("model.gguf")
        >>> metrics = benchmark_inference(model, ["Test"] * 10)
        >>> metrics.tokens_per_second > 0
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement benchmark_inference")


def count_tokens(model: Any, text: str) -> int:
    """
    Count tokens in text using model's tokenizer.

    Args:
        model: Loaded llama.cpp model
        text: Text to tokenize

    Returns:
        Number of tokens

    Examples:
        >>> model = load_model("model.gguf")
        >>> count = count_tokens(model, "Hello, world!")
        >>> count > 0
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement count_tokens")


def is_gguf_file(path: str) -> bool:
    """
    Check if a file is a valid GGUF file.

    Args:
        path: File path

    Returns:
        True if file exists and has .gguf extension
    """
    return os.path.exists(path) and path.lower().endswith('.gguf')


def get_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Extract quantization type from filename.

    Common patterns: Q4_K_M, Q5_K_S, Q8_0, F16, etc.

    Args:
        filename: Model filename

    Returns:
        Quantization type or None if not found

    Examples:
        >>> get_quantization_from_filename("llama-7b.Q4_K_M.gguf")
        'Q4_K_M'
        >>> get_quantization_from_filename("model-f16.gguf")
        'F16'
    """
    import re
    # Common patterns
    patterns = [
        r'[._-](Q[0-9]_K_[MS])[._-]',
        r'[._-](Q[0-9]_[0-9])[._-]',
        r'[._-](Q[0-9])[._-]',
        r'[._-](F16|F32)[._-]',
        r'[._-](f16|f32)[._-]',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def llamacpp_available() -> bool:
    """Check if llama-cpp-python is installed."""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False
