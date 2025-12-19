"""Lab 05: Framework Benchmarking - Compare inference frameworks.

This module provides tools for benchmarking LLM inference frameworks,
measuring throughput, latency, and memory usage.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union
import time


@dataclass
class LatencyResult:
    """Results from latency measurement.

    Attributes:
        mean_ms: Average latency in milliseconds
        std_ms: Standard deviation in milliseconds
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
        ttft_ms: Time to first token in milliseconds
        measurements: List of individual measurements
    """
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ttft_ms: float
    measurements: List[float] = field(default_factory=list)


@dataclass
class ThroughputResult:
    """Results from throughput measurement.

    Attributes:
        tokens_per_second: Generation throughput
        total_tokens: Total tokens generated
        total_time_s: Total time in seconds
        batch_size: Number of prompts processed
    """
    tokens_per_second: float
    total_tokens: int
    total_time_s: float
    batch_size: int


@dataclass
class MemoryResult:
    """Results from memory measurement.

    Attributes:
        peak_memory_mb: Peak memory usage in MB
        allocated_memory_mb: Currently allocated memory in MB
        reserved_memory_mb: Reserved memory in MB (GPU)
        memory_efficiency: Tokens per MB of peak memory
    """
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    memory_efficiency: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a framework.

    Attributes:
        framework_name: Name of the framework benchmarked
        model_name: Name/path of the model used
        latency: Latency measurement results
        throughput: Throughput measurement results
        memory: Memory measurement results
        config: Configuration used for benchmarking
        metadata: Additional metadata (hardware, versions, etc.)
    """
    framework_name: str
    model_name: str
    latency: Optional[LatencyResult] = None
    throughput: Optional[ThroughputResult] = None
    memory: Optional[MemoryResult] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def measure_latency(
    generate_fn: Callable,
    prompt: str,
    num_tokens: int = 50,
    warmup_runs: int = 2,
    num_runs: int = 5,
    **generate_kwargs,
) -> LatencyResult:
    """Measure generation latency with warmup.

    Performs warmup runs to exclude cold-start effects, then measures
    latency across multiple runs.

    Args:
        generate_fn: Function that generates text. Should accept
            (prompt, max_tokens, **kwargs) and return generated text.
        prompt: Input prompt for generation
        num_tokens: Number of tokens to generate
        warmup_runs: Number of warmup runs to perform (not measured)
        num_runs: Number of measurement runs
        **generate_kwargs: Additional arguments for generate_fn

    Returns:
        LatencyResult with timing statistics

    Example:
        >>> def mock_generate(prompt, max_tokens, **kwargs):
        ...     time.sleep(0.1)  # Simulate generation
        ...     return "Generated text"
        >>> result = measure_latency(mock_generate, "Hello", num_tokens=10)
        >>> result.mean_ms > 0
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement measure_latency")


def measure_throughput(
    generate_fn: Callable,
    prompts: List[str],
    num_tokens: int = 50,
    **generate_kwargs,
) -> ThroughputResult:
    """Measure generation throughput for batch processing.

    Measures tokens per second for processing a batch of prompts.

    Args:
        generate_fn: Function that generates text for batch.
            Should accept (prompts, max_tokens, **kwargs) and
            return list of generated texts.
        prompts: List of input prompts
        num_tokens: Number of tokens to generate per prompt
        **generate_kwargs: Additional arguments for generate_fn

    Returns:
        ThroughputResult with throughput metrics

    Example:
        >>> def mock_batch_generate(prompts, max_tokens, **kwargs):
        ...     time.sleep(0.01 * len(prompts))
        ...     return ["text"] * len(prompts)
        >>> result = measure_throughput(mock_batch_generate, ["a", "b"], 10)
        >>> result.batch_size == 2
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement measure_throughput")


def measure_memory(
    generate_fn: Callable,
    prompt: str,
    num_tokens: int = 50,
    device: str = "cuda",
    **generate_kwargs,
) -> MemoryResult:
    """Measure memory usage during generation.

    Tracks peak memory usage while generating text. For GPU, uses
    torch.cuda memory tracking. For CPU, uses process memory.

    Args:
        generate_fn: Function that generates text
        prompt: Input prompt for generation
        num_tokens: Number of tokens to generate
        device: Device type ("cuda" or "cpu")
        **generate_kwargs: Additional arguments for generate_fn

    Returns:
        MemoryResult with memory usage metrics

    Example:
        >>> result = measure_memory(lambda p, n: "text", "Hello", device="cpu")
        >>> result.peak_memory_mb >= 0
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement measure_memory")


def benchmark_single(
    generate_fn: Callable,
    prompt: str,
    num_tokens: int = 50,
    warmup_runs: int = 2,
    num_runs: int = 5,
    measure_mem: bool = True,
    device: str = "cuda",
    **generate_kwargs,
) -> Dict[str, Any]:
    """Run complete benchmark for a single prompt.

    Measures latency, estimates throughput, and optionally measures memory
    for a single prompt.

    Args:
        generate_fn: Function that generates text
        prompt: Input prompt
        num_tokens: Tokens to generate
        warmup_runs: Warmup iterations
        num_runs: Measurement iterations
        measure_mem: Whether to measure memory
        device: Device for memory measurement
        **generate_kwargs: Additional arguments

    Returns:
        Dictionary with 'latency', 'throughput', and optionally 'memory' keys

    Example:
        >>> def gen(p, n): return "x" * n
        >>> result = benchmark_single(gen, "Hello", measure_mem=False)
        >>> 'latency' in result
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement benchmark_single")


def benchmark_batch(
    generate_fn: Callable,
    prompts: List[str],
    num_tokens: int = 50,
    warmup_runs: int = 1,
    measure_mem: bool = True,
    device: str = "cuda",
    **generate_kwargs,
) -> Dict[str, Any]:
    """Run complete benchmark for batch processing.

    Measures throughput and optionally memory for batch generation.

    Args:
        generate_fn: Function that generates text for batches
        prompts: List of input prompts
        num_tokens: Tokens to generate per prompt
        warmup_runs: Warmup iterations
        measure_mem: Whether to measure memory
        device: Device for memory measurement
        **generate_kwargs: Additional arguments

    Returns:
        Dictionary with 'throughput' and optionally 'memory' keys

    Example:
        >>> def gen(ps, n): return ["x"] * len(ps)
        >>> result = benchmark_batch(gen, ["a", "b"], measure_mem=False)
        >>> 'throughput' in result
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement benchmark_batch")


def run_latency_sweep(
    generate_fn: Callable,
    prompt: str,
    token_counts: List[int],
    warmup_runs: int = 2,
    num_runs: int = 3,
    **generate_kwargs,
) -> Dict[int, LatencyResult]:
    """Run latency benchmark across different output lengths.

    Tests how latency scales with the number of generated tokens.

    Args:
        generate_fn: Function that generates text
        prompt: Input prompt
        token_counts: List of token counts to test
        warmup_runs: Warmup iterations per test
        num_runs: Measurement iterations per test
        **generate_kwargs: Additional arguments

    Returns:
        Dictionary mapping token count to LatencyResult

    Example:
        >>> def gen(p, n): return "x"
        >>> results = run_latency_sweep(gen, "Hi", [10, 20])
        >>> len(results) == 2
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement run_latency_sweep")


def run_throughput_sweep(
    generate_fn: Callable,
    prompt: str,
    num_tokens: int,
    batch_sizes: List[int],
    **generate_kwargs,
) -> Dict[int, ThroughputResult]:
    """Run throughput benchmark across different batch sizes.

    Tests how throughput scales with batch size.

    Args:
        generate_fn: Function that generates text for batches
        prompt: Base prompt (will be replicated for batch)
        num_tokens: Tokens to generate per prompt
        batch_sizes: List of batch sizes to test
        **generate_kwargs: Additional arguments

    Returns:
        Dictionary mapping batch size to ThroughputResult

    Example:
        >>> def gen(ps, n): return ["x"] * len(ps)
        >>> results = run_throughput_sweep(gen, "Hi", 10, [1, 2])
        >>> len(results) == 2
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement run_throughput_sweep")


def compare_results(
    results: Dict[str, BenchmarkResult],
) -> Dict[str, Dict[str, Any]]:
    """Compare benchmark results across frameworks.

    Creates a comparison table showing relative performance.

    Args:
        results: Dictionary mapping framework name to BenchmarkResult

    Returns:
        Comparison dictionary with metrics for each framework and
        relative rankings

    Example:
        >>> r1 = BenchmarkResult("fw1", "model", throughput=ThroughputResult(100, 1000, 10, 1))
        >>> r2 = BenchmarkResult("fw2", "model", throughput=ThroughputResult(200, 2000, 10, 1))
        >>> comp = compare_results({"fw1": r1, "fw2": r2})
        >>> "fw1" in comp and "fw2" in comp
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_results")


def generate_report(
    comparison: Dict[str, Dict[str, Any]],
    output_format: str = "markdown",
    include_charts: bool = False,
) -> str:
    """Generate a formatted benchmark report.

    Creates a human-readable report from comparison data.

    Args:
        comparison: Comparison data from compare_results()
        output_format: "markdown", "text", or "html"
        include_charts: Whether to include ASCII charts (markdown/text only)

    Returns:
        Formatted report string

    Example:
        >>> comp = {"fw1": {"throughput_tps": 100}, "fw2": {"throughput_tps": 200}}
        >>> report = generate_report(comp, output_format="text")
        >>> "fw1" in report
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement generate_report")


def plot_comparison(
    results: Dict[str, BenchmarkResult],
    metric: str = "throughput",
    save_path: Optional[str] = None,
) -> Any:
    """Create visualization of benchmark comparison.

    Generates a bar chart comparing frameworks on the specified metric.

    Args:
        results: Dictionary mapping framework name to BenchmarkResult
        metric: Metric to plot ("throughput", "latency", or "memory")
        save_path: If provided, save plot to this path

    Returns:
        Matplotlib figure object (or None if matplotlib unavailable)

    Example:
        >>> r1 = BenchmarkResult("fw1", "model", throughput=ThroughputResult(100, 1000, 10, 1))
        >>> fig = plot_comparison({"fw1": r1}, metric="throughput")
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement plot_comparison")


# Utility functions

def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark metadata.

    Collects CPU, GPU, memory, and software version information.

    Returns:
        Dictionary with system information

    Example:
        >>> info = get_system_info()
        >>> "python_version" in info
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_system_info")


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text.

    Simple estimation based on character count. For accurate counts,
    use the actual tokenizer.

    Args:
        text: Text to estimate tokens for
        chars_per_token: Average characters per token (default 4.0)

    Returns:
        Estimated token count

    Example:
        >>> estimate_tokens("Hello world", chars_per_token=4.0)
        3
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement estimate_tokens")


def format_time(seconds: float) -> str:
    """Format time duration for display.

    Converts seconds to human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1.5s", "150ms", "1500us")

    Example:
        >>> format_time(1.5)
        '1.50s'
        >>> format_time(0.015)
        '15.00ms'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement format_time")


def format_memory(mb: float) -> str:
    """Format memory size for display.

    Converts MB to appropriate unit.

    Args:
        mb: Memory in megabytes

    Returns:
        Formatted string (e.g., "1.5GB", "512MB")

    Example:
        >>> format_memory(1536)
        '1.50GB'
        >>> format_memory(512)
        '512.00MB'
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement format_memory")
