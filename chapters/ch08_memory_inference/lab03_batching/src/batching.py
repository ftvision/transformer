"""
Lab 03: Batching Strategies

Implement and compare static vs continuous batching for LLM serving.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class Request:
    """
    Represents a single inference request.

    Attributes:
        id: Unique identifier for the request
        prompt_length: Number of tokens in the prompt
        output_length: Number of tokens to generate
        arrival_time: When the request arrived (simulation time)
        start_time: When processing started (None if not started)
        end_time: When processing completed (None if not completed)
        tokens_generated: Number of tokens generated so far
    """
    id: int
    prompt_length: int
    output_length: int
    arrival_time: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tokens_generated: int = 0

    def is_complete(self) -> bool:
        """Check if request has generated all tokens."""
        return self.tokens_generated >= self.output_length

    def remaining_tokens(self) -> int:
        """Number of tokens left to generate."""
        return max(0, self.output_length - self.tokens_generated)


def calculate_request_latency(request: Request) -> Dict[str, float]:
    """
    Calculate latency metrics for a completed request.

    Args:
        request: A completed Request object (must have start_time and end_time)

    Returns:
        Dictionary with:
            - queue_time: Time spent waiting before processing started
            - processing_time: Time from start to completion
            - total_latency: Total time from arrival to completion
            - time_to_first_token: Time until first token generated (≈ start_time - arrival_time + prefill_time)
                                   For simplicity, assume TTFT ≈ queue_time + small constant
            - tokens_generated: Number of tokens generated

    Examples:
        >>> req = Request(id=0, prompt_length=10, output_length=50,
        ...               arrival_time=0.0, start_time=0.5, end_time=1.5)
        >>> req.tokens_generated = 50
        >>> metrics = calculate_request_latency(req)
        >>> metrics['queue_time']
        0.5
        >>> metrics['processing_time']
        1.0
        >>> metrics['total_latency']
        1.5
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_request_latency")


def static_batch_simulation(
    requests: List[Request],
    batch_size: int,
    time_per_token: float = 0.01
) -> Tuple[List[Request], float, float]:
    """
    Simulate static batching strategy.

    Static batching:
    1. Wait until batch_size requests are available (or no more requests coming)
    2. Process the batch until ALL requests in the batch complete
    3. Requests that finish early must wait for the longest request
    4. Only then can a new batch start

    Args:
        requests: List of Request objects (sorted by arrival_time)
        batch_size: Number of requests to process together
        time_per_token: Time to generate one token (same for all batch members)

    Returns:
        completed_requests: List of completed Request objects with timing info
        total_time: Total simulation time
        throughput: Tokens per second (total_tokens / total_time)

    Example:
        >>> requests = [
        ...     Request(id=0, prompt_length=10, output_length=50, arrival_time=0.0),
        ...     Request(id=1, prompt_length=10, output_length=100, arrival_time=0.0),
        ... ]
        >>> completed, total_time, throughput = static_batch_simulation(
        ...     requests, batch_size=2, time_per_token=0.01
        ... )
        >>> # Both requests process for 100 steps (longest output)
        >>> # Request 0 finishes early but waits
        >>> total_time
        1.0  # 100 tokens × 0.01s

    Note:
        - Assume prefill is instant (or include a small fixed prefill time)
        - time_per_token applies per decode step for the whole batch
        - If fewer than batch_size requests, process what's available
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement static_batch_simulation")


def continuous_batch_simulation(
    requests: List[Request],
    max_batch_size: int,
    time_per_token: float = 0.01
) -> Tuple[List[Request], float, float]:
    """
    Simulate continuous (iteration-level) batching strategy.

    Continuous batching:
    1. Start processing requests as they arrive (up to max_batch_size)
    2. At each decode step:
       - Generate one token for each active request
       - Remove completed requests immediately
       - Add new requests if slots available
    3. Requests don't wait for others to finish

    Args:
        requests: List of Request objects (sorted by arrival_time)
        max_batch_size: Maximum number of concurrent requests
        time_per_token: Time to generate one token per decode step

    Returns:
        completed_requests: List of completed Request objects with timing info
        total_time: Total simulation time
        throughput: Tokens per second

    Example:
        >>> requests = [
        ...     Request(id=0, prompt_length=10, output_length=20, arrival_time=0.0),
        ...     Request(id=1, prompt_length=10, output_length=100, arrival_time=0.0),
        ... ]
        >>> completed, total_time, throughput = continuous_batch_simulation(
        ...     requests, max_batch_size=2, time_per_token=0.01
        ... )
        >>> # Request 0 completes at 0.2s, slot opens for new request
        >>> # Request 1 continues until 1.0s

    Note:
        - New requests join at iteration boundaries
        - Completed requests leave immediately
        - Time advances by time_per_token per iteration
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement continuous_batch_simulation")


def calculate_batch_metrics(
    completed_requests: List[Request],
    total_time: float
) -> Dict[str, float]:
    """
    Calculate aggregate metrics for completed requests.

    Args:
        completed_requests: List of completed Request objects
        total_time: Total simulation time

    Returns:
        Dictionary with:
            - throughput: Total tokens generated / total time
            - avg_latency: Average total latency across requests
            - p50_latency: 50th percentile latency
            - p95_latency: 95th percentile latency
            - p99_latency: 99th percentile latency
            - avg_queue_time: Average time requests spent in queue
            - total_tokens: Total tokens generated

    Examples:
        >>> requests = [...]  # List of completed requests
        >>> metrics = calculate_batch_metrics(requests, total_time=10.0)
        >>> metrics['throughput']
        150.0  # tokens/sec
        >>> metrics['p95_latency']
        0.95  # 95th percentile latency
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_batch_metrics")


def calculate_arithmetic_intensity_with_batch(
    batch_size: int,
    model_memory_bytes: float,
    flops_per_token: float = 2.0
) -> float:
    """
    Calculate how batching improves arithmetic intensity.

    Without batching (batch_size=1):
    - Load all weights once, generate 1 token
    - Intensity ≈ flops_per_token per byte (very low)

    With batching:
    - Load all weights once, generate batch_size tokens
    - Intensity ≈ flops_per_token × batch_size per byte

    Args:
        batch_size: Number of sequences processed together
        model_memory_bytes: Size of model weights in bytes
        flops_per_token: FLOPs per token per weight byte (default: 2 for mul-add)

    Returns:
        Arithmetic intensity (FLOPS per byte)

    Examples:
        >>> # Single sequence
        >>> calculate_arithmetic_intensity_with_batch(1, 14e9)
        2.0

        >>> # Batch of 8
        >>> calculate_arithmetic_intensity_with_batch(8, 14e9)
        16.0

        >>> # Batch of 64
        >>> calculate_arithmetic_intensity_with_batch(64, 14e9)
        128.0

    Note:
        This is a simplified model. Real intensity depends on:
        - Activation memory accesses
        - KV-cache accesses
        - Memory layout and caching effects
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_arithmetic_intensity_with_batch")


def optimal_batch_size(
    memory_budget: float,
    model_memory: float,
    kv_cache_per_token: float,
    max_seq_len: int
) -> int:
    """
    Calculate the maximum batch size given memory constraints.

    Total memory usage:
        model_memory + batch_size × max_seq_len × kv_cache_per_token

    We need this to be <= memory_budget.

    Args:
        memory_budget: Total available GPU memory (bytes)
        model_memory: Memory for model weights (bytes)
        kv_cache_per_token: KV-cache memory per token (bytes)
        max_seq_len: Maximum sequence length

    Returns:
        Maximum batch size that fits in memory (at least 1)

    Examples:
        >>> # A100 80GB, 7B model (14GB), 512 bytes/token KV, 2048 seq len
        >>> optimal_batch_size(80e9, 14e9, 512, 2048)
        62  # (80GB - 14GB) / (2048 × 512 bytes) ≈ 62

        >>> # Same but with 4096 seq len
        >>> optimal_batch_size(80e9, 14e9, 512, 4096)
        31  # Half the batch size for 2x sequence length

    Note:
        This is a simplified calculation. Real systems also need memory for:
        - Activations
        - Optimizer states (if training)
        - Framework overhead
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement optimal_batch_size")


def compare_batching_strategies(
    requests: List[Request],
    batch_size: int,
    time_per_token: float = 0.01
) -> Dict[str, Any]:
    """
    Compare static vs continuous batching on the same workload.

    Args:
        requests: List of Request objects
        batch_size: Batch size for both strategies
        time_per_token: Time per token

    Returns:
        Dictionary with comparison results:
            - static_throughput: Tokens/sec for static batching
            - continuous_throughput: Tokens/sec for continuous batching
            - static_avg_latency: Average latency for static
            - continuous_avg_latency: Average latency for continuous
            - throughput_improvement: continuous / static ratio
            - latency_improvement: static / continuous ratio (>1 means continuous is better)

    Examples:
        >>> requests = [...]  # Requests with varying output lengths
        >>> comparison = compare_batching_strategies(requests, batch_size=4)
        >>> comparison['throughput_improvement']
        1.3  # Continuous is 30% higher throughput
        >>> comparison['latency_improvement']
        1.5  # Continuous has 50% lower latency
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_batching_strategies")


def simulate_workload(
    num_requests: int,
    prompt_length_range: Tuple[int, int] = (50, 200),
    output_length_range: Tuple[int, int] = (20, 500),
    arrival_rate: float = 10.0,  # requests per second
    seed: int = 42
) -> List[Request]:
    """
    Generate a realistic workload of requests.

    Args:
        num_requests: Number of requests to generate
        prompt_length_range: (min, max) prompt lengths
        output_length_range: (min, max) output lengths
        arrival_rate: Average requests per second (Poisson process)
        seed: Random seed for reproducibility

    Returns:
        List of Request objects sorted by arrival time

    Examples:
        >>> requests = simulate_workload(100, arrival_rate=10.0)
        >>> len(requests)
        100
        >>> all(r.arrival_time <= requests[i+1].arrival_time
        ...     for i, r in enumerate(requests[:-1]))
        True  # Sorted by arrival time
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement simulate_workload")
