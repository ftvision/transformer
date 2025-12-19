"""
Lab 03: Continuous Batching Simulation

Implement a simplified scheduler for continuous batching, the key technique
that allows high-throughput LLM serving.

Your task: Complete the functions and classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq


class RequestState(Enum):
    """Possible states for a generation request."""
    WAITING = "waiting"      # In queue, not yet scheduled
    RUNNING = "running"      # Actively generating tokens
    PREEMPTED = "preempted"  # Temporarily swapped out
    FINISHED = "finished"    # Generation complete


@dataclass
class Request:
    """
    A single generation request.

    Attributes:
        request_id: Unique identifier
        prompt_tokens: Number of tokens in the prompt
        max_tokens: Maximum tokens to generate
        arrival_time: When the request arrived
        state: Current state of the request
        tokens_generated: Number of output tokens generated so far
        completion_time: When the request finished (if finished)
    """
    request_id: int
    prompt_tokens: int
    max_tokens: int
    arrival_time: float
    state: RequestState = RequestState.WAITING
    tokens_generated: int = 0
    completion_time: Optional[float] = None

    def is_complete(self) -> bool:
        """Check if this request has finished generating."""
        return self.tokens_generated >= self.max_tokens

    def remaining_tokens(self) -> int:
        """Number of tokens left to generate."""
        return max(0, self.max_tokens - self.tokens_generated)


@dataclass
class BatchMetrics:
    """Metrics for a single batch iteration."""
    step: int
    batch_size: int
    tokens_generated: int
    requests_completed: int
    requests_started: int
    gpu_utilization: float  # 0-1


class ContinuousBatchingScheduler:
    """
    Scheduler implementing continuous (iteration-level) batching.

    Unlike static batching (which waits for all requests to complete),
    continuous batching adds/removes requests at each iteration step.

    Key concepts:
    - At each step, we generate ONE token per active request
    - When a request finishes, a new one can start immediately
    - GPU stays fully utilized even with varying generation lengths

    Attributes:
        max_batch_size: Maximum number of concurrent requests
        max_total_tokens: Maximum total tokens (prompt + generated) across all requests
    """

    def __init__(self, max_batch_size: int, max_total_tokens: int = 4096):
        """
        Initialize the scheduler.

        Args:
            max_batch_size: Maximum concurrent requests
            max_total_tokens: Memory budget in tokens

        Example:
            >>> scheduler = ContinuousBatchingScheduler(max_batch_size=32)
        """
        # YOUR CODE HERE
        # 1. Store max_batch_size and max_total_tokens
        # 2. Initialize waiting queue (list or priority queue)
        # 3. Initialize running set/dict
        # 4. Initialize completed list
        # 5. Initialize step counter
        raise NotImplementedError("Implement __init__")

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the waiting queue.

        Args:
            request: The request to queue

        The request should be added to the waiting queue and will be
        scheduled when there's capacity.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement add_request")

    def _can_schedule(self, request: Request) -> bool:
        """
        Check if we have capacity to schedule a request.

        Conditions to check:
        1. Running batch size < max_batch_size
        2. Total tokens (including this request) < max_total_tokens

        Args:
            request: The request to potentially schedule

        Returns:
            True if the request can be scheduled
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _can_schedule")

    def _current_total_tokens(self) -> int:
        """
        Calculate total tokens currently being processed.

        This includes:
        - prompt_tokens for each running request
        - tokens_generated for each running request
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement _current_total_tokens")

    def step(self, current_time: float) -> BatchMetrics:
        """
        Execute one iteration of the scheduler.

        One step does:
        1. Add waiting requests to running batch (if capacity)
        2. Generate one token for each running request
        3. Check for completed requests and remove them
        4. Return metrics for this step

        Args:
            current_time: Current simulation time

        Returns:
            Metrics for this iteration

        This is the core of continuous batching - at each step we can
        add new requests or remove completed ones.
        """
        # YOUR CODE HERE
        #
        # Steps:
        # 1. Try to schedule waiting requests
        #    for req in waiting_queue:
        #        if can_schedule(req):
        #            move to running
        #
        # 2. Generate one token for each running request
        #    for req in running:
        #        req.tokens_generated += 1
        #
        # 3. Check for completed requests
        #    for req in running:
        #        if req.is_complete():
        #            req.state = FINISHED
        #            req.completion_time = current_time
        #            move to completed
        #
        # 4. Return BatchMetrics
        raise NotImplementedError("Implement step")

    def run_until_empty(self, time_per_step: float = 0.01) -> List[BatchMetrics]:
        """
        Run the scheduler until all requests are completed.

        Args:
            time_per_step: Simulated time for each iteration

        Returns:
            List of metrics for each step
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement run_until_empty")

    def get_completed_requests(self) -> List[Request]:
        """Return list of completed requests."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_completed_requests")

    def get_waiting_count(self) -> int:
        """Return number of requests waiting to be scheduled."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_waiting_count")

    def get_running_count(self) -> int:
        """Return number of currently running requests."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_running_count")


class StaticBatchingScheduler:
    """
    Scheduler implementing static batching for comparison.

    Static batching waits for ALL requests in a batch to complete
    before starting the next batch. This is less efficient because
    fast requests must wait for slow ones.
    """

    def __init__(self, batch_size: int):
        """
        Initialize static batching scheduler.

        Args:
            batch_size: Fixed batch size
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement __init__")

    def add_request(self, request: Request) -> None:
        """Add a request to the queue."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement add_request")

    def step(self, current_time: float) -> BatchMetrics:
        """
        Execute one step of static batching.

        In static batching:
        - Fill batch to batch_size from queue
        - Generate one token for each
        - Only when ALL batch requests are done, move to next batch
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement step")

    def run_until_empty(self, time_per_step: float = 0.01) -> List[BatchMetrics]:
        """Run until all requests complete."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement run_until_empty")

    def get_completed_requests(self) -> List[Request]:
        """Return completed requests."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement get_completed_requests")


def compare_batching_strategies(
    requests: List[Request],
    batch_size: int,
    max_total_tokens: int = 8192
) -> Dict[str, Any]:
    """
    Compare continuous vs static batching performance.

    This demonstrates the efficiency advantage of continuous batching.

    Args:
        requests: List of requests to process
        batch_size: Batch size for both strategies
        max_total_tokens: Token budget for continuous batching

    Returns:
        Dictionary with:
        - 'continuous': Dict with metrics for continuous batching
        - 'static': Dict with metrics for static batching
        - 'speedup': Ratio of static_time / continuous_time

    Metrics for each strategy include:
        - 'total_steps': Number of scheduler steps
        - 'avg_latency': Average request completion time
        - 'throughput': Tokens generated per step
        - 'avg_gpu_utilization': Average GPU utilization
    """
    # YOUR CODE HERE
    #
    # 1. Clone requests for each strategy (so they can be modified independently)
    # 2. Run continuous batching scheduler
    # 3. Run static batching scheduler
    # 4. Calculate metrics for each
    # 5. Compare and return results
    raise NotImplementedError("Implement compare_batching_strategies")


def simulate_request_arrival(
    num_requests: int,
    arrival_rate: float,
    prompt_len_range: tuple = (10, 100),
    output_len_range: tuple = (10, 200)
) -> List[Request]:
    """
    Simulate request arrivals with Poisson distribution.

    Args:
        num_requests: Total number of requests to generate
        arrival_rate: Average requests per time unit (Poisson rate)
        prompt_len_range: (min, max) prompt length
        output_len_range: (min, max) output length

    Returns:
        List of Request objects with realistic arrival patterns

    Example:
        >>> requests = simulate_request_arrival(100, arrival_rate=10)
        >>> len(requests)
        100
    """
    # YOUR CODE HERE
    #
    # Use np.random.exponential for inter-arrival times (Poisson process)
    # Use np.random.randint for prompt and output lengths
    raise NotImplementedError("Implement simulate_request_arrival")


def calculate_queue_wait_time(
    scheduler: ContinuousBatchingScheduler
) -> Dict[str, float]:
    """
    Calculate queue waiting statistics from completed requests.

    Args:
        scheduler: Scheduler with completed requests

    Returns:
        Dictionary with:
        - 'mean_wait': Average time from arrival to start
        - 'max_wait': Maximum wait time
        - 'p99_wait': 99th percentile wait time
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement calculate_queue_wait_time")
