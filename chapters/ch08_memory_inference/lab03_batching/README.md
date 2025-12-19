# Lab 03: Batching Strategies

## Objective

Implement and compare static vs continuous batching strategies for LLM serving.

## What You'll Build

A simulation of different batching strategies:
- Static batching (wait for all requests to finish)
- Continuous batching (add/remove requests dynamically)
- Metrics calculation (throughput, latency, utilization)

## Prerequisites

Read these docs first:
- `../docs/03_batching_strategies.md`
- Completed Lab 01 and Lab 02

## Instructions

1. Open `src/batching.py`
2. Implement the functions and classes marked with `# YOUR CODE HERE`
3. Run tests to verify: `uv run pytest tests/`

## Classes and Functions to Implement

### `Request` Class

Represents a single inference request.

```python
@dataclass
class Request:
    id: int
    prompt_length: int      # Number of prompt tokens
    output_length: int      # Number of tokens to generate
    arrival_time: float     # When the request arrived
    start_time: float = None    # When processing started
    end_time: float = None      # When processing completed
```

### `calculate_request_latency(request)`
Calculate latency metrics for a completed request.

Returns dict with:
- `queue_time`: Time waiting before processing started
- `processing_time`: Time from start to completion
- `total_latency`: Total time from arrival to completion
- `time_to_first_token`: Time to first generated token
- `tokens_generated`: Number of tokens generated

### `static_batch_simulation(requests, batch_size, time_per_token)`
Simulate static batching.
- Collect requests until batch is full
- Process until ALL requests in batch complete
- Requests that finish early wait for others

Returns:
- `completed_requests`: List of completed Request objects
- `total_time`: Total simulation time
- `throughput`: Tokens per second

### `continuous_batch_simulation(requests, max_batch_size, time_per_token)`
Simulate continuous batching.
- New requests can join mid-generation
- Completed requests leave immediately
- Always keeps batch full if requests available

Returns:
- `completed_requests`: List of completed Request objects
- `total_time`: Total simulation time
- `throughput`: Tokens per second

### `calculate_batch_metrics(completed_requests, total_time)`
Calculate aggregate metrics for a batch of completed requests.

Returns dict with:
- `throughput`: Total tokens / total time
- `avg_latency`: Average total latency
- `p50_latency`, `p95_latency`, `p99_latency`: Latency percentiles
- `avg_queue_time`: Average time in queue
- `gpu_utilization`: Fraction of time GPU was processing

### `calculate_arithmetic_intensity_with_batch(batch_size, model_memory_bytes)`
Calculate how batching improves arithmetic intensity.
- Single request: intensity ≈ 2 FLOPS/byte
- Batched: intensity ≈ 2 × batch_size FLOPS/byte

Returns: Arithmetic intensity

### `optimal_batch_size(memory_budget, model_memory, kv_cache_per_token, max_seq_len)`
Calculate the optimal batch size given memory constraints.

Memory usage: `model_memory + batch_size × max_seq_len × kv_cache_per_token`

Returns: Maximum batch size that fits in memory

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_batching.py::TestStaticBatching

# Run with verbose output
uv run pytest tests/ -v
```

## Example Usage

```python
from batching import (
    Request,
    static_batch_simulation,
    continuous_batch_simulation,
    calculate_batch_metrics
)

# Create requests with varying output lengths
requests = [
    Request(id=0, prompt_length=100, output_length=50, arrival_time=0.0),
    Request(id=1, prompt_length=100, output_length=200, arrival_time=0.1),
    Request(id=2, prompt_length=100, output_length=30, arrival_time=0.2),
    Request(id=3, prompt_length=100, output_length=150, arrival_time=0.3),
]

# Static batching
static_results, static_time, static_throughput = static_batch_simulation(
    requests, batch_size=2, time_per_token=0.01
)
static_metrics = calculate_batch_metrics(static_results, static_time)
print(f"Static batching:")
print(f"  Throughput: {static_throughput:.1f} tokens/sec")
print(f"  Avg latency: {static_metrics['avg_latency']:.3f}s")

# Continuous batching
cont_results, cont_time, cont_throughput = continuous_batch_simulation(
    requests, max_batch_size=2, time_per_token=0.01
)
cont_metrics = calculate_batch_metrics(cont_results, cont_time)
print(f"Continuous batching:")
print(f"  Throughput: {cont_throughput:.1f} tokens/sec")
print(f"  Avg latency: {cont_metrics['avg_latency']:.3f}s")
```

## Hints

- In static batching, the batch processes for `max(output_lengths)` steps
- In continuous batching, a slot is freed as soon as a request completes
- Track both wall-clock time and per-request metrics
- Throughput = total_tokens / total_time
- Latency is per-request (from arrival to completion)

## Key Insights

1. **Static batching** wastes compute when requests finish at different times
2. **Continuous batching** maximizes GPU utilization
3. **Throughput vs Latency tradeoff**: Larger batches improve throughput but can hurt latency
4. **Memory limits batch size**: KV-cache memory grows with batch size

## Verification

All tests pass = you understand batching tradeoffs!

Expected result: Continuous batching should achieve higher throughput AND lower average latency than static batching when request output lengths vary.
