"""Tests for Lab 03: Continuous Batching Simulation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from continuous_batching import (
    Request,
    RequestState,
    BatchMetrics,
    ContinuousBatchingScheduler,
    StaticBatchingScheduler,
    compare_batching_strategies,
    simulate_request_arrival,
    calculate_queue_wait_time,
)


class TestRequest:
    """Tests for the Request class."""

    def test_request_initialization(self):
        """Request should initialize with correct attributes."""
        req = Request(
            request_id=1,
            prompt_tokens=50,
            max_tokens=100,
            arrival_time=0.0
        )

        assert req.request_id == 1
        assert req.prompt_tokens == 50
        assert req.max_tokens == 100
        assert req.state == RequestState.WAITING
        assert req.tokens_generated == 0

    def test_is_complete(self):
        """is_complete should return True when all tokens generated."""
        req = Request(request_id=1, prompt_tokens=10, max_tokens=50, arrival_time=0.0)

        assert not req.is_complete()

        req.tokens_generated = 49
        assert not req.is_complete()

        req.tokens_generated = 50
        assert req.is_complete()

    def test_remaining_tokens(self):
        """remaining_tokens should return correct count."""
        req = Request(request_id=1, prompt_tokens=10, max_tokens=100, arrival_time=0.0)

        assert req.remaining_tokens() == 100

        req.tokens_generated = 30
        assert req.remaining_tokens() == 70

        req.tokens_generated = 100
        assert req.remaining_tokens() == 0


class TestContinuousBatchingScheduler:
    """Tests for the continuous batching scheduler."""

    def test_initialization(self):
        """Scheduler should initialize with empty queues."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=32)

        assert scheduler.get_waiting_count() == 0
        assert scheduler.get_running_count() == 0
        assert len(scheduler.get_completed_requests()) == 0

    def test_add_request(self):
        """Adding request should increase waiting count."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=32)

        req = Request(request_id=1, prompt_tokens=10, max_tokens=50, arrival_time=0.0)
        scheduler.add_request(req)

        assert scheduler.get_waiting_count() == 1

    def test_step_schedules_request(self):
        """Step should move waiting requests to running."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=32)

        req = Request(request_id=1, prompt_tokens=10, max_tokens=50, arrival_time=0.0)
        scheduler.add_request(req)

        scheduler.step(current_time=0.0)

        assert scheduler.get_waiting_count() == 0
        assert scheduler.get_running_count() == 1

    def test_step_generates_token(self):
        """Each step should generate one token per running request."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=32)

        req = Request(request_id=1, prompt_tokens=10, max_tokens=50, arrival_time=0.0)
        scheduler.add_request(req)

        # First step: schedule and generate
        scheduler.step(current_time=0.0)
        # Second step: generate another token
        scheduler.step(current_time=0.01)

        # Request should have 2 tokens generated
        assert req.tokens_generated == 2

    def test_step_completes_request(self):
        """Step should move completed requests to finished."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=32)

        # Request that needs only 3 tokens
        req = Request(request_id=1, prompt_tokens=10, max_tokens=3, arrival_time=0.0)
        scheduler.add_request(req)

        # Run 3 steps
        for i in range(3):
            scheduler.step(current_time=i * 0.01)

        assert scheduler.get_running_count() == 0
        assert len(scheduler.get_completed_requests()) == 1
        assert req.state == RequestState.FINISHED

    def test_continuous_replacement(self):
        """New requests should be scheduled immediately when others finish."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=2)

        # Two short requests
        req1 = Request(request_id=1, prompt_tokens=10, max_tokens=2, arrival_time=0.0)
        req2 = Request(request_id=2, prompt_tokens=10, max_tokens=5, arrival_time=0.0)
        # One request that arrives slightly later
        req3 = Request(request_id=3, prompt_tokens=10, max_tokens=3, arrival_time=0.01)

        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.add_request(req3)

        # Step 1: req1 and req2 start
        metrics = scheduler.step(current_time=0.0)
        assert metrics.batch_size == 2  # Should have 2 running

        # Step 2: req1 completes, req3 should start
        metrics = scheduler.step(current_time=0.01)
        # req1 should be done now

        # After req1 finishes, req3 should be scheduled
        metrics = scheduler.step(current_time=0.02)
        assert scheduler.get_running_count() == 2  # req2 and req3

    def test_respects_max_batch_size(self):
        """Scheduler should not exceed max_batch_size."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=2)

        for i in range(5):
            req = Request(request_id=i, prompt_tokens=10, max_tokens=100, arrival_time=0.0)
            scheduler.add_request(req)

        scheduler.step(current_time=0.0)

        assert scheduler.get_running_count() <= 2
        assert scheduler.get_waiting_count() == 3

    def test_run_until_empty(self):
        """run_until_empty should process all requests."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=4)

        for i in range(10):
            req = Request(request_id=i, prompt_tokens=10, max_tokens=5, arrival_time=0.0)
            scheduler.add_request(req)

        metrics = scheduler.run_until_empty()

        assert scheduler.get_waiting_count() == 0
        assert scheduler.get_running_count() == 0
        assert len(scheduler.get_completed_requests()) == 10


class TestStaticBatchingScheduler:
    """Tests for static batching scheduler."""

    def test_initialization(self):
        """Static scheduler should initialize correctly."""
        scheduler = StaticBatchingScheduler(batch_size=4)
        assert len(scheduler.get_completed_requests()) == 0

    def test_waits_for_batch_completion(self):
        """Static batching should wait for all in batch to complete."""
        scheduler = StaticBatchingScheduler(batch_size=2)

        # One fast, one slow request
        fast = Request(request_id=1, prompt_tokens=10, max_tokens=2, arrival_time=0.0)
        slow = Request(request_id=2, prompt_tokens=10, max_tokens=10, arrival_time=0.0)
        waiting = Request(request_id=3, prompt_tokens=10, max_tokens=5, arrival_time=0.0)

        scheduler.add_request(fast)
        scheduler.add_request(slow)
        scheduler.add_request(waiting)

        # Run until fast could have finished (2 steps)
        scheduler.step(0.0)
        scheduler.step(0.01)

        # Fast is done, but batch isn't complete yet
        # In static batching, waiting request shouldn't start until slow finishes
        assert fast.is_complete()
        assert not slow.is_complete()
        # waiting should NOT be running yet in static batching

    def test_run_until_empty(self):
        """Static scheduler should eventually complete all requests."""
        scheduler = StaticBatchingScheduler(batch_size=2)

        for i in range(4):
            req = Request(request_id=i, prompt_tokens=10, max_tokens=5, arrival_time=0.0)
            scheduler.add_request(req)

        scheduler.run_until_empty()

        assert len(scheduler.get_completed_requests()) == 4


class TestCompareBatchingStrategies:
    """Tests for comparing batching strategies."""

    def test_continuous_faster_for_varying_lengths(self):
        """Continuous batching should be faster when request lengths vary."""
        # Create requests with highly varying output lengths
        requests = [
            Request(request_id=0, prompt_tokens=10, max_tokens=5, arrival_time=0.0),
            Request(request_id=1, prompt_tokens=10, max_tokens=100, arrival_time=0.0),
            Request(request_id=2, prompt_tokens=10, max_tokens=5, arrival_time=0.0),
            Request(request_id=3, prompt_tokens=10, max_tokens=100, arrival_time=0.0),
        ]

        result = compare_batching_strategies(requests, batch_size=2)

        # Continuous should be faster (speedup > 1)
        assert result['speedup'] >= 1.0

        # Continuous should have higher average GPU utilization
        assert result['continuous']['avg_gpu_utilization'] >= result['static']['avg_gpu_utilization']

    def test_similar_for_uniform_lengths(self):
        """Both strategies should be similar when all requests have same length."""
        # All requests have same output length
        requests = [
            Request(request_id=i, prompt_tokens=10, max_tokens=50, arrival_time=0.0)
            for i in range(4)
        ]

        result = compare_batching_strategies(requests, batch_size=2)

        # With uniform lengths, speedup should be close to 1
        assert 0.8 < result['speedup'] < 1.5

    def test_returns_all_metrics(self):
        """Compare should return all expected metrics."""
        requests = [
            Request(request_id=i, prompt_tokens=10, max_tokens=20, arrival_time=0.0)
            for i in range(4)
        ]

        result = compare_batching_strategies(requests, batch_size=2)

        # Check continuous metrics
        assert 'total_steps' in result['continuous']
        assert 'avg_latency' in result['continuous']
        assert 'throughput' in result['continuous']
        assert 'avg_gpu_utilization' in result['continuous']

        # Check static metrics
        assert 'total_steps' in result['static']
        assert 'avg_latency' in result['static']
        assert 'throughput' in result['static']

        # Check comparison
        assert 'speedup' in result


class TestSimulateRequestArrival:
    """Tests for request arrival simulation."""

    def test_generates_correct_count(self):
        """Should generate requested number of requests."""
        requests = simulate_request_arrival(num_requests=100, arrival_rate=10)
        assert len(requests) == 100

    def test_arrival_times_increase(self):
        """Request arrival times should be monotonically increasing."""
        requests = simulate_request_arrival(num_requests=50, arrival_rate=10)

        arrival_times = [r.arrival_time for r in requests]
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] >= arrival_times[i-1]

    def test_respects_length_ranges(self):
        """Prompt and output lengths should be within specified ranges."""
        requests = simulate_request_arrival(
            num_requests=100,
            arrival_rate=10,
            prompt_len_range=(20, 50),
            output_len_range=(30, 80)
        )

        for req in requests:
            assert 20 <= req.prompt_tokens <= 50
            assert 30 <= req.max_tokens <= 80

    def test_unique_request_ids(self):
        """All request IDs should be unique."""
        requests = simulate_request_arrival(num_requests=100, arrival_rate=10)

        ids = [r.request_id for r in requests]
        assert len(set(ids)) == 100


class TestCalculateQueueWaitTime:
    """Tests for queue wait time calculation."""

    def test_wait_time_calculation(self):
        """Should calculate correct wait time statistics."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=1)

        # Add requests with staggered arrivals
        req1 = Request(request_id=1, prompt_tokens=10, max_tokens=5, arrival_time=0.0)
        req2 = Request(request_id=2, prompt_tokens=10, max_tokens=5, arrival_time=0.01)
        req3 = Request(request_id=3, prompt_tokens=10, max_tokens=5, arrival_time=0.02)

        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.add_request(req3)

        scheduler.run_until_empty(time_per_step=0.01)

        stats = calculate_queue_wait_time(scheduler)

        assert 'mean_wait' in stats
        assert 'max_wait' in stats
        assert 'p99_wait' in stats

        # Max wait should be >= mean wait
        assert stats['max_wait'] >= stats['mean_wait']
        # p99 should be between mean and max
        assert stats['mean_wait'] <= stats['p99_wait'] <= stats['max_wait']

    def test_no_wait_for_single_request(self):
        """Single request should have zero wait time."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=4)

        req = Request(request_id=1, prompt_tokens=10, max_tokens=5, arrival_time=0.0)
        scheduler.add_request(req)

        scheduler.run_until_empty(time_per_step=0.01)

        stats = calculate_queue_wait_time(scheduler)

        # First request should start immediately
        assert stats['mean_wait'] < 0.001  # Essentially zero


class TestBatchMetrics:
    """Tests for batch metrics tracking."""

    def test_metrics_returned_each_step(self):
        """Each step should return BatchMetrics."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=4)

        req = Request(request_id=1, prompt_tokens=10, max_tokens=10, arrival_time=0.0)
        scheduler.add_request(req)

        metrics = scheduler.step(current_time=0.0)

        assert isinstance(metrics, BatchMetrics)
        assert metrics.step >= 0
        assert metrics.batch_size >= 0
        assert 0 <= metrics.gpu_utilization <= 1

    def test_gpu_utilization_calculation(self):
        """GPU utilization should reflect batch fullness."""
        scheduler = ContinuousBatchingScheduler(max_batch_size=4)

        # Add 2 requests (half capacity)
        for i in range(2):
            req = Request(request_id=i, prompt_tokens=10, max_tokens=10, arrival_time=0.0)
            scheduler.add_request(req)

        metrics = scheduler.step(current_time=0.0)

        # With 2/4 capacity, utilization should be ~0.5
        assert 0.4 <= metrics.gpu_utilization <= 0.6
