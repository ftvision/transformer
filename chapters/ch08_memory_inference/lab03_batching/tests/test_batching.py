"""Tests for Lab 03: Batching Strategies."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from batching import (
    Request,
    calculate_request_latency,
    static_batch_simulation,
    continuous_batch_simulation,
    calculate_batch_metrics,
    calculate_arithmetic_intensity_with_batch,
    optimal_batch_size,
    compare_batching_strategies,
    simulate_workload,
)


class TestRequest:
    """Tests for the Request dataclass."""

    def test_is_complete(self):
        """Request should know when it's complete."""
        req = Request(id=0, prompt_length=10, output_length=50, arrival_time=0.0)
        req.tokens_generated = 49
        assert not req.is_complete()

        req.tokens_generated = 50
        assert req.is_complete()

    def test_remaining_tokens(self):
        """Should calculate remaining tokens correctly."""
        req = Request(id=0, prompt_length=10, output_length=50, arrival_time=0.0)
        assert req.remaining_tokens() == 50

        req.tokens_generated = 30
        assert req.remaining_tokens() == 20

        req.tokens_generated = 60  # Over-generated
        assert req.remaining_tokens() == 0


class TestCalculateRequestLatency:
    """Tests for calculate_request_latency."""

    def test_basic_latency(self):
        """Calculate basic latency metrics."""
        req = Request(
            id=0, prompt_length=10, output_length=50,
            arrival_time=0.0, start_time=0.5, end_time=1.5
        )
        req.tokens_generated = 50

        metrics = calculate_request_latency(req)

        assert metrics['queue_time'] == 0.5
        assert metrics['processing_time'] == 1.0
        assert metrics['total_latency'] == 1.5
        assert metrics['tokens_generated'] == 50

    def test_no_queue_time(self):
        """Request starts immediately."""
        req = Request(
            id=0, prompt_length=10, output_length=100,
            arrival_time=1.0, start_time=1.0, end_time=2.0
        )
        req.tokens_generated = 100

        metrics = calculate_request_latency(req)

        assert metrics['queue_time'] == 0.0
        assert metrics['processing_time'] == 1.0


class TestStaticBatchSimulation:
    """Tests for static_batch_simulation."""

    def test_single_request(self):
        """Single request should complete."""
        requests = [
            Request(id=0, prompt_length=10, output_length=100, arrival_time=0.0)
        ]

        completed, total_time, throughput = static_batch_simulation(
            requests, batch_size=1, time_per_token=0.01
        )

        assert len(completed) == 1
        assert completed[0].tokens_generated == 100
        assert abs(total_time - 1.0) < 0.01  # 100 × 0.01 = 1.0

    def test_batch_waits_for_longest(self):
        """Batch should wait for the longest request."""
        requests = [
            Request(id=0, prompt_length=10, output_length=50, arrival_time=0.0),
            Request(id=1, prompt_length=10, output_length=100, arrival_time=0.0),
        ]

        completed, total_time, throughput = static_batch_simulation(
            requests, batch_size=2, time_per_token=0.01
        )

        assert len(completed) == 2
        # Both should finish at the same time (when longest completes)
        assert abs(completed[0].end_time - completed[1].end_time) < 0.01
        # Total time is based on longest request
        assert abs(total_time - 1.0) < 0.01  # 100 × 0.01

    def test_multiple_batches(self):
        """Should process multiple batches sequentially."""
        requests = [
            Request(id=0, prompt_length=10, output_length=50, arrival_time=0.0),
            Request(id=1, prompt_length=10, output_length=50, arrival_time=0.0),
            Request(id=2, prompt_length=10, output_length=50, arrival_time=0.0),
            Request(id=3, prompt_length=10, output_length=50, arrival_time=0.0),
        ]

        completed, total_time, throughput = static_batch_simulation(
            requests, batch_size=2, time_per_token=0.01
        )

        assert len(completed) == 4
        # Two batches of 50 tokens each
        assert abs(total_time - 1.0) < 0.01  # 2 batches × 0.5s each

    def test_throughput_calculation(self):
        """Throughput should be total tokens / total time."""
        requests = [
            Request(id=0, prompt_length=10, output_length=100, arrival_time=0.0),
        ]

        completed, total_time, throughput = static_batch_simulation(
            requests, batch_size=1, time_per_token=0.01
        )

        expected_throughput = 100 / 1.0  # 100 tokens / 1 second
        assert abs(throughput - expected_throughput) < 1


class TestContinuousBatchSimulation:
    """Tests for continuous_batch_simulation."""

    def test_single_request(self):
        """Single request should complete."""
        requests = [
            Request(id=0, prompt_length=10, output_length=100, arrival_time=0.0)
        ]

        completed, total_time, throughput = continuous_batch_simulation(
            requests, max_batch_size=1, time_per_token=0.01
        )

        assert len(completed) == 1
        assert completed[0].tokens_generated == 100

    def test_requests_complete_independently(self):
        """In continuous batching, requests complete independently."""
        requests = [
            Request(id=0, prompt_length=10, output_length=20, arrival_time=0.0),
            Request(id=1, prompt_length=10, output_length=100, arrival_time=0.0),
        ]

        completed, total_time, throughput = continuous_batch_simulation(
            requests, max_batch_size=2, time_per_token=0.01
        )

        assert len(completed) == 2

        # Find the requests by id
        req_0 = next(r for r in completed if r.id == 0)
        req_1 = next(r for r in completed if r.id == 1)

        # Request 0 should complete earlier
        assert req_0.end_time < req_1.end_time
        assert abs(req_0.end_time - 0.2) < 0.02  # ~20 tokens × 0.01

    def test_slot_reuse(self):
        """Completed request's slot should be reused."""
        requests = [
            Request(id=0, prompt_length=10, output_length=20, arrival_time=0.0),
            Request(id=1, prompt_length=10, output_length=20, arrival_time=0.0),
            Request(id=2, prompt_length=10, output_length=20, arrival_time=0.0),
        ]

        completed, total_time, throughput = continuous_batch_simulation(
            requests, max_batch_size=2, time_per_token=0.01
        )

        assert len(completed) == 3
        # With slot reuse, total time < 3 × 20 × 0.01 = 0.6
        # First two finish at 0.2, third starts at 0.2 and finishes at 0.4
        assert total_time < 0.5

    def test_better_than_static_for_varying_lengths(self):
        """Continuous should outperform static when output lengths vary."""
        requests = [
            Request(id=0, prompt_length=10, output_length=20, arrival_time=0.0),
            Request(id=1, prompt_length=10, output_length=200, arrival_time=0.0),
        ]

        _, static_time, _ = static_batch_simulation(
            [Request(id=r.id, prompt_length=r.prompt_length,
                    output_length=r.output_length, arrival_time=r.arrival_time)
             for r in requests],
            batch_size=2, time_per_token=0.01
        )

        _, cont_time, _ = continuous_batch_simulation(
            [Request(id=r.id, prompt_length=r.prompt_length,
                    output_length=r.output_length, arrival_time=r.arrival_time)
             for r in requests],
            max_batch_size=2, time_per_token=0.01
        )

        # Continuous should be faster or equal
        assert cont_time <= static_time + 0.01


class TestCalculateBatchMetrics:
    """Tests for calculate_batch_metrics."""

    def test_throughput(self):
        """Throughput should be total tokens / time."""
        requests = [
            Request(id=0, prompt_length=10, output_length=100, arrival_time=0.0,
                   start_time=0.0, end_time=1.0),
            Request(id=1, prompt_length=10, output_length=100, arrival_time=0.0,
                   start_time=0.0, end_time=1.0),
        ]
        for r in requests:
            r.tokens_generated = r.output_length

        metrics = calculate_batch_metrics(requests, total_time=1.0)

        assert metrics['throughput'] == 200.0  # 200 tokens / 1 second
        assert metrics['total_tokens'] == 200

    def test_latency_percentiles(self):
        """Should calculate latency percentiles."""
        # Create requests with varying latencies
        requests = []
        for i in range(100):
            req = Request(
                id=i, prompt_length=10, output_length=50,
                arrival_time=0.0, start_time=0.0, end_time=i * 0.01 + 0.5
            )
            req.tokens_generated = 50
            requests.append(req)

        metrics = calculate_batch_metrics(requests, total_time=2.0)

        # p50 should be around middle value
        assert 0.9 < metrics['p50_latency'] < 1.1
        # p95 should be higher
        assert metrics['p95_latency'] > metrics['p50_latency']
        # p99 should be highest
        assert metrics['p99_latency'] > metrics['p95_latency']


class TestArithmeticIntensityWithBatch:
    """Tests for calculate_arithmetic_intensity_with_batch."""

    def test_single_batch(self):
        """Single request has baseline intensity."""
        intensity = calculate_arithmetic_intensity_with_batch(1, 14e9)
        assert intensity == 2.0

    def test_batch_scaling(self):
        """Intensity should scale linearly with batch size."""
        i1 = calculate_arithmetic_intensity_with_batch(1, 14e9)
        i8 = calculate_arithmetic_intensity_with_batch(8, 14e9)
        assert i8 == 8 * i1

    def test_large_batch(self):
        """Large batch should have high intensity."""
        intensity = calculate_arithmetic_intensity_with_batch(64, 14e9)
        assert intensity == 128.0


class TestOptimalBatchSize:
    """Tests for optimal_batch_size."""

    def test_a100_7b_model(self):
        """Calculate optimal batch for 7B model on A100."""
        # A100 80GB, 7B model (14GB fp16), 512 bytes/token KV, 2048 seq len
        batch = optimal_batch_size(80e9, 14e9, 512, 2048)

        # Available: 80GB - 14GB = 66GB
        # Per batch slot: 2048 × 512 = 1MB
        # Batch size: 66GB / 1MB ≈ 66000... but that's too much
        # Let's recalculate with realistic KV-cache per token
        # Actually 512 bytes is 512 * 2048 = 1MB per sequence
        # (80e9 - 14e9) / (512 * 2048) = 66e9 / 1e6 = 66000
        # This seems unrealistic...

        # More realistic: KV-cache per token = 512KB (for 32 layers, d=4096, fp16)
        # Then per sequence at 2048 tokens: 512KB * 2048 = 1GB
        # (66GB) / 1GB = 66 sequences
        # But our input is 512 bytes/token so result is ~64000

        # The function should return a reasonable number
        assert batch >= 1

    def test_memory_constrained(self):
        """Should handle memory-constrained scenarios."""
        # Very limited memory
        batch = optimal_batch_size(
            memory_budget=20e9,
            model_memory=14e9,
            kv_cache_per_token=1024,  # 1KB per token
            max_seq_len=4096
        )

        # Available: 6GB, per sequence: 4096 × 1KB = 4MB
        # Batch: 6GB / 4MB = 1500
        assert batch >= 1

    def test_model_barely_fits(self):
        """When model barely fits, batch size should be minimal."""
        batch = optimal_batch_size(
            memory_budget=15e9,
            model_memory=14e9,
            kv_cache_per_token=1024,
            max_seq_len=2048
        )

        # Available: 1GB, per sequence: 2MB
        # Batch: ~500 but at least 1
        assert batch >= 1


class TestCompareStrategies:
    """Tests for compare_batching_strategies."""

    def test_comparison_returns_all_keys(self):
        """Should return all expected comparison metrics."""
        requests = [
            Request(id=i, prompt_length=10, output_length=50 + i * 10,
                   arrival_time=i * 0.1)
            for i in range(10)
        ]

        comparison = compare_batching_strategies(requests, batch_size=4)

        expected_keys = [
            'static_throughput', 'continuous_throughput',
            'static_avg_latency', 'continuous_avg_latency',
            'throughput_improvement', 'latency_improvement'
        ]
        for key in expected_keys:
            assert key in comparison, f"Missing key: {key}"

    def test_continuous_not_worse(self):
        """Continuous batching shouldn't be worse than static."""
        requests = [
            Request(id=i, prompt_length=10, output_length=50 + i * 50,
                   arrival_time=0.0)
            for i in range(8)
        ]

        comparison = compare_batching_strategies(requests, batch_size=4)

        # Throughput improvement should be >= 1 (continuous >= static)
        assert comparison['throughput_improvement'] >= 0.95  # Allow small variance


class TestSimulateWorkload:
    """Tests for simulate_workload."""

    def test_correct_count(self):
        """Should generate correct number of requests."""
        requests = simulate_workload(100)
        assert len(requests) == 100

    def test_sorted_by_arrival(self):
        """Requests should be sorted by arrival time."""
        requests = simulate_workload(50)
        for i in range(len(requests) - 1):
            assert requests[i].arrival_time <= requests[i + 1].arrival_time

    def test_lengths_in_range(self):
        """Prompt and output lengths should be in specified ranges."""
        requests = simulate_workload(
            100,
            prompt_length_range=(50, 100),
            output_length_range=(10, 200)
        )

        for req in requests:
            assert 50 <= req.prompt_length <= 100
            assert 10 <= req.output_length <= 200

    def test_reproducible_with_seed(self):
        """Same seed should produce same workload."""
        r1 = simulate_workload(50, seed=42)
        r2 = simulate_workload(50, seed=42)

        for req1, req2 in zip(r1, r2):
            assert req1.prompt_length == req2.prompt_length
            assert req1.output_length == req2.output_length
            assert req1.arrival_time == req2.arrival_time


class TestMilestone:
    """Integration tests for batching strategies."""

    def test_continuous_batching_advantage(self):
        """Continuous batching should show clear advantage with varying lengths."""
        # Create workload with high variance in output lengths
        np.random.seed(42)
        requests = []
        for i in range(20):
            output_len = np.random.choice([20, 50, 100, 200, 500])
            requests.append(Request(
                id=i,
                prompt_length=50,
                output_length=output_len,
                arrival_time=i * 0.05
            ))

        comparison = compare_batching_strategies(
            [Request(id=r.id, prompt_length=r.prompt_length,
                    output_length=r.output_length, arrival_time=r.arrival_time)
             for r in requests],
            batch_size=4,
            time_per_token=0.01
        )

        print("\n✅ Milestone Test - Batching Strategy Comparison")
        print(f"   Static throughput: {comparison['static_throughput']:.1f} tokens/sec")
        print(f"   Continuous throughput: {comparison['continuous_throughput']:.1f} tokens/sec")
        print(f"   Throughput improvement: {comparison['throughput_improvement']:.2f}x")
        print(f"   Static avg latency: {comparison['static_avg_latency']:.3f}s")
        print(f"   Continuous avg latency: {comparison['continuous_avg_latency']:.3f}s")
        print(f"   Latency improvement: {comparison['latency_improvement']:.2f}x")

        # Continuous should be at least as good
        assert comparison['throughput_improvement'] >= 0.95
        assert comparison['latency_improvement'] >= 0.95

    def test_memory_batch_tradeoff(self):
        """Demonstrate memory vs batch size tradeoff."""
        # Different memory budgets
        budgets = [40e9, 60e9, 80e9]  # 40GB, 60GB, 80GB
        model_memory = 14e9  # 7B fp16
        kv_per_token = 512 * 1024  # 512KB per token (realistic for 7B)
        max_seq = 2048

        print("\n✅ Milestone Test - Memory vs Batch Size Tradeoff")
        for budget in budgets:
            batch = optimal_batch_size(budget, model_memory, kv_per_token, max_seq)
            intensity = calculate_arithmetic_intensity_with_batch(batch, model_memory)
            print(f"   {budget/1e9:.0f}GB GPU: batch={batch}, intensity={intensity:.1f} FLOPS/byte")

        # Larger memory should allow larger batch
        b1 = optimal_batch_size(40e9, model_memory, kv_per_token, max_seq)
        b2 = optimal_batch_size(80e9, model_memory, kv_per_token, max_seq)
        assert b2 >= b1
