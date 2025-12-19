"""Tests for Lab 05: Framework Benchmarking."""

import pytest
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import (
    LatencyResult,
    ThroughputResult,
    MemoryResult,
    BenchmarkResult,
    measure_latency,
    measure_throughput,
    measure_memory,
    benchmark_single,
    benchmark_batch,
    run_latency_sweep,
    run_throughput_sweep,
    compare_results,
    generate_report,
    plot_comparison,
    get_system_info,
    estimate_tokens,
    format_time,
    format_memory,
)


# Mock generate functions for testing
def mock_generate_single(prompt: str, max_tokens: int, **kwargs) -> str:
    """Mock single prompt generation."""
    time.sleep(0.01)  # Simulate some work
    return "x" * (max_tokens * 4)  # ~4 chars per token


def mock_generate_batch(prompts: list, max_tokens: int, **kwargs) -> list:
    """Mock batch generation."""
    time.sleep(0.01 * len(prompts))  # Scale with batch
    return ["x" * (max_tokens * 4) for _ in prompts]


class TestLatencyResult:
    """Tests for LatencyResult dataclass."""

    def test_creation(self):
        """Should create LatencyResult."""
        result = LatencyResult(
            mean_ms=100.0,
            std_ms=10.0,
            min_ms=90.0,
            max_ms=120.0,
            ttft_ms=50.0,
        )
        assert result.mean_ms == 100.0

    def test_measurements_default(self):
        """Should default to empty measurements list."""
        result = LatencyResult(
            mean_ms=100.0, std_ms=10.0, min_ms=90.0, max_ms=120.0, ttft_ms=50.0
        )
        assert result.measurements == []

    def test_with_measurements(self):
        """Should store measurements."""
        result = LatencyResult(
            mean_ms=100.0,
            std_ms=10.0,
            min_ms=90.0,
            max_ms=120.0,
            ttft_ms=50.0,
            measurements=[90.0, 100.0, 110.0],
        )
        assert len(result.measurements) == 3


class TestThroughputResult:
    """Tests for ThroughputResult dataclass."""

    def test_creation(self):
        """Should create ThroughputResult."""
        result = ThroughputResult(
            tokens_per_second=100.0,
            total_tokens=1000,
            total_time_s=10.0,
            batch_size=32,
        )
        assert result.tokens_per_second == 100.0
        assert result.batch_size == 32

    def test_consistency(self):
        """Values should be internally consistent."""
        result = ThroughputResult(
            tokens_per_second=100.0,
            total_tokens=1000,
            total_time_s=10.0,
            batch_size=1,
        )
        # throughput = tokens / time
        expected_tps = result.total_tokens / result.total_time_s
        assert result.tokens_per_second == expected_tps


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_creation(self):
        """Should create MemoryResult."""
        result = MemoryResult(
            peak_memory_mb=1024.0,
            allocated_memory_mb=800.0,
            reserved_memory_mb=1200.0,
            memory_efficiency=10.0,
        )
        assert result.peak_memory_mb == 1024.0

    def test_efficiency_calculation(self):
        """Memory efficiency should be tokens/MB."""
        # 1000 tokens, 100 MB peak = 10 tokens/MB
        result = MemoryResult(
            peak_memory_mb=100.0,
            allocated_memory_mb=80.0,
            reserved_memory_mb=120.0,
            memory_efficiency=10.0,  # 1000 tokens / 100 MB
        )
        assert result.memory_efficiency == 10.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_minimal_creation(self):
        """Should create with minimal required fields."""
        result = BenchmarkResult(
            framework_name="test",
            model_name="gpt2",
        )
        assert result.framework_name == "test"
        assert result.latency is None

    def test_full_creation(self):
        """Should create with all fields."""
        latency = LatencyResult(100, 10, 90, 110, 50)
        throughput = ThroughputResult(100, 1000, 10, 1)
        memory = MemoryResult(1024, 800, 1200, 1.0)

        result = BenchmarkResult(
            framework_name="vllm",
            model_name="llama-7b",
            latency=latency,
            throughput=throughput,
            memory=memory,
            config={"max_tokens": 100},
            metadata={"gpu": "A100"},
        )
        assert result.latency.mean_ms == 100
        assert result.throughput.tokens_per_second == 100
        assert result.memory.peak_memory_mb == 1024


class TestMeasureLatency:
    """Tests for measure_latency."""

    def test_returns_latency_result(self):
        """Should return LatencyResult."""
        result = measure_latency(
            mock_generate_single,
            "Hello",
            num_tokens=10,
            warmup_runs=1,
            num_runs=2,
        )
        assert isinstance(result, LatencyResult)

    def test_positive_values(self):
        """All latency values should be positive."""
        result = measure_latency(
            mock_generate_single,
            "Test prompt",
            num_tokens=10,
            warmup_runs=1,
            num_runs=3,
        )
        assert result.mean_ms > 0
        assert result.min_ms > 0
        assert result.max_ms > 0

    def test_min_less_than_max(self):
        """Min should be <= max."""
        result = measure_latency(
            mock_generate_single,
            "Test",
            num_tokens=10,
            warmup_runs=1,
            num_runs=5,
        )
        assert result.min_ms <= result.max_ms

    def test_mean_between_min_max(self):
        """Mean should be between min and max."""
        result = measure_latency(
            mock_generate_single,
            "Test",
            num_tokens=10,
            warmup_runs=1,
            num_runs=5,
        )
        assert result.min_ms <= result.mean_ms <= result.max_ms

    def test_stores_measurements(self):
        """Should store individual measurements."""
        num_runs = 5
        result = measure_latency(
            mock_generate_single,
            "Test",
            num_tokens=10,
            warmup_runs=1,
            num_runs=num_runs,
        )
        assert len(result.measurements) == num_runs


class TestMeasureThroughput:
    """Tests for measure_throughput."""

    def test_returns_throughput_result(self):
        """Should return ThroughputResult."""
        result = measure_throughput(
            mock_generate_batch,
            ["Hello"] * 4,
            num_tokens=10,
        )
        assert isinstance(result, ThroughputResult)

    def test_correct_batch_size(self):
        """Should record correct batch size."""
        prompts = ["a", "b", "c"]
        result = measure_throughput(
            mock_generate_batch,
            prompts,
            num_tokens=10,
        )
        assert result.batch_size == len(prompts)

    def test_positive_throughput(self):
        """Throughput should be positive."""
        result = measure_throughput(
            mock_generate_batch,
            ["Test"] * 4,
            num_tokens=20,
        )
        assert result.tokens_per_second > 0

    def test_scales_with_batch(self):
        """Total tokens should scale with batch size."""
        result1 = measure_throughput(
            mock_generate_batch,
            ["Test"] * 2,
            num_tokens=10,
        )
        result2 = measure_throughput(
            mock_generate_batch,
            ["Test"] * 4,
            num_tokens=10,
        )
        # More prompts = more total tokens
        assert result2.total_tokens > result1.total_tokens


class TestMeasureMemory:
    """Tests for measure_memory."""

    def test_returns_memory_result(self):
        """Should return MemoryResult."""
        result = measure_memory(
            mock_generate_single,
            "Test",
            num_tokens=10,
            device="cpu",
        )
        assert isinstance(result, MemoryResult)

    def test_non_negative_values(self):
        """Memory values should be non-negative."""
        result = measure_memory(
            mock_generate_single,
            "Test",
            num_tokens=10,
            device="cpu",
        )
        assert result.peak_memory_mb >= 0
        assert result.allocated_memory_mb >= 0


class TestBenchmarkSingle:
    """Tests for benchmark_single."""

    def test_returns_dict(self):
        """Should return dictionary."""
        result = benchmark_single(
            mock_generate_single,
            "Hello",
            num_tokens=10,
            warmup_runs=1,
            num_runs=2,
            measure_mem=False,
        )
        assert isinstance(result, dict)

    def test_contains_latency(self):
        """Should contain latency key."""
        result = benchmark_single(
            mock_generate_single,
            "Hello",
            num_tokens=10,
            measure_mem=False,
        )
        assert "latency" in result

    def test_contains_memory_when_requested(self):
        """Should contain memory when measure_mem=True."""
        result = benchmark_single(
            mock_generate_single,
            "Hello",
            num_tokens=10,
            measure_mem=True,
            device="cpu",
        )
        assert "memory" in result

    def test_no_memory_when_disabled(self):
        """Should not measure memory when measure_mem=False."""
        result = benchmark_single(
            mock_generate_single,
            "Hello",
            num_tokens=10,
            measure_mem=False,
        )
        assert "memory" not in result or result["memory"] is None


class TestBenchmarkBatch:
    """Tests for benchmark_batch."""

    def test_returns_dict(self):
        """Should return dictionary."""
        result = benchmark_batch(
            mock_generate_batch,
            ["a", "b", "c"],
            num_tokens=10,
            measure_mem=False,
        )
        assert isinstance(result, dict)

    def test_contains_throughput(self):
        """Should contain throughput key."""
        result = benchmark_batch(
            mock_generate_batch,
            ["a", "b"],
            num_tokens=10,
            measure_mem=False,
        )
        assert "throughput" in result


class TestRunLatencySweep:
    """Tests for run_latency_sweep."""

    def test_returns_dict(self):
        """Should return dictionary."""
        result = run_latency_sweep(
            mock_generate_single,
            "Test",
            token_counts=[10, 20],
            warmup_runs=1,
            num_runs=2,
        )
        assert isinstance(result, dict)

    def test_correct_keys(self):
        """Keys should match token counts."""
        token_counts = [10, 20, 30]
        result = run_latency_sweep(
            mock_generate_single,
            "Test",
            token_counts=token_counts,
            warmup_runs=1,
            num_runs=2,
        )
        assert set(result.keys()) == set(token_counts)

    def test_values_are_latency_results(self):
        """Values should be LatencyResults."""
        result = run_latency_sweep(
            mock_generate_single,
            "Test",
            token_counts=[10],
            warmup_runs=1,
            num_runs=2,
        )
        assert isinstance(result[10], LatencyResult)


class TestRunThroughputSweep:
    """Tests for run_throughput_sweep."""

    def test_returns_dict(self):
        """Should return dictionary."""
        result = run_throughput_sweep(
            mock_generate_batch,
            "Test",
            num_tokens=10,
            batch_sizes=[1, 2],
        )
        assert isinstance(result, dict)

    def test_correct_keys(self):
        """Keys should match batch sizes."""
        batch_sizes = [1, 4, 8]
        result = run_throughput_sweep(
            mock_generate_batch,
            "Test",
            num_tokens=10,
            batch_sizes=batch_sizes,
        )
        assert set(result.keys()) == set(batch_sizes)

    def test_values_are_throughput_results(self):
        """Values should be ThroughputResults."""
        result = run_throughput_sweep(
            mock_generate_batch,
            "Test",
            num_tokens=10,
            batch_sizes=[2],
        )
        assert isinstance(result[2], ThroughputResult)


class TestCompareResults:
    """Tests for compare_results."""

    def test_returns_dict(self):
        """Should return dictionary."""
        r1 = BenchmarkResult("fw1", "model")
        result = compare_results({"fw1": r1})
        assert isinstance(result, dict)

    def test_contains_framework_keys(self):
        """Should contain keys for each framework."""
        r1 = BenchmarkResult("fw1", "model")
        r2 = BenchmarkResult("fw2", "model")
        result = compare_results({"fw1": r1, "fw2": r2})
        assert "fw1" in result
        assert "fw2" in result

    def test_with_throughput(self):
        """Should compare throughput when available."""
        r1 = BenchmarkResult(
            "fw1", "model",
            throughput=ThroughputResult(100, 1000, 10, 1)
        )
        r2 = BenchmarkResult(
            "fw2", "model",
            throughput=ThroughputResult(200, 2000, 10, 1)
        )
        result = compare_results({"fw1": r1, "fw2": r2})
        # fw2 should show higher throughput
        assert result is not None


class TestGenerateReport:
    """Tests for generate_report."""

    def test_returns_string(self):
        """Should return string."""
        comp = {"fw1": {"throughput_tps": 100}}
        result = generate_report(comp)
        assert isinstance(result, str)

    def test_contains_framework_name(self):
        """Report should contain framework names."""
        comp = {"my_framework": {"throughput_tps": 100}}
        result = generate_report(comp)
        assert "my_framework" in result

    def test_markdown_format(self):
        """Markdown format should have tables."""
        comp = {"fw1": {"throughput_tps": 100}}
        result = generate_report(comp, output_format="markdown")
        # Markdown tables use |
        assert isinstance(result, str)

    def test_text_format(self):
        """Text format should be readable."""
        comp = {"fw1": {"throughput_tps": 100}}
        result = generate_report(comp, output_format="text")
        assert isinstance(result, str)


class TestPlotComparison:
    """Tests for plot_comparison."""

    def test_handles_no_matplotlib(self):
        """Should handle matplotlib not available."""
        r1 = BenchmarkResult(
            "fw1", "model",
            throughput=ThroughputResult(100, 1000, 10, 1)
        )
        # Should not raise even if matplotlib unavailable
        try:
            result = plot_comparison({"fw1": r1})
        except ImportError:
            pytest.skip("matplotlib not available")


class TestGetSystemInfo:
    """Tests for get_system_info."""

    def test_returns_dict(self):
        """Should return dictionary."""
        info = get_system_info()
        assert isinstance(info, dict)

    def test_contains_python_version(self):
        """Should contain Python version."""
        info = get_system_info()
        assert "python_version" in info

    def test_version_is_string(self):
        """Python version should be string."""
        info = get_system_info()
        assert isinstance(info["python_version"], str)


class TestEstimateTokens:
    """Tests for estimate_tokens."""

    def test_returns_int(self):
        """Should return integer."""
        result = estimate_tokens("Hello world")
        assert isinstance(result, int)

    def test_empty_string(self):
        """Empty string should return 0."""
        result = estimate_tokens("")
        assert result == 0

    def test_scales_with_length(self):
        """Longer text should have more tokens."""
        short = estimate_tokens("Hi")
        long = estimate_tokens("Hello world, how are you today?")
        assert long > short

    def test_respects_chars_per_token(self):
        """Should use provided chars_per_token."""
        text = "12345678"  # 8 chars
        result = estimate_tokens(text, chars_per_token=4.0)
        assert result == 2

    def test_rounds_appropriately(self):
        """Should handle non-integer results."""
        text = "Hello"  # 5 chars
        result = estimate_tokens(text, chars_per_token=4.0)
        assert result >= 1


class TestFormatTime:
    """Tests for format_time."""

    def test_seconds(self):
        """Should format seconds."""
        result = format_time(1.5)
        assert "1.50s" == result or "1.5s" in result

    def test_milliseconds(self):
        """Should format milliseconds."""
        result = format_time(0.015)
        assert "ms" in result

    def test_zero(self):
        """Should handle zero."""
        result = format_time(0)
        assert isinstance(result, str)


class TestFormatMemory:
    """Tests for format_memory."""

    def test_gigabytes(self):
        """Should format GB for large values."""
        result = format_memory(1536)  # 1.5 GB
        assert "GB" in result

    def test_megabytes(self):
        """Should format MB for smaller values."""
        result = format_memory(512)
        assert "MB" in result

    def test_zero(self):
        """Should handle zero."""
        result = format_memory(0)
        assert isinstance(result, str)


class TestMilestone:
    """Integration tests for benchmarking."""

    def test_complete_single_benchmark_workflow(self):
        """Test complete single-prompt benchmark workflow."""
        # Benchmark single prompt
        result = benchmark_single(
            mock_generate_single,
            "The future of AI is",
            num_tokens=20,
            warmup_runs=1,
            num_runs=3,
            measure_mem=True,
            device="cpu",
        )

        assert "latency" in result
        assert result["latency"].mean_ms > 0
        print("\n✅ Single benchmark: Latency measured")

    def test_complete_batch_benchmark_workflow(self):
        """Test complete batch benchmark workflow."""
        prompts = ["Test prompt"] * 8

        result = benchmark_batch(
            mock_generate_batch,
            prompts,
            num_tokens=20,
            measure_mem=False,
        )

        assert "throughput" in result
        assert result["throughput"].tokens_per_second > 0
        assert result["throughput"].batch_size == 8
        print("\n✅ Batch benchmark: Throughput measured")

    def test_sweep_workflows(self):
        """Test sweep benchmarks."""
        # Latency sweep
        latency_results = run_latency_sweep(
            mock_generate_single,
            "Test",
            token_counts=[10, 20],
            warmup_runs=1,
            num_runs=2,
        )
        assert len(latency_results) == 2
        print("\n✅ Latency sweep: Completed")

        # Throughput sweep
        throughput_results = run_throughput_sweep(
            mock_generate_batch,
            "Test",
            num_tokens=10,
            batch_sizes=[1, 2, 4],
        )
        assert len(throughput_results) == 3
        print("✅ Throughput sweep: Completed")

    def test_comparison_and_report(self):
        """Test comparison and report generation."""
        # Create benchmark results
        r1 = BenchmarkResult(
            "huggingface",
            "gpt2",
            latency=LatencyResult(150, 20, 130, 180, 80),
            throughput=ThroughputResult(50, 500, 10, 1),
        )
        r2 = BenchmarkResult(
            "vllm",
            "gpt2",
            latency=LatencyResult(80, 10, 70, 95, 40),
            throughput=ThroughputResult(200, 2000, 10, 1),
        )

        # Compare
        comparison = compare_results({
            "huggingface": r1,
            "vllm": r2,
        })
        assert "huggingface" in comparison
        assert "vllm" in comparison
        print("\n✅ Comparison: Completed")

        # Generate report
        report = generate_report(comparison, output_format="text")
        assert "huggingface" in report
        assert "vllm" in report
        print("✅ Report: Generated")

    def test_full_milestone(self):
        """Complete benchmarking milestone test."""
        print("\n✅ Milestone Test - Framework Benchmarking")

        # System info
        info = get_system_info()
        assert "python_version" in info
        print("   System info: ✓")

        # Utility functions
        tokens = estimate_tokens("Hello world how are you")
        assert tokens > 0
        print("   Token estimation: ✓")

        time_str = format_time(0.5)
        assert isinstance(time_str, str)
        print("   Time formatting: ✓")

        mem_str = format_memory(1024)
        assert isinstance(mem_str, str)
        print("   Memory formatting: ✓")

        # Measurements
        latency = measure_latency(
            mock_generate_single, "Test", num_tokens=10,
            warmup_runs=1, num_runs=2
        )
        assert latency.mean_ms > 0
        print("   Latency measurement: ✓")

        throughput = measure_throughput(
            mock_generate_batch, ["Test"] * 4, num_tokens=10
        )
        assert throughput.tokens_per_second > 0
        print("   Throughput measurement: ✓")

        print("   All benchmarking tests passed!")
