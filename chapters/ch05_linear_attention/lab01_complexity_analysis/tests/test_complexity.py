"""Tests for Lab 01: Complexity Analysis."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from complexity import (
    standard_attention,
    measure_attention_time,
    measure_attention_memory,
    fit_complexity_curve,
    find_max_seq_length,
    benchmark_attention_scaling,
)


class TestStandardAttention:
    """Tests for standard_attention function."""

    def test_output_shape_2d(self):
        """Output should have correct shape for 2D input."""
        seq_len, d_model = 10, 64
        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        output = standard_attention(Q, K, V)

        assert output.shape == (seq_len, d_model)

    def test_output_shape_3d(self):
        """Output should have correct shape for 3D (batched) input."""
        batch, seq_len, d_model = 2, 10, 64
        Q = np.random.randn(batch, seq_len, d_model)
        K = np.random.randn(batch, seq_len, d_model)
        V = np.random.randn(batch, seq_len, d_model)

        output = standard_attention(Q, K, V)

        assert output.shape == (batch, seq_len, d_model)

    def test_output_not_nan(self):
        """Output should not contain NaN values."""
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 64)

        output = standard_attention(Q, K, V)

        assert not np.any(np.isnan(output))

    def test_attention_is_weighted_sum(self):
        """Output should be a weighted sum of values."""
        # With identity attention, output should equal V
        seq_len, d_model = 5, 8

        # Make Q and K such that attention is uniform
        Q = np.ones((seq_len, d_model))
        K = np.ones((seq_len, d_model))
        V = np.random.randn(seq_len, d_model)

        output = standard_attention(Q, K, V)

        # With uniform attention, each output is mean of all V rows
        expected = V.mean(axis=0, keepdims=True).repeat(seq_len, axis=0)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_scaling_applied(self):
        """Scaling by sqrt(d_k) should be applied."""
        np.random.seed(42)
        d_k = 64
        Q = np.random.randn(10, d_k)
        K = np.random.randn(10, d_k)
        V = np.random.randn(10, d_k)

        output = standard_attention(Q, K, V)

        # Without scaling, attention would be too peaked
        # Check output is reasonable (not all same row)
        row_variance = np.var(output, axis=0).mean()
        assert row_variance > 0.01, "Output seems too uniform, check scaling"


class TestMeasureTime:
    """Tests for measure_attention_time function."""

    def test_returns_tuple(self):
        """Should return (mean, std) tuple."""
        result = measure_attention_time(64, 32, num_runs=3)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_positive_times(self):
        """Times should be positive."""
        mean_time, std_time = measure_attention_time(64, 32, num_runs=3)

        assert mean_time > 0
        assert std_time >= 0

    def test_longer_seq_takes_longer(self):
        """Longer sequences should take more time."""
        time_short, _ = measure_attention_time(64, 32, num_runs=3)
        time_long, _ = measure_attention_time(256, 32, num_runs=3)

        assert time_long > time_short

    def test_reasonable_std(self):
        """Standard deviation should be reasonable (not larger than mean)."""
        mean_time, std_time = measure_attention_time(128, 32, num_runs=5)

        # Std shouldn't be more than the mean (that would indicate problems)
        assert std_time < mean_time * 2


class TestMeasureMemory:
    """Tests for measure_attention_memory function."""

    def test_memory_calculation(self):
        """Memory should be seq_len² × 4 bytes."""
        seq_len = 100
        memory = measure_attention_memory(seq_len, 64)

        expected = seq_len * seq_len * 4  # float32 = 4 bytes
        assert memory == expected

    def test_quadratic_scaling(self):
        """Memory should scale quadratically."""
        mem_100 = measure_attention_memory(100, 64)
        mem_200 = measure_attention_memory(200, 64)

        # Doubling seq_len should quadruple memory
        ratio = mem_200 / mem_100
        np.testing.assert_allclose(ratio, 4.0, rtol=0.01)

    def test_known_values(self):
        """Test against known memory values."""
        # 1024² × 4 = 4,194,304 bytes = 4 MB
        memory = measure_attention_memory(1024, 64)
        assert memory == 1024 * 1024 * 4

        # 4096² × 4 = 67,108,864 bytes = 64 MB
        memory = measure_attention_memory(4096, 64)
        assert memory == 4096 * 4096 * 4


class TestFitComplexity:
    """Tests for fit_complexity_curve function."""

    def test_perfect_quadratic(self):
        """Should detect O(n²) for perfect quadratic data."""
        seq_lengths = [100, 200, 400, 800]
        times = [1.0, 4.0, 16.0, 64.0]  # Perfect n²

        exponent = fit_complexity_curve(seq_lengths, times)

        np.testing.assert_allclose(exponent, 2.0, atol=0.1)

    def test_perfect_linear(self):
        """Should detect O(n) for perfect linear data."""
        seq_lengths = [100, 200, 400, 800]
        times = [1.0, 2.0, 4.0, 8.0]  # Perfect n

        exponent = fit_complexity_curve(seq_lengths, times)

        np.testing.assert_allclose(exponent, 1.0, atol=0.1)

    def test_noisy_quadratic(self):
        """Should approximately detect O(n²) with noisy data."""
        np.random.seed(42)
        seq_lengths = [100, 200, 400, 800, 1600]
        # Quadratic with some noise
        times = [(n/100)**2 * (1 + 0.1 * np.random.randn())
                 for n in seq_lengths]

        exponent = fit_complexity_curve(seq_lengths, times)

        # Should be close to 2.0, allow some tolerance for noise
        assert 1.5 < exponent < 2.5

    def test_returns_float(self):
        """Should return a float."""
        seq_lengths = [100, 200, 400]
        times = [1.0, 4.0, 16.0]

        exponent = fit_complexity_curve(seq_lengths, times)

        assert isinstance(exponent, (float, np.floating))


class TestFindMaxSeqLength:
    """Tests for find_max_seq_length function."""

    def test_known_values(self):
        """Test against known memory/seq_len pairs."""
        # 64 MB = 64 × 10^6 bytes
        # seq_len² × 4 = 64 × 10^6
        # seq_len = sqrt(16 × 10^6) = 4000
        max_len = find_max_seq_length(64.0, 64)
        assert max_len == 4000

    def test_small_memory(self):
        """Small memory should give small seq_len."""
        max_len = find_max_seq_length(1.0, 64)  # 1 MB
        # seq_len = sqrt(1e6 / 4) = 500
        assert max_len == 500

    def test_returns_int(self):
        """Should return an integer."""
        max_len = find_max_seq_length(64.0, 64)
        assert isinstance(max_len, (int, np.integer))

    def test_memory_calculation_correct(self):
        """Returned seq_len should fit in memory budget."""
        memory_limit_mb = 16.0
        max_len = find_max_seq_length(memory_limit_mb, 64)

        # Memory used should be <= limit
        memory_used = max_len * max_len * 4  # bytes
        memory_used_mb = memory_used / 1e6

        assert memory_used_mb <= memory_limit_mb


class TestBenchmarkScaling:
    """Tests for benchmark_attention_scaling function."""

    def test_returns_dict(self):
        """Should return a dictionary with expected keys."""
        results = benchmark_attention_scaling([64, 128], d_model=32, num_runs=2)

        assert isinstance(results, dict)
        assert 'seq_lengths' in results
        assert 'times_ms' in results
        assert 'times_std' in results
        assert 'memory_mb' in results
        assert 'fitted_exponent' in results

    def test_correct_lengths(self):
        """Output lists should match input lengths."""
        seq_lengths = [64, 128, 256]
        results = benchmark_attention_scaling(seq_lengths, d_model=32, num_runs=2)

        assert len(results['seq_lengths']) == len(seq_lengths)
        assert len(results['times_ms']) == len(seq_lengths)
        assert len(results['memory_mb']) == len(seq_lengths)

    def test_times_increase(self):
        """Times should increase with sequence length."""
        results = benchmark_attention_scaling([64, 128, 256], d_model=32, num_runs=2)

        times = results['times_ms']
        assert times[1] > times[0]
        assert times[2] > times[1]

    def test_exponent_near_quadratic(self):
        """Fitted exponent should be close to 2.0."""
        results = benchmark_attention_scaling(
            [64, 128, 256, 512],
            d_model=32,
            num_runs=3
        )

        exponent = results['fitted_exponent']
        # Allow some tolerance due to overhead and noise
        assert 1.5 < exponent < 2.5, f"Exponent {exponent} not close to 2.0"


class TestComplexityMilestone:
    """
    Milestone test: Verify understanding of O(n²) complexity.
    """

    def test_quadratic_complexity_understood(self):
        """
        MILESTONE: Demonstrate that attention is O(n²).

        This test verifies that:
        1. Your attention implementation works correctly
        2. You can measure its complexity
        3. The measured complexity is indeed O(n²)
        """
        # Run benchmark
        results = benchmark_attention_scaling(
            [128, 256, 512, 1024],
            d_model=64,
            num_runs=3
        )

        # Verify O(n²) complexity
        exponent = results['fitted_exponent']
        assert 1.7 < exponent < 2.3, (
            f"Measured exponent is {exponent:.2f}, expected ~2.0 for O(n²)"
        )

        # Verify memory scaling
        mem_128 = results['memory_mb'][0]
        mem_1024 = results['memory_mb'][3]
        ratio = mem_1024 / mem_128

        # 1024/128 = 8, so memory should scale by 8² = 64
        assert 50 < ratio < 80, (
            f"Memory ratio is {ratio:.1f}, expected ~64 for O(n²)"
        )

        print(f"\n{'='*60}")
        print("MILESTONE: O(n²) Complexity Understood!")
        print(f"Fitted exponent: {exponent:.2f} (expected ~2.0)")
        print(f"Memory scaling: {ratio:.1f}x (expected ~64x)")
        print(f"{'='*60}\n")
