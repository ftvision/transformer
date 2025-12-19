"""Tests for Lab 05: Profiling and Optimization."""

import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profiling import (
    benchmark_function,
    profile_attention_components,
    identify_bottleneck,
    measure_memory_usage,
    attention_baseline,
    attention_optimized,
    compare_optimizations,
    generate_profile_report,
    compute_arithmetic_intensity,
    ProfileResult,
)


# Check if CUDA is available
HAS_CUDA = torch.cuda.is_available()


class TestBenchmarkFunction:
    """Tests for benchmark_function."""

    def test_benchmark_returns_positive_time(self):
        """Benchmark should return positive time."""
        def simple_fn(x):
            return x + 1

        x = torch.randn(100, 100)
        time_s = benchmark_function(simple_fn, (x,), warmup=2, iterations=10, sync=False)

        assert time_s > 0

    def test_benchmark_warmup_runs(self):
        """Warmup should run specified number of times."""
        call_count = [0]

        def counting_fn(x):
            call_count[0] += 1
            return x

        x = torch.randn(10)
        benchmark_function(counting_fn, (x,), warmup=5, iterations=10, sync=False)

        assert call_count[0] == 15  # 5 warmup + 10 iterations

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_benchmark_cuda_sync(self):
        """CUDA benchmark should synchronize properly."""
        def matmul_fn(x):
            return x @ x.T

        x = torch.randn(500, 500, device='cuda')
        time_s = benchmark_function(matmul_fn, (x,), warmup=3, iterations=20, sync=True)

        assert time_s > 0
        # Should be reasonable (less than 1 second for this size)
        assert time_s < 1.0


class TestProfileAttentionComponents:
    """Tests for profile_attention_components."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_profile_returns_results(self):
        """Profiling should return results for each component."""
        results = profile_attention_components(64, 128, 4, device='cuda')

        assert len(results) == 5
        expected_names = [
            'qkv_projection',
            'score_computation',
            'softmax',
            'weighted_sum',
            'output_projection',
        ]
        for r, name in zip(results, expected_names):
            assert r.name == name
            assert r.time_ms >= 0

    def test_profile_cpu_works(self):
        """Profiling should work on CPU."""
        results = profile_attention_components(32, 64, 2, device='cpu')

        assert len(results) == 5
        for r in results:
            assert r.time_ms >= 0


class TestIdentifyBottleneck:
    """Tests for identify_bottleneck."""

    def test_identify_correct_bottleneck(self):
        """Should identify the component with maximum time."""
        results = [
            ProfileResult("a", 1.0),
            ProfileResult("b", 5.0),
            ProfileResult("c", 2.0),
        ]

        name, pct = identify_bottleneck(results)

        assert name == "b"
        assert abs(pct - 62.5) < 0.1  # 5/8 * 100 = 62.5%

    def test_identify_single_component(self):
        """Should handle single component."""
        results = [ProfileResult("only", 10.0)]

        name, pct = identify_bottleneck(results)

        assert name == "only"
        assert pct == 100.0

    def test_identify_tied_components(self):
        """Should handle tied components (return first max)."""
        results = [
            ProfileResult("a", 5.0),
            ProfileResult("b", 5.0),
        ]

        name, pct = identify_bottleneck(results)

        assert name in ["a", "b"]
        assert pct == 50.0


class TestMeasureMemoryUsage:
    """Tests for measure_memory_usage."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_memory_usage_returns_dict(self):
        """Should return dictionary with expected keys."""
        def simple_fn(x):
            return x @ x.T

        x = torch.randn(1000, 1000, device='cuda')
        mem = measure_memory_usage(simple_fn, (x,))

        assert 'peak_memory_mb' in mem
        assert 'current_memory_mb' in mem
        assert 'cached_memory_mb' in mem

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_memory_usage_positive(self):
        """Memory usage should be positive."""
        def alloc_fn(x):
            y = x @ x.T
            z = y @ y
            return z

        x = torch.randn(500, 500, device='cuda')
        mem = measure_memory_usage(alloc_fn, (x,))

        assert mem['peak_memory_mb'] > 0

    def test_memory_usage_cpu_fallback(self):
        """Should return zeros for CPU."""
        def simple_fn(x):
            return x + 1

        x = torch.randn(100)
        mem = measure_memory_usage(simple_fn, (x,), device='cpu')

        # Should not crash, returns zeros for non-CUDA
        assert isinstance(mem, dict)


class TestAttentionOptimized:
    """Tests for attention optimization."""

    def get_test_tensors(self, device='cpu'):
        """Create test tensors for attention."""
        batch, heads, seq, d_k = 2, 4, 32, 16
        Q = torch.randn(batch, heads, seq, d_k, device=device)
        K = torch.randn(batch, heads, seq, d_k, device=device)
        V = torch.randn(batch, heads, seq, d_k, device=device)
        return Q, K, V

    def test_baseline_output_shape(self):
        """Baseline attention should produce correct shape."""
        Q, K, V = self.get_test_tensors()
        output = attention_baseline(Q, K, V)
        assert output.shape == Q.shape

    def test_optimized_level0_matches_baseline(self):
        """Level 0 should match baseline."""
        Q, K, V = self.get_test_tensors()

        baseline = attention_baseline(Q, K, V)
        optimized = attention_optimized(Q, K, V, optimization_level=0)

        assert torch.allclose(baseline, optimized, atol=1e-5)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_optimized_level1_shape(self):
        """Level 1 (SDPA) should produce correct shape."""
        Q, K, V = self.get_test_tensors(device='cuda')
        output = attention_optimized(Q, K, V, optimization_level=1)
        assert output.shape == Q.shape

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_optimized_level1_correctness(self):
        """Level 1 should produce similar results to baseline."""
        Q, K, V = self.get_test_tensors(device='cuda')

        baseline = attention_baseline(Q, K, V)
        optimized = attention_optimized(Q, K, V, optimization_level=1)

        # SDPA may have slightly different numerics
        assert torch.allclose(baseline, optimized, atol=1e-4)


class TestCompareOptimizations:
    """Tests for compare_optimizations."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_compare_returns_dict(self):
        """Should return dictionary with optimization results."""
        results = compare_optimizations(2, 4, 64, 16, device='cuda')

        assert isinstance(results, dict)
        assert 'baseline' in results
        assert 'sdpa' in results

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_compare_positive_times(self):
        """All times should be positive."""
        results = compare_optimizations(2, 4, 64, 16, device='cuda')

        for name, time_ms in results.items():
            if not (time_ms != time_ms):  # Check for NaN
                assert time_ms > 0


class TestGenerateProfileReport:
    """Tests for generate_profile_report."""

    def test_report_is_string(self):
        """Report should be a string."""
        report = generate_profile_report(32, 64, 2, device='cpu')
        assert isinstance(report, str)

    def test_report_contains_components(self):
        """Report should mention all components."""
        report = generate_profile_report(32, 64, 2, device='cpu')

        assert 'qkv_projection' in report
        assert 'softmax' in report
        assert 'Bottleneck' in report

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_report_cuda(self):
        """Report should work on CUDA."""
        report = generate_profile_report(64, 128, 4, device='cuda')
        assert len(report) > 0


class TestArithmeticIntensity:
    """Tests for compute_arithmetic_intensity."""

    def test_matmul_ai(self):
        """Matrix multiply should have known AI."""
        # For square matrices: AI ≈ M/3 (when M=N=K)
        ai = compute_arithmetic_intensity('matmul', {
            'A': (1024, 1024),
            'B': (1024, 1024)
        })

        # AI should be positive and reasonable
        assert ai > 0
        # For 1024x1024: AI ≈ 341 FLOPS/byte
        assert 100 < ai < 500

    def test_softmax_ai(self):
        """Softmax should be memory-bound (low AI)."""
        ai = compute_arithmetic_intensity('softmax', {
            'x': (1024, 1024)
        })

        # Softmax is memory-bound, AI should be low (< 10)
        assert ai > 0
        assert ai < 10

    def test_layernorm_ai(self):
        """LayerNorm should be memory-bound (low AI)."""
        ai = compute_arithmetic_intensity('layernorm', {
            'x': (1024, 768)
        })

        assert ai > 0
        assert ai < 20

    def test_gelu_ai(self):
        """GELU should be memory-bound."""
        ai = compute_arithmetic_intensity('gelu', {
            'x': (1024, 768)
        })

        assert ai > 0
        assert ai < 20


class TestMilestone:
    """Chapter 12 Lab 05 Milestone."""

    def test_profiling_milestone(self):
        """
        MILESTONE: Profiling and optimization tools working correctly.

        This demonstrates understanding of:
        - Proper benchmarking with warmup and sync
        - Component-level profiling
        - Memory measurement
        - Optimization techniques
        - Arithmetic intensity analysis
        """
        # Test benchmarking
        def simple_fn(x):
            return x @ x.T

        x = torch.randn(100, 100)
        time_s = benchmark_function(simple_fn, (x,), warmup=2, iterations=10, sync=False)
        assert time_s > 0

        # Test bottleneck identification
        results = [
            ProfileResult("compute", 10.0),
            ProfileResult("memory", 5.0),
            ProfileResult("other", 2.0),
        ]
        name, pct = identify_bottleneck(results)
        assert name == "compute"
        assert abs(pct - 58.8) < 1.0  # 10/17 * 100

        # Test arithmetic intensity
        ai_matmul = compute_arithmetic_intensity('matmul', {
            'A': (512, 512),
            'B': (512, 512)
        })
        ai_softmax = compute_arithmetic_intensity('softmax', {
            'x': (512, 512)
        })
        # Matmul should have higher AI than softmax
        assert ai_matmul > ai_softmax

        # Test attention optimization
        Q = torch.randn(2, 4, 32, 16)
        K = torch.randn(2, 4, 32, 16)
        V = torch.randn(2, 4, 32, 16)

        baseline = attention_baseline(Q, K, V)
        optimized = attention_optimized(Q, K, V, optimization_level=0)
        assert torch.allclose(baseline, optimized, atol=1e-5)

        # Test report generation
        report = generate_profile_report(32, 64, 2, device='cpu')
        assert isinstance(report, str)
        assert len(report) > 0

        print("\n" + "=" * 60)
        print("Lab 05 Milestone Achieved!")
        print("Profiling and optimization tools working correctly.")
        print("=" * 60)
        print("\nChapter 12 Complete!")
        print("You have learned:")
        print("  - Triton kernel programming")
        print("  - Kernel fusion for attention")
        print("  - JAX fundamentals")
        print("  - JAX transformations (jit, vmap, grad)")
        print("  - Performance profiling and optimization")
        print("=" * 60 + "\n")
