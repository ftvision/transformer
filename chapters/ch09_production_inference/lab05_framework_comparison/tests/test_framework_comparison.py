"""Tests for Lab 05: Framework Comparison."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from framework_comparison import (
    Framework,
    HardwareConfig,
    WorkloadConfig,
    FrameworkCapabilities,
    FRAMEWORK_SPECS,
    get_compatible_frameworks,
    rank_frameworks,
    estimate_throughput,
    estimate_memory_requirement,
    generate_deployment_recommendation,
    compare_frameworks_for_scenario,
    simulate_benchmark,
    format_recommendation_report,
    BenchmarkResult,
)


class TestGetCompatibleFrameworks:
    """Tests for framework compatibility checking."""

    def test_no_gpu_excludes_gpu_only(self):
        """Without GPU, GPU-only frameworks should be excluded."""
        hardware = HardwareConfig(has_gpu=False, cpu_memory_gb=32)
        compatible = get_compatible_frameworks(hardware)

        # vLLM, SGLang, TensorRT-LLM require GPU
        assert Framework.VLLM not in compatible
        assert Framework.SGLANG not in compatible
        assert Framework.TENSORRT_LLM not in compatible

        # HuggingFace and llama.cpp work on CPU
        assert Framework.HUGGINGFACE in compatible
        assert Framework.LLAMA_CPP in compatible

    def test_with_gpu_includes_gpu_frameworks(self):
        """With sufficient GPU, GPU frameworks should be available."""
        hardware = HardwareConfig(
            has_gpu=True,
            gpu_memory_gb=24,
            gpu_type="nvidia"
        )
        compatible = get_compatible_frameworks(hardware)

        assert Framework.VLLM in compatible
        assert Framework.SGLANG in compatible
        assert Framework.HUGGINGFACE in compatible

    def test_low_gpu_memory_limits_options(self):
        """Low GPU memory should exclude memory-hungry frameworks."""
        hardware = HardwareConfig(
            has_gpu=True,
            gpu_memory_gb=6,  # Less than vLLM's 8GB minimum
            gpu_type="nvidia"
        )
        compatible = get_compatible_frameworks(hardware)

        assert Framework.VLLM not in compatible  # Needs 8GB+
        assert Framework.TENSORRT_LLM not in compatible  # Needs 16GB+

    def test_apple_silicon_excludes_tensorrt(self):
        """Apple Silicon should exclude TensorRT-LLM (NVIDIA only)."""
        hardware = HardwareConfig(
            has_gpu=True,
            gpu_memory_gb=32,
            gpu_type="apple_silicon"
        )
        compatible = get_compatible_frameworks(hardware)

        assert Framework.TENSORRT_LLM not in compatible
        assert Framework.LLAMA_CPP in compatible  # Works well on Apple


class TestRankFrameworks:
    """Tests for framework ranking."""

    def test_returns_sorted_list(self):
        """Should return frameworks sorted by score."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=100,
            avg_prompt_tokens=100,
            avg_output_tokens=200,
            throughput_priority=True
        )

        rankings = rank_frameworks(hardware, workload)

        # Should be sorted by score descending
        scores = [r['score'] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_structured_output_prefers_sglang(self):
        """When structured output required, SGLang should rank higher."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=10,
            avg_prompt_tokens=100,
            avg_output_tokens=50,
            requires_structured_output=True
        )

        rankings = rank_frameworks(hardware, workload)

        # SGLang should be top or near top
        sglang_rank = next(
            i for i, r in enumerate(rankings)
            if r['framework'] == Framework.SGLANG
        )
        assert sglang_rank < 2  # Top 2

    def test_includes_reasoning(self):
        """Rankings should include reasoning."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rankings = rank_frameworks(hardware, workload)

        for ranking in rankings:
            assert 'framework' in ranking
            assert 'score' in ranking
            assert 'reasons' in ranking
            assert len(ranking['reasons']) > 0

    def test_cpu_only_ranks_llama_cpp_high(self):
        """For CPU-only, llama.cpp should rank well."""
        hardware = HardwareConfig(has_gpu=False, cpu_memory_gb=32)
        workload = WorkloadConfig(
            concurrent_users=1,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rankings = rank_frameworks(hardware, workload)

        # llama.cpp should be available and ranked
        frameworks = [r['framework'] for r in rankings]
        assert Framework.LLAMA_CPP in frameworks


class TestEstimateThroughput:
    """Tests for throughput estimation."""

    def test_returns_all_metrics(self):
        """Should return all expected metrics."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        metrics = estimate_throughput(Framework.VLLM, hardware, workload)

        assert 'tokens_per_second' in metrics
        assert 'requests_per_second' in metrics
        assert 'time_to_first_token_ms' in metrics
        assert 'batch_efficiency' in metrics

    def test_vllm_higher_throughput_than_hf(self):
        """vLLM should have higher throughput than HuggingFace."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        vllm_metrics = estimate_throughput(Framework.VLLM, hardware, workload)
        hf_metrics = estimate_throughput(Framework.HUGGINGFACE, hardware, workload)

        assert vllm_metrics['tokens_per_second'] > hf_metrics['tokens_per_second']

    def test_batch_efficiency_reasonable(self):
        """Batch efficiency should be between 0 and 1."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        metrics = estimate_throughput(Framework.VLLM, hardware, workload)

        assert 0 < metrics['batch_efficiency'] <= 1


class TestEstimateMemoryRequirement:
    """Tests for memory estimation."""

    def test_returns_all_components(self):
        """Should return all memory components."""
        memory = estimate_memory_requirement(
            model_params_b=7.0,
            framework=Framework.VLLM,
            max_batch_size=32,
            max_seq_len=2048
        )

        assert 'model_memory_gb' in memory
        assert 'kv_cache_memory_gb' in memory
        assert 'activation_memory_gb' in memory
        assert 'total_memory_gb' in memory

    def test_model_memory_scales_with_params(self):
        """Model memory should scale with parameter count."""
        memory_7b = estimate_memory_requirement(7.0, Framework.VLLM, 32)
        memory_13b = estimate_memory_requirement(13.0, Framework.VLLM, 32)

        assert memory_13b['model_memory_gb'] > memory_7b['model_memory_gb']

    def test_quantization_reduces_model_memory(self):
        """Quantization should reduce model memory."""
        memory_fp16 = estimate_memory_requirement(
            7.0, Framework.VLLM, 32, quantization_bits=16
        )
        memory_int4 = estimate_memory_requirement(
            7.0, Framework.VLLM, 32, quantization_bits=4
        )

        assert memory_int4['model_memory_gb'] < memory_fp16['model_memory_gb']

    def test_paged_attention_more_efficient(self):
        """vLLM/SGLang should have lower KV cache memory."""
        memory_vllm = estimate_memory_requirement(7.0, Framework.VLLM, 32)
        memory_hf = estimate_memory_requirement(7.0, Framework.HUGGINGFACE, 32)

        assert memory_vllm['kv_cache_memory_gb'] < memory_hf['kv_cache_memory_gb']


class TestGenerateDeploymentRecommendation:
    """Tests for deployment recommendation generation."""

    def test_returns_complete_recommendation(self):
        """Should return all recommendation components."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rec = generate_deployment_recommendation(hardware, workload)

        assert 'recommended_framework' in rec
        assert 'reasoning' in rec
        assert 'configuration' in rec
        assert 'expected_performance' in rec
        assert 'warnings' in rec
        assert 'alternatives' in rec

    def test_recommends_compatible_framework(self):
        """Recommendation should be a compatible framework."""
        hardware = HardwareConfig(has_gpu=False, cpu_memory_gb=16)
        workload = WorkloadConfig(
            concurrent_users=1,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rec = generate_deployment_recommendation(hardware, workload)

        # Should recommend a CPU-compatible framework
        assert rec['recommended_framework'] in [Framework.HUGGINGFACE, Framework.LLAMA_CPP]

    def test_structured_output_recommends_sglang(self):
        """Structured output requirement should recommend SGLang."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=10,
            avg_prompt_tokens=100,
            avg_output_tokens=50,
            requires_structured_output=True
        )

        rec = generate_deployment_recommendation(hardware, workload)

        assert rec['recommended_framework'] == Framework.SGLANG

    def test_includes_warnings_for_edge_cases(self):
        """Should warn about potential issues."""
        hardware = HardwareConfig(
            has_gpu=True,
            gpu_memory_gb=8,  # Tight on memory
            gpu_type="nvidia"
        )
        workload = WorkloadConfig(
            concurrent_users=100,  # Many users for limited memory
            avg_prompt_tokens=1000,
            avg_output_tokens=1000
        )

        rec = generate_deployment_recommendation(hardware, workload, model_params_b=13.0)

        # Should warn about memory constraints
        assert len(rec['warnings']) > 0


class TestCompareFrameworksForScenario:
    """Tests for framework comparison."""

    def test_compares_all_frameworks(self):
        """Should compare all frameworks."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        comparison = compare_frameworks_for_scenario(hardware, workload)

        # Should have entry for each framework
        for framework in Framework:
            assert framework.value in comparison or framework.name.lower() in str(comparison)

    def test_includes_compatibility_flag(self):
        """Each comparison should indicate compatibility."""
        hardware = HardwareConfig(has_gpu=False, cpu_memory_gb=16)
        workload = WorkloadConfig(
            concurrent_users=1,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        comparison = compare_frameworks_for_scenario(hardware, workload)

        # Check that compatible flag is present
        for framework_name, data in comparison.items():
            assert 'compatible' in data

    def test_includes_pros_and_cons(self):
        """Comparison should include pros and cons."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        comparison = compare_frameworks_for_scenario(hardware, workload)

        for framework_name, data in comparison.items():
            if data.get('compatible', False):
                assert 'pros' in data
                assert 'cons' in data


class TestSimulateBenchmark:
    """Tests for benchmark simulation."""

    def test_returns_benchmark_result(self):
        """Should return a BenchmarkResult."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")

        result = simulate_benchmark(Framework.VLLM, hardware, num_requests=100)

        assert isinstance(result, BenchmarkResult)
        assert result.framework == Framework.VLLM

    def test_positive_metrics(self):
        """All metrics should be positive."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")

        result = simulate_benchmark(Framework.VLLM, hardware)

        assert result.tokens_per_second > 0
        assert result.time_to_first_token_ms > 0
        assert result.requests_completed > 0
        assert result.total_time_seconds > 0
        assert result.memory_peak_gb > 0

    def test_framework_affects_performance(self):
        """Different frameworks should show different performance."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")

        vllm_result = simulate_benchmark(Framework.VLLM, hardware)
        hf_result = simulate_benchmark(Framework.HUGGINGFACE, hardware)

        # vLLM should be faster
        assert vllm_result.tokens_per_second > hf_result.tokens_per_second


class TestFormatRecommendationReport:
    """Tests for report formatting."""

    def test_returns_string(self):
        """Should return a formatted string."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rec = generate_deployment_recommendation(hardware, workload)
        report = format_recommendation_report(rec)

        assert isinstance(report, str)
        assert len(report) > 100  # Should have substantial content

    def test_includes_framework_name(self):
        """Report should include the recommended framework."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rec = generate_deployment_recommendation(hardware, workload)
        report = format_recommendation_report(rec)

        # Should mention the framework
        assert rec['recommended_framework'].name.lower() in report.lower() or \
               rec['recommended_framework'].value in report.lower()

    def test_includes_key_sections(self):
        """Report should include key sections."""
        hardware = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        workload = WorkloadConfig(
            concurrent_users=50,
            avg_prompt_tokens=100,
            avg_output_tokens=200
        )

        rec = generate_deployment_recommendation(hardware, workload)
        report = format_recommendation_report(rec)

        # Should have recognizable sections
        report_lower = report.lower()
        assert 'recommend' in report_lower
        assert 'reason' in report_lower or 'why' in report_lower
        assert 'performance' in report_lower or 'throughput' in report_lower
