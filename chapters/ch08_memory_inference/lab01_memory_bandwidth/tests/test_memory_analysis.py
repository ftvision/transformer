"""Tests for Lab 01: Memory Bandwidth Analysis."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_analysis import (
    calculate_model_memory,
    calculate_max_tokens_per_second,
    calculate_arithmetic_intensity,
    is_memory_bound,
    calculate_kv_cache_memory,
    calculate_attention_flops,
    analyze_inference_bottleneck,
)


class TestModelMemory:
    """Tests for calculate_model_memory."""

    def test_7b_fp16(self):
        """7B model in fp16 should be 14 GB."""
        memory = calculate_model_memory(7e9, dtype_bytes=2)
        assert memory == 14e9, f"Expected 14e9, got {memory}"

    def test_7b_int8(self):
        """7B model in int8 should be 7 GB."""
        memory = calculate_model_memory(7e9, dtype_bytes=1)
        assert memory == 7e9, f"Expected 7e9, got {memory}"

    def test_7b_int4(self):
        """7B model in int4 should be 3.5 GB."""
        memory = calculate_model_memory(7e9, dtype_bytes=0.5)
        assert memory == 3.5e9, f"Expected 3.5e9, got {memory}"

    def test_70b_fp16(self):
        """70B model in fp16 should be 140 GB."""
        memory = calculate_model_memory(70e9, dtype_bytes=2)
        assert memory == 140e9, f"Expected 140e9, got {memory}"

    def test_fp32(self):
        """Test fp32 precision."""
        memory = calculate_model_memory(1e9, dtype_bytes=4)
        assert memory == 4e9, f"Expected 4e9, got {memory}"

    def test_default_dtype(self):
        """Default dtype should be fp16 (2 bytes)."""
        memory = calculate_model_memory(1e9)
        assert memory == 2e9, "Default should be fp16 (2 bytes)"


class TestMaxTokensPerSecond:
    """Tests for calculate_max_tokens_per_second."""

    def test_7b_on_a100(self):
        """7B fp16 model on A100 (2 TB/s) should get ~143 tokens/sec."""
        model_memory = 14e9  # 7B in fp16
        bandwidth = 2e12  # A100 bandwidth
        tps = calculate_max_tokens_per_second(model_memory, bandwidth)
        assert abs(tps - 142.857) < 1, f"Expected ~143, got {tps}"

    def test_7b_on_h100(self):
        """7B fp16 model on H100 (3.35 TB/s) should get ~239 tokens/sec."""
        model_memory = 14e9
        bandwidth = 3.35e12  # H100 bandwidth
        tps = calculate_max_tokens_per_second(model_memory, bandwidth)
        assert abs(tps - 239.286) < 1, f"Expected ~239, got {tps}"

    def test_smaller_model_faster(self):
        """Smaller models should achieve higher throughput."""
        bandwidth = 2e12
        tps_7b = calculate_max_tokens_per_second(14e9, bandwidth)
        tps_3b = calculate_max_tokens_per_second(6e9, bandwidth)
        assert tps_3b > tps_7b, "Smaller model should be faster"

    def test_quantized_model_faster(self):
        """Quantized model (int8) should be faster than fp16."""
        bandwidth = 2e12
        tps_fp16 = calculate_max_tokens_per_second(14e9, bandwidth)
        tps_int8 = calculate_max_tokens_per_second(7e9, bandwidth)
        assert tps_int8 == 2 * tps_fp16, "INT8 should be 2x faster than fp16"


class TestArithmeticIntensity:
    """Tests for calculate_arithmetic_intensity."""

    def test_basic_calculation(self):
        """Basic arithmetic intensity calculation."""
        intensity = calculate_arithmetic_intensity(1000, 100)
        assert intensity == 10.0, f"Expected 10.0, got {intensity}"

    def test_low_intensity(self):
        """Low intensity typical of memory-bound operations."""
        intensity = calculate_arithmetic_intensity(14e9, 14e9)
        assert intensity == 1.0, f"Expected 1.0, got {intensity}"

    def test_high_intensity(self):
        """High intensity typical of compute-bound operations."""
        intensity = calculate_arithmetic_intensity(1e12, 1e9)
        assert intensity == 1000.0, f"Expected 1000.0, got {intensity}"

    def test_fractional_intensity(self):
        """Intensity can be fractional."""
        intensity = calculate_arithmetic_intensity(100, 200)
        assert intensity == 0.5, f"Expected 0.5, got {intensity}"


class TestIsMemoryBound:
    """Tests for is_memory_bound."""

    def test_low_intensity_memory_bound(self):
        """Low arithmetic intensity should be memory-bound."""
        # A100: 312 TFLOPS, 2 TB/s → ridge = 156
        result = is_memory_bound(1.0, 312e12, 2e12)
        assert result is True, "Intensity 1 should be memory-bound on A100"

    def test_high_intensity_compute_bound(self):
        """High arithmetic intensity should be compute-bound."""
        result = is_memory_bound(200.0, 312e12, 2e12)
        assert result is False, "Intensity 200 should be compute-bound on A100"

    def test_at_ridge_point(self):
        """At exactly the ridge point, should be compute-bound (or borderline)."""
        ridge = 312e12 / 2e12  # = 156
        result = is_memory_bound(ridge, 312e12, 2e12)
        # At ridge point, technically not memory-bound anymore
        assert result is False, "At ridge point should be compute-bound"

    def test_just_below_ridge(self):
        """Just below ridge point should be memory-bound."""
        ridge = 312e12 / 2e12
        result = is_memory_bound(ridge - 1, 312e12, 2e12)
        assert result is True, "Below ridge should be memory-bound"

    def test_single_token_inference(self):
        """Single token inference has very low intensity, always memory-bound."""
        # Approximation: ~2 FLOPS per weight for forward pass
        intensity = 2.0  # typical for single-token decode
        result = is_memory_bound(intensity, 312e12, 2e12)
        assert result is True, "Single-token inference should be memory-bound"


class TestKVCacheMemory:
    """Tests for calculate_kv_cache_memory."""

    def test_llama2_7b_single_sequence(self):
        """Llama-2-7B: single sequence, 2048 tokens."""
        # 32 layers, d_model=4096
        memory = calculate_kv_cache_memory(1, 2048, 32, 4096, 2)
        expected = 1 * 2048 * 32 * 2 * 4096 * 2  # 1 GB
        assert memory == expected, f"Expected {expected}, got {memory}"

    def test_batch_scaling(self):
        """KV-cache should scale linearly with batch size."""
        memory_1 = calculate_kv_cache_memory(1, 2048, 32, 4096, 2)
        memory_8 = calculate_kv_cache_memory(8, 2048, 32, 4096, 2)
        assert memory_8 == 8 * memory_1, "Should scale linearly with batch"

    def test_seq_len_scaling(self):
        """KV-cache should scale linearly with sequence length."""
        memory_1k = calculate_kv_cache_memory(1, 1024, 32, 4096, 2)
        memory_4k = calculate_kv_cache_memory(1, 4096, 32, 4096, 2)
        assert memory_4k == 4 * memory_1k, "Should scale linearly with seq_len"

    def test_quantized_kv_cache(self):
        """INT8 KV-cache should be half the size of fp16."""
        memory_fp16 = calculate_kv_cache_memory(1, 2048, 32, 4096, dtype_bytes=2)
        memory_int8 = calculate_kv_cache_memory(1, 2048, 32, 4096, dtype_bytes=1)
        assert memory_int8 == memory_fp16 / 2, "INT8 should be half of fp16"

    def test_larger_model(self):
        """Larger models have larger KV-cache per token."""
        # Smaller model: 12 layers, d_model=768
        memory_small = calculate_kv_cache_memory(1, 1024, 12, 768, 2)
        # Larger model: 32 layers, d_model=4096
        memory_large = calculate_kv_cache_memory(1, 1024, 32, 4096, 2)
        assert memory_large > memory_small, "Larger model should have larger KV-cache"


class TestAttentionFlops:
    """Tests for calculate_attention_flops."""

    def test_single_token_decode(self):
        """Single token attending to 1024 keys."""
        flops = calculate_attention_flops(1, 1, 1024, 4096, 32)
        # Should be relatively small
        assert flops > 0, "FLOPs should be positive"
        assert flops < 1e9, "Single token decode should be < 1B FLOPs"

    def test_prefill_larger_than_decode(self):
        """Prefill should have much more FLOPs than decode."""
        decode_flops = calculate_attention_flops(1, 1, 512, 4096, 32)
        prefill_flops = calculate_attention_flops(1, 512, 512, 4096, 32)
        # Prefill has seq_len^2 scaling
        assert prefill_flops > 100 * decode_flops, "Prefill should be >> decode"

    def test_batch_scaling(self):
        """FLOPs should scale linearly with batch size."""
        flops_1 = calculate_attention_flops(1, 10, 100, 4096, 32)
        flops_4 = calculate_attention_flops(4, 10, 100, 4096, 32)
        assert flops_4 == 4 * flops_1, "Should scale linearly with batch"

    def test_quadratic_seq_len_scaling(self):
        """FLOPs should scale quadratically with sequence length (self-attention)."""
        flops_100 = calculate_attention_flops(1, 100, 100, 4096, 32)
        flops_200 = calculate_attention_flops(1, 200, 200, 4096, 32)
        # 200^2 / 100^2 = 4
        ratio = flops_200 / flops_100
        assert 3.5 < ratio < 4.5, f"Should scale ~4x, got {ratio}x"


class TestAnalyzeInferenceBottleneck:
    """Tests for analyze_inference_bottleneck."""

    @pytest.fixture
    def llama_7b_config(self):
        """Llama-2-7B model configuration."""
        return {
            'num_params': 7e9,
            'num_layers': 32,
            'd_model': 4096,
            'num_heads': 32,
            'dtype_bytes': 2
        }

    @pytest.fixture
    def a100_config(self):
        """A100 GPU configuration."""
        return {
            'bandwidth_bytes_per_sec': 2e12,
            'compute_flops': 312e12,
            'memory_bytes': 80e9
        }

    def test_returns_all_keys(self, llama_7b_config, a100_config):
        """Should return all expected keys."""
        result = analyze_inference_bottleneck(llama_7b_config, a100_config)
        expected_keys = [
            'model_memory_bytes',
            'max_tokens_per_sec',
            'kv_cache_memory_bytes',
            'total_memory_bytes',
            'fits_in_memory',
            'arithmetic_intensity',
            'is_memory_bound',
            'ridge_point'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_7b_on_a100_memory_bound(self, llama_7b_config, a100_config):
        """7B model on A100 should be memory-bound."""
        result = analyze_inference_bottleneck(llama_7b_config, a100_config)
        assert result['is_memory_bound'] is True

    def test_7b_fits_in_a100(self, llama_7b_config, a100_config):
        """7B model should fit in A100 80GB."""
        result = analyze_inference_bottleneck(llama_7b_config, a100_config)
        assert result['fits_in_memory'] is True
        assert result['model_memory_bytes'] < 80e9

    def test_large_batch_memory(self, llama_7b_config, a100_config):
        """Large batch + long sequence should increase memory usage."""
        result = analyze_inference_bottleneck(
            llama_7b_config, a100_config,
            batch_size=16, seq_len=4096
        )
        # Model (14GB) + KV-cache (16 * 4096 * 32 * 2 * 4096 * 2 = 32GB)
        assert result['total_memory_bytes'] > 40e9

    def test_max_tokens_per_sec_reasonable(self, llama_7b_config, a100_config):
        """Max tokens/sec should be in reasonable range for 7B on A100."""
        result = analyze_inference_bottleneck(llama_7b_config, a100_config)
        # Should be around 143 tokens/sec
        assert 100 < result['max_tokens_per_sec'] < 200

    def test_ridge_point_calculation(self, llama_7b_config, a100_config):
        """Ridge point should be compute/bandwidth."""
        result = analyze_inference_bottleneck(llama_7b_config, a100_config)
        expected_ridge = 312e12 / 2e12  # = 156
        assert abs(result['ridge_point'] - expected_ridge) < 1


class TestMilestone:
    """Integration test for the complete analysis."""

    def test_full_analysis_7b(self):
        """Complete analysis for Llama-2-7B on A100."""
        model_config = {
            'num_params': 7e9,
            'num_layers': 32,
            'd_model': 4096,
            'num_heads': 32,
            'dtype_bytes': 2
        }
        gpu_config = {
            'bandwidth_bytes_per_sec': 2e12,
            'compute_flops': 312e12,
            'memory_bytes': 80e9
        }

        result = analyze_inference_bottleneck(
            model_config, gpu_config,
            batch_size=1, seq_len=2048
        )

        # Verify key properties
        assert result['model_memory_bytes'] == 14e9, "Model should be 14 GB"
        assert result['is_memory_bound'] is True, "Should be memory-bound"
        assert result['fits_in_memory'] is True, "Should fit in 80 GB"
        assert 100 < result['max_tokens_per_sec'] < 200, "Reasonable throughput"

        print("\n✅ Milestone Test Passed!")
        print(f"   Model memory: {result['model_memory_bytes'] / 1e9:.1f} GB")
        print(f"   KV-cache memory: {result['kv_cache_memory_bytes'] / 1e9:.2f} GB")
        print(f"   Max tokens/sec: {result['max_tokens_per_sec']:.1f}")
        print(f"   Memory-bound: {result['is_memory_bound']}")
        print(f"   Ridge point: {result['ridge_point']:.1f} FLOPS/byte")
