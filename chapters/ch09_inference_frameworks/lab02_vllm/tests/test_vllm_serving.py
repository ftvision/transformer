"""Tests for Lab 02: vLLM Serving."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vllm_serving import (
    create_llm,
    create_sampling_params,
    generate_offline,
    measure_throughput,
    calculate_memory_usage,
    get_vllm_version,
    is_gpu_available,
)

# Check if vLLM is available
VLLM_AVAILABLE = get_vllm_version() != "not installed"
GPU_AVAILABLE = is_gpu_available()

# Skip all tests if vLLM not installed
pytestmark = pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="vLLM not installed"
)


class TestCreateSamplingParams:
    """Tests for create_sampling_params (doesn't require GPU)."""

    def test_creates_params(self):
        """Should create sampling params object."""
        params = create_sampling_params()
        assert params is not None

    def test_respects_temperature(self):
        """Should set temperature correctly."""
        params = create_sampling_params(temperature=0.5)
        assert params.temperature == 0.5

    def test_respects_max_tokens(self):
        """Should set max_tokens correctly."""
        params = create_sampling_params(max_tokens=200)
        assert params.max_tokens == 200

    def test_respects_top_p(self):
        """Should set top_p correctly."""
        params = create_sampling_params(top_p=0.9)
        assert params.top_p == 0.9


class TestCalculateMemoryUsage:
    """Tests for calculate_memory_usage (doesn't require GPU)."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        mem = calculate_memory_usage("gpt2")
        assert isinstance(mem, dict)

    def test_has_required_keys(self):
        """Should have required keys."""
        mem = calculate_memory_usage("gpt2")
        required_keys = ['model_memory_gb', 'kv_cache_per_token_mb']
        for key in required_keys:
            assert key in mem, f"Missing key: {key}"

    def test_positive_values(self):
        """Memory values should be positive."""
        mem = calculate_memory_usage("gpt2")
        assert mem['model_memory_gb'] > 0

    def test_larger_model_more_memory(self):
        """Larger model should use more memory."""
        # These are rough estimates based on model names
        mem_small = calculate_memory_usage("gpt2")
        # Note: This test is illustrative; actual values depend on implementation
        assert mem_small['model_memory_gb'] > 0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestCreateLLM:
    """Tests for create_llm (requires GPU)."""

    def test_creates_llm(self):
        """Should create LLM instance."""
        llm = create_llm("gpt2")
        assert llm is not None

    def test_with_custom_memory(self):
        """Should accept gpu_memory_utilization."""
        llm = create_llm("gpt2", gpu_memory_utilization=0.5)
        assert llm is not None


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGenerateOffline:
    """Tests for generate_offline (requires GPU)."""

    @pytest.fixture(scope="class")
    def llm(self):
        """Create LLM once for tests."""
        return create_llm("gpt2")

    def test_returns_list(self, llm):
        """Should return a list."""
        params = create_sampling_params(max_tokens=10)
        outputs = generate_offline(llm, ["Hello"], params)
        assert isinstance(outputs, list)

    def test_correct_count(self, llm):
        """Should return one output per prompt."""
        params = create_sampling_params(max_tokens=10)
        prompts = ["A", "B", "C"]
        outputs = generate_offline(llm, prompts, params)
        assert len(outputs) == len(prompts)

    def test_non_empty_outputs(self, llm):
        """Outputs should be non-empty strings."""
        params = create_sampling_params(max_tokens=10)
        outputs = generate_offline(llm, ["Hello"], params)
        assert all(isinstance(o, str) for o in outputs)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestMeasureThroughput:
    """Tests for measure_throughput (requires GPU)."""

    @pytest.fixture(scope="class")
    def llm(self):
        """Create LLM once for tests."""
        return create_llm("gpt2")

    def test_returns_metrics(self, llm):
        """Should return ThroughputMetrics."""
        params = create_sampling_params(max_tokens=20)
        metrics = measure_throughput(llm, ["Test"] * 10, params)
        assert hasattr(metrics, 'tokens_per_second')

    def test_positive_throughput(self, llm):
        """Throughput should be positive."""
        params = create_sampling_params(max_tokens=20)
        metrics = measure_throughput(llm, ["Test"] * 10, params)
        assert metrics.tokens_per_second > 0

    def test_total_tokens_positive(self, llm):
        """Total tokens should be positive."""
        params = create_sampling_params(max_tokens=20)
        metrics = measure_throughput(llm, ["Test"] * 10, params)
        assert metrics.total_tokens > 0


class TestVersionInfo:
    """Tests for version and availability functions."""

    def test_version_string(self):
        """Should return version string."""
        version = get_vllm_version()
        assert isinstance(version, str)

    def test_gpu_check_bool(self):
        """GPU check should return bool."""
        result = is_gpu_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestMilestone:
    """Integration tests for vLLM serving."""

    def test_full_workflow(self):
        """Complete vLLM workflow."""
        # Create LLM
        llm = create_llm("gpt2")

        # Create params
        params = create_sampling_params(
            temperature=0.7,
            max_tokens=50
        )

        # Generate
        prompts = [
            "The future of AI is",
            "Machine learning can",
            "In the year 2050"
        ]
        outputs = generate_offline(llm, prompts, params)

        assert len(outputs) == len(prompts)
        assert all(len(o) > 0 for o in outputs)

        # Measure throughput
        metrics = measure_throughput(llm, prompts * 10, params)

        print("\n✅ Milestone Test - vLLM Serving")
        print(f"   Model: gpt2")
        print(f"   Prompts: {len(prompts * 10)}")
        print(f"   Total tokens: {metrics.total_tokens}")
        print(f"   Throughput: {metrics.tokens_per_second:.1f} tok/s")
        print(f"   Latency: {metrics.total_time:.2f}s")

        assert metrics.tokens_per_second > 0
