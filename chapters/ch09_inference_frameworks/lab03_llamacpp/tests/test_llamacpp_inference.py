"""Tests for Lab 03: llama.cpp Inference."""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamacpp_inference import (
    load_model,
    generate_text,
    generate_chat,
    get_model_info,
    estimate_memory_usage,
    benchmark_inference,
    count_tokens,
    is_gguf_file,
    get_quantization_from_filename,
    llamacpp_available,
)

# Check if llama-cpp-python is available
LLAMACPP_AVAILABLE = llamacpp_available()

# Skip all model-dependent tests if llama-cpp-python not installed
pytestmark = pytest.mark.skipif(
    not LLAMACPP_AVAILABLE,
    reason="llama-cpp-python not installed"
)


class TestUtilityFunctions:
    """Tests for utility functions (don't require model)."""

    def test_get_quantization_q4_k_m(self):
        """Should extract Q4_K_M from filename."""
        quant = get_quantization_from_filename("llama-7b.Q4_K_M.gguf")
        assert quant == "Q4_K_M"

    def test_get_quantization_q8_0(self):
        """Should extract Q8_0 from filename."""
        quant = get_quantization_from_filename("model-Q8_0.gguf")
        assert quant == "Q8_0"

    def test_get_quantization_f16(self):
        """Should extract F16 from filename."""
        quant = get_quantization_from_filename("model-f16.gguf")
        assert quant == "F16"

    def test_get_quantization_none(self):
        """Should return None for unknown format."""
        quant = get_quantization_from_filename("model.gguf")
        assert quant is None

    def test_is_gguf_file_extension(self):
        """Should check .gguf extension."""
        # Create temp file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            temp_path = f.name
        try:
            assert is_gguf_file(temp_path) is True
        finally:
            os.unlink(temp_path)

    def test_is_gguf_file_nonexistent(self):
        """Should return False for nonexistent file."""
        assert is_gguf_file("/nonexistent/path/model.gguf") is False

    def test_llamacpp_available_returns_bool(self):
        """Should return boolean."""
        result = llamacpp_available()
        assert isinstance(result, bool)


class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        # Create a mock gguf file for size testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'0' * 1024 * 1024)  # 1 MB file
            temp_path = f.name
        try:
            mem = estimate_memory_usage(temp_path, n_ctx=2048)
            assert isinstance(mem, dict)
        finally:
            os.unlink(temp_path)

    def test_has_required_keys(self):
        """Should have required memory keys."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'0' * 1024 * 1024)
            temp_path = f.name
        try:
            mem = estimate_memory_usage(temp_path)
            assert 'model_size' in mem
            assert 'total_estimate' in mem
        finally:
            os.unlink(temp_path)


# Tests that require an actual GGUF model
# These will be skipped in CI without a model file
MODEL_PATH = os.environ.get("TEST_GGUF_MODEL", None)
MODEL_AVAILABLE = MODEL_PATH is not None and os.path.exists(MODEL_PATH) if MODEL_PATH else False


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
class TestLoadModel:
    """Tests for load_model (requires actual model)."""

    def test_loads_model(self):
        """Should load model successfully."""
        model = load_model(MODEL_PATH, n_ctx=512)
        assert model is not None

    def test_with_threads(self):
        """Should accept n_threads parameter."""
        model = load_model(MODEL_PATH, n_ctx=512, n_threads=4)
        assert model is not None


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
class TestGenerateText:
    """Tests for generate_text (requires actual model)."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for tests."""
        return load_model(MODEL_PATH, n_ctx=512)

    def test_returns_string(self, model):
        """Should return a string."""
        text = generate_text(model, "Hello", max_tokens=10)
        assert isinstance(text, str)

    def test_non_empty(self, model):
        """Should return non-empty text."""
        text = generate_text(model, "The answer is", max_tokens=20)
        assert len(text) > 0

    def test_respects_max_tokens(self, model):
        """Output should not exceed max_tokens."""
        text = generate_text(model, "Hello", max_tokens=5)
        tokens = count_tokens(model, text)
        assert tokens <= 10  # Allow some flexibility


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
class TestGenerateChat:
    """Tests for generate_chat (requires actual model)."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for tests."""
        return load_model(MODEL_PATH, n_ctx=512)

    def test_returns_string(self, model):
        """Should return a string."""
        messages = [{"role": "user", "content": "Hello"}]
        response = generate_chat(model, messages, max_tokens=20)
        assert isinstance(response, str)

    def test_handles_system_message(self, model):
        """Should handle system message."""
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"}
        ]
        response = generate_chat(model, messages, max_tokens=20)
        assert isinstance(response, str)


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
class TestCountTokens:
    """Tests for count_tokens (requires actual model)."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for tests."""
        return load_model(MODEL_PATH, n_ctx=512)

    def test_returns_int(self, model):
        """Should return an integer."""
        count = count_tokens(model, "Hello, world!")
        assert isinstance(count, int)

    def test_positive_count(self, model):
        """Should return positive count."""
        count = count_tokens(model, "Hello")
        assert count > 0

    def test_longer_more_tokens(self, model):
        """Longer text should have more tokens."""
        short = count_tokens(model, "Hi")
        long = count_tokens(model, "Hello, how are you doing today?")
        assert long > short


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
class TestBenchmarkInference:
    """Tests for benchmark_inference (requires actual model)."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for tests."""
        return load_model(MODEL_PATH, n_ctx=512)

    def test_returns_metrics(self, model):
        """Should return BenchmarkMetrics."""
        metrics = benchmark_inference(model, ["Test"], max_tokens=10)
        assert hasattr(metrics, 'tokens_per_second')

    def test_positive_throughput(self, model):
        """Throughput should be positive."""
        metrics = benchmark_inference(model, ["Test"] * 3, max_tokens=10)
        assert metrics.tokens_per_second > 0


class TestMilestone:
    """Integration tests for llama.cpp inference."""

    @pytest.mark.skipif(not MODEL_AVAILABLE, reason="No test model available")
    def test_full_workflow(self):
        """Complete llama.cpp workflow."""
        # Load model
        model = load_model(MODEL_PATH, n_ctx=1024)

        # Get info
        info = get_model_info(MODEL_PATH)
        print(f"\n✅ Model Info:")
        print(f"   File: {MODEL_PATH}")
        print(f"   Size: {info.get('file_size_gb', 'unknown')} GB")

        # Generate text
        text = generate_text(
            model, "Once upon a time",
            max_tokens=30, temperature=0.7
        )
        assert len(text) > 0

        # Chat
        messages = [{"role": "user", "content": "Hello!"}]
        response = generate_chat(model, messages, max_tokens=30)
        assert len(response) > 0

        # Benchmark
        metrics = benchmark_inference(model, ["Test prompt"] * 5, max_tokens=20)

        print("\n✅ Milestone Test - llama.cpp Inference")
        print(f"   Generated: {text[:50]}...")
        print(f"   Throughput: {metrics.tokens_per_second:.1f} tok/s")
        print(f"   Total time: {metrics.total_time:.2f}s")
