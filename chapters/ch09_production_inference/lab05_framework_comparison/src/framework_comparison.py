"""
Lab 05: Framework Comparison

Compare different inference frameworks by simulating their characteristics
and building a decision helper.

Your task: Complete the functions and classes below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class Framework(Enum):
    """Supported inference frameworks."""
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt_llm"


@dataclass
class HardwareConfig:
    """Hardware configuration for deployment."""
    has_gpu: bool
    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 16.0
    gpu_type: str = "none"  # "none", "nvidia", "apple_silicon", "amd"
    num_gpus: int = 0


@dataclass
class WorkloadConfig:
    """Workload characteristics."""
    concurrent_users: int
    avg_prompt_tokens: int
    avg_output_tokens: int
    requires_structured_output: bool = False
    latency_sensitive: bool = False
    throughput_priority: bool = True


@dataclass
class FrameworkCapabilities:
    """Capabilities and characteristics of a framework."""
    name: Framework
    supports_gpu: bool
    supports_cpu: bool
    supports_quantization: bool
    supports_continuous_batching: bool
    supports_structured_output: bool
    supports_tensor_parallel: bool
    min_gpu_memory_gb: float
    complexity: str  # "low", "medium", "high"
    throughput_multiplier: float  # Relative to HuggingFace baseline
    latency_multiplier: float  # Lower is better


# Framework specifications (provided for reference)
FRAMEWORK_SPECS: Dict[Framework, FrameworkCapabilities] = {
    Framework.HUGGINGFACE: FrameworkCapabilities(
        name=Framework.HUGGINGFACE,
        supports_gpu=True,
        supports_cpu=True,
        supports_quantization=True,
        supports_continuous_batching=False,
        supports_structured_output=False,
        supports_tensor_parallel=False,
        min_gpu_memory_gb=0,
        complexity="low",
        throughput_multiplier=1.0,
        latency_multiplier=1.0,
    ),
    Framework.VLLM: FrameworkCapabilities(
        name=Framework.VLLM,
        supports_gpu=True,
        supports_cpu=False,
        supports_quantization=True,
        supports_continuous_batching=True,
        supports_structured_output=False,
        supports_tensor_parallel=True,
        min_gpu_memory_gb=8,
        complexity="medium",
        throughput_multiplier=5.0,
        latency_multiplier=0.8,
    ),
    Framework.LLAMA_CPP: FrameworkCapabilities(
        name=Framework.LLAMA_CPP,
        supports_gpu=True,
        supports_cpu=True,
        supports_quantization=True,
        supports_continuous_batching=False,
        supports_structured_output=False,
        supports_tensor_parallel=False,
        min_gpu_memory_gb=0,
        complexity="low",
        throughput_multiplier=0.5,  # CPU is slower
        latency_multiplier=1.5,
    ),
    Framework.SGLANG: FrameworkCapabilities(
        name=Framework.SGLANG,
        supports_gpu=True,
        supports_cpu=False,
        supports_quantization=True,
        supports_continuous_batching=True,
        supports_structured_output=True,
        supports_tensor_parallel=True,
        min_gpu_memory_gb=8,
        complexity="medium",
        throughput_multiplier=4.5,
        latency_multiplier=0.5,  # With RadixAttention caching
    ),
    Framework.TENSORRT_LLM: FrameworkCapabilities(
        name=Framework.TENSORRT_LLM,
        supports_gpu=True,
        supports_cpu=False,
        supports_quantization=True,
        supports_continuous_batching=True,
        supports_structured_output=False,
        supports_tensor_parallel=True,
        min_gpu_memory_gb=16,
        complexity="high",
        throughput_multiplier=7.0,
        latency_multiplier=0.6,
    ),
}


def get_compatible_frameworks(
    hardware: HardwareConfig
) -> List[Framework]:
    """
    Get list of frameworks compatible with given hardware.

    Args:
        hardware: Hardware configuration

    Returns:
        List of compatible frameworks

    Example:
        >>> hw = HardwareConfig(has_gpu=False, cpu_memory_gb=16)
        >>> compatible = get_compatible_frameworks(hw)
        >>> Framework.LLAMA_CPP in compatible
        True
        >>> Framework.VLLM in compatible
        False
    """
    # YOUR CODE HERE
    #
    # For each framework, check:
    # 1. If hardware has no GPU, framework must support CPU
    # 2. If hardware has GPU, check GPU memory >= min_gpu_memory_gb
    # 3. If hardware is Apple Silicon, exclude TensorRT-LLM
    raise NotImplementedError("Implement get_compatible_frameworks")


def rank_frameworks(
    hardware: HardwareConfig,
    workload: WorkloadConfig
) -> List[Dict[str, Any]]:
    """
    Rank frameworks based on hardware and workload requirements.

    Args:
        hardware: Hardware configuration
        workload: Workload characteristics

    Returns:
        List of dicts with 'framework', 'score', 'reasons' keys,
        sorted by score (highest first)

    Scoring criteria:
    - +10 if supports required features (structured output, etc.)
    - +5 * throughput_multiplier if throughput priority
    - +5 / latency_multiplier if latency sensitive
    - -5 if complexity is "high"
    - -10 if doesn't meet minimum requirements

    Example:
        >>> hw = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        >>> workload = WorkloadConfig(concurrent_users=100, avg_prompt_tokens=100,
        ...                           avg_output_tokens=200, throughput_priority=True)
        >>> rankings = rank_frameworks(hw, workload)
        >>> rankings[0]['framework']  # Top recommendation
    """
    # YOUR CODE HERE
    #
    # Steps:
    # 1. Get compatible frameworks
    # 2. Score each based on workload match
    # 3. Sort by score descending
    # 4. Include reasoning for each
    raise NotImplementedError("Implement rank_frameworks")


def estimate_throughput(
    framework: Framework,
    hardware: HardwareConfig,
    workload: WorkloadConfig,
    model_params_b: float = 7.0
) -> Dict[str, float]:
    """
    Estimate throughput metrics for a framework configuration.

    Args:
        framework: The framework to evaluate
        hardware: Hardware configuration
        workload: Workload characteristics
        model_params_b: Model size in billions of parameters

    Returns:
        Dictionary with:
        - 'tokens_per_second': Estimated output tokens per second
        - 'requests_per_second': Estimated request throughput
        - 'time_to_first_token_ms': Estimated TTFT
        - 'batch_efficiency': Utilization factor (0-1)

    Example:
        >>> framework = Framework.VLLM
        >>> hw = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        >>> workload = WorkloadConfig(concurrent_users=50, avg_prompt_tokens=100,
        ...                           avg_output_tokens=200)
        >>> metrics = estimate_throughput(framework, hw, workload, model_params_b=7.0)
    """
    # YOUR CODE HERE
    #
    # Base calculations (rough estimates):
    # - Single GPU baseline: ~50 tokens/sec for 7B model with HuggingFace
    # - Apply framework's throughput_multiplier
    # - Scale with GPU memory (more memory = larger batches = higher throughput)
    # - For CPU-only (llama.cpp): ~20 tokens/sec baseline
    #
    # Batch efficiency depends on:
    # - Whether framework supports continuous batching
    # - Number of concurrent users vs max batch size
    raise NotImplementedError("Implement estimate_throughput")


def estimate_memory_requirement(
    model_params_b: float,
    framework: Framework,
    max_batch_size: int,
    max_seq_len: int = 2048,
    quantization_bits: int = 16
) -> Dict[str, float]:
    """
    Estimate memory requirements for a deployment.

    Args:
        model_params_b: Model size in billions of parameters
        framework: The framework to use
        max_batch_size: Maximum concurrent requests
        max_seq_len: Maximum sequence length
        quantization_bits: Quantization level (16=FP16, 8=INT8, 4=INT4)

    Returns:
        Dictionary with:
        - 'model_memory_gb': Memory for model weights
        - 'kv_cache_memory_gb': Memory for KV cache
        - 'activation_memory_gb': Memory for activations
        - 'total_memory_gb': Total memory requirement

    The KV cache memory depends heavily on the framework:
    - vLLM/SGLang: Efficient with PagedAttention
    - HuggingFace: Must allocate max_seq_len per request
    """
    # YOUR CODE HERE
    #
    # Model memory = params * bytes_per_param / 1e9
    # bytes_per_param = quantization_bits / 8
    #
    # KV cache (per request) = 2 * num_layers * num_heads * head_dim * seq_len * 2 bytes
    # Approximate: 2 * (model_params / hidden_dim) * max_seq_len * 2
    # For 7B model with hidden_dim=4096: ~512 bytes per token
    #
    # With PagedAttention: KV cache is ~3-5x more efficient
    raise NotImplementedError("Implement estimate_memory_requirement")


def generate_deployment_recommendation(
    hardware: HardwareConfig,
    workload: WorkloadConfig,
    model_params_b: float = 7.0
) -> Dict[str, Any]:
    """
    Generate a comprehensive deployment recommendation.

    Args:
        hardware: Hardware configuration
        workload: Workload characteristics
        model_params_b: Model size in billions

    Returns:
        Dictionary with:
        - 'recommended_framework': Best framework for the scenario
        - 'reasoning': List of reasons for the recommendation
        - 'configuration': Suggested configuration parameters
        - 'expected_performance': Throughput/latency estimates
        - 'warnings': List of potential issues
        - 'alternatives': Other viable options with tradeoffs

    Example:
        >>> hw = HardwareConfig(has_gpu=True, gpu_memory_gb=16, gpu_type="nvidia")
        >>> workload = WorkloadConfig(concurrent_users=20, avg_prompt_tokens=500,
        ...                           avg_output_tokens=100, requires_structured_output=True)
        >>> rec = generate_deployment_recommendation(hw, workload)
        >>> rec['recommended_framework']
        Framework.SGLANG  # Because of structured output requirement
    """
    # YOUR CODE HERE
    #
    # This should combine:
    # 1. Framework compatibility check
    # 2. Framework ranking
    # 3. Memory requirement estimation
    # 4. Throughput estimation
    # 5. Generate actionable configuration
    raise NotImplementedError("Implement generate_deployment_recommendation")


def compare_frameworks_for_scenario(
    hardware: HardwareConfig,
    workload: WorkloadConfig,
    model_params_b: float = 7.0
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all compatible frameworks for a given scenario.

    Args:
        hardware: Hardware configuration
        workload: Workload characteristics
        model_params_b: Model size in billions

    Returns:
        Dictionary mapping framework name to metrics:
        - 'compatible': Whether framework works for this scenario
        - 'throughput': Estimated throughput metrics
        - 'memory': Memory requirements
        - 'score': Overall score for this workload
        - 'pros': List of advantages
        - 'cons': List of disadvantages

    Example:
        >>> hw = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        >>> workload = WorkloadConfig(concurrent_users=100, avg_prompt_tokens=100,
        ...                           avg_output_tokens=200)
        >>> comparison = compare_frameworks_for_scenario(hw, workload)
        >>> comparison['vllm']['compatible']
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compare_frameworks_for_scenario")


@dataclass
class BenchmarkResult:
    """Result from a simulated benchmark."""
    framework: Framework
    tokens_per_second: float
    time_to_first_token_ms: float
    requests_completed: int
    total_time_seconds: float
    memory_peak_gb: float


def simulate_benchmark(
    framework: Framework,
    hardware: HardwareConfig,
    num_requests: int = 100,
    prompt_tokens: int = 100,
    output_tokens: int = 200
) -> BenchmarkResult:
    """
    Simulate a benchmark run for a framework.

    This doesn't run actual inference, but simulates expected
    performance based on framework characteristics.

    Args:
        framework: Framework to benchmark
        hardware: Hardware configuration
        num_requests: Number of requests to simulate
        prompt_tokens: Tokens per prompt
        output_tokens: Tokens to generate per request

    Returns:
        BenchmarkResult with simulated metrics

    Example:
        >>> hw = HardwareConfig(has_gpu=True, gpu_memory_gb=24, gpu_type="nvidia")
        >>> result = simulate_benchmark(Framework.VLLM, hw, num_requests=100)
        >>> result.tokens_per_second > 0
        True
    """
    # YOUR CODE HERE
    #
    # Simulate based on framework characteristics:
    # - Apply throughput/latency multipliers from FRAMEWORK_SPECS
    # - Account for batching effects
    # - Add some random variance for realism
    raise NotImplementedError("Implement simulate_benchmark")


def format_recommendation_report(
    recommendation: Dict[str, Any]
) -> str:
    """
    Format a deployment recommendation as a readable report.

    Args:
        recommendation: Output from generate_deployment_recommendation

    Returns:
        Formatted string report

    Example output:
    ```
    === Deployment Recommendation ===

    Recommended Framework: vLLM

    Reasoning:
    - High concurrent user count requires continuous batching
    - GPU available with sufficient memory
    - Throughput is prioritized over latency

    Expected Performance:
    - Throughput: ~1500 tokens/second
    - Time to First Token: ~80ms
    - Requests/second: ~7.5

    Configuration:
    - Max batch size: 32
    - GPU memory utilization: 90%
    - Tensor parallel: No (single GPU)

    Warnings:
    - Consider monitoring GPU memory under peak load

    Alternatives:
    1. SGLang - Similar performance, better for future structured output needs
    2. TensorRT-LLM - Higher throughput but more complex setup
    ```
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement format_recommendation_report")
