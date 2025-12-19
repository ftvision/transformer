# Lab 05: Framework Benchmarking

## Learning Objectives

By completing this lab, you will:
1. Design fair benchmarks for LLM inference
2. Measure throughput, latency, and memory across frameworks
3. Understand the trade-offs between different inference solutions
4. Create reproducible benchmark reports

## Background

Comparing LLM inference frameworks requires careful methodology:
- **Throughput**: Tokens generated per second (higher is better)
- **Latency**: Time to first token and total generation time (lower is better)
- **Memory**: Peak GPU/CPU memory usage (lower is better)
- **Quality**: Output consistency and correctness

### Key Metrics

```
Throughput = Total Output Tokens / Total Time

Time to First Token (TTFT) = Time until first token generated

Inter-Token Latency (ITL) = Average time between tokens

Memory Efficiency = Tokens Generated / Peak Memory
```

### Benchmark Considerations

1. **Warmup**: First inference is often slower (model loading, JIT compilation)
2. **Batching**: Different batch sizes reveal different characteristics
3. **Sequence Length**: Input and output lengths affect performance
4. **Quantization**: Compare same precision levels fairly
5. **Hardware**: Document GPU/CPU specs precisely

## Instructions

Implement the functions in `src/benchmark.py`:

### Part 1: Metric Collection

1. `BenchmarkResult` - Dataclass to store benchmark results
2. `measure_latency()` - Measure generation latency with warmup
3. `measure_throughput()` - Measure tokens per second
4. `measure_memory()` - Track peak memory usage

### Part 2: Benchmark Runners

5. `benchmark_single()` - Benchmark a single prompt
6. `benchmark_batch()` - Benchmark batch processing
7. `run_latency_sweep()` - Test across different sequence lengths
8. `run_throughput_sweep()` - Test across different batch sizes

### Part 3: Analysis and Reporting

9. `compare_results()` - Compare results across frameworks
10. `generate_report()` - Create a benchmark report
11. `plot_comparison()` - Visualize benchmark results

## Example Usage

```python
from benchmark import (
    BenchmarkResult,
    measure_latency,
    measure_throughput,
    benchmark_single,
    run_throughput_sweep,
    compare_results,
    generate_report,
)

# Measure single prompt latency
latency = measure_latency(
    generate_fn=model.generate,
    prompt="The future of AI is",
    num_tokens=50,
    warmup_runs=2,
    num_runs=5,
)
print(f"Average latency: {latency.mean_ms:.1f}ms")
print(f"Time to first token: {latency.ttft_ms:.1f}ms")

# Benchmark batch throughput
throughput = measure_throughput(
    generate_fn=model.generate,
    prompts=["Hello"] * 32,
    num_tokens=100,
)
print(f"Throughput: {throughput.tokens_per_second:.1f} tok/s")

# Run sweep across batch sizes
results = run_throughput_sweep(
    generate_fn=model.generate,
    prompt="Test prompt",
    num_tokens=50,
    batch_sizes=[1, 4, 8, 16, 32],
)

# Compare frameworks
comparison = compare_results({
    "huggingface": hf_results,
    "vllm": vllm_results,
    "llamacpp": llamacpp_results,
})

# Generate report
report = generate_report(comparison, output_format="markdown")
print(report)
```

## Running Tests

```bash
cd chapters/ch09_inference_frameworks
python -m pytest lab05_benchmark/tests/ -v
```

## Milestone

When complete, you should be able to:
1. Measure latency with proper warmup handling
2. Calculate throughput for single and batch processing
3. Track memory usage during inference
4. Compare multiple frameworks fairly
5. Generate readable benchmark reports

## Tips

- Always use warmup runs to exclude cold-start effects
- Use the same prompts and generation settings across frameworks
- Document hardware and software versions
- Consider variance in measurements (report std dev)
- Memory measurement may require framework-specific methods
