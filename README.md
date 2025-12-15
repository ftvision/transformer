# Transformer Learning Repository

A comprehensive learning resource for understanding transformer architectures and recent developments in the field. This repository combines theoretical understanding with practical implementations, covering everything from fundamental concepts to advanced optimization techniques.

## Learning Objectives

- Master fundamental transformer architecture and attention mechanisms
- Implement various attention variants and understand their trade-offs
- Explore inference optimization techniques and frameworks
- Learn about hardware acceleration and kernel implementations
- Understand practical training and deployment strategies

## Major Topics

### 1. Attention Mechanisms

Core attention concepts and advanced variants:

- **Standard Attention**: Multi-head self-attention, cross-attention mechanisms
- **Linear Attention**: Efficient attention variants reducing computational complexity
  - Flash Linear Attention
  - Linear attention from Kimi (e.g., DeltaNet, GLA variants)
- **Sparse Attention**: Structured sparsity patterns
  - DeepSeek sparse attention implementations
  - Other efficient sparse patterns
- **Flash Attention**: Hardware-aware optimized attention implementations

### 2. Transformer Architecture

- Basic transformer blocks and layer design
- Positional encodings and rotary embeddings
- Feed-forward networks and layer normalization strategies
- Modern architectural variants and improvements

### 3. Implementation & Learning

#### Python Fundamentals
- From-scratch implementations of core transformer components
- Educational code to understand mechanisms deeply
- Direct tensor operations without abstractions

#### Libraries & Frameworks
- PyTorch and JAX implementations
- High-level framework usage for training
- Integration with modern optimization techniques

### 4. Training

- Pretraining strategies and data handling
- Post-training and fine-tuning techniques
- Efficient training approaches (LoRA, quantization, etc.)
- Distributed training considerations

### 5. Inference Optimization

Industrial-grade inference frameworks:

- **vLLM**: High-throughput inference engine with optimized memory management
- **SGLang**: Structured generation and language engine
- **llama.cpp**: Efficient CPU and quantized inference
- Batch processing and dynamic scheduling
- Quantization and model compression for inference

### 6. Hardware & Kernels

- **JAX**: Composable transformations and functional programming for ML
- **TPU**: Tensor Processing Unit programming and optimization
- Custom kernel development for bottleneck operations
- Hardware-aware algorithm design
- CUDA/HIP kernel implementations

## Repository Structure

```
belgrade-v2/
├── README.md                 # This file
├── basics/                   # Python implementations from scratch
│   ├── attention.py
│   ├── transformer.py
│   └── ...
├── attention_variants/       # Different attention mechanisms
│   ├── flash_attention/
│   ├── linear_attention/
│   ├── sparse_attention/
│   └── ...
├── training/                 # Training implementations
│   ├── pretraining/
│   ├── finetuning/
│   └── ...
├── inference/                # Inference optimization
│   ├── vllm_integration/
│   ├── sglang_integration/
│   ├── llamacpp_integration/
│   └── ...
├── hardware/                 # Hardware-specific optimizations
│   ├── jax_implementations/
│   ├── tpu_kernels/
│   ├── cuda_kernels/
│   └── ...
└── experiments/              # Reproducible experiments and benchmarks
```

## Key Resources & References

### Papers
- Attention is All You Need (Vaswani et al., 2017)
- Flash Attention (Dao et al., 2022)
- DeepSeek variants and sparse attention papers
- Kimi attention mechanism papers

### Frameworks & Tools
- PyTorch: https://pytorch.org/
- JAX: https://jax.readthedocs.io/
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/hpcaitech/SGLang
- llama.cpp: https://github.com/ggerganov/llama.cpp

## Getting Started

1. **Understand the Basics**: Start with fundamental implementations in `basics/`
2. **Explore Attention Variants**: Study different attention mechanisms in `attention_variants/`
3. **Practice Training**: Implement training loops in `training/`
4. **Optimize Inference**: Learn inference optimization with `inference/`
5. **Hardware Acceleration**: Dive into `hardware/` for performance optimization

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (to be updated as we add code)
pip install torch jax numpy
```

## Contributing

As this is a learning repository, feel free to:
- Add new implementations and explanations
- Document learnings and insights
- Include benchmarks and comparisons
- Share optimization techniques

---

**Status**: Repository initialization in progress. Major topics identified, structure ready for implementation.
