# Chapter 12: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### Triton

**Triton Official Documentation**
- Docs: https://triton-lang.org/
- **Priority**: ⭐⭐⭐ Must read
- **Time**: 2-3 hours
- **What to focus on**:
  - Getting Started tutorial
  - Matrix multiplication tutorial
  - Fused softmax tutorial
- **Key insight**: Understanding the block-level programming model is essential before writing kernels.

**"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** (Tillet et al., 2019)
- Paper: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- **Priority**: ⭐⭐ Important
- **Time**: 1-2 hours
- **What to focus on**: Sections 2-3 (programming model, compilation)
- **Why read it**: Understand the design decisions behind Triton.

---

## Core Libraries

### JAX

**JAX Documentation**
- Docs: https://jax.readthedocs.io/
- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - "JAX quickstart" for basics
  - "How to Think in JAX" for the functional paradigm
  - "Parallel Evaluation in JAX" for pmap/pjit

**JAX Tutorials**
- URL: https://jax.readthedocs.io/en/latest/tutorials.html
- **Priority**: ⭐⭐⭐ Must read
- **Recommended order**:
  1. Quickstart
  2. JIT compilation
  3. Automatic vectorization (vmap)
  4. Parallel evaluation (pmap)

### PyTorch Compilation

**torch.compile Documentation**
- Docs: https://pytorch.org/docs/stable/torch.compiler.html
- **Priority**: ⭐⭐ Important
- **What to focus on**:
  - Basic usage patterns
  - Compilation modes (default, reduce-overhead, max-autotune)
  - Debugging and troubleshooting

**PyTorch Profiler**
- Docs: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **Priority**: ⭐⭐⭐ Must understand
- **Time**: 30 minutes
- **Lab connection**: Essential for Lab 05

---

## Recommended Reading

### GPU Architecture

**"CUDA C++ Programming Guide"** (NVIDIA)
- Docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Priority**: ⭐⭐ Important background
- **Time**: 2-3 hours for essentials
- **What to focus on**:
  - Chapter 2: Programming Model (grids, blocks, threads)
  - Chapter 4: Hardware Implementation (SMs, memory hierarchy)
  - Chapter 5: Performance Guidelines

**"GPU Performance Background"** (NVIDIA)
- Docs: https://docs.nvidia.com/deeplearning/performance/index.html
- **Priority**: ⭐⭐ Highly recommended
- **What to focus on**: Memory bandwidth analysis, Tensor Core utilization

### TPU

**"In-Datacenter Performance Analysis of a Tensor Processing Unit"** (Jouppi et al., 2017)
- Paper: https://arxiv.org/abs/1704.04760
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 3: TPU Architecture
  - Section 4: Performance comparison methodology
- **Key insight**: Understand why systolic arrays work for ML.

**Cloud TPU Documentation**
- Docs: https://cloud.google.com/tpu/docs
- **Priority**: ⭐⭐ Important for TPU users
- **What to focus on**: Performance guide, best practices

### XLA

**XLA Overview**
- Docs: https://www.tensorflow.org/xla
- **Priority**: ⭐⭐ Important
- **What to focus on**: How XLA optimizes computations, fusion strategies

**"XLA: Optimizing Compiler for Machine Learning"**
- Blog: https://www.tensorflow.org/xla/overview
- **Priority**: ⭐ Optional but enriching
- **Why read it**: High-level overview of XLA's optimization passes

---

## Code References

### Triton Examples

**Triton Tutorials Repository**
- Code: https://github.com/triton-lang/triton/tree/main/python/tutorials
- **Priority**: ⭐⭐⭐ Must study
- **Key files**:
  - `01-vector-add.py` - Basic kernel structure
  - `02-fused-softmax.py` - Fusion example
  - `03-matrix-multiplication.py` - Tiled matmul
  - `06-fused-attention.py` - Flash Attention in Triton

**Flash Attention Triton Implementation**
- Code: https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/flash_attn_triton.py
- **Priority**: ⭐⭐ Highly recommended
- **Why study it**: Production-quality Triton kernel for attention
- **Lab connection**: Reference for Lab 02

### JAX Examples

**JAX Examples Repository**
- Code: https://github.com/google/jax/tree/main/examples
- **Priority**: ⭐⭐ Highly recommended
- **Key files**:
  - `mnist_classifier.py` - Basic training loop
  - `differentially_private_sgd.py` - Advanced gradients

**Flax (JAX neural network library)**
- Code: https://github.com/google/flax
- **Priority**: ⭐⭐ Important for practical JAX
- **What to focus on**: How to structure JAX models

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Kernel Optimization

**"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"** (Jia et al., 2018)
- Paper: https://arxiv.org/abs/1804.06826
- **Why read it**: Deep understanding of GPU microarchitecture

**"Optimizing CUDA Applications"** (NVIDIA)
- Docs: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Why read it**: Detailed optimization techniques

### Compiler Internals

**MLIR (Multi-Level Intermediate Representation)**
- Docs: https://mlir.llvm.org/
- **Why read it**: Understand modern ML compiler infrastructure (XLA, Triton use MLIR)

**"TorchInductor: A PyTorch Native Compiler"**
- Blog: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler/977
- **Why read it**: How PyTorch 2.0's compiler works

### Research Papers

**"Flash Attention: Fast and Memory-Efficient Exact Attention"** (Dao et al., 2022)
- Paper: https://arxiv.org/abs/2205.14135
- **Priority**: ⭐⭐ Highly recommended
- **Why read it**: The algorithm you implement in Lab 02

**"Flash Attention 2"** (Dao, 2023)
- Paper: https://arxiv.org/abs/2307.08691
- **Why read it**: Optimizations beyond Flash Attention 1

---

## Tools Reference

### Profiling Tools

| Tool | Use Case | Platform |
|------|----------|----------|
| PyTorch Profiler | Quick PyTorch profiling | Any |
| Nsight Systems | System-wide GPU profiling | NVIDIA |
| Nsight Compute | Kernel-level analysis | NVIDIA |
| JAX Profiler | JAX/TPU profiling | Any |
| TensorBoard | Visualization | Any |

### Installation Commands

```bash
# Triton (comes with PyTorch 2.0+)
pip install triton

# JAX (CPU)
pip install jax jaxlib

# JAX (GPU - check CUDA version)
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Nsight Systems
# Download from NVIDIA developer site

# PyTorch profiler (included with PyTorch)
pip install torch tensorboard
```

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| Triton basics | Triton tutorials | 2-3 hours |
| Kernel fusion | Fused softmax tutorial | 1 hour |
| JAX transformations | JAX quickstart | 1 hour |
| TPU architecture | Jouppi et al. paper | 1-2 hours |
| Profiling | PyTorch profiler docs | 30 min |
| GPU memory | CUDA programming guide Ch. 5 | 1 hour |

---

## What's NOT Covered Here

These topics are covered in earlier chapters:

- **Flash Attention algorithm** → Chapter 10
- **Distributed training** → Chapter 11
- **Basic transformer architecture** → Chapters 1-3
- **Memory optimization (KV-cache)** → Chapter 8

This chapter focuses on writing efficient code. The earlier chapters cover the algorithms themselves.

---

## Lab Connections

| Lab | Key References |
|-----|----------------|
| Lab 01: Triton Basics | Triton tutorials 01-02 |
| Lab 02: Fused Attention | Flash Attention paper, Triton tutorial 06 |
| Lab 03: JAX Intro | JAX quickstart |
| Lab 04: JAX JIT & vmap | JAX transformations docs |
| Lab 05: Profiling | PyTorch profiler tutorial, Nsight docs |
