# Chapter 11: Extended Reading & References

This document provides carefully scoped references for deeper understanding. Each resource is rated by priority and includes guidance on what to focus on.

---

## Essential Reading

### ZeRO Paper

**"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (Rajbhandari et al., 2019)
- Paper: https://arxiv.org/abs/1910.02054
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~2 hours
- **What to focus on**:
  - Section 3: The three ZeRO stages explained
  - Section 4: Memory analysis and communication costs
  - Figure 1: Memory comparison across approaches
- **Key insight**: Understanding why partitioning optimizer states gives such large memory savings (often the largest memory consumer!).

### Megatron-LM Paper

**"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (Shoeybi et al., 2019)
- Paper: https://arxiv.org/abs/1909.08053
- **Priority**: ⭐⭐⭐ Must read
- **Time**: ~2 hours
- **What to focus on**:
  - Section 3: Tensor parallelism for transformers
  - Section 3.2: How to split MLP and attention layers
  - Figure 3: The clever column/row split pattern
- **Key insight**: How to minimize communication by pairing column-parallel and row-parallel layers.

---

## Core Libraries

### PyTorch DDP Documentation

**Documentation**: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - The `find_unused_parameters` flag and when to use it
  - Gradient bucketing and `bucket_cap_mb`
  - The `DistributedSampler` requirements
- **Lab connection**: Lab 02 implements DDP training from scratch

### PyTorch FSDP Documentation

**Documentation**: https://pytorch.org/docs/stable/fsdp.html

- **Priority**: ⭐⭐⭐ Must understand
- **What to focus on**:
  - `ShardingStrategy` options (FULL_SHARD, SHARD_GRAD_OP, etc.)
  - `auto_wrap_policy` for transformer models
  - `MixedPrecision` configuration
  - State dict handling for checkpoints
- **Lab connection**: Lab 03 uses FSDP for memory-efficient training

### DeepSpeed Documentation

**Documentation**: https://www.deepspeed.ai/docs/config-json/

- **Priority**: ⭐⭐ Important
- **What to focus on**:
  - ZeRO configuration (stages 1, 2, 3)
  - Offload options (CPU, NVMe)
  - Integration with HuggingFace Trainer
- **Lab connection**: Lab 05 integrates DeepSpeed

---

## Recommended Reading

### Mixed Precision Training

**"Mixed Precision Training"** (Micikevicius et al., 2017)
- Paper: https://arxiv.org/abs/1710.03740
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 3: Loss scaling technique
  - Section 4: Which operations need fp32
  - Table 1: Operations that are safe in fp16
- **Key insight**: Why loss scaling is necessary for fp16 but not bf16.

### Pipeline Parallelism

**"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"** (Huang et al., 2019)
- Paper: https://arxiv.org/abs/1811.06965
- **Priority**: ⭐⭐ Highly recommended
- **Time**: 1-2 hours
- **What to focus on**:
  - Section 2: The pipeline bubble problem
  - Section 3: Micro-batching solution
  - Figure 2: Pipeline schedule visualization
- **Key insight**: The trade-off between bubble size and memory (more micro-batches = smaller bubble but more activation memory).

**"PipeDream: Generalized Pipeline Parallelism for DNN Training"** (Narayanan et al., 2019)
- Paper: https://arxiv.org/abs/1806.03377
- **Priority**: ⭐ Optional but enriching
- **What to focus on**: The 1F1B schedule that reduces memory requirements

---

## Code References

### Reference Implementations

**Megatron-LM (NVIDIA)**
- Code: https://github.com/NVIDIA/Megatron-LM
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Production-quality tensor and pipeline parallelism
- **Files to read**:
  - `megatron/core/tensor_parallel/` - Column/row parallel layers
  - `megatron/core/pipeline_parallel/` - Pipeline schedules

**DeepSpeed Examples**
- Code: https://github.com/microsoft/DeepSpeedExamples
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Practical examples of ZeRO stages, offloading
- **Folders to explore**:
  - `training/` - Various training examples
  - `inference/` - Inference optimization examples

**PyTorch FSDP Tutorial**
- Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **Priority**: ⭐⭐⭐ Must do
- **Time**: 1-2 hours
- **What you'll learn**: Hands-on FSDP implementation

**Llama Training (Torchtune)**
- Code: https://github.com/pytorch/torchtune
- **Priority**: ⭐⭐ Highly recommended
- **Why use it**: Modern PyTorch training recipes with FSDP
- **Files to read**: `torchtune/training/` - Full distributed training examples

---

## Deep Dives (Optional)

These are for students who want to go deeper. Not required for the labs.

### Sequence Parallelism

**"Reducing Activation Recomputation in Large Transformer Models"** (Korthikanti et al., 2022)
- Paper: https://arxiv.org/abs/2205.05198
- **Why read it**: Extends tensor parallelism to sequence dimension, further reducing memory
- **Key sections**: Section 3 (sequence parallelism), Section 4 (selective activation recomputation)

### Communication Optimization

**"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"** (Li et al., 2020)
- Paper: https://arxiv.org/abs/2006.15704
- **Why read it**: Deep dive into DDP implementation details
- **Key sections**: Gradient bucketing, communication-computation overlap

### ZeRO-Infinity

**"ZeRO-Infinity: Breaking the GPU Memory Wall"** (Rajbhandari et al., 2021)
- Paper: https://arxiv.org/abs/2104.07857
- **Why read it**: How to train models larger than GPU+CPU memory using NVMe
- **Key insight**: Heterogeneous memory systems for extreme scale

### 3D Parallelism

**"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (Narayanan et al., 2021)
- Paper: https://arxiv.org/abs/2104.04473
- **Why read it**: How to combine DP, TP, and PP effectively
- **Key sections**: Section 4 (parallelism mapping), Section 5 (performance analysis)

---

## Quick Reference Card

| Concept | Primary Resource | Time |
|---------|------------------|------|
| DDP basics | PyTorch DDP docs | 30 min |
| ZeRO stages | ZeRO paper §3 | 1 hour |
| Tensor parallelism | Megatron-LM paper §3 | 1 hour |
| Pipeline parallelism | GPipe paper §2-3 | 1 hour |
| Mixed precision | Mixed Precision paper | 1 hour |
| FSDP usage | PyTorch FSDP tutorial | 2 hours |

---

## Tools and Utilities

### Profiling

**PyTorch Profiler**
- Docs: https://pytorch.org/docs/stable/profiler.html
- **Use for**: Finding communication bottlenecks, kernel timing

**NVIDIA Nsight Systems**
- Docs: https://developer.nvidia.com/nsight-systems
- **Use for**: GPU timeline analysis, NCCL communication profiling

### Debugging

**torch.distributed.debug**
- Enable with: `TORCH_DISTRIBUTED_DEBUG=DETAIL`
- **Use for**: Debugging hangs, finding which rank failed

**NCCL Debug**
- Enable with: `NCCL_DEBUG=INFO`
- **Use for**: Communication issues, network problems

---

## What's NOT Covered Here

These topics are covered in other chapters:

- **Flash Attention** → Chapter 10
- **Quantization** → Chapter 8
- **Inference optimization** → Chapter 9
- **Custom CUDA kernels** → Chapter 12

Stay focused on understanding distributed training patterns first. Optimization techniques build on this foundation.

---

## Lab Connections

| Doc | Related Labs |
|-----|--------------|
| `01_parallelism_strategies.md` | Lab 01: Multi-GPU setup |
| `02_ddp.md` | Lab 02: DDP Training |
| `03_model_parallelism.md` | (concepts used in Labs 03-05) |
| `04_zero_and_fsdp.md` | Lab 03: FSDP |
| `05_mixed_precision.md` | Lab 04: Mixed Precision |
| All docs | Lab 05: DeepSpeed integration |
