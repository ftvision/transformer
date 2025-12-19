# Flash Attention v2 and v3: Improvements and Optimizations

## Evolution of Flash Attention

```
Timeline:
- Flash Attention v1 (2022): Original algorithm
- Flash Attention v2 (2023): 2x speedup through better parallelism
- Flash Attention v3 (2024): Hopper GPU optimizations (H100)
```

## Flash Attention v1 Limitations

The original Flash Attention had some inefficiencies:

1. **Sequential K, V iteration**: Outer loop over K, V blocks
2. **Limited parallelism**: Couldn't fully utilize all SMs
3. **Non-matmul FLOPs**: Softmax, rescaling not utilizing tensor cores
4. **Suboptimal memory access**: Some redundant HBM reads

## Flash Attention v2: Key Improvements

### 1. Reversed Loop Order

**v1**: Outer loop over K, V; inner loop over Q
**v2**: Outer loop over Q; inner loop over K, V

```python
# v1 loop order
for j in range(T_c):  # K, V blocks
    for i in range(T_r):  # Q blocks
        ...

# v2 loop order
for i in range(T_r):  # Q blocks - OUTER
    for j in range(T_c):  # K, V blocks - INNER
        ...
```

**Why this matters:**
- Each thread block now "owns" a Q block
- O accumulator stays in registers/SRAM throughout
- No need to repeatedly load/store O to HBM

### 2. Better Work Partitioning

**v1**: One thread block per (Q_block, head) pair
**v2**: Split work across more thread blocks

```
v1 parallelism: batch × heads × T_r
v2 parallelism: batch × heads × T_r × (more splits)
```

For long sequences, v2 can use more thread blocks, better utilizing the GPU.

### 3. Reduced Non-Matmul FLOPs

The v1 algorithm had many scalar operations per element:
- exp()
- max reduction
- sum reduction
- division

v2 reorganizes computation to:
- Batch these operations
- Use warp-level primitives
- Reduce register pressure

### 4. Better Warp Specialization

```
v2 warp roles:
- Warp 0-3: Compute Q @ K^T (tensor cores)
- Warp 4-7: Compute softmax, update statistics
- Warp 0-3: Compute P @ V (tensor cores)

Overlapped execution:
[QK^T compute] [softmax] [PV compute]
              [QK^T next] [softmax] [PV next]
```

### Performance Comparison: v1 vs v2

| Sequence Length | v1 (TFLOPS) | v2 (TFLOPS) | Speedup |
|-----------------|-------------|-------------|---------|
| 512 | 142 | 193 | 1.36x |
| 1024 | 156 | 216 | 1.38x |
| 2048 | 163 | 227 | 1.39x |
| 4096 | 168 | 231 | 1.38x |
| 8192 | 170 | 233 | 1.37x |

*Numbers for A100, FP16, causal attention*

v2 achieves **50-70%** of theoretical peak FLOPS (vs ~35% for v1).

## Flash Attention v3: Hopper Optimizations

Flash Attention v3 is specifically optimized for NVIDIA Hopper GPUs (H100).

### Key H100 Features Utilized

**1. Warp Group Level Parallelism**

H100 introduces new WGMMA (Warp Group Matrix Multiply-Accumulate) instructions:
- 4 warps (128 threads) work together on one matrix multiply
- Higher throughput than v2's warp-level operations

```
v2: Each warp does independent matmul
v3: Warp groups cooperate on larger matmuls
```

**2. TMA (Tensor Memory Accelerator)**

Hardware unit for efficient tensor loads:
- Async memory copies without occupying compute units
- Automatic address calculation
- Reduces software overhead

```python
# v2 (software loads)
for i in range(B_r):
    for j in range(d):
        sram[i][j] = hbm[global_offset + i * stride + j]

# v3 (TMA)
tma_load(sram, hbm, descriptor)  # Hardware handles it!
```

**3. FP8 Support**

H100 adds FP8 tensor cores:
- 2x throughput vs FP16
- Flash Attention v3 supports FP8 attention

```
FP8 attention:
- Q, K in FP8: 2x faster QK^T
- P in FP16 (for softmax stability)
- V in FP8: 2x faster PV
```

### v3 Algorithm Changes

**1. Two-Stage Pipelining**

```
Stage 1: Load next K, V (TMA) | Compute current block
Stage 2: Compute current block | Load next K, V (TMA)

Timeline:
[Load K1,V1] [Compute K1,V1] [Compute K2,V2] [Compute K3,V3] ...
             [Load K2,V2]    [Load K3,V3]    [Load K4,V4]
```

Memory loads and compute are fully overlapped!

**2. Ping-Pong Buffering**

Two SRAM buffers for K, V:
- Buffer A: Being computed
- Buffer B: Being loaded

```
Time T:   [Buffer A: compute] [Buffer B: load]
Time T+1: [Buffer A: load]    [Buffer B: compute]
```

**3. Low-Precision Accumulation**

Careful mixed-precision strategy:
- FP8 matmuls accumulate to FP32
- Softmax in FP32
- Convert back to FP16/BF16 for output

### Performance: v2 vs v3 on H100

| Sequence Length | v2 (TFLOPS) | v3 (TFLOPS) | Speedup |
|-----------------|-------------|-------------|---------|
| 512 | 320 | 440 | 1.38x |
| 1024 | 380 | 560 | 1.47x |
| 2048 | 410 | 620 | 1.51x |
| 4096 | 420 | 675 | 1.61x |
| 8192 | 425 | 700 | 1.65x |

*Numbers for H100, FP16, causal attention*

v3 achieves **70-75%** of H100's theoretical peak!

## Using Flash Attention in Practice

### PyTorch Integration

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ has built-in Flash Attention
# Automatically uses best backend (Flash, Memory-efficient, or Math)

Q = torch.randn(batch, heads, seq_len, d, device='cuda')
K = torch.randn(batch, heads, seq_len, d, device='cuda')
V = torch.randn(batch, heads, seq_len, d, device='cuda')

# This automatically uses Flash Attention when possible
output = F.scaled_dot_product_attention(Q, K, V)

# Causal masking
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Check which backend is used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
```

### Direct Flash Attention Library

```python
# pip install flash-attn
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Separate Q, K, V
# Note: expects (batch, seq_len, heads, d) format
output = flash_attn_func(Q, K, V, causal=True)

# Packed QKV (more efficient)
QKV = torch.stack([Q, K, V], dim=2)  # (batch, seq_len, 3, heads, d)
output = flash_attn_qkvpacked_func(QKV, causal=True)
```

### Checking Availability

```python
# Check if Flash Attention is available
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # True/False

# Requirements for Flash Attention:
# - NVIDIA GPU (Ampere or later recommended)
# - Head dimension <= 256 (v2) or <= 128 (v3 for best perf)
# - CUDA 11.6+
# - cuDNN 8.5+
```

## When to Use Which Version

| Use Case | Recommendation |
|----------|----------------|
| A100 GPU | Flash Attention v2 |
| H100 GPU | Flash Attention v3 |
| PyTorch only | F.scaled_dot_product_attention |
| Custom head dims | May need v2 (v3 has restrictions) |
| FP8 training | v3 only |

## Comparison Summary

| Feature | v1 | v2 | v3 |
|---------|----|----|----|
| Loop order | KV outer | Q outer | Q outer |
| Peak utilization | ~35% | ~55% | ~75% |
| Warp strategy | Independent | Specialized | Warp groups |
| Memory loads | Sync | Async | TMA |
| FP8 support | No | No | Yes |
| Target GPU | Any CUDA | Ampere+ | Hopper |

## What's Next

Flash Attention handles the forward pass efficiently. But for training, we also want to reduce memory for storing activations. See `06_gradient_checkpointing.md` for techniques that trade compute for memory in the backward pass.
