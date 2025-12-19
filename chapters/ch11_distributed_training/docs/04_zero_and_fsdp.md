# ZeRO and FSDP: Memory-Efficient Data Parallelism

![ZeRO Stages](vis/zero_stages.svg)

## The Redundancy Problem

Standard DDP replicates everything on every GPU:

```
DDP with 4 GPUs training a 7B model:

GPU 0                    GPU 1                    GPU 2                    GPU 3
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Parameters (14GB)│    │ Parameters (14GB)│    │ Parameters (14GB)│    │ Parameters (14GB)│
│ Gradients  (14GB)│    │ Gradients  (14GB)│    │ Gradients  (14GB)│    │ Gradients  (14GB)│
│ Optimizer  (56GB)│    │ Optimizer  (56GB)│    │ Optimizer  (56GB)│    │ Optimizer  (56GB)│
│   (Adam m,v,fp32)│    │   (Adam m,v,fp32)│    │   (Adam m,v,fp32)│    │   (Adam m,v,fp32)│
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
Total per GPU: 84GB      Total per GPU: 84GB      Total per GPU: 84GB      Total per GPU: 84GB

Total memory used: 84GB × 4 = 336GB
But unique data: only 84GB!
```

**The insight**: Why store 4 copies when we only need 1?

## ZeRO: Zero Redundancy Optimizer

ZeRO (from DeepSpeed) progressively shards model states across GPUs:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZeRO Stages                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│     ZeRO-1      │     ZeRO-2      │          ZeRO-3             │
│  Shard Optimizer│ + Shard Gradients│ + Shard Parameters          │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Memory: 4× → 1× │ Memory: 8× → 1× │ Memory: 12× → 1×            │
│ (optimizer)     │ (opt + grad)    │ (opt + grad + params)       │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Comm: same as   │ Comm: same as   │ Comm: 1.5× DDP              │
│ DDP             │ DDP             │ (extra all-gather)          │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## ZeRO Stage 1: Optimizer State Partitioning

Partition optimizer states (Adam's m, v, fp32 master weights):

```
Before ZeRO-1 (4 GPUs, 7B model):

GPU 0: optimizer states for ALL 7B params → 56GB
GPU 1: optimizer states for ALL 7B params → 56GB
GPU 2: optimizer states for ALL 7B params → 56GB
GPU 3: optimizer states for ALL 7B params → 56GB

After ZeRO-1:

GPU 0: optimizer for params 0-1.75B    → 14GB
GPU 1: optimizer for params 1.75B-3.5B → 14GB
GPU 2: optimizer for params 3.5B-5.25B → 14GB
GPU 3: optimizer for params 5.25B-7B   → 14GB

Savings: 56GB → 14GB per GPU (4× reduction)
```

**How it works**:

```
Training step with ZeRO-1:

1. Forward pass: same as DDP (full model on each GPU)
2. Backward pass: same as DDP (compute all gradients)
3. All-reduce gradients: same as DDP
4. Optimizer step:
   - Each GPU only updates its assigned parameters
   - GPU 0 updates params 0-1.75B using its optimizer shard
   - GPU 1 updates params 1.75B-3.5B using its optimizer shard
   - etc.
5. All-gather parameters: broadcast updated params to all GPUs
```

## ZeRO Stage 2: Gradient Partitioning

Also partition gradients (in addition to optimizer states):

```
After ZeRO-2:

GPU 0: grads + opt for params 0-1.75B    → 14GB + 14GB = 28GB
GPU 1: grads + opt for params 1.75B-3.5B → 14GB + 14GB = 28GB
GPU 2: grads + opt for params 3.5B-5.25B → 14GB + 14GB = 28GB
GPU 3: grads + opt for params 5.25B-7B   → 14GB + 14GB = 28GB

(vs 70GB per GPU with DDP: 14 params + 14 grads + 56 opt)
```

**How it works**:

```
Training step with ZeRO-2:

1. Forward pass: same as DDP
2. Backward pass:
   - Compute gradients as usual
   - BUT: use reduce-scatter instead of all-reduce
   - Each GPU ends up with gradients for its param partition only
3. Optimizer step: each GPU updates its params
4. All-gather: broadcast updated params
```

**Reduce-scatter vs All-reduce**:
```
All-reduce: everyone gets full averaged gradients
Reduce-scatter: each GPU gets 1/N of averaged gradients

           All-Reduce                     Reduce-Scatter
GPU 0: [g0, g1, g2, g3]              GPU 0: [sum(g0)]
GPU 1: [g0, g1, g2, g3]     →        GPU 1: [sum(g1)]
GPU 2: [g0, g1, g2, g3]              GPU 2: [sum(g2)]
GPU 3: [g0, g1, g2, g3]              GPU 3: [sum(g3)]
```

## ZeRO Stage 3: Parameter Partitioning

Also partition model parameters. Each GPU only stores 1/N of the model!

```
After ZeRO-3:

GPU 0: params + grads + opt for shard 0 → ~21GB total
GPU 1: params + grads + opt for shard 1 → ~21GB total
GPU 2: params + grads + opt for shard 2 → ~21GB total
GPU 3: params + grads + opt for shard 3 → ~21GB total

vs 84GB per GPU with DDP!
```

**How it works**:

```
Training step with ZeRO-3:

1. Forward pass:
   - For each layer:
     - All-gather: collect full layer weights from all GPUs
     - Compute forward for this layer
     - Discard weights (don't store full model!)
2. Backward pass:
   - For each layer (reverse order):
     - All-gather: collect full layer weights
     - Compute gradients
     - Reduce-scatter: each GPU keeps its gradient shard
     - Discard weights
3. Optimizer step: update local param shard
4. No additional all-gather needed (params already partitioned)

Communication overhead: 1.5× DDP (extra all-gathers)
```

## FSDP: Fully Sharded Data Parallel

PyTorch's native implementation of ZeRO-3 concepts.

### Basic FSDP Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = TransformerModel()

# Wrap with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    # or: ShardingStrategy.SHARD_GRAD_OP,           # ZeRO-2
    # or: ShardingStrategy.NO_SHARD,                # DDP
    device_id=torch.cuda.current_device(),
)

# Training loop unchanged
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP Wrapping Strategies

FSDP can wrap at different granularities:

```python
# Option 1: Wrap the whole model (coarse-grained)
model = FSDP(model)  # One big shard

# Option 2: Wrap individual layers (fine-grained)
class TransformerModel(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([
            FSDP(TransformerBlock())  # Each block wrapped
            for _ in range(num_layers)
        ])

# Option 3: Auto-wrap policy (recommended)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},  # Your block class
)
```

**Why wrapping granularity matters**:

```
Coarse-grained (whole model):
- All-gather entire model at once
- Large memory spike during forward
- Fewer communication calls

Fine-grained (per-layer):
- All-gather one layer at a time
- Smooth memory usage
- More communication calls
- Can overlap communication with compute
```

### FSDP Configuration Options

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)

# Mixed precision
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,      # Compute dtype
    reduce_dtype=torch.float16,     # Gradient reduction dtype
    buffer_dtype=torch.float16,     # Buffer dtype
)

# CPU offload (for very large models)
cpu_offload = CPUOffload(offload_params=True)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision,
    cpu_offload=cpu_offload,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch next layer
    device_id=torch.cuda.current_device(),
)
```

### Sharding Strategies Comparison

```
┌─────────────────────┬──────────────┬─────────────────┬─────────────────┐
│ Strategy            │ Memory       │ Communication   │ Use Case        │
├─────────────────────┼──────────────┼─────────────────┼─────────────────┤
│ FULL_SHARD          │ Lowest       │ Highest         │ Largest models  │
│ (ZeRO-3)            │ ~1/N         │ 1.5× DDP        │                 │
├─────────────────────┼──────────────┼─────────────────┼─────────────────┤
│ SHARD_GRAD_OP       │ Medium       │ Same as DDP     │ Large models,   │
│ (ZeRO-2)            │ ~2/N         │                 │ fast comm       │
├─────────────────────┼──────────────┼─────────────────┼─────────────────┤
│ NO_SHARD            │ Highest      │ Same as DDP     │ Small models,   │
│ (DDP)               │ ~1×          │                 │ baseline        │
├─────────────────────┼──────────────┼─────────────────┼─────────────────┤
│ HYBRID_SHARD        │ Configurable │ Configurable    │ Multi-node with │
│ (within node only)  │              │                 │ slow network    │
└─────────────────────┴──────────────┴─────────────────┴─────────────────┘
```

## DeepSpeed ZeRO

Microsoft's DeepSpeed library offers ZeRO with additional features:

```python
import deepspeed

# Config file approach
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 3,  # ZeRO stage (1, 2, or 3)
        "offload_optimizer": {
            "device": "cpu",  # Offload optimizer to CPU
        },
        "offload_param": {
            "device": "cpu",  # Offload params to CPU
        },
    },
    "fp16": {
        "enabled": True,
    },
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

# Training loop
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)  # DeepSpeed handles gradient scaling
    model.step()          # DeepSpeed handles optimizer step
```

### DeepSpeed ZeRO-Infinity

Offload to NVMe for models larger than CPU RAM:

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
        },
    },
}

# Can train trillion-parameter models on limited hardware!
```

## Memory Breakdown Comparison

For a 7B parameter model with Adam optimizer on 4 GPUs:

```
                    DDP         ZeRO-1      ZeRO-2      ZeRO-3
────────────────────────────────────────────────────────────────
Parameters (fp16)   14 GB       14 GB       14 GB       3.5 GB
Gradients (fp16)    14 GB       14 GB       3.5 GB      3.5 GB
Optimizer:
  - fp32 params     28 GB       7 GB        7 GB        7 GB
  - momentum (m)    28 GB       7 GB        7 GB        7 GB
  - variance (v)    28 GB       7 GB        7 GB        7 GB
────────────────────────────────────────────────────────────────
Total per GPU       112 GB      49 GB       38.5 GB     28 GB
────────────────────────────────────────────────────────────────
Fits on 80GB GPU?   No          Yes         Yes         Yes
Fits on 40GB GPU?   No          No          Yes         Yes
```

## FSDP Best Practices

### 1. Choose the Right Wrapping

```python
# Bad: too fine-grained (too much communication)
for layer in model.layers:
    for sublayer in layer:  # Don't wrap every tiny module
        FSDP(sublayer)

# Good: wrap at transformer block level
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)
```

### 2. Use Activation Checkpointing

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)

# Checkpoint transformer blocks
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda m: isinstance(m, TransformerBlock),
)

model = FSDP(model, ...)
```

### 3. Save and Load Checkpoints Correctly

```python
from torch.distributed.fsdp import StateDictType

# Saving
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,  # Gather full state dict
):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "checkpoint.pt")

# Loading
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state_dict = torch.load("checkpoint.pt")
    model.load_state_dict(state_dict)
```

### 4. Handle Gradient Clipping

```python
# With FSDP, use clip_grad_norm_ carefully
model.clip_grad_norm_(max_norm=1.0)  # FSDP method, not torch.nn.utils
```

## Common Issues and Solutions

### Issue 1: OOM with FSDP

```python
# Problem: Memory spike during all-gather
# Solution: Use smaller FSDP units

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,  # Wrap each block
    limit_all_gathers=True,  # Limit concurrent all-gathers
)
```

### Issue 2: Slow Training

```python
# Problem: Too much communication
# Solution: Use SHARD_GRAD_OP if memory allows

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # Less communication
)
```

### Issue 3: Inconsistent Results

```python
# Problem: Different random states across GPUs
# Solution: Sync random seeds

torch.manual_seed(seed + rank)  # Different per GPU (for data loading)
# But sync model initialization before FSDP wrapping
```

## What's Next

FSDP and ZeRO handle memory efficiently, but we can save even more with reduced precision:
- `05_mixed_precision.md` - fp16, bf16, and loss scaling
