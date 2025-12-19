# Lab 03: FSDP - Fully Sharded Data Parallel

## Objectives

By the end of this lab, you will:
1. Understand the memory redundancy problem in standard DDP
2. Learn how FSDP shards model states across GPUs
3. Implement training with FSDP using different sharding strategies
4. Apply mixed precision with FSDP
5. Master checkpoint saving/loading with FSDP

## Prerequisites

- Completed Lab 01 (Multi-GPU Setup)
- Completed Lab 02 (DDP Training)
- Understanding of ZeRO concepts from `docs/04_zero_and_fsdp.md`
- PyTorch >= 2.0 with FSDP support

## Background

### The Memory Problem with DDP

Standard DDP replicates everything on every GPU:
```
7B model with Adam optimizer on 4 GPUs:
- Parameters: 14GB × 4 = 56GB
- Gradients: 14GB × 4 = 56GB
- Optimizer (m, v, fp32): 56GB × 4 = 224GB
Total: 336GB (but only 84GB of unique data!)
```

### FSDP Solution

FSDP implements ZeRO-3 concepts, sharding:
- Parameters (gathered on-demand for computation)
- Gradients (reduce-scattered after backward)
- Optimizer states (each GPU updates its shard)

```
Same 7B model with FSDP on 4 GPUs:
- Parameters: 14GB / 4 = 3.5GB per GPU
- Gradients: 14GB / 4 = 3.5GB per GPU
- Optimizer: 56GB / 4 = 14GB per GPU
Total per GPU: ~21GB (vs 84GB with DDP!)
```

### Sharding Strategies

```
FULL_SHARD (ZeRO-3):
- Maximum memory savings
- Most communication
- Use for largest models

SHARD_GRAD_OP (ZeRO-2):
- Shard gradients + optimizer
- Less communication
- Good balance

NO_SHARD (DDP-like):
- No sharding
- Minimal communication
- For comparison/debugging
```

## Instructions

### Step 1: Implement FSDP Wrapping

Complete `wrap_model_fsdp()` in `src/fsdp_trainer.py`:

```python
def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy,
    auto_wrap_policy: Optional[Callable] = None,
    mixed_precision: Optional[MixedPrecision] = None,
) -> FSDP:
    """Wrap model with FSDP."""
```

Key considerations:
- Choose appropriate sharding strategy
- Apply auto-wrap policy for transformer models
- Configure mixed precision if provided

### Step 2: Implement Mixed Precision Policy

Complete `create_mixed_precision_policy()`:

```python
def create_mixed_precision_policy(
    param_dtype: torch.dtype = torch.float16,
    reduce_dtype: torch.dtype = torch.float16,
    buffer_dtype: torch.dtype = torch.float16,
) -> MixedPrecision:
    """Create FSDP mixed precision policy."""
```

### Step 3: Implement the FSDP Trainer

Complete the `FSDPTrainer` class with methods:
- `train_step()` - Single training step
- `train_epoch()` - Full epoch training
- `save_checkpoint()` - FSDP-aware checkpoint saving
- `load_checkpoint()` - FSDP-aware checkpoint loading

### Step 4: Implement Checkpoint Utilities

FSDP checkpointing is different from DDP! You need to handle:
- `FULL_STATE_DICT` - Gather to rank 0 for saving
- `SHARDED_STATE_DICT` - Each rank saves its shard

```python
def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    **kwargs
):
    """Save FSDP checkpoint with proper state dict handling."""
```

## Running the Lab

```bash
# Test single-process FSDP wrapping
pytest tests/test_fsdp_trainer.py -v

# Run multi-GPU training (requires 2+ GPUs)
torchrun --nproc_per_node=2 src/fsdp_trainer.py

# Run with specific sharding strategy
torchrun --nproc_per_node=4 src/fsdp_trainer.py --sharding full_shard
```

## Hints

### Hint 1: Auto-Wrap Policy
For transformer models, use `transformer_auto_wrap_policy`:
```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)
```

### Hint 2: State Dict Context Manager
Use FSDP's state dict context manager:
```python
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state_dict = model.state_dict()
```

### Hint 3: Mixed Precision
Create MixedPrecision policy:
```python
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
```

### Hint 4: Gradient Clipping
Use FSDP's built-in clip_grad_norm_:
```python
model.clip_grad_norm_(max_norm=1.0)  # Not torch.nn.utils!
```

## Expected Output

```
============================================================
Lab 03 Milestone: FSDP Training Complete!
Sharding Strategy: FULL_SHARD
Memory per GPU: 2.3 GB (vs 8.1 GB with DDP)
Initial Loss: 2.3456
Final Loss:   0.4567
Memory Savings: 71.6%
============================================================
```

## Common Issues

### Issue 1: OOM during all-gather
```python
# Solution: Use smaller FSDP units
model = FSDP(model, limit_all_gathers=True)
```

### Issue 2: Checkpoint size
```python
# FULL_STATE_DICT gathers to rank 0 - needs memory
# Use SHARDED_STATE_DICT for distributed checkpointing
```

### Issue 3: Module wrapping order
```python
# Always wrap inner modules before outer modules
# Use auto_wrap_policy to handle this automatically
```

## Milestone

Complete the following test to verify your implementation:

```python
def test_milestone_fsdp_training():
    """MILESTONE: Complete FSDP training with memory savings."""
    # Train with FSDP
    # Verify loss decreases
    # Verify memory usage is lower than DDP baseline
    # Test checkpoint save/load
```

## Next Steps

After completing this lab:
1. Try different sharding strategies and compare memory/speed
2. Experiment with `HYBRID_SHARD` for multi-node training
3. Proceed to Lab 04 (Mixed Precision) for more memory optimization
