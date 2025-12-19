# Lab 05: DeepSpeed Integration

## Objectives

By the end of this lab, you will:
1. Understand DeepSpeed's ZeRO optimization stages
2. Configure DeepSpeed using JSON config files
3. Train models with DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3
4. Use CPU offloading for very large models
5. Integrate DeepSpeed with Hugging Face Transformers

## Prerequisites

- Completed Labs 01-04
- Understanding of ZeRO from `docs/04_zero_and_fsdp.md`
- DeepSpeed installed (`pip install deepspeed`)

## Background

### What is DeepSpeed?

DeepSpeed is Microsoft's deep learning optimization library that provides:
- ZeRO (Zero Redundancy Optimizer) for memory efficiency
- Mixed precision training
- CPU and NVMe offloading
- Gradient accumulation and checkpointing
- Easy configuration via JSON

### DeepSpeed vs FSDP

```
┌─────────────────┬─────────────────────┬─────────────────────┐
│ Feature         │ DeepSpeed           │ FSDP                │
├─────────────────┼─────────────────────┼─────────────────────┤
│ ZeRO Stages     │ 1, 2, 3, Infinity   │ 2, 3 (NO_SHARD=DDP) │
│ CPU Offload     │ Yes                 │ Yes                 │
│ NVMe Offload    │ Yes (ZeRO-Infinity) │ No                  │
│ Config Style    │ JSON file           │ Python API          │
│ Integration     │ HuggingFace native  │ PyTorch native      │
└─────────────────┴─────────────────────┴─────────────────────┘
```

### ZeRO Stages in DeepSpeed

```
ZeRO-1: Partition optimizer states
- Memory: ~4× reduction in optimizer memory
- Communication: Same as DDP

ZeRO-2: + Partition gradients
- Memory: ~8× reduction
- Communication: Same as DDP

ZeRO-3: + Partition parameters
- Memory: Linear scaling with GPUs
- Communication: ~1.5× DDP

ZeRO-Infinity: + NVMe offloading
- Memory: Limited by NVMe, not GPU
- Can train trillion-parameter models!
```

## Instructions

### Step 1: Create DeepSpeed Config

Complete `create_deepspeed_config()` in `src/deepspeed_trainer.py`:

```python
def create_deepspeed_config(
    stage: int = 2,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    train_batch_size: int = 32,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> dict:
    """Create DeepSpeed configuration dict."""
```

### Step 2: Initialize DeepSpeed

Complete `initialize_deepspeed()`:

```python
def initialize_deepspeed(
    model: nn.Module,
    config: dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[DeepSpeedEngine, torch.optim.Optimizer, Any, Any]:
    """Initialize model with DeepSpeed."""
```

### Step 3: Implement Training Loop

Complete the `DeepSpeedTrainer` class:

```python
class DeepSpeedTrainer:
    def train_step(self, batch) -> float:
        """Single training step with DeepSpeed."""

    def train_epoch(self, dataloader, ...) -> float:
        """Full epoch training."""

    def save_checkpoint(self, path, client_state):
        """Save DeepSpeed checkpoint."""

    def load_checkpoint(self, path) -> dict:
        """Load DeepSpeed checkpoint."""
```

### Step 4: CPU Offloading

Implement CPU offloading for large models:

```python
def create_offload_config(
    optimizer_offload: bool = True,
    param_offload: bool = False,
    offload_device: str = "cpu",
) -> dict:
    """Create configuration for CPU/NVMe offloading."""
```

## Running the Lab

```bash
# Install DeepSpeed
pip install deepspeed

# Test DeepSpeed components
pytest tests/test_deepspeed.py -v

# Run with ZeRO-2
deepspeed src/deepspeed_trainer.py --stage 2

# Run with ZeRO-3 and offloading
deepspeed src/deepspeed_trainer.py --stage 3 --offload

# Multi-node training
deepspeed --hostfile hostfile.txt src/deepspeed_trainer.py
```

## DeepSpeed Configuration Examples

### ZeRO-2 Config

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": true
    }
}
```

### ZeRO-3 with CPU Offload

```json
{
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

## Hints

### Hint 1: Basic DeepSpeed Initialization
```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)
```

### Hint 2: Training Step
```python
# DeepSpeed handles gradient scaling automatically
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()
```

### Hint 3: Checkpoint Save/Load
```python
# Save
model_engine.save_checkpoint(path, client_state={'epoch': epoch})

# Load
_, client_state = model_engine.load_checkpoint(path)
epoch = client_state['epoch']
```

### Hint 4: bf16 Instead of fp16
```python
ds_config = {
    "bf16": {
        "enabled": True
    },
    # Don't include fp16 when using bf16
}
```

## Expected Output

```
============================================================
Lab 05 Milestone: DeepSpeed Training Complete!
ZeRO Stage: 3
CPU Offload: Enabled
Initial Loss: 2.3456
Final Loss:   0.4567
Memory per GPU: 4.2 GB
Training Speed: 1250 samples/sec
============================================================
```

## Common Issues

### Issue 1: DeepSpeed Not Installed
```bash
pip install deepspeed
# For CUDA support:
DS_BUILD_OPS=1 pip install deepspeed
```

### Issue 2: Config Validation Error
```python
# Ensure batch sizes are consistent
config = {
    "train_batch_size": batch_size * gradient_accumulation * world_size,
    "train_micro_batch_size_per_gpu": batch_size,
    "gradient_accumulation_steps": gradient_accumulation,
}
```

### Issue 3: Checkpoint Loading Failed
```python
# DeepSpeed checkpoints are directories, not files
model_engine.load_checkpoint(
    checkpoint_dir,  # Directory path, not file
    load_optimizer_states=True,
    load_lr_scheduler_states=True,
)
```

## Milestone

Complete the following test:

```python
def test_milestone_deepspeed_training():
    """MILESTONE: Train with DeepSpeed ZeRO."""
    # Initialize DeepSpeed with ZeRO-2
    # Train for multiple epochs
    # Verify loss decreases
    # Test checkpoint save/load
```

## Integration with HuggingFace

DeepSpeed integrates seamlessly with HuggingFace Trainer:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # Just add this!
    per_device_train_batch_size=8,
    ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

## Comparing ZeRO Stages

Run the comparison script:

```bash
python src/deepspeed_trainer.py --compare

# Output:
# Stage | Memory/GPU | Speed    | Best For
# ------+------------+----------+-------------------
# 0     | 100%       | 1.0x     | Small models (DDP)
# 1     | 75%        | 0.98x    | Medium models
# 2     | 50%        | 0.95x    | Large models
# 3     | 25%        | 0.85x    | Very large models
```

## Next Steps

Congratulations! You've completed all labs in Chapter 11: Distributed Training.

You now understand:
- Multi-GPU setup and communication primitives
- DDP for data parallelism
- FSDP for memory-efficient sharding
- Mixed precision training (fp16/bf16)
- DeepSpeed for advanced ZeRO optimization

Next: Apply these techniques to train your own large models!
