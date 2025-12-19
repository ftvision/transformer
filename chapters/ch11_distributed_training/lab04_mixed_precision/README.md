# Lab 04: Mixed Precision Training

## Objectives

By the end of this lab, you will:
1. Understand fp16 vs bf16 number formats and their trade-offs
2. Implement mixed precision training using PyTorch AMP
3. Use GradScaler for fp16 loss scaling
4. Know which operations should stay in fp32
5. Combine mixed precision with DDP and FSDP

## Prerequisites

- Completed Lab 02 (DDP Training)
- Understanding of gradient computation
- Read `docs/05_mixed_precision.md`

## Background

### Why Mixed Precision?

Single precision (fp32) uses 32 bits per number. Half precision uses 16 bits:

```
Memory savings:
- 7B model in fp32: 28 GB
- 7B model in fp16: 14 GB
- 50% memory reduction!

Speed improvement:
- Tensor cores optimized for fp16/bf16
- 2-8× faster matrix multiplications
```

### fp16 vs bf16

```
┌─────────┬─────────┬──────────────┬───────────────────────┐
│ Format  │ Bits    │ Range        │ Precision             │
├─────────┼─────────┼──────────────┼───────────────────────┤
│ fp32    │ 32      │ ±3.4×10³⁸    │ ~7 decimal digits     │
│ fp16    │ 16      │ ±65504       │ ~3.3 decimal digits   │
│ bf16    │ 16      │ ±3.4×10³⁸    │ ~2.4 decimal digits   │
└─────────┴─────────┴──────────────┴───────────────────────┘

fp16: More precision, limited range → needs loss scaling
bf16: Less precision, full range → drop-in replacement for fp32
```

### The Loss Scaling Solution

fp16's limited range causes gradient underflow. Loss scaling fixes this:

```
Standard training:
loss → backward → gradients → optimizer step

With loss scaling:
loss × scale → backward → gradients ÷ scale → optimizer step
                                    ↑
                    Gradients are larger during backward,
                    avoiding underflow
```

## Instructions

### Step 1: Implement AMP Context Manager

Complete `mixed_precision_forward()` in `src/mixed_precision.py`:

```python
def mixed_precision_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run forward pass with automatic mixed precision."""
```

Use `torch.autocast` for automatic dtype casting.

### Step 2: Implement GradScaler Wrapper

Complete `scaled_backward()`:

```python
def scaled_backward(
    loss: torch.Tensor,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: Optional[float] = None,
) -> bool:
    """Perform scaled backward pass and optimizer step."""
```

This should:
1. Scale the loss
2. Backward pass
3. Unscale gradients
4. Optionally clip gradients
5. Check for inf/nan and skip step if found
6. Update scaler

### Step 3: Implement the Mixed Precision Trainer

Complete the `MixedPrecisionTrainer` class:

```python
class MixedPrecisionTrainer:
    """Trainer with automatic mixed precision support."""

    def train_step(self, batch, optimizer, criterion) -> Tuple[float, bool]:
        """Returns (loss, step_skipped)."""

    def train_epoch(self, dataloader, optimizer, criterion, ...) -> Dict:
        """Returns stats including skipped steps."""
```

### Step 4: Implement Precision-Sensitive Operations

Some operations should stay in fp32. Complete:

```python
def fp32_layer_norm(x: torch.Tensor, normalized_shape, ...) -> torch.Tensor:
    """Layer norm that always uses fp32 for numerical stability."""

def fp32_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that uses fp32 to avoid overflow."""
```

## Running the Lab

```bash
# Test mixed precision components
pytest tests/test_mixed_precision.py -v

# Run training with fp16
python src/mixed_precision.py --dtype fp16

# Run training with bf16 (if supported)
python src/mixed_precision.py --dtype bf16

# Multi-GPU with mixed precision
torchrun --nproc_per_node=2 src/mixed_precision.py --dtype fp16
```

## Hints

### Hint 1: Using autocast
```python
from torch.cuda.amp import autocast

with autocast(dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
# backward is outside autocast
loss.backward()
```

### Hint 2: GradScaler Usage
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# In training loop:
with autocast(dtype=torch.float16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # For gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

### Hint 3: bf16 Doesn't Need Scaler
```python
# bf16 has full range, no scaling needed
with autocast(dtype=torch.bfloat16):
    loss = model(batch)

loss.backward()  # No scaler!
optimizer.step()
```

### Hint 4: Force fp32 for Specific Ops
```python
with autocast(dtype=torch.float16):
    x = model.layer1(input)  # fp16

    # Force fp32 for sensitive operation
    with autocast(enabled=False):
        x = x.float()
        x = sensitive_op(x)
        x = x.half()

    output = model.layer2(x)  # back to fp16
```

## Expected Output

```
============================================================
Lab 04 Milestone: Mixed Precision Training Complete!
Precision: float16
Using GradScaler: True
Initial Loss: 2.3456
Final Loss:   0.5678
Skipped Steps: 3 (due to gradient overflow)
Memory Reduction: 45.2%
============================================================
```

## Common Issues

### Issue 1: NaN/Inf Loss
```python
# Check GradScaler state
print(f"Current scale: {scaler.get_scale()}")

# Solution: Scale might need to decrease
# GradScaler handles this automatically
```

### Issue 2: Training Instability
```python
# Keep problematic layers in fp32
class StableModel(nn.Module):
    def forward(self, x):
        with autocast(enabled=False):
            x = self.sensitive_layer(x.float())
        return x
```

### Issue 3: bf16 Not Available
```python
# Check hardware support
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16  # Fall back
```

## Milestone

Complete the following test:

```python
def test_milestone_mixed_precision_training():
    """MILESTONE: Train with mixed precision."""
    # Train with fp16 and GradScaler
    # Verify loss decreases
    # Verify memory usage is lower than fp32
    # Handle any skipped steps gracefully
```

## Comparing Precision Formats

Run the comparison script to see the differences:

```bash
python src/mixed_precision.py --compare

# Output:
# Format    | Memory  | Speed   | Stability
# fp32      | 100%    | 1.0x    | Stable
# fp16      | 50%     | 2.1x    | Needs scaling
# bf16      | 50%     | 2.0x    | Stable
```

## Next Steps

After completing this lab:
1. Combine mixed precision with FSDP (see Lab 03)
2. Experiment with which operations benefit most from fp16
3. Proceed to Lab 05 (DeepSpeed) for more advanced features
