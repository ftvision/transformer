# Lab 02: DDP Training

## Objective

Implement a complete training loop using PyTorch's DistributedDataParallel (DDP).

## What You'll Build

1. A DDP wrapper for any PyTorch model
2. A DistributedSampler-aware data loading pipeline
3. A complete training loop with gradient synchronization
4. Checkpoint saving/loading for distributed training

## Prerequisites

- Complete Lab 01 (multi-GPU setup)
- Read `../docs/02_ddp.md`

## Key Concepts

DDP replicates your model on each GPU and synchronizes gradients after backward:

```
GPU 0: model(batch_0) → loss_0 → backward → ∇_0 ──┐
GPU 1: model(batch_1) → loss_1 → backward → ∇_1 ──┼→ All-Reduce → avg(∇)
GPU 2: model(batch_2) → loss_2 → backward → ∇_2 ──┘
                                                       ↓
                                               optimizer.step()
```

## Instructions

1. Open `src/ddp_trainer.py`
2. Implement the classes and functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Classes to Implement

### `DDPTrainer`

A trainer class that handles DDP training:

```python
class DDPTrainer:
    def __init__(self, model, rank, world_size):
        """Wrap model with DDP."""

    def train_epoch(self, dataloader, optimizer, criterion):
        """Run one training epoch with gradient sync."""

    def save_checkpoint(self, path, epoch, optimizer):
        """Save model checkpoint (rank 0 only)."""

    def load_checkpoint(self, path, optimizer):
        """Load checkpoint and broadcast to all ranks."""
```

### Functions

- `create_distributed_dataloader()` - Create DataLoader with DistributedSampler
- `reduce_mean()` - Average a value across all processes
- `sync_gradients()` - Manually synchronize gradients (for understanding)

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_ddp_trainer.py::TestDDPWrapper

# Run with verbose output
uv run pytest tests/ -v
```

## Key DDP Patterns

### 1. Wrapping the Model

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(rank)
model = DDP(model, device_ids=[rank])
```

### 2. Using DistributedSampler

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

# IMPORTANT: Set epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Different shuffle each epoch
    train_epoch(dataloader)
```

### 3. Saving Checkpoints

```python
# Only save on rank 0 to avoid corruption
if rank == 0:
    torch.save({
        'model': model.module.state_dict(),  # .module to get unwrapped model
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, 'checkpoint.pt')

# Synchronize before continuing
dist.barrier()
```

### 4. Loading Checkpoints

```python
# Load on all ranks (map_location handles device)
map_location = {'cuda:0': f'cuda:{rank}'}
checkpoint = torch.load('checkpoint.pt', map_location=map_location)

model.module.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

## Common Pitfalls

1. **Forgetting sampler.set_epoch()**: Same data order each epoch
2. **Saving with model instead of model.module**: Wrong state dict
3. **Not using barrier after save**: Other ranks may read incomplete file
4. **Unequal batch sizes**: Last batch may differ across ranks

## Example Training Script Structure

```python
def main(rank, world_size):
    # Setup
    setup_distributed(rank, world_size)

    # Model
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Data
    sampler = DistributedSampler(dataset, world_size, rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Training
    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            loss = model(batch).mean()
            loss.backward()  # Gradients synced automatically!
            optimizer.step()

        # Checkpoint (rank 0 only)
        if rank == 0:
            torch.save(model.module.state_dict(), f'epoch_{epoch}.pt')
        dist.barrier()

    cleanup_distributed()
```

## Verification

All tests pass = you can train models with DDP!

Next up: Lab 03 where you'll use FSDP for memory-efficient training.
