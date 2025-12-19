# Lab 01: Multi-GPU Setup

## Objective

Configure and verify a multi-GPU environment for distributed training.

## What You'll Build

Functions and utilities to:
1. Detect available GPUs
2. Initialize PyTorch distributed process groups
3. Set up environment variables for distributed training
4. Verify GPU communication works correctly

## Prerequisites

Read these docs first:
- `../docs/01_parallelism_strategies.md`
- `../docs/02_ddp.md`

## Hardware Requirements

This lab requires:
- At least 1 CUDA-capable GPU (2+ recommended)
- NCCL library installed (comes with PyTorch)

**Note**: If you only have 1 GPU, tests will still pass but will simulate single-GPU distributed training.

## Instructions

1. Open `src/setup.py`
2. Implement the functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Functions to Implement

### `get_device_info()`
Detect available GPUs and return information about them.

Returns dict with:
- `num_gpus`: Number of available CUDA devices
- `devices`: List of device properties
- `cuda_available`: Boolean

### `setup_distributed(rank, world_size, backend='nccl')`
Initialize PyTorch distributed process group.

- `rank`: This process's rank (0 to world_size-1)
- `world_size`: Total number of processes
- `backend`: Communication backend ('nccl' for GPU, 'gloo' for CPU)

### `cleanup_distributed()`
Clean up distributed process group.

### `verify_gpu_communication(rank, world_size)`
Verify that GPUs can communicate by doing a simple all-reduce.

### `get_local_rank()`
Get the local rank from environment variables.

Used when running with `torchrun`.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Test single-GPU setup
uv run pytest tests/test_setup.py::TestDeviceInfo
```

## Environment Variables

When running distributed training, these environment variables are set:

| Variable | Description | Example |
|----------|-------------|---------|
| `WORLD_SIZE` | Total number of processes | 4 |
| `RANK` | Global rank of this process | 0, 1, 2, 3 |
| `LOCAL_RANK` | Rank within this node | 0, 1 (if 2 GPUs per node) |
| `MASTER_ADDR` | IP of rank 0 process | `localhost` |
| `MASTER_PORT` | Port for communication | `29500` |

## Example Usage

```python
import torch
from setup import (
    get_device_info,
    setup_distributed,
    cleanup_distributed,
    verify_gpu_communication
)

# Check what's available
info = get_device_info()
print(f"Found {info['num_gpus']} GPUs")

# In a distributed script (would be launched with torchrun)
def main(rank, world_size):
    setup_distributed(rank, world_size)

    # Your training code here
    device = torch.device(f'cuda:{rank}')

    # Verify communication works
    success = verify_gpu_communication(rank, world_size)
    print(f"Rank {rank}: Communication {'OK' if success else 'FAILED'}")

    cleanup_distributed()
```

## Running with torchrun

To run a distributed script:

```bash
# Single node, 2 GPUs
torchrun --nproc_per_node=2 my_script.py

# Single node, all available GPUs
torchrun --nproc_per_node=auto my_script.py
```

## Hints

- `torch.cuda.device_count()` returns number of GPUs
- `torch.cuda.get_device_properties(i)` returns GPU properties
- `torch.distributed.init_process_group()` initializes distributed
- `torch.distributed.all_reduce()` is the key communication primitive
- Always call `cleanup_distributed()` at the end (use try/finally)

## Verification

All tests pass = your environment is ready for distributed training!

Next up: Lab 02 where you'll implement actual DDP training.
