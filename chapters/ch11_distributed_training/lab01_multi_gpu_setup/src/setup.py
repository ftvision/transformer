"""
Lab 01: Multi-GPU Setup

Configure and verify multi-GPU distributed environment.

Your task: Complete the functions below to make all tests pass.
Run: uv run pytest tests/
"""

import os
from typing import Dict, List, Any, Optional

import torch
import torch.distributed as dist


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available CUDA devices.

    Returns:
        Dictionary containing:
            - 'cuda_available': bool, whether CUDA is available
            - 'num_gpus': int, number of CUDA devices
            - 'devices': list of dicts, one per device with:
                - 'name': str, device name
                - 'total_memory_gb': float, total memory in GB
                - 'compute_capability': tuple, (major, minor)
            - 'current_device': int, current default device index

    Example:
        >>> info = get_device_info()
        >>> print(f"Found {info['num_gpus']} GPUs")
        >>> if info['num_gpus'] > 0:
        ...     print(f"GPU 0: {info['devices'][0]['name']}")
    """
    # YOUR CODE HERE
    # Hints:
    # - torch.cuda.is_available()
    # - torch.cuda.device_count()
    # - torch.cuda.get_device_properties(i) for device i
    # - properties.name, properties.total_memory, properties.major, properties.minor
    raise NotImplementedError("Implement get_device_info")


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
    master_addr: str = 'localhost',
    master_port: str = '29500'
) -> None:
    """
    Initialize PyTorch distributed process group.

    This sets up the communication infrastructure for distributed training.
    Must be called once per process before any distributed operations.

    Args:
        rank: Global rank of this process (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend
                 - 'nccl': NVIDIA GPU communication (fastest for GPU)
                 - 'gloo': CPU fallback, also works for GPU
        master_addr: IP address of rank 0 process
        master_port: Port for communication

    Raises:
        RuntimeError: If distributed is already initialized

    Example:
        >>> # In process with rank 0
        >>> setup_distributed(rank=0, world_size=4)
        >>> # Now can use dist.all_reduce(), dist.broadcast(), etc.
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Set environment variables: MASTER_ADDR, MASTER_PORT
    # 2. Call dist.init_process_group(backend, rank=rank, world_size=world_size)
    # 3. Set CUDA device for this rank: torch.cuda.set_device(rank % num_gpus)
    raise NotImplementedError("Implement setup_distributed")


def cleanup_distributed() -> None:
    """
    Clean up distributed process group.

    Should be called at the end of distributed training.
    Safe to call even if distributed is not initialized.

    Example:
        >>> try:
        ...     setup_distributed(rank, world_size)
        ...     # training code
        ... finally:
        ...     cleanup_distributed()
    """
    # YOUR CODE HERE
    # Hint: dist.destroy_process_group()
    # Check dist.is_initialized() first
    raise NotImplementedError("Implement cleanup_distributed")


def is_distributed() -> bool:
    """
    Check if distributed training is initialized.

    Returns:
        True if distributed process group is initialized, False otherwise
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement is_distributed")


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Global rank (0 to world_size-1)
        Returns 0 if distributed is not initialized (single process)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_rank")


def get_world_size() -> int:
    """
    Get the total number of processes.

    Returns:
        World size (total number of processes)
        Returns 1 if distributed is not initialized (single process)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_world_size")


def get_local_rank() -> int:
    """
    Get the local rank from environment variables.

    Local rank is the rank within the current node (machine).
    Set by torchrun/torch.distributed.launch.

    Returns:
        Local rank from LOCAL_RANK environment variable
        Returns 0 if not set

    Example:
        # 2 nodes, 4 GPUs each
        # Node 0: local_ranks 0,1,2,3 (global ranks 0,1,2,3)
        # Node 1: local_ranks 0,1,2,3 (global ranks 4,5,6,7)
    """
    # YOUR CODE HERE
    # Hint: os.environ.get('LOCAL_RANK', '0')
    raise NotImplementedError("Implement get_local_rank")


def verify_gpu_communication(rank: int, world_size: int) -> bool:
    """
    Verify that GPU communication works via a simple all-reduce.

    Each process creates a tensor with its rank, then all-reduce sums them.
    Expected result: sum of ranks = 0 + 1 + ... + (world_size-1)

    Args:
        rank: This process's rank
        world_size: Total number of processes

    Returns:
        True if all-reduce produces correct result, False otherwise

    Example:
        >>> # With world_size=4
        >>> # Each process starts with tensor([rank])
        >>> # After all-reduce: tensor([0+1+2+3]) = tensor([6])
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Create tensor with rank value on correct device
    # 2. dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # 3. Check result equals expected sum
    raise NotImplementedError("Implement verify_gpu_communication")


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast a tensor from source rank to all other ranks.

    Args:
        tensor: Tensor to broadcast (only src's tensor matters for input)
        src: Source rank to broadcast from

    Returns:
        The broadcasted tensor (same on all ranks)

    Example:
        >>> # On rank 0: tensor = torch.tensor([1, 2, 3])
        >>> # On rank 1: tensor = torch.tensor([0, 0, 0])
        >>> result = broadcast_tensor(tensor, src=0)
        >>> # On all ranks: result = torch.tensor([1, 2, 3])
    """
    # YOUR CODE HERE
    # Hint: dist.broadcast(tensor, src=src)
    raise NotImplementedError("Implement broadcast_tensor")


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all ranks into a list.

    Args:
        tensor: Local tensor from this rank

    Returns:
        List of tensors, one from each rank (same on all ranks)

    Example:
        >>> # Rank 0: tensor = torch.tensor([0])
        >>> # Rank 1: tensor = torch.tensor([1])
        >>> result = all_gather_tensors(tensor)
        >>> # On all ranks: result = [tensor([0]), tensor([1])]
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Create output list with empty tensors (same shape as input)
    # 2. dist.all_gather(output_list, tensor)
    raise NotImplementedError("Implement all_gather_tensors")


def print_rank_0(message: str) -> None:
    """
    Print message only on rank 0.

    Useful for logging in distributed training to avoid duplicate prints.

    Args:
        message: Message to print

    Example:
        >>> print_rank_0("Starting training...")  # Only prints on rank 0
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement print_rank_0")
