"""
Lab 02: DDP Training

Implement distributed data parallel training.

Your task: Complete the classes and functions below to make all tests pass.
Run: uv run pytest tests/
"""

import os
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Create a DataLoader with DistributedSampler for DDP training.

    The DistributedSampler ensures each process gets a different subset of data.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size PER GPU (not total)
        rank: This process's rank
        world_size: Total number of processes
        shuffle: Whether to shuffle (handled by sampler)
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete last batch (recommended for DDP)

    Returns:
        Tuple of (DataLoader, DistributedSampler)
        Keep the sampler reference to call set_epoch() each epoch

    Example:
        >>> dataloader, sampler = create_distributed_dataloader(
        ...     dataset, batch_size=32, rank=0, world_size=4
        ... )
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)  # Important!
        ...     for batch in dataloader:
        ...         train_step(batch)
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Create DistributedSampler with dataset, num_replicas=world_size, rank=rank
    # 2. Create DataLoader with sampler (not shuffle - sampler handles it)
    # 3. Return both dataloader and sampler
    raise NotImplementedError("Implement create_distributed_dataloader")


def reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce a tensor by averaging across all processes.

    Useful for computing global metrics (loss, accuracy) across all GPUs.

    Args:
        tensor: Local tensor to reduce
        world_size: Number of processes

    Returns:
        Averaged tensor (same on all ranks)

    Example:
        >>> local_loss = torch.tensor(0.5)  # Different on each rank
        >>> global_loss = reduce_mean(local_loss, world_size=4)
        >>> # global_loss is average of all local losses
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Clone tensor to avoid modifying input
    # 2. dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # 3. Divide by world_size
    raise NotImplementedError("Implement reduce_mean")


class DDPTrainer:
    """
    Distributed Data Parallel Trainer.

    Wraps a model with DDP and provides training utilities.

    Attributes:
        model: The DDP-wrapped model
        rank: This process's rank
        world_size: Total number of processes
        device: CUDA device for this rank
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        find_unused_parameters: bool = False,
    ):
        """
        Initialize DDP trainer.

        Args:
            model: PyTorch model (will be wrapped with DDP)
            rank: This process's rank
            world_size: Total number of processes
            find_unused_parameters: Set True if some params don't get gradients

        Example:
            >>> model = TransformerModel()
            >>> trainer = DDPTrainer(model, rank=0, world_size=4)
            >>> # trainer.model is now DDP-wrapped
        """
        # YOUR CODE HERE
        # Steps:
        # 1. Store rank, world_size
        # 2. Determine device (cuda:rank or cpu)
        # 3. Move model to device
        # 4. Wrap with DDP: DDP(model, device_ids=[rank], find_unused_parameters=...)
        raise NotImplementedError("Implement DDPTrainer.__init__")

    @property
    def unwrapped_model(self) -> nn.Module:
        """
        Get the underlying model without DDP wrapper.

        Useful for saving checkpoints or accessing model attributes.

        Returns:
            The original model (model.module for DDP)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement unwrapped_model")

    def train_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Perform a single training step.

        Args:
            batch: Input batch (will be moved to device)
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Loss value (scalar)

        Note:
            DDP automatically synchronizes gradients during backward()
        """
        # YOUR CODE HERE
        # Steps:
        # 1. Move batch to device
        # 2. Zero gradients
        # 3. Forward pass
        # 4. Compute loss
        # 5. Backward (gradients auto-synced by DDP)
        # 6. Optimizer step
        # 7. Return loss.item()
        raise NotImplementedError("Implement train_step")

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        sampler: Optional[DistributedSampler] = None,
        epoch: Optional[int] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            criterion: Loss function
            sampler: DistributedSampler (to set epoch for shuffling)
            epoch: Current epoch number (for sampler)

        Returns:
            Average loss over the epoch (reduced across all ranks)
        """
        # YOUR CODE HERE
        # Steps:
        # 1. If sampler and epoch provided, call sampler.set_epoch(epoch)
        # 2. Set model to train mode
        # 3. Loop over batches, accumulate loss
        # 4. Compute average loss
        # 5. Reduce mean across all ranks
        # 6. Return averaged loss
        raise NotImplementedError("Implement train_epoch")

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        """
        Save a training checkpoint.

        Only saves on rank 0 to avoid file corruption.
        Other ranks wait at a barrier.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer (to save state)
            **kwargs: Additional items to save

        Example:
            >>> trainer.save_checkpoint(
            ...     'checkpoint.pt',
            ...     epoch=5,
            ...     optimizer=optimizer,
            ...     best_loss=0.1
            ... )
        """
        # YOUR CODE HERE
        # Steps:
        # 1. If rank == 0:
        #    - Create checkpoint dict with model state, optimizer state, epoch, kwargs
        #    - Use self.unwrapped_model.state_dict() (not self.model)
        #    - torch.save(checkpoint, path)
        # 2. dist.barrier() to synchronize
        raise NotImplementedError("Implement save_checkpoint")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        All ranks load the checkpoint (with proper device mapping).

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to load state into (optional)

        Returns:
            Checkpoint dictionary (for accessing epoch, etc.)

        Example:
            >>> checkpoint = trainer.load_checkpoint('checkpoint.pt', optimizer)
            >>> start_epoch = checkpoint['epoch'] + 1
        """
        # YOUR CODE HERE
        # Steps:
        # 1. Create map_location for proper device mapping
        # 2. torch.load with map_location
        # 3. Load model state into self.unwrapped_model
        # 4. If optimizer provided, load optimizer state
        # 5. Return checkpoint dict
        raise NotImplementedError("Implement load_checkpoint")

    def synchronize(self) -> None:
        """
        Synchronize all processes.

        Blocks until all ranks reach this point.
        Useful after checkpointing or before evaluation.
        """
        # YOUR CODE HERE
        # Hint: dist.barrier()
        raise NotImplementedError("Implement synchronize")


def sync_gradients_manual(model: nn.Module, world_size: int) -> None:
    """
    Manually synchronize gradients across all processes.

    This is what DDP does automatically. Implemented here for understanding.

    Args:
        model: Model with gradients to synchronize
        world_size: Number of processes

    Note:
        You normally don't need this - DDP handles it.
        This is for educational purposes.
    """
    # YOUR CODE HERE
    # Steps:
    # 1. Loop over model.parameters()
    # 2. For each param with grad:
    #    - dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    #    - param.grad /= world_size
    raise NotImplementedError("Implement sync_gradients_manual")


class SimpleDataset(Dataset):
    """
    Simple dataset for testing DDP.

    Generates random input-target pairs.
    """

    def __init__(self, size: int, input_dim: int, output_dim: int):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Generate fixed random data
        torch.manual_seed(42)
        self.inputs = torch.randn(size, input_dim)
        self.targets = torch.randn(size, output_dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """
    Simple model for testing DDP.

    Two linear layers with ReLU.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
