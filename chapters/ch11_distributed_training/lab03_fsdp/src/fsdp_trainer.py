"""Lab 03: FSDP - Fully Sharded Data Parallel.

This lab implements memory-efficient distributed training using FSDP,
which shards model parameters, gradients, and optimizer states across GPUs.

Key concepts:
- FSDP wrapping with different sharding strategies
- Mixed precision with FSDP
- Checkpoint saving/loading with state dict types
- Memory-efficient training for large models
"""

import os
from typing import Optional, Callable, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from functools import partial


# =============================================================================
# Simple Model and Dataset for Testing
# =============================================================================

class TransformerBlock(nn.Module):
    """Simple transformer block for FSDP wrapping demonstration."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ffn_out)

        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for FSDP testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 1000, seq_len: int = 32, vocab_size: int = 1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random input sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Target is shifted input (language modeling)
        target = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, target


# =============================================================================
# FSDP Wrapping Functions
# =============================================================================

def create_mixed_precision_policy(
    param_dtype: torch.dtype = torch.float16,
    reduce_dtype: torch.dtype = torch.float16,
    buffer_dtype: torch.dtype = torch.float16,
) -> MixedPrecision:
    """Create FSDP mixed precision policy.

    Args:
        param_dtype: Data type for parameters during forward/backward
        reduce_dtype: Data type for gradient reductions
        buffer_dtype: Data type for buffers

    Returns:
        MixedPrecision policy for FSDP

    Example:
        >>> policy = create_mixed_precision_policy(torch.bfloat16)
        >>> isinstance(policy, MixedPrecision)
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_mixed_precision_policy")


def create_transformer_wrap_policy(
    transformer_layer_cls: type = TransformerBlock,
) -> Callable:
    """Create auto-wrap policy for transformer models.

    FSDP can automatically wrap modules at specific granularities.
    For transformers, we typically wrap each transformer block.

    Args:
        transformer_layer_cls: The transformer block class to wrap

    Returns:
        A callable wrap policy for FSDP

    Example:
        >>> policy = create_transformer_wrap_policy(TransformerBlock)
        >>> callable(policy)
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_transformer_wrap_policy")


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    auto_wrap_policy: Optional[Callable] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    device_id: Optional[int] = None,
) -> FSDP:
    """Wrap model with FSDP for distributed training.

    Args:
        model: The model to wrap
        sharding_strategy: How to shard model states
            - FULL_SHARD: Shard everything (ZeRO-3)
            - SHARD_GRAD_OP: Shard grads and optimizer (ZeRO-2)
            - NO_SHARD: No sharding (like DDP)
        auto_wrap_policy: Policy for automatic sub-module wrapping
        mixed_precision: Mixed precision configuration
        device_id: CUDA device ID

    Returns:
        FSDP-wrapped model

    Example:
        >>> model = SimpleTransformer()
        >>> # In distributed context:
        >>> # wrapped = wrap_model_fsdp(model, ShardingStrategy.FULL_SHARD)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement wrap_model_fsdp")


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int,
    epoch: int,
    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    **kwargs,
) -> None:
    """Save FSDP checkpoint with proper state dict handling.

    FSDP checkpoints require special handling:
    - FULL_STATE_DICT: Gather full state to rank 0, save single file
    - SHARDED_STATE_DICT: Each rank saves its shard

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        path: Path to save checkpoint
        rank: Current process rank
        epoch: Current epoch number
        state_dict_type: Type of state dict to save
        **kwargs: Additional items to save

    Example:
        >>> # save_fsdp_checkpoint(model, optimizer, "ckpt.pt", rank=0, epoch=5)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement save_fsdp_checkpoint")


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
) -> Dict[str, Any]:
    """Load FSDP checkpoint with proper state dict handling.

    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer (optional)
        path: Path to checkpoint
        state_dict_type: Type of state dict used when saving

    Returns:
        Checkpoint dict with epoch and any extra saved items

    Example:
        >>> # checkpoint = load_fsdp_checkpoint(model, optimizer, "ckpt.pt")
        >>> # epoch = checkpoint['epoch']
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement load_fsdp_checkpoint")


# =============================================================================
# FSDP Trainer Class
# =============================================================================

class FSDPTrainer:
    """Trainer class for FSDP distributed training.

    This trainer handles:
    - FSDP model wrapping
    - Training loop with proper gradient handling
    - Checkpoint save/load with FSDP state dicts
    - Memory tracking

    Args:
        model: Model to train (will be wrapped with FSDP)
        rank: Process rank
        world_size: Total number of processes
        sharding_strategy: FSDP sharding strategy
        mixed_precision: Mixed precision policy (optional)

    Example:
        >>> model = SimpleTransformer()
        >>> # In distributed context:
        >>> # trainer = FSDPTrainer(model, rank=0, world_size=2)
        >>> # loss = trainer.train_epoch(dataloader, optimizer, criterion)
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        mixed_precision: Optional[MixedPrecision] = None,
    ):
        """Initialize FSDP trainer."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.__init__")

    @property
    def unwrapped_model(self) -> nn.Module:
        """Get the underlying model without FSDP wrapper.

        Returns:
            The original unwrapped model
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.unwrapped_model")

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Perform a single training step.

        Args:
            batch: Tuple of (input, target) tensors
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Loss value as float
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.train_step")

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        sampler: Optional[DistributedSampler] = None,
        epoch: int = 0,
        max_grad_norm: Optional[float] = None,
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            sampler: Distributed sampler (for setting epoch)
            epoch: Current epoch number
            max_grad_norm: Max gradient norm for clipping (optional)

        Returns:
            Average loss for the epoch
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.train_epoch")

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ) -> None:
        """Save FSDP checkpoint.

        Only rank 0 saves when using FULL_STATE_DICT.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer
            **kwargs: Additional items to save
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.save_checkpoint")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load FSDP checkpoint.

        Args:
            path: Path to checkpoint
            optimizer: Optimizer to restore (optional)

        Returns:
            Checkpoint dict
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.load_checkpoint")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics.

        Returns:
            Dict with allocated and reserved memory in GB
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement FSDPTrainer.get_memory_stats")


# =============================================================================
# Utility Functions
# =============================================================================

def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DistributedSampler]:
    """Create dataloader with distributed sampler.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        rank: Process rank
        world_size: Total processes
        num_workers: Number of data loading workers

    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, sampler


def compare_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    sharding_strategies: list,
) -> Dict[str, float]:
    """Compare memory usage across different sharding strategies.

    Args:
        model: Model to test
        batch_size: Batch size
        seq_len: Sequence length
        sharding_strategies: List of strategies to compare

    Returns:
        Dict mapping strategy name to memory usage in GB
    """
    # This is a utility for demonstration - actual implementation
    # would require running distributed processes
    results = {}

    for strategy in sharding_strategies:
        # Estimate based on strategy
        base_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        base_memory_gb = base_memory / (1024 ** 3)

        if strategy == ShardingStrategy.NO_SHARD:
            results["NO_SHARD"] = base_memory_gb * 4  # params + grads + optimizer
        elif strategy == ShardingStrategy.SHARD_GRAD_OP:
            results["SHARD_GRAD_OP"] = base_memory_gb * 2.5
        elif strategy == ShardingStrategy.FULL_SHARD:
            results["FULL_SHARD"] = base_memory_gb * 1.5

    return results


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Training with {world_size} GPUs")
        print("=" * 60)

    # Create model
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
    )

    # Create mixed precision policy
    mixed_precision = create_mixed_precision_policy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # Create trainer with FSDP
    trainer = FSDPTrainer(
        model,
        rank=rank,
        world_size=world_size,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
    )

    # Create dataset and dataloader
    dataset = SimpleDataset(size=10000, seq_len=128, vocab_size=10000)
    dataloader, sampler = create_distributed_dataloader(
        dataset, batch_size=16, rank=rank, world_size=world_size
    )

    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5
    initial_loss = None
    final_loss = None

    for epoch in range(num_epochs):
        loss = trainer.train_epoch(
            dataloader, optimizer, criterion,
            sampler=sampler, epoch=epoch, max_grad_norm=1.0
        )

        if epoch == 0:
            initial_loss = loss
        final_loss = loss

        if rank == 0:
            memory_stats = trainer.get_memory_stats()
            print(f"Epoch {epoch}: Loss = {loss:.4f}, "
                  f"Memory = {memory_stats['allocated']:.2f} GB")

    # Save checkpoint
    trainer.save_checkpoint("fsdp_checkpoint.pt", epoch=num_epochs-1, optimizer=optimizer)

    # Print results
    if rank == 0:
        print(f"\n{'='*60}")
        print("Lab 03 Milestone: FSDP Training Complete!")
        print(f"Sharding Strategy: FULL_SHARD")
        memory_stats = trainer.get_memory_stats()
        print(f"Memory per GPU: {memory_stats['allocated']:.2f} GB")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss:   {final_loss:.4f}")
        print(f"{'='*60}\n")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
