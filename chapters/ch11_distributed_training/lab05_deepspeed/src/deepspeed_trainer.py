"""Lab 05: DeepSpeed Integration.

This lab implements distributed training using DeepSpeed's ZeRO
optimization for memory-efficient training of large models.

Key concepts:
- DeepSpeed ZeRO stages (1, 2, 3)
- Configuration via JSON/dict
- CPU and NVMe offloading
- Gradient checkpointing
"""

import os
import argparse
import json
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# DeepSpeed import - handle case where not installed
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedEngine = None


# =============================================================================
# Simple Model and Dataset for Testing
# =============================================================================

class TransformerBlock(nn.Module):
    """Simple transformer block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ffn_out)
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for DeepSpeed testing."""

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
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        target = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, target


# =============================================================================
# DeepSpeed Configuration
# =============================================================================

def create_deepspeed_config(
    stage: int = 2,
    fp16: bool = True,
    bf16: bool = False,
    gradient_accumulation_steps: int = 1,
    train_batch_size: int = 32,
    train_micro_batch_size_per_gpu: int = 8,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    offload_device: str = "cpu",
) -> dict:
    """Create DeepSpeed configuration dictionary.

    Args:
        stage: ZeRO optimization stage (0, 1, 2, or 3)
        fp16: Enable fp16 training
        bf16: Enable bf16 training (mutually exclusive with fp16)
        gradient_accumulation_steps: Number of accumulation steps
        train_batch_size: Global batch size
        train_micro_batch_size_per_gpu: Batch size per GPU per step
        offload_optimizer: Offload optimizer states to CPU
        offload_param: Offload parameters to CPU (ZeRO-3 only)
        offload_device: Device for offloading ("cpu" or "nvme")

    Returns:
        DeepSpeed configuration dictionary

    Example:
        >>> config = create_deepspeed_config(stage=2, fp16=True)
        >>> config['zero_optimization']['stage']
        2
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_deepspeed_config")


def create_offload_config(
    optimizer_offload: bool = True,
    param_offload: bool = False,
    offload_device: str = "cpu",
    pin_memory: bool = True,
    nvme_path: Optional[str] = None,
) -> dict:
    """Create configuration for CPU/NVMe offloading.

    Args:
        optimizer_offload: Offload optimizer states
        param_offload: Offload parameters (ZeRO-3 only)
        offload_device: "cpu" or "nvme"
        pin_memory: Pin memory for faster transfers
        nvme_path: Path for NVMe offloading

    Returns:
        Offload configuration dictionary

    Example:
        >>> config = create_offload_config(optimizer_offload=True)
        >>> 'offload_optimizer' in config or 'offload_param' in config
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_offload_config")


def save_config_to_file(config: dict, path: str) -> None:
    """Save DeepSpeed config to JSON file.

    Args:
        config: Configuration dictionary
        path: Output file path

    Example:
        >>> config = create_deepspeed_config(stage=2)
        >>> save_config_to_file(config, "ds_config.json")
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config_from_file(path: str) -> dict:
    """Load DeepSpeed config from JSON file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# DeepSpeed Initialization
# =============================================================================

def initialize_deepspeed(
    model: nn.Module,
    config: dict,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_parameters: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    """Initialize model with DeepSpeed.

    Args:
        model: PyTorch model
        config: DeepSpeed configuration dict
        optimizer: Optional optimizer (DeepSpeed can create one)
        model_parameters: Model parameters for optimizer
        lr_scheduler: Optional learning rate scheduler

    Returns:
        Tuple of (model_engine, optimizer, dataloader, lr_scheduler)

    Example:
        >>> model = SimpleTransformer()
        >>> config = create_deepspeed_config(stage=2)
        >>> # engine, opt, _, _ = initialize_deepspeed(model, config)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement initialize_deepspeed")


# =============================================================================
# DeepSpeed Trainer Class
# =============================================================================

class DeepSpeedTrainer:
    """Trainer class for DeepSpeed distributed training.

    This trainer handles:
    - DeepSpeed model engine initialization
    - Training loop with automatic gradient handling
    - Checkpoint save/load
    - Memory and performance tracking

    Args:
        model: Model to train
        config: DeepSpeed configuration dict or path
        optimizer: Optional optimizer
        lr_scheduler: Optional learning rate scheduler

    Example:
        >>> model = SimpleTransformer()
        >>> config = create_deepspeed_config(stage=2)
        >>> # trainer = DeepSpeedTrainer(model, config)
        >>> # loss = trainer.train_epoch(dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        """Initialize DeepSpeed trainer."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.__init__")

    @property
    def device(self) -> torch.device:
        """Get the device for this process."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.device")

    @property
    def rank(self) -> int:
        """Get the rank of this process."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.rank")

    @property
    def world_size(self) -> int:
        """Get the total number of processes."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.world_size")

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
    ) -> float:
        """Perform a single training step.

        DeepSpeed handles gradient scaling and accumulation automatically.

        Args:
            batch: Tuple of (input, target) tensors
            criterion: Loss function

        Returns:
            Loss value as float
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.train_step")

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        sampler: Optional[DistributedSampler] = None,
        epoch: int = 0,
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            criterion: Loss function
            sampler: Distributed sampler (for setting epoch)
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.train_epoch")

    def save_checkpoint(
        self,
        path: str,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save DeepSpeed checkpoint.

        DeepSpeed checkpoints are directories containing:
        - Model state
        - Optimizer state
        - Scheduler state
        - Client state (custom data)

        Args:
            path: Directory path to save checkpoint
            client_state: Custom state dict to save
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.save_checkpoint")

    def load_checkpoint(
        self,
        path: str,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> Dict[str, Any]:
        """Load DeepSpeed checkpoint.

        Args:
            path: Directory path to checkpoint
            load_optimizer_states: Whether to load optimizer state
            load_lr_scheduler_states: Whether to load scheduler state

        Returns:
            Client state dict
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.load_checkpoint")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics.

        Returns:
            Dict with allocated and reserved memory in GB
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement DeepSpeedTrainer.get_memory_stats")


# =============================================================================
# Utility Functions
# =============================================================================

def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DistributedSampler]:
    """Create dataloader with distributed sampler.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers

    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    if DEEPSPEED_AVAILABLE:
        rank = deepspeed.comm.get_rank() if deepspeed.comm.is_initialized() else 0
        world_size = deepspeed.comm.get_world_size() if deepspeed.comm.is_initialized() else 1
    else:
        rank = 0
        world_size = 1

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


def compare_zero_stages(
    model_fn,
    dataset: Dataset,
    batch_size: int = 8,
    num_steps: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Compare memory and speed across ZeRO stages.

    Args:
        model_fn: Function that returns a new model
        dataset: Dataset to use
        batch_size: Batch size per GPU
        num_steps: Number of training steps

    Returns:
        Dict mapping stage to stats
    """
    results = {}

    # Note: This would need to be run in separate processes
    # for accurate comparison. This is a simplified version.
    for stage in [0, 1, 2, 3]:
        results[f"stage_{stage}"] = {
            'memory_gb': 0.0,
            'steps_per_sec': 0.0,
            'description': _get_stage_description(stage),
        }

    return results


def _get_stage_description(stage: int) -> str:
    """Get description for ZeRO stage."""
    descriptions = {
        0: "No ZeRO (DDP-like)",
        1: "Optimizer state partitioning",
        2: "Optimizer + gradient partitioning",
        3: "Full parameter partitioning",
    }
    return descriptions.get(stage, "Unknown")


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--compare', action='store_true', help='Compare ZeRO stages')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=-1)  # For deepspeed launcher
    args = parser.parse_args()

    if not DEEPSPEED_AVAILABLE:
        print("DeepSpeed not installed. Run: pip install deepspeed")
        return

    # Handle comparison mode
    if args.compare:
        print("Comparing ZeRO stages...")
        results = compare_zero_stages(
            lambda: SimpleTransformer(vocab_size=1000, d_model=256, num_layers=4),
            SimpleDataset(size=1000),
            batch_size=args.batch_size,
        )

        print(f"\n{'Stage':<10} | {'Description':<35}")
        print("-" * 50)
        for stage, stats in results.items():
            print(f"{stage:<10} | {stats['description']:<35}")
        return

    # Create DeepSpeed config
    config = create_deepspeed_config(
        stage=args.stage,
        fp16=True,
        gradient_accumulation_steps=4,
        train_batch_size=args.batch_size * 4,  # batch * accum
        train_micro_batch_size_per_gpu=args.batch_size,
        offload_optimizer=args.offload,
        offload_param=args.offload and args.stage == 3,
    )

    print(f"Training with ZeRO Stage {args.stage}")
    if args.offload:
        print("CPU offloading enabled")
    print("=" * 60)

    # Create model
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
    )

    # Create trainer
    trainer = DeepSpeedTrainer(model, config)

    # Create dataset and dataloader
    dataset = SimpleDataset(size=10000, seq_len=128, vocab_size=10000)
    dataloader, sampler = create_distributed_dataloader(
        dataset, batch_size=args.batch_size
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Training loop
    initial_loss = None
    final_loss = None

    for epoch in range(args.epochs):
        loss = trainer.train_epoch(
            dataloader, criterion,
            sampler=sampler, epoch=epoch
        )

        if epoch == 0:
            initial_loss = loss
        final_loss = loss

        if trainer.rank == 0:
            memory_stats = trainer.get_memory_stats()
            print(f"Epoch {epoch}: Loss = {loss:.4f}, "
                  f"Memory = {memory_stats['allocated']:.2f} GB")

    # Save checkpoint
    trainer.save_checkpoint(
        "deepspeed_checkpoint",
        client_state={'epoch': args.epochs - 1}
    )

    # Print results
    if trainer.rank == 0:
        memory_stats = trainer.get_memory_stats()
        print(f"\n{'='*60}")
        print("Lab 05 Milestone: DeepSpeed Training Complete!")
        print(f"ZeRO Stage: {args.stage}")
        print(f"CPU Offload: {'Enabled' if args.offload else 'Disabled'}")
        print(f"Memory per GPU: {memory_stats['allocated']:.2f} GB")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss:   {final_loss:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
