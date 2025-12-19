"""Lab 04: Mixed Precision Training.

This lab implements mixed precision training using PyTorch's AMP
(Automatic Mixed Precision) with GradScaler for fp16 training.

Key concepts:
- fp16 vs bf16 number formats
- Automatic mixed precision with autocast
- Loss scaling with GradScaler
- Precision-sensitive operations
"""

import os
import argparse
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# =============================================================================
# Simple Model and Dataset for Testing
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for mixed precision testing."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 1000, input_dim: int = 784, output_dim: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.output_dim, (1,)).squeeze()
        return x, y


# =============================================================================
# Mixed Precision Utilities
# =============================================================================

def check_amp_available(dtype: torch.dtype = torch.float16) -> bool:
    """Check if AMP is available for the given dtype.

    Args:
        dtype: The target dtype (torch.float16 or torch.bfloat16)

    Returns:
        True if AMP with the dtype is supported

    Example:
        >>> check_amp_available(torch.float16)
        True  # On CUDA devices
    """
    if not torch.cuda.is_available():
        return False

    if dtype == torch.bfloat16:
        return torch.cuda.is_bf16_supported()

    return True  # fp16 always supported on CUDA


def mixed_precision_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run forward pass with automatic mixed precision.

    Uses torch.autocast to automatically cast operations to the
    specified dtype for faster computation.

    Args:
        model: The model to run
        inputs: Input tensor
        dtype: Target dtype (float16 or bfloat16)

    Returns:
        Model output tensor

    Example:
        >>> model = SimpleModel()
        >>> x = torch.randn(4, 784)
        >>> # On CUDA:
        >>> # output = mixed_precision_forward(model, x, torch.float16)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement mixed_precision_forward")


def scaled_backward(
    loss: torch.Tensor,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: Optional[float] = None,
) -> bool:
    """Perform scaled backward pass and optimizer step.

    This function:
    1. Scales the loss to prevent gradient underflow
    2. Performs backward pass
    3. Unscales gradients (for clipping)
    4. Optionally clips gradients
    5. Steps optimizer (skipping if inf/nan)
    6. Updates scaler

    Args:
        loss: Loss tensor
        scaler: GradScaler instance
        optimizer: Optimizer
        model: Model (for gradient clipping)
        max_grad_norm: Max gradient norm for clipping (optional)

    Returns:
        True if step was taken, False if skipped (overflow)

    Example:
        >>> scaler = GradScaler()
        >>> # step_taken = scaled_backward(loss, scaler, optimizer, model)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement scaled_backward")


def create_grad_scaler(
    init_scale: float = 65536.0,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True,
) -> GradScaler:
    """Create a GradScaler with custom parameters.

    Args:
        init_scale: Initial scale value
        growth_factor: Factor to grow scale on successful steps
        backoff_factor: Factor to reduce scale on overflow
        growth_interval: Steps between growth attempts
        enabled: Whether scaling is enabled

    Returns:
        Configured GradScaler

    Example:
        >>> scaler = create_grad_scaler(init_scale=2**16)
        >>> scaler.get_scale()
        65536.0
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_grad_scaler")


# =============================================================================
# Precision-Sensitive Operations
# =============================================================================

def fp32_layer_norm(
    x: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalization that always uses fp32 for stability.

    LayerNorm's variance calculation can be unstable in fp16.
    This function ensures the computation happens in fp32.

    Args:
        x: Input tensor
        normalized_shape: Shape for normalization
        weight: Scale parameter (optional)
        bias: Bias parameter (optional)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor in the original dtype

    Example:
        >>> x = torch.randn(4, 256, dtype=torch.float16)
        >>> out = fp32_layer_norm(x, (256,))
        >>> out.dtype
        torch.float16
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement fp32_layer_norm")


def fp32_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that uses fp32 to avoid overflow.

    Softmax uses exp() which can overflow in fp16 for large values.
    This function computes softmax in fp32 and converts back.

    Args:
        x: Input tensor
        dim: Dimension for softmax

    Returns:
        Softmax output in the original dtype

    Example:
        >>> x = torch.randn(4, 1000, dtype=torch.float16)
        >>> out = fp32_softmax(x, dim=-1)
        >>> out.dtype
        torch.float16
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement fp32_softmax")


def fp32_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Cross entropy loss computed in fp32.

    Cross entropy uses log() which needs numerical precision.

    Args:
        input: Logits tensor
        target: Target tensor
        weight: Class weights (optional)
        reduction: Reduction method

    Returns:
        Loss tensor in fp32

    Example:
        >>> logits = torch.randn(4, 10, dtype=torch.float16)
        >>> targets = torch.randint(0, 10, (4,))
        >>> loss = fp32_cross_entropy(logits, targets)
        >>> loss.dtype
        torch.float32
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement fp32_cross_entropy")


# =============================================================================
# Mixed Precision Trainer Class
# =============================================================================

class MixedPrecisionTrainer:
    """Trainer with automatic mixed precision support.

    This trainer handles:
    - Automatic mixed precision with autocast
    - Loss scaling with GradScaler (for fp16)
    - Gradient clipping with scaled gradients
    - Tracking skipped steps

    Args:
        model: Model to train
        dtype: Precision dtype (float16 or bfloat16)
        use_scaler: Whether to use GradScaler (auto for fp16)

    Example:
        >>> model = SimpleModel().cuda()
        >>> trainer = MixedPrecisionTrainer(model, dtype=torch.float16)
        >>> loss, skipped = trainer.train_step(batch, optimizer, criterion)
    """

    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float16,
        use_scaler: Optional[bool] = None,
    ):
        """Initialize mixed precision trainer."""
        # YOUR CODE HERE
        raise NotImplementedError("Implement MixedPrecisionTrainer.__init__")

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """Perform a single training step with mixed precision.

        Args:
            batch: Tuple of (input, target) tensors
            optimizer: Optimizer
            criterion: Loss function
            max_grad_norm: Max gradient norm for clipping

        Returns:
            Tuple of (loss_value, step_skipped)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MixedPrecisionTrainer.train_step")

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        max_grad_norm: Optional[float] = None,
        sampler: Optional[DistributedSampler] = None,
        epoch: int = 0,
    ) -> Dict[str, Any]:
        """Train for one epoch with mixed precision.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            max_grad_norm: Max gradient norm for clipping
            sampler: Distributed sampler (for setting epoch)
            epoch: Current epoch number

        Returns:
            Dict with 'loss', 'skipped_steps', 'total_steps'
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MixedPrecisionTrainer.train_epoch")

    def get_scaler_state(self) -> Dict[str, Any]:
        """Get current GradScaler state.

        Returns:
            Dict with scale, growth_tracker, etc.
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement MixedPrecisionTrainer.get_scaler_state")


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_precision_formats(
    model_fn,
    input_shape: Tuple[int, ...],
    num_steps: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Compare training with different precision formats.

    Args:
        model_fn: Function that returns a new model
        input_shape: Shape of input data
        num_steps: Number of training steps

    Returns:
        Dict mapping format to stats (memory, speed, etc.)

    Example:
        >>> def model_fn():
        ...     return SimpleModel()
        >>> results = compare_precision_formats(model_fn, (32, 784))
    """
    results = {}

    formats = [
        ('fp32', torch.float32, False),
        ('fp16', torch.float16, True),
    ]

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        formats.append(('bf16', torch.bfloat16, False))

    for name, dtype, use_scaler in formats:
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        model = model_fn()
        if torch.cuda.is_available():
            model = model.cuda()

        # Create fake data
        batch_size = input_shape[0]
        x = torch.randn(*input_shape)
        y = torch.randint(0, 10, (batch_size,))
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        scaler = GradScaler(enabled=use_scaler) if use_scaler else None

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            if dtype != torch.float32:
                with autocast(dtype=dtype):
                    out = model(x)
                    loss = criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        for _ in range(num_steps):
            optimizer.zero_grad()
            if dtype != torch.float32:
                with autocast(dtype=dtype):
                    out = model(x)
                    loss = criterion(out, y)
            else:
                out = model(x)
                loss = criterion(out, y)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)

            results[name] = {
                'time_ms': elapsed_ms,
                'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'steps_per_sec': num_steps / (elapsed_ms / 1000),
            }
        else:
            results[name] = {
                'time_ms': 0,
                'peak_memory_mb': 0,
                'steps_per_sec': 0,
            }

    return results


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage.

    Returns:
        Dict with allocated and reserved memory in GB
    """
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
            'peak_allocated_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'peak_allocated_gb': 0}


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--compare', action='store_true', help='Compare precision formats')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        print("Comparing precision formats...")
        results = compare_precision_formats(
            lambda: SimpleModel(input_dim=784, hidden_dim=512, num_layers=4),
            input_shape=(args.batch_size, 784),
            num_steps=100,
        )

        print(f"\n{'Format':<10} | {'Memory (MB)':<12} | {'Speed (steps/s)':<15}")
        print("-" * 45)
        for name, stats in results.items():
            print(f"{name:<10} | {stats['peak_memory_mb']:<12.1f} | {stats['steps_per_sec']:<15.1f}")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Training with {args.dtype} on {device}")
    print("=" * 60)

    # Check bf16 support
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bf16 not supported, falling back to fp16")
        dtype = torch.float16

    # Create model
    model = SimpleModel(input_dim=784, hidden_dim=512, num_layers=4).to(device)

    # Create trainer
    trainer = MixedPrecisionTrainer(model, dtype=dtype)

    # Create dataset and dataloader
    dataset = SimpleDataset(size=10000, input_dim=784, output_dim=10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    initial_loss = None
    final_loss = None
    total_skipped = 0

    for epoch in range(args.epochs):
        stats = trainer.train_epoch(
            dataloader, optimizer, criterion, max_grad_norm=1.0, epoch=epoch
        )

        if epoch == 0:
            initial_loss = stats['loss']
        final_loss = stats['loss']
        total_skipped += stats['skipped_steps']

        print(f"Epoch {epoch}: Loss = {stats['loss']:.4f}, "
              f"Skipped = {stats['skipped_steps']}/{stats['total_steps']}")

    # Print results
    memory = get_memory_usage()
    scaler_state = trainer.get_scaler_state()

    print(f"\n{'='*60}")
    print(f"Lab 04 Milestone: Mixed Precision Training Complete!")
    print(f"Precision: {args.dtype}")
    print(f"Using GradScaler: {scaler_state.get('enabled', False)}")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Total Skipped Steps: {total_skipped}")
    print(f"Peak Memory: {memory['peak_allocated_gb']:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
