"""
Lab 03: Training Loop

Implement a complete training loop for language models.

Your task: Complete the classes and functions below to make all tests pass.
Run: uv run pytest tests/
"""

import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math


class AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    AdamW fixes a subtle bug in how Adam handles weight decay:
    - Original Adam: applies weight decay to the gradient before the adaptive step
    - AdamW: applies weight decay directly to the weights after the adaptive step

    This decoupling leads to better generalization in practice.

    Algorithm:
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t²
        m_hat = m_t / (1 - β1^t)
        v_hat = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - lr * (m_hat / (sqrt(v_hat) + ε) + weight_decay * θ_{t-1})

    Attributes:
        params: List of parameter arrays to optimize
        lr: Learning rate
        betas: Coefficients for computing running averages (β1, β2)
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay coefficient
        m: First moment estimates (one per parameter)
        v: Second moment estimates (one per parameter)
        t: Current timestep
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: List of parameter arrays to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for running averages (default: (0.9, 0.999))
            eps: Numerical stability term (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)

        Example:
            >>> params = [np.random.randn(10, 10), np.random.randn(10)]
            >>> optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
        """
        # YOUR CODE HERE
        # 1. Store hyperparameters
        # 2. Initialize m and v as zeros with same shapes as params
        # 3. Initialize timestep t = 0
        raise NotImplementedError("Implement AdamW.__init__")

    def step(self, gradients: List[np.ndarray]) -> None:
        """
        Perform single optimization step.

        Updates each parameter using its corresponding gradient.

        Args:
            gradients: List of gradient arrays (same length as params)
                      Each gradient has same shape as corresponding parameter

        Note:
            - Updates params in-place
            - Updates m, v, and t

        Example:
            >>> optimizer = AdamW([param], lr=1e-3)
            >>> gradients = [np.random.randn(*param.shape)]
            >>> optimizer.step(gradients)
            >>> # param has been updated
        """
        # YOUR CODE HERE
        # 1. Increment timestep
        # 2. For each (param, grad, m, v):
        #    a. Update m = β1 * m + (1 - β1) * grad
        #    b. Update v = β2 * v + (1 - β2) * grad²
        #    c. Compute bias-corrected m_hat and v_hat
        #    d. Update param with AdamW rule (decoupled weight decay!)
        raise NotImplementedError("Implement AdamW.step")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr


class LRScheduler:
    """
    Learning rate scheduler with warmup and decay.

    Supports:
    - Linear warmup: LR increases linearly from 0 to base_lr
    - Cosine decay: LR decreases following a cosine curve
    - Linear decay: LR decreases linearly

    The schedule is:
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            if schedule == 'cosine':
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
            elif schedule == 'linear':
                lr = base_lr - (base_lr - min_lr) * progress
    """

    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        schedule: str = 'cosine'
    ):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: The optimizer to adjust LR for
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate after decay
            schedule: 'cosine' or 'linear'

        Raises:
            ValueError: If schedule is not 'cosine' or 'linear'

        Example:
            >>> optimizer = AdamW(params, lr=1e-3)
            >>> scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)
        """
        # YOUR CODE HERE
        # 1. Store optimizer and hyperparameters
        # 2. Store base_lr from optimizer
        # 3. Initialize current_step = 0
        # 4. Validate schedule parameter
        raise NotImplementedError("Implement LRScheduler.__init__")

    def step(self) -> float:
        """
        Update learning rate and return current value.

        Should be called once per training step.

        Returns:
            Current learning rate after update

        Example:
            >>> for step in range(total_steps):
            ...     train_step()
            ...     lr = scheduler.step()
            ...     print(f"Step {step}, LR: {lr}")
        """
        # YOUR CODE HERE
        # 1. Compute new LR based on current_step
        # 2. Update optimizer's LR
        # 3. Increment current_step
        # 4. Return the new LR
        raise NotImplementedError("Implement LRScheduler.step")

    def get_lr(self) -> float:
        """
        Get current learning rate without updating.

        Returns:
            Current learning rate
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement LRScheduler.get_lr")

    def _compute_lr(self, step: int) -> float:
        """
        Compute learning rate for a given step.

        Args:
            step: Training step number

        Returns:
            Learning rate for that step
        """
        # YOUR CODE HERE
        # Implement warmup + decay logic
        raise NotImplementedError("Implement LRScheduler._compute_lr")


class Trainer:
    """
    Main trainer class for language models.

    Handles:
    - Forward pass through model
    - Loss computation
    - Gradient clipping
    - Optimizer step
    - Learning rate scheduling
    - Metric logging
    """

    def __init__(
        self,
        model: Any,
        optimizer: AdamW,
        scheduler: Optional[LRScheduler] = None,
        max_grad_norm: float = 1.0,
        loss_fn: Optional[callable] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model with forward() and parameters() methods
            optimizer: AdamW optimizer
            scheduler: Optional LR scheduler
            max_grad_norm: Maximum gradient norm for clipping
            loss_fn: Loss function (logits, targets) -> loss
                    If None, assumes model returns loss directly

        Example:
            >>> trainer = Trainer(model, optimizer, scheduler, max_grad_norm=1.0)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement Trainer.__init__")

    def train_step(
        self,
        batch: Dict[str, np.ndarray],
        compute_gradients_fn: callable
    ) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Dictionary containing:
                - 'input_ids': Input token IDs (batch, seq_len)
                - 'labels': Target token IDs (batch, seq_len)
            compute_gradients_fn: Function that takes (model, batch) and returns
                                 (loss, gradients_list)

        Returns:
            Dictionary containing:
            - 'loss': Training loss
            - 'grad_norm': Gradient norm (before clipping)
            - 'lr': Current learning rate

        Note:
            This design separates gradient computation (which requires autograd)
            from the training loop mechanics. In practice, you'd use PyTorch's
            autograd, but for learning purposes we pass in a gradient function.
        """
        # YOUR CODE HERE
        # 1. Compute loss and gradients using compute_gradients_fn
        # 2. Compute gradient norm
        # 3. Clip gradients if needed
        # 4. Optimizer step
        # 5. Scheduler step (if present)
        # 6. Return metrics
        raise NotImplementedError("Implement Trainer.train_step")

    def train_epoch(
        self,
        dataloader: Iterable,
        compute_gradients_fn: callable
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Iterable of batches
            compute_gradients_fn: Function for gradient computation

        Returns:
            Dictionary with average metrics over epoch:
            - 'avg_loss': Average training loss
            - 'avg_grad_norm': Average gradient norm
            - 'final_lr': Learning rate at end of epoch
            - 'num_steps': Number of steps in epoch
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement Trainer.train_epoch")


def create_simple_dataloader(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True
) -> Iterable[Dict[str, np.ndarray]]:
    """
    Create a simple dataloader for language modeling.

    For language modeling, we create input-output pairs where:
    - input_ids: tokens [0:seq_len]
    - labels: tokens [1:seq_len+1] (shifted by 1 for next-token prediction)

    Args:
        data: 1D array of token IDs
        batch_size: Number of sequences per batch
        seq_len: Sequence length
        shuffle: Whether to shuffle the data

    Yields:
        Dictionaries with 'input_ids' and 'labels'

    Example:
        >>> data = np.arange(1000)  # Token IDs
        >>> for batch in create_simple_dataloader(data, batch_size=4, seq_len=64):
        ...     print(batch['input_ids'].shape)  # (4, 64)
        ...     print(batch['labels'].shape)     # (4, 64)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement create_simple_dataloader")


def compute_num_params(model: Any) -> int:
    """
    Count total number of trainable parameters in model.

    Args:
        model: Model with parameters() method

    Returns:
        Total number of parameters
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement compute_num_params")


def estimate_memory_usage(
    model: Any,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 4  # float32 = 4 bytes
) -> Dict[str, float]:
    """
    Estimate memory usage for training.

    This is a rough estimate for understanding memory requirements.

    Args:
        model: Model with parameters() method
        batch_size: Batch size
        seq_len: Sequence length
        dtype_bytes: Bytes per element (4 for float32)

    Returns:
        Dictionary with estimates in MB:
        - 'params': Model parameters
        - 'gradients': Same as params (one gradient per param)
        - 'optimizer_state': 2x params for AdamW (m and v)
        - 'activations': Rough estimate for activations
        - 'total': Sum of above
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement estimate_memory_usage")
