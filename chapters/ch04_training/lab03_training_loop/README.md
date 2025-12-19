# Lab 03: Training Loop

## Objective

Implement a complete training loop for language models with proper learning rate scheduling and gradient handling.

## What You'll Build

A `Trainer` class that:
1. Manages the training loop with proper batching
2. Implements AdamW optimizer from scratch
3. Supports learning rate warmup and cosine decay
4. Handles gradient clipping
5. Logs training metrics

## Prerequisites

- Complete Lab 01 (loss functions)
- Complete Lab 02 (gradient visualization)
- Read `../docs/03_optimizers.md`
- Read `../docs/04_lr_schedules.md`

## Instructions

1. Open `src/trainer.py`
2. Implement the classes and functions marked with `# YOUR CODE HERE`
3. Run tests: `uv run pytest tests/`

## Classes to Implement

### `AdamW`
Implement the AdamW optimizer with decoupled weight decay.

```python
class AdamW:
    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        """Initialize AdamW optimizer."""

    def step(self, gradients: List[np.ndarray]) -> None:
        """Perform single optimization step."""

    def zero_grad(self) -> None:
        """Reset gradient accumulators (if any)."""
```

### `LRScheduler`
Implement learning rate scheduling with warmup and decay.

```python
class LRScheduler:
    def __init__(
        self,
        optimizer: AdamW,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        schedule: str = 'cosine'  # or 'linear'
    ):
        """Initialize learning rate scheduler."""

    def step(self) -> float:
        """Update learning rate and return current value."""

    def get_lr(self) -> float:
        """Get current learning rate."""
```

### `Trainer`
The main training class.

```python
class Trainer:
    def __init__(
        self,
        model: Any,  # Any model with forward() method
        optimizer: AdamW,
        scheduler: LRScheduler,
        max_grad_norm: float = 1.0
    ):
        """Initialize trainer."""

    def train_step(self, batch: dict) -> dict:
        """
        Execute single training step.
        Returns dict with 'loss', 'grad_norm', 'lr'
        """

    def train_epoch(self, dataloader: Iterable) -> dict:
        """Train for one epoch, return average metrics."""
```

## The Training Loop

A proper training loop follows this pattern:

```python
for step, batch in enumerate(dataloader):
    # 1. Forward pass
    logits = model.forward(batch['input_ids'])

    # 2. Compute loss
    loss = cross_entropy_loss(logits, batch['labels'])

    # 3. Backward pass (compute gradients)
    gradients = compute_gradients(loss, model.parameters())

    # 4. Gradient clipping
    clip_gradients(gradients, max_norm=1.0)

    # 5. Optimizer step
    optimizer.step(gradients)

    # 6. Learning rate update
    scheduler.step()

    # 7. Logging
    log_metrics(step, loss, lr)
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test class
uv run pytest tests/test_trainer.py::TestAdamW

# Run with verbose output
uv run pytest tests/ -v
```

## AdamW Algorithm

```python
# Initialize:
m = 0  # First moment estimate
v = 0  # Second moment estimate
t = 0  # Timestep

# For each step:
t = t + 1
m = β1 * m + (1 - β1) * gradient
v = β2 * v + (1 - β2) * gradient²

# Bias correction
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)

# Update (note: weight decay is decoupled!)
param = param - lr * (m_hat / (sqrt(v_hat) + ε) + weight_decay * param)
```

## Learning Rate Schedule

### Linear Warmup + Cosine Decay

```python
if step < warmup_steps:
    # Linear warmup
    lr = base_lr * step / warmup_steps
else:
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

## Hints

- Store `m` and `v` for each parameter (not per-element)
- Remember to handle the bias correction term `(1 - β^t)`
- Gradient clipping should use the global norm (from Lab 02)
- The `model` object needs `parameters()` and `forward()` methods

## Example Usage

```python
import numpy as np

# Dummy model
class SimpleModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01

    def parameters(self):
        return [self.W1, self.W2]

    def forward(self, x):
        h = np.tanh(x @ self.W1)
        return h @ self.W2

# Setup
model = SimpleModel(100, 64, 50)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)
trainer = Trainer(model, optimizer, scheduler)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        metrics = trainer.train_step(batch)
        print(f"Loss: {metrics['loss']:.4f}, LR: {metrics['lr']:.6f}")
```

## Verification

All tests pass = you've built a complete training infrastructure!

Now you're ready to train your first model in Lab 04.
