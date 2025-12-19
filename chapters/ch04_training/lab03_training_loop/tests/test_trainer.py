"""Tests for Lab 03: Training Loop."""

import numpy as np
import pytest
import sys
from pathlib import Path
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer import (
    AdamW,
    LRScheduler,
    Trainer,
    create_simple_dataloader,
    compute_num_params,
    estimate_memory_usage,
)


class DummyModel:
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=5, output_dim=10):
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01

    def parameters(self):
        return [self.W1, self.W2]

    def forward(self, x):
        h = np.tanh(x @ self.W1)
        return h @ self.W2


class TestAdamW:
    """Tests for AdamW optimizer."""

    def test_initialization(self):
        """Should initialize with correct attributes."""
        params = [np.random.randn(10, 10), np.random.randn(10)]
        optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)

        assert optimizer.lr == 1e-3
        assert len(optimizer.m) == 2
        assert len(optimizer.v) == 2
        assert optimizer.t == 0

    def test_m_v_shapes(self):
        """m and v should match parameter shapes."""
        params = [np.random.randn(10, 20), np.random.randn(5, 5, 5)]
        optimizer = AdamW(params)

        assert optimizer.m[0].shape == (10, 20)
        assert optimizer.m[1].shape == (5, 5, 5)
        assert optimizer.v[0].shape == (10, 20)
        assert optimizer.v[1].shape == (5, 5, 5)

    def test_step_updates_params(self):
        """Optimizer step should update parameters."""
        params = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        params_before = params[0].copy()

        optimizer = AdamW(params, lr=0.1)
        gradients = [np.ones_like(params[0])]

        optimizer.step(gradients)

        assert not np.allclose(params[0], params_before), "Parameters should be updated"

    def test_step_increments_t(self):
        """Each step should increment timestep."""
        params = [np.random.randn(5, 5)]
        optimizer = AdamW(params)

        assert optimizer.t == 0
        optimizer.step([np.random.randn(5, 5)])
        assert optimizer.t == 1
        optimizer.step([np.random.randn(5, 5)])
        assert optimizer.t == 2

    def test_weight_decay_effect(self):
        """Weight decay should shrink parameters."""
        # Large positive weights
        params_with_decay = [np.ones((10, 10)) * 10.0]
        params_no_decay = [np.ones((10, 10)) * 10.0]

        opt_with_decay = AdamW(params_with_decay, lr=0.1, weight_decay=0.1)
        opt_no_decay = AdamW(params_no_decay, lr=0.1, weight_decay=0.0)

        # Zero gradient - only weight decay should affect params
        zero_grad = [np.zeros((10, 10))]

        for _ in range(10):
            opt_with_decay.step(zero_grad)
            opt_no_decay.step(zero_grad)

        # With decay, params should be smaller
        assert np.mean(np.abs(params_with_decay[0])) < np.mean(np.abs(params_no_decay[0]))

    def test_momentum_effect(self):
        """Momentum should smooth updates."""
        np.random.seed(42)
        params = [np.zeros((10, 10))]
        optimizer = AdamW(params, lr=0.01, betas=(0.9, 0.999))

        # Alternating gradients
        for i in range(20):
            grad = np.ones((10, 10)) if i % 2 == 0 else -np.ones((10, 10))
            optimizer.step([grad])

        # With momentum, updates should be small due to cancellation
        assert np.abs(params[0]).mean() < 0.1

    def test_bias_correction(self):
        """Early steps should apply bias correction."""
        params = [np.zeros((5, 5))]
        optimizer = AdamW(params, lr=1.0, betas=(0.9, 0.999))

        grad = np.ones((5, 5))
        optimizer.step([grad])

        # Without bias correction, update would be ~0.1 (m=0.1, sqrt(v)~0.03)
        # With bias correction, it should be larger in early steps
        # After 1 step: m_hat = 0.1 / 0.1 = 1.0
        assert np.abs(params[0]).mean() > 0.5


class TestLRScheduler:
    """Tests for learning rate scheduler."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)

        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.base_lr == 1e-3

    def test_warmup_starts_at_zero(self):
        """LR should start near zero during warmup."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)

        initial_lr = scheduler.get_lr()
        assert initial_lr < 1e-4, f"Initial LR should be near 0, got {initial_lr}"

    def test_warmup_linear_increase(self):
        """LR should increase linearly during warmup."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())

        # Should be approximately linear
        expected = [1e-3 * (i + 1) / 100 for i in range(100)]
        np.testing.assert_allclose(lrs, expected, rtol=1e-5)

    def test_warmup_reaches_base_lr(self):
        """LR should reach base_lr at end of warmup."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=100, total_steps=1000)

        for _ in range(100):
            scheduler.step()

        lr_after_warmup = scheduler.get_lr()
        np.testing.assert_allclose(lr_after_warmup, 1e-3, rtol=1e-5)

    def test_cosine_decay(self):
        """LR should decay following cosine after warmup."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(
            optimizer, warmup_steps=0, total_steps=100,
            min_lr=0.0, schedule='cosine'
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())

        # First LR should be base_lr
        np.testing.assert_allclose(lrs[0], 1e-3, rtol=1e-5)

        # Last LR should be min_lr
        np.testing.assert_allclose(lrs[-1], 0.0, atol=1e-6)

        # Mid-point should be ~half
        assert 0.4e-3 < lrs[50] < 0.6e-3

    def test_linear_decay(self):
        """LR should decay linearly with linear schedule."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(
            optimizer, warmup_steps=0, total_steps=100,
            min_lr=0.0, schedule='linear'
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())

        # Should be exactly linear
        expected = [1e-3 * (1 - i / 100) for i in range(100)]
        np.testing.assert_allclose(lrs, expected, rtol=1e-5)

    def test_min_lr_respected(self):
        """LR should not go below min_lr."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(
            optimizer, warmup_steps=0, total_steps=100,
            min_lr=1e-4, schedule='cosine'
        )

        for _ in range(200):  # Go beyond total_steps
            lr = scheduler.step()
            assert lr >= 1e-4 - 1e-10, f"LR {lr} below min_lr"

    def test_invalid_schedule_raises(self):
        """Should raise error for invalid schedule."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params)

        with pytest.raises(ValueError):
            LRScheduler(optimizer, warmup_steps=10, total_steps=100, schedule='invalid')


class TestTrainer:
    """Tests for Trainer class."""

    def test_initialization(self):
        """Should initialize with model and optimizer."""
        model = DummyModel()
        optimizer = AdamW(model.parameters())
        trainer = Trainer(model, optimizer)

        assert trainer.model is model
        assert trainer.optimizer is optimizer

    def test_train_step_returns_metrics(self):
        """train_step should return loss, grad_norm, lr."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=10, total_steps=100)
        trainer = Trainer(model, optimizer, scheduler)

        def compute_grads(model, batch):
            # Dummy gradient computation
            loss = 1.0
            grads = [np.random.randn(*p.shape) * 0.01 for p in model.parameters()]
            return loss, grads

        batch = {'input_ids': np.zeros((2, 10)), 'labels': np.zeros((2, 10))}
        metrics = trainer.train_step(batch, compute_grads)

        assert 'loss' in metrics
        assert 'grad_norm' in metrics
        assert 'lr' in metrics

    def test_train_step_updates_model(self):
        """train_step should update model parameters."""
        model = DummyModel()
        params_before = [p.copy() for p in model.parameters()]

        optimizer = AdamW(model.parameters(), lr=0.1)
        trainer = Trainer(model, optimizer)

        def compute_grads(model, batch):
            grads = [np.ones_like(p) for p in model.parameters()]
            return 1.0, grads

        batch = {'input_ids': np.zeros((2, 10)), 'labels': np.zeros((2, 10))}
        trainer.train_step(batch, compute_grads)

        # Parameters should have changed
        for p_before, p_after in zip(params_before, model.parameters()):
            assert not np.allclose(p_before, p_after)

    def test_gradient_clipping(self):
        """Should clip gradients to max_grad_norm."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=0.1)
        trainer = Trainer(model, optimizer, max_grad_norm=1.0)

        def compute_grads(model, batch):
            # Large gradients
            grads = [np.ones_like(p) * 100 for p in model.parameters()]
            return 1.0, grads

        batch = {'input_ids': np.zeros((2, 10)), 'labels': np.zeros((2, 10))}
        metrics = trainer.train_step(batch, compute_grads)

        # Original grad norm should be > 1.0
        assert metrics['grad_norm'] > 1.0

    def test_train_epoch(self):
        """train_epoch should iterate over all batches."""
        model = DummyModel()
        optimizer = AdamW(model.parameters())
        trainer = Trainer(model, optimizer)

        def compute_grads(model, batch):
            return 1.0, [np.random.randn(*p.shape) * 0.01 for p in model.parameters()]

        # Create dummy dataloader with 5 batches
        dataloader = [
            {'input_ids': np.zeros((2, 10)), 'labels': np.zeros((2, 10))}
            for _ in range(5)
        ]

        metrics = trainer.train_epoch(dataloader, compute_grads)

        assert metrics['num_steps'] == 5
        assert 'avg_loss' in metrics
        assert 'avg_grad_norm' in metrics


class TestDataloader:
    """Tests for create_simple_dataloader."""

    def test_batch_shapes(self):
        """Batches should have correct shapes."""
        data = np.arange(1000)
        dataloader = create_simple_dataloader(data, batch_size=4, seq_len=64)

        batch = next(iter(dataloader))

        assert batch['input_ids'].shape == (4, 64)
        assert batch['labels'].shape == (4, 64)

    def test_labels_shifted(self):
        """Labels should be shifted by 1 from inputs."""
        data = np.arange(1000)
        dataloader = create_simple_dataloader(data, batch_size=1, seq_len=10, shuffle=False)

        batch = next(iter(dataloader))

        # Labels should be inputs shifted by 1
        np.testing.assert_array_equal(
            batch['labels'][0, :-1],
            batch['input_ids'][0, 1:]
        )

    def test_generates_multiple_batches(self):
        """Should generate multiple batches."""
        data = np.arange(1000)
        dataloader = list(create_simple_dataloader(data, batch_size=4, seq_len=64))

        assert len(dataloader) > 1


class TestUtilities:
    """Tests for utility functions."""

    def test_compute_num_params(self):
        """Should count parameters correctly."""
        model = DummyModel(input_dim=10, hidden_dim=5, output_dim=10)

        num_params = compute_num_params(model)

        # W1: 10x5 = 50, W2: 5x10 = 50
        assert num_params == 100

    def test_estimate_memory(self):
        """Should estimate memory usage."""
        model = DummyModel(input_dim=10, hidden_dim=5, output_dim=10)

        estimate = estimate_memory_usage(model, batch_size=4, seq_len=64)

        assert 'params' in estimate
        assert 'gradients' in estimate
        assert 'optimizer_state' in estimate
        assert 'total' in estimate

        # Total should be sum of components
        expected_total = (
            estimate['params'] +
            estimate['gradients'] +
            estimate['optimizer_state'] +
            estimate['activations']
        )
        np.testing.assert_allclose(estimate['total'], expected_total, rtol=1e-5)


class TestIntegration:
    """Integration tests for complete training."""

    def test_training_reduces_loss(self):
        """Training should reduce loss on simple problem."""
        np.random.seed(42)

        # Simple model
        model = DummyModel(input_dim=10, hidden_dim=20, output_dim=10)

        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        scheduler = LRScheduler(optimizer, warmup_steps=10, total_steps=100)
        trainer = Trainer(model, optimizer, scheduler, max_grad_norm=1.0)

        # Simple loss: make output match target
        def compute_grads(model, batch):
            x = batch['input_ids']
            target = batch['labels']

            # Forward
            h = np.tanh(x @ model.W1)
            output = h @ model.W2

            # Loss (MSE for simplicity)
            loss = np.mean((output - target) ** 2)

            # Gradients (analytical)
            d_output = 2 * (output - target) / output.size
            d_W2 = h.T @ d_output
            d_h = d_output @ model.W2.T
            d_W1 = x.T @ (d_h * (1 - h ** 2))

            return loss, [d_W1, d_W2]

        # Training data: learn identity-ish mapping
        losses = []
        for step in range(100):
            x = np.random.randn(8, 10).astype(np.float32)
            y = x  # Try to learn identity

            batch = {'input_ids': x, 'labels': y}
            metrics = trainer.train_step(batch, compute_grads)
            losses.append(metrics['loss'])

        # Loss should decrease
        assert losses[-1] < losses[0], "Training should reduce loss"
        assert losses[-1] < losses[0] / 2, "Loss should decrease significantly"

    def test_warmup_then_decay(self):
        """LR should increase during warmup then decay."""
        params = [np.random.randn(10, 10)]
        optimizer = AdamW(params, lr=1e-3)
        scheduler = LRScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=1e-5,
            schedule='cosine'
        )

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.step())

        # Warmup phase: should increase
        for i in range(1, 10):
            assert lrs[i] > lrs[i-1], f"LR should increase during warmup, step {i}"

        # Decay phase: should generally decrease
        assert lrs[50] < lrs[10], "LR should decrease after warmup"
        assert lrs[99] < lrs[50], "LR should continue decreasing"

    def test_full_training_loop_structure(self):
        """Test complete training loop matches expected structure."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = LRScheduler(optimizer, warmup_steps=5, total_steps=20)
        trainer = Trainer(model, optimizer, scheduler, max_grad_norm=1.0)

        def compute_grads(model, batch):
            return 0.5, [np.random.randn(*p.shape) * 0.01 for p in model.parameters()]

        # Simulate 2 epochs of 10 batches each
        all_metrics = []
        for epoch in range(2):
            dataloader = [
                {'input_ids': np.zeros((2, 10)), 'labels': np.zeros((2, 10))}
                for _ in range(10)
            ]

            epoch_metrics = trainer.train_epoch(dataloader, compute_grads)
            all_metrics.append(epoch_metrics)

        assert len(all_metrics) == 2
        assert all_metrics[0]['num_steps'] == 10
        assert all_metrics[1]['num_steps'] == 10
