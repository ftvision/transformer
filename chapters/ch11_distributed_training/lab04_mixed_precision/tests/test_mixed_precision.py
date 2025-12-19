"""Tests for Lab 04: Mixed Precision Training."""

import pytest
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mixed_precision import (
    check_amp_available,
    mixed_precision_forward,
    scaled_backward,
    create_grad_scaler,
    fp32_layer_norm,
    fp32_softmax,
    fp32_cross_entropy,
    MixedPrecisionTrainer,
    SimpleModel,
    SimpleDataset,
)


class TestCheckAMPAvailable:
    """Tests for check_amp_available function."""

    def test_returns_bool(self):
        """Should return a boolean."""
        result = check_amp_available(torch.float16)
        assert isinstance(result, bool)

    def test_fp16_on_cpu(self):
        """fp16 AMP not available on CPU."""
        # If no CUDA, should return False
        if not torch.cuda.is_available():
            assert check_amp_available(torch.float16) is False


class TestMixedPrecisionForward:
    """Tests for mixed_precision_forward function."""

    def test_returns_tensor(self):
        """Should return a tensor."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(4, 10)

        # CPU doesn't support autocast with float16, use float32
        output = mixed_precision_forward(model, x, dtype=torch.float32)
        assert isinstance(output, torch.Tensor)

    def test_output_shape(self):
        """Output shape should match model output."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        x = torch.randn(4, 10)

        output = mixed_precision_forward(model, x, dtype=torch.float32)
        assert output.shape == (4, 5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fp16_on_cuda(self):
        """fp16 forward should work on CUDA."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5).cuda()
        x = torch.randn(4, 10).cuda()

        output = mixed_precision_forward(model, x, dtype=torch.float16)
        assert isinstance(output, torch.Tensor)


class TestScaledBackward:
    """Tests for scaled_backward function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_bool(self):
        """Should return True/False for step taken."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        x = torch.randn(4, 10).cuda()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
            loss = output.sum()

        step_taken = scaled_backward(loss, scaler, optimizer, model)
        assert isinstance(step_taken, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_updates_scaler(self):
        """Scaler state should be updated."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        initial_scale = scaler.get_scale()

        x = torch.randn(4, 10).cuda()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(x)
            loss = output.sum()

        scaled_backward(loss, scaler, optimizer, model)

        # Scale might change (up or down)
        assert scaler.get_scale() is not None


class TestCreateGradScaler:
    """Tests for create_grad_scaler function."""

    def test_returns_grad_scaler(self):
        """Should return a GradScaler."""
        scaler = create_grad_scaler()
        assert isinstance(scaler, GradScaler)

    def test_custom_init_scale(self):
        """Should accept custom init_scale."""
        scaler = create_grad_scaler(init_scale=1024.0)
        assert scaler.get_scale() == 1024.0

    def test_disabled_scaler(self):
        """Should accept enabled=False."""
        scaler = create_grad_scaler(enabled=False)
        # When disabled, scale operations are no-ops
        assert scaler is not None


class TestFP32LayerNorm:
    """Tests for fp32_layer_norm function."""

    def test_preserves_dtype(self):
        """Should preserve input dtype."""
        x = torch.randn(4, 256, dtype=torch.float32)
        out = fp32_layer_norm(x, (256,))
        assert out.dtype == torch.float32

    def test_output_shape(self):
        """Output shape should match input."""
        x = torch.randn(4, 256)
        out = fp32_layer_norm(x, (256,))
        assert out.shape == x.shape

    def test_normalizes_values(self):
        """Output should be normalized."""
        x = torch.randn(4, 256)
        out = fp32_layer_norm(x, (256,))

        # Mean should be ~0, std should be ~1 along normalized dim
        assert torch.abs(out.mean(dim=-1)).max() < 0.1
        assert torch.abs(out.std(dim=-1) - 1.0).max() < 0.1


class TestFP32Softmax:
    """Tests for fp32_softmax function."""

    def test_preserves_dtype(self):
        """Should preserve input dtype."""
        x = torch.randn(4, 1000, dtype=torch.float32)
        out = fp32_softmax(x, dim=-1)
        assert out.dtype == torch.float32

    def test_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = torch.randn(4, 1000)
        out = fp32_softmax(x, dim=-1)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_positive_values(self):
        """All outputs should be positive."""
        x = torch.randn(4, 1000)
        out = fp32_softmax(x, dim=-1)
        assert (out >= 0).all()


class TestFP32CrossEntropy:
    """Tests for fp32_cross_entropy function."""

    def test_returns_scalar(self):
        """Should return scalar loss."""
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = fp32_cross_entropy(logits, targets)
        assert loss.dim() == 0

    def test_returns_fp32(self):
        """Should return fp32 regardless of input dtype."""
        logits = torch.randn(4, 10, dtype=torch.float32)
        targets = torch.randint(0, 10, (4,))
        loss = fp32_cross_entropy(logits, targets)
        assert loss.dtype == torch.float32

    def test_positive_loss(self):
        """Loss should be positive."""
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = fp32_cross_entropy(logits, targets)
        assert loss > 0


class TestMixedPrecisionTrainer:
    """Tests for MixedPrecisionTrainer class."""

    def test_init(self):
        """Should initialize trainer."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)
        assert trainer is not None

    def test_train_step_returns_tuple(self):
        """train_step should return (loss, skipped) tuple."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        batch = (torch.randn(4, 10), torch.randint(0, 5, (4,)))
        loss, skipped = trainer.train_step(batch, optimizer, criterion)

        assert isinstance(loss, float)
        assert isinstance(skipped, bool)

    def test_train_step_updates_model(self):
        """train_step should update model parameters."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]

        batch = (torch.randn(4, 10), torch.randint(0, 5, (4,)))
        trainer.train_step(batch, optimizer, criterion)

        # Check params changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)


class TestTrainEpoch:
    """Tests for train_epoch method."""

    def test_returns_dict(self):
        """train_epoch should return stats dict."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        dataset = SimpleDataset(size=100, input_dim=10, output_dim=5)
        dataloader = DataLoader(dataset, batch_size=16)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        stats = trainer.train_epoch(dataloader, optimizer, criterion)

        assert isinstance(stats, dict)
        assert 'loss' in stats
        assert 'skipped_steps' in stats
        assert 'total_steps' in stats

    def test_loss_is_positive(self):
        """Loss should be positive."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        dataset = SimpleDataset(size=100, input_dim=10, output_dim=5)
        dataloader = DataLoader(dataset, batch_size=16)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        stats = trainer.train_epoch(dataloader, optimizer, criterion)
        assert stats['loss'] > 0


class TestScalerState:
    """Tests for get_scaler_state method."""

    def test_returns_dict(self):
        """get_scaler_state should return a dict."""
        model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        state = trainer.get_scaler_state()
        assert isinstance(state, dict)


class TestMilestone:
    """
    Lab 04 Milestone: Mixed Precision Training.
    """

    def test_milestone_mixed_precision_fp32(self):
        """MILESTONE: Train with fp32 (baseline)."""
        model = SimpleModel(input_dim=100, hidden_dim=64, output_dim=10)
        trainer = MixedPrecisionTrainer(model, dtype=torch.float32)

        dataset = SimpleDataset(size=500, input_dim=100, output_dim=10)
        dataloader = DataLoader(dataset, batch_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train for multiple epochs
        initial_loss = None
        final_loss = None
        total_skipped = 0

        for epoch in range(5):
            stats = trainer.train_epoch(
                dataloader, optimizer, criterion, max_grad_norm=1.0, epoch=epoch
            )

            if epoch == 0:
                initial_loss = stats['loss']
            final_loss = stats['loss']
            total_skipped += stats['skipped_steps']

        # Verify training worked
        assert final_loss < initial_loss, "Loss should decrease during training"

        print(f"\n{'='*60}")
        print("Lab 04 Milestone: Mixed Precision Training Complete!")
        print(f"Precision: fp32 (baseline)")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss:   {final_loss:.4f}")
        print(f"Skipped Steps: {total_skipped}")
        print(f"Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        print(f"{'='*60}\n")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_milestone_mixed_precision_fp16(self):
        """MILESTONE: Train with fp16 and GradScaler."""
        model = SimpleModel(input_dim=100, hidden_dim=64, output_dim=10).cuda()
        trainer = MixedPrecisionTrainer(model, dtype=torch.float16)

        dataset = SimpleDataset(size=500, input_dim=100, output_dim=10)

        def collate_fn(batch):
            inputs = torch.stack([b[0] for b in batch]).cuda()
            targets = torch.stack([b[1] for b in batch]).cuda()
            return inputs, targets

        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train for multiple epochs
        initial_loss = None
        final_loss = None
        total_skipped = 0

        for epoch in range(5):
            stats = trainer.train_epoch(
                dataloader, optimizer, criterion, max_grad_norm=1.0, epoch=epoch
            )

            if epoch == 0:
                initial_loss = stats['loss']
            final_loss = stats['loss']
            total_skipped += stats['skipped_steps']

        # Verify training worked
        assert final_loss < initial_loss, "Loss should decrease during training"

        # Get scaler state
        scaler_state = trainer.get_scaler_state()

        print(f"\n{'='*60}")
        print("Lab 04 Milestone: Mixed Precision Training Complete!")
        print(f"Precision: fp16 with GradScaler")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss:   {final_loss:.4f}")
        print(f"Skipped Steps: {total_skipped}")
        print(f"Final Scale: {scaler_state.get('scale', 'N/A')}")
        print(f"Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        print(f"{'='*60}\n")
