"""Tests for Lab 02: DDP Training."""

import os
import tempfile
import pytest
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ddp_trainer import (
    create_distributed_dataloader,
    reduce_mean,
    DDPTrainer,
    sync_gradients_manual,
    SimpleDataset,
    SimpleModel,
)


class TestCreateDistributedDataloader:
    """Tests for create_distributed_dataloader function."""

    def test_returns_tuple(self):
        """Should return (DataLoader, DistributedSampler) tuple."""
        dataset = TensorDataset(torch.randn(100, 10))

        # Mock single-process distributed
        dataloader, sampler = create_distributed_dataloader(
            dataset, batch_size=8, rank=0, world_size=1
        )

        assert isinstance(dataloader, DataLoader)
        from torch.utils.data import DistributedSampler
        assert isinstance(sampler, DistributedSampler)

    def test_batch_size(self):
        """DataLoader should have correct batch size."""
        dataset = TensorDataset(torch.randn(100, 10))

        dataloader, _ = create_distributed_dataloader(
            dataset, batch_size=16, rank=0, world_size=1
        )

        batch = next(iter(dataloader))
        # batch_size might be less for last batch if drop_last=False
        assert batch[0].shape[0] <= 16

    def test_sampler_rank(self):
        """Sampler should have correct rank."""
        dataset = TensorDataset(torch.randn(100, 10))

        _, sampler = create_distributed_dataloader(
            dataset, batch_size=8, rank=2, world_size=4
        )

        assert sampler.rank == 2
        assert sampler.num_replicas == 4


class TestReduceMean:
    """Tests for reduce_mean function."""

    def test_single_process(self):
        """In single process, reduce_mean should return input."""
        if dist.is_initialized():
            dist.destroy_process_group()

        # Initialize single-process distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29510'

        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

            tensor = torch.tensor(0.5)
            result = reduce_mean(tensor, world_size=1)

            assert torch.isclose(result, tensor)

        finally:
            dist.destroy_process_group()

    def test_does_not_modify_input(self):
        """reduce_mean should not modify the input tensor."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29511'

        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

            tensor = torch.tensor(0.5)
            original = tensor.clone()
            _ = reduce_mean(tensor, world_size=1)

            assert torch.equal(tensor, original)

        finally:
            dist.destroy_process_group()


class TestDDPTrainer:
    """Tests for DDPTrainer class."""

    def setup_method(self):
        """Setup for each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29512'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    def teardown_method(self):
        """Cleanup after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_init_wraps_model(self):
        """DDPTrainer should wrap model with DDP."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        # Model should be DDP wrapped
        from torch.nn.parallel import DistributedDataParallel
        assert isinstance(trainer.model, DistributedDataParallel)

    def test_unwrapped_model(self):
        """unwrapped_model should return original model."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        unwrapped = trainer.unwrapped_model
        assert isinstance(unwrapped, SimpleModel)

    def test_train_step_returns_loss(self):
        """train_step should return a scalar loss."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        batch = (torch.randn(4, 10), torch.randn(4, 5))
        loss = trainer.train_step(batch, optimizer, criterion)

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_updates_model(self):
        """train_step should update model parameters."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        # Get initial params
        initial_params = [p.clone() for p in trainer.model.parameters()]

        batch = (torch.randn(4, 10), torch.randn(4, 5))
        trainer.train_step(batch, optimizer, criterion)

        # Check params changed
        for initial, current in zip(initial_params, trainer.model.parameters()):
            assert not torch.equal(initial, current)


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def setup_method(self):
        """Setup for each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29513'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    def teardown_method(self):
        """Cleanup after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_save_creates_file(self):
        """save_checkpoint should create a file."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path, epoch=5, optimizer=optimizer)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_restores_state(self):
        """load_checkpoint should restore model state."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

        # Save initial state
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path, epoch=5, optimizer=optimizer)

            # Modify model
            for p in trainer.model.parameters():
                p.data.fill_(999)

            # Load checkpoint
            checkpoint = trainer.load_checkpoint(path, optimizer)

            # Verify epoch
            assert checkpoint['epoch'] == 5

            # Verify model state restored (not 999)
            for p in trainer.model.parameters():
                assert not torch.all(p == 999)

        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_checkpoint_contains_extras(self):
        """save_checkpoint should include extra kwargs."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(
                path, epoch=5, optimizer=optimizer,
                best_loss=0.1, learning_rate=0.001
            )

            checkpoint = trainer.load_checkpoint(path)
            assert checkpoint['best_loss'] == 0.1
            assert checkpoint['learning_rate'] == 0.001

        finally:
            if os.path.exists(path):
                os.remove(path)


class TestTrainEpoch:
    """Tests for train_epoch method."""

    def setup_method(self):
        """Setup for each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29514'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    def teardown_method(self):
        """Cleanup after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_train_epoch_returns_loss(self):
        """train_epoch should return average loss."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        dataset = SimpleDataset(size=100, input_dim=10, output_dim=5)
        dataloader, sampler = create_distributed_dataloader(
            dataset, batch_size=16, rank=0, world_size=1
        )

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss = trainer.train_epoch(
            dataloader, optimizer, criterion,
            sampler=sampler, epoch=0
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_reduces_loss(self):
        """Training should reduce loss over multiple epochs."""
        model = SimpleModel(10, 20, 5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        dataset = SimpleDataset(size=100, input_dim=10, output_dim=5)
        dataloader, sampler = create_distributed_dataloader(
            dataset, batch_size=16, rank=0, world_size=1
        )

        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(5):
            loss = trainer.train_epoch(
                dataloader, optimizer, criterion,
                sampler=sampler, epoch=epoch
            )
            losses.append(loss)

        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestSyncGradients:
    """Tests for manual gradient sync (educational)."""

    def setup_method(self):
        """Setup for each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29515'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    def teardown_method(self):
        """Cleanup after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_sync_gradients_single_process(self):
        """sync_gradients_manual should work with single process."""
        model = SimpleModel(10, 20, 5)

        # Create some gradients
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        # Store original gradients
        original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        # Sync (with world_size=1, should be unchanged)
        sync_gradients_manual(model, world_size=1)

        # Verify gradients still exist and are same (world_size=1)
        for orig, p in zip(original_grads, model.parameters()):
            if p.grad is not None:
                assert torch.allclose(orig, p.grad)


class TestMilestone:
    """
    Lab 02 Milestone: Can train a model with DDP.
    """

    def setup_method(self):
        """Setup for each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29516'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    def teardown_method(self):
        """Cleanup after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_milestone_ddp_training(self):
        """MILESTONE: Complete DDP training loop."""
        # Create model and trainer
        model = SimpleModel(input_dim=10, hidden_dim=32, output_dim=5)
        trainer = DDPTrainer(model, rank=0, world_size=1)

        # Create dataset and dataloader
        dataset = SimpleDataset(size=200, input_dim=10, output_dim=5)
        dataloader, sampler = create_distributed_dataloader(
            dataset, batch_size=32, rank=0, world_size=1
        )

        # Setup training
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train for multiple epochs
        initial_loss = None
        final_loss = None

        for epoch in range(10):
            loss = trainer.train_epoch(
                dataloader, optimizer, criterion,
                sampler=sampler, epoch=epoch
            )

            if epoch == 0:
                initial_loss = loss
            final_loss = loss

        # Verify training worked
        assert final_loss < initial_loss, "Loss should decrease during training"

        # Test checkpointing
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path, epoch=9, optimizer=optimizer)
            checkpoint = trainer.load_checkpoint(path, optimizer)
            assert checkpoint['epoch'] == 9

        finally:
            if os.path.exists(path):
                os.remove(path)

        print(f"\n{'='*60}")
        print("Lab 02 Milestone: DDP Training Complete!")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss:   {final_loss:.4f}")
        print(f"Improvement:  {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        print(f"{'='*60}\n")
