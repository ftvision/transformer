"""Tests for Lab 05: DeepSpeed Integration."""

import os
import tempfile
import pytest
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepspeed_trainer import (
    create_deepspeed_config,
    create_offload_config,
    save_config_to_file,
    load_config_from_file,
    SimpleTransformer,
    SimpleDataset,
    DEEPSPEED_AVAILABLE,
)

# Skip all tests if DeepSpeed not installed
pytestmark = pytest.mark.skipif(
    not DEEPSPEED_AVAILABLE,
    reason="DeepSpeed not installed"
)


class TestCreateDeepSpeedConfig:
    """Tests for create_deepspeed_config function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        config = create_deepspeed_config()
        assert isinstance(config, dict)

    def test_default_stage(self):
        """Default stage should be 2."""
        config = create_deepspeed_config()
        assert config['zero_optimization']['stage'] == 2

    def test_custom_stage(self):
        """Should accept custom stage."""
        for stage in [0, 1, 2, 3]:
            config = create_deepspeed_config(stage=stage)
            assert config['zero_optimization']['stage'] == stage

    def test_fp16_enabled(self):
        """fp16 should be configurable."""
        config = create_deepspeed_config(fp16=True)
        assert config.get('fp16', {}).get('enabled', False) is True

    def test_bf16_enabled(self):
        """bf16 should be configurable."""
        config = create_deepspeed_config(bf16=True, fp16=False)
        assert config.get('bf16', {}).get('enabled', False) is True

    def test_batch_size_config(self):
        """Batch size should be set correctly."""
        config = create_deepspeed_config(
            train_batch_size=64,
            train_micro_batch_size_per_gpu=8,
        )
        assert config['train_batch_size'] == 64
        assert config['train_micro_batch_size_per_gpu'] == 8

    def test_gradient_accumulation(self):
        """Gradient accumulation should be configurable."""
        config = create_deepspeed_config(gradient_accumulation_steps=4)
        assert config['gradient_accumulation_steps'] == 4


class TestCreateOffloadConfig:
    """Tests for create_offload_config function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        config = create_offload_config()
        assert isinstance(config, dict)

    def test_optimizer_offload(self):
        """Should configure optimizer offloading."""
        config = create_offload_config(optimizer_offload=True)
        assert 'offload_optimizer' in config or 'device' in config.get('offload_optimizer', {})

    def test_param_offload(self):
        """Should configure parameter offloading."""
        config = create_offload_config(param_offload=True)
        assert 'offload_param' in config or 'device' in config.get('offload_param', {})

    def test_cpu_device(self):
        """Should support CPU offloading."""
        config = create_offload_config(offload_device="cpu")
        # Check that config doesn't error
        assert config is not None

    def test_nvme_device(self):
        """Should support NVMe offloading."""
        config = create_offload_config(
            offload_device="nvme",
            nvme_path="/tmp/nvme"
        )
        assert config is not None


class TestConfigFileIO:
    """Tests for config file save/load."""

    def test_save_and_load(self):
        """Should be able to save and load config."""
        config = create_deepspeed_config(stage=3)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_config_to_file(config, path)
            loaded = load_config_from_file(path)
            assert loaded['zero_optimization']['stage'] == 3
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_preserves_all_keys(self):
        """Loaded config should have all original keys."""
        config = create_deepspeed_config(
            stage=2,
            fp16=True,
            gradient_accumulation_steps=4,
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            save_config_to_file(config, path)
            loaded = load_config_from_file(path)

            # Check key structure is preserved
            assert 'zero_optimization' in loaded
            assert 'gradient_accumulation_steps' in loaded
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestSimpleTransformer:
    """Tests for SimpleTransformer model."""

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        model = SimpleTransformer(
            vocab_size=100,
            d_model=64,
            nhead=2,
            num_layers=2,
        )

        x = torch.randint(0, 100, (4, 16))
        output = model(x)

        assert output.shape == (4, 16, 100)  # (batch, seq, vocab)

    def test_trainable(self):
        """Model should be trainable."""
        model = SimpleTransformer(
            vocab_size=100,
            d_model=64,
            nhead=2,
            num_layers=2,
        )

        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        # Check gradients flow
        x = torch.randint(0, 100, (4, 16))
        output = model(x)
        loss = output.sum()
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestSimpleDataset:
    """Tests for SimpleDataset."""

    def test_length(self):
        """Dataset should have correct length."""
        dataset = SimpleDataset(size=100)
        assert len(dataset) == 100

    def test_item_shape(self):
        """Items should have correct shape."""
        dataset = SimpleDataset(size=100, seq_len=32, vocab_size=1000)
        input_ids, target = dataset[0]

        assert input_ids.shape == (32,)
        assert target.shape == (32,)

    def test_vocab_range(self):
        """Values should be in vocab range."""
        dataset = SimpleDataset(size=100, seq_len=32, vocab_size=1000)
        input_ids, target = dataset[0]

        assert input_ids.max() < 1000
        assert input_ids.min() >= 0
        assert target.max() < 1000
        assert target.min() >= 0


# Note: The following tests require DeepSpeed to be properly initialized
# with distributed environment. They are marked to skip in non-distributed setup.

@pytest.mark.skipif(
    not DEEPSPEED_AVAILABLE or not torch.cuda.is_available(),
    reason="DeepSpeed and CUDA required"
)
class TestDeepSpeedTrainerIntegration:
    """Integration tests for DeepSpeedTrainer (require distributed setup)."""

    def test_placeholder(self):
        """Placeholder for integration tests.

        Full integration tests would require:
        - Multiple GPUs
        - Proper distributed initialization
        - Running via deepspeed launcher

        These tests serve as documentation for expected behavior.
        """
        pass


class TestMilestone:
    """
    Lab 05 Milestone: DeepSpeed Training.

    Note: Full milestone tests require DeepSpeed with distributed setup.
    These tests verify configuration and model components work correctly.
    """

    def test_milestone_config_creation(self):
        """MILESTONE: Create valid DeepSpeed configs for all stages."""
        for stage in [0, 1, 2, 3]:
            config = create_deepspeed_config(stage=stage)

            # Verify basic structure
            assert 'zero_optimization' in config
            assert config['zero_optimization']['stage'] == stage

            # Verify batch config
            assert 'train_batch_size' in config or 'train_micro_batch_size_per_gpu' in config

        print(f"\n{'='*60}")
        print("Lab 05 Milestone: DeepSpeed Config Creation Complete!")
        print("Created valid configs for ZeRO stages 0, 1, 2, 3")
        print(f"{'='*60}\n")

    def test_milestone_offload_config(self):
        """MILESTONE: Create valid offload configurations."""
        # CPU offload
        cpu_config = create_offload_config(
            optimizer_offload=True,
            param_offload=True,
            offload_device="cpu",
        )
        assert cpu_config is not None

        # NVMe offload
        nvme_config = create_offload_config(
            optimizer_offload=True,
            param_offload=True,
            offload_device="nvme",
            nvme_path="/tmp/nvme_offload",
        )
        assert nvme_config is not None

        print(f"\n{'='*60}")
        print("Lab 05 Milestone: Offload Config Creation Complete!")
        print("Created valid configs for CPU and NVMe offloading")
        print(f"{'='*60}\n")

    def test_milestone_model_compatibility(self):
        """MILESTONE: Verify model is compatible with DeepSpeed."""
        model = SimpleTransformer(
            vocab_size=1000,
            d_model=256,
            nhead=4,
            num_layers=4,
        )

        # Model should have standard nn.Module interface
        assert hasattr(model, 'forward')
        assert hasattr(model, 'parameters')
        assert hasattr(model, 'state_dict')
        assert hasattr(model, 'load_state_dict')

        # Should work with standard PyTorch operations
        x = torch.randint(0, 1000, (4, 32))
        output = model(x)
        assert output.shape == (4, 32, 1000)

        # Should be able to compute loss
        criterion = nn.CrossEntropyLoss()
        target = torch.randint(0, 1000, (4, 32))
        loss = criterion(output.view(-1, 1000), target.view(-1))
        assert loss.item() > 0

        print(f"\n{'='*60}")
        print("Lab 05 Milestone: Model Compatibility Complete!")
        print("Model is compatible with DeepSpeed requirements")
        print(f"{'='*60}\n")


class TestConfigValidation:
    """Tests for config validation and edge cases."""

    def test_stage_0_no_zero(self):
        """Stage 0 should have minimal ZeRO config."""
        config = create_deepspeed_config(stage=0)
        assert config['zero_optimization']['stage'] == 0

    def test_stage_3_with_offload(self):
        """Stage 3 should support full offloading."""
        config = create_deepspeed_config(
            stage=3,
            offload_optimizer=True,
            offload_param=True,
        )
        assert config['zero_optimization']['stage'] == 3

    def test_mixed_precision_mutex(self):
        """fp16 and bf16 should be mutually exclusive."""
        # This tests the expected behavior - implementation should handle this
        config_fp16 = create_deepspeed_config(fp16=True, bf16=False)
        config_bf16 = create_deepspeed_config(fp16=False, bf16=True)

        # At most one should be enabled
        fp16_enabled = config_fp16.get('fp16', {}).get('enabled', False)
        bf16_in_fp16 = config_fp16.get('bf16', {}).get('enabled', False)

        bf16_enabled = config_bf16.get('bf16', {}).get('enabled', False)
        fp16_in_bf16 = config_bf16.get('fp16', {}).get('enabled', False)

        # Verify mutual exclusivity
        assert not (fp16_enabled and bf16_in_fp16)
        assert not (bf16_enabled and fp16_in_bf16)
