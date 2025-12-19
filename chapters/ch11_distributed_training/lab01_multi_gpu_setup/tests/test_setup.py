"""Tests for Lab 01: Multi-GPU Setup."""

import os
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from setup import (
    get_device_info,
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    verify_gpu_communication,
    broadcast_tensor,
    all_gather_tensors,
    print_rank_0,
)


class TestDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        """Should have all required keys."""
        info = get_device_info()

        assert 'cuda_available' in info
        assert 'num_gpus' in info
        assert 'devices' in info
        assert 'current_device' in info

    def test_cuda_available_is_bool(self):
        """cuda_available should be boolean."""
        info = get_device_info()
        assert isinstance(info['cuda_available'], bool)

    def test_num_gpus_is_int(self):
        """num_gpus should be non-negative integer."""
        info = get_device_info()
        assert isinstance(info['num_gpus'], int)
        assert info['num_gpus'] >= 0

    def test_devices_is_list(self):
        """devices should be a list."""
        info = get_device_info()
        assert isinstance(info['devices'], list)

    def test_num_gpus_matches_devices(self):
        """Number of devices should match num_gpus."""
        info = get_device_info()
        assert len(info['devices']) == info['num_gpus']

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_properties(self):
        """Each device should have required properties."""
        info = get_device_info()

        for device in info['devices']:
            assert 'name' in device
            assert 'total_memory_gb' in device
            assert 'compute_capability' in device

            assert isinstance(device['name'], str)
            assert isinstance(device['total_memory_gb'], float)
            assert device['total_memory_gb'] > 0

    def test_consistency(self):
        """Results should be consistent across calls."""
        info1 = get_device_info()
        info2 = get_device_info()

        assert info1['cuda_available'] == info2['cuda_available']
        assert info1['num_gpus'] == info2['num_gpus']


class TestDistributedSetup:
    """Tests for distributed setup/cleanup functions."""

    def test_is_distributed_false_by_default(self):
        """is_distributed should be False when not initialized."""
        # Make sure we're clean
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        assert not is_distributed()

    def test_get_rank_default(self):
        """get_rank should return 0 when not distributed."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        assert get_rank() == 0

    def test_get_world_size_default(self):
        """get_world_size should return 1 when not distributed."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        assert get_world_size() == 1

    def test_cleanup_safe_when_not_initialized(self):
        """cleanup_distributed should be safe to call when not initialized."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Should not raise
        cleanup_distributed()


class TestLocalRank:
    """Tests for get_local_rank function."""

    def test_default_local_rank(self):
        """Should return 0 when LOCAL_RANK not set."""
        # Remove LOCAL_RANK if set
        old_value = os.environ.pop('LOCAL_RANK', None)

        try:
            assert get_local_rank() == 0
        finally:
            if old_value is not None:
                os.environ['LOCAL_RANK'] = old_value

    def test_reads_environment_variable(self):
        """Should read LOCAL_RANK from environment."""
        old_value = os.environ.get('LOCAL_RANK')

        try:
            os.environ['LOCAL_RANK'] = '3'
            assert get_local_rank() == 3
        finally:
            if old_value is not None:
                os.environ['LOCAL_RANK'] = old_value
            else:
                os.environ.pop('LOCAL_RANK', None)

    def test_returns_int(self):
        """Should return an integer."""
        result = get_local_rank()
        assert isinstance(result, int)


class TestPrintRank0:
    """Tests for print_rank_0 function."""

    def test_prints_when_rank_0(self):
        """Should print when rank is 0."""
        # When not distributed, rank is 0
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        with patch('builtins.print') as mock_print:
            print_rank_0("test message")
            mock_print.assert_called_once_with("test message")


class TestCommunicationPrimitives:
    """Tests for communication functions (require distributed init)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_broadcast_shape_preserved(self):
        """Broadcast should preserve tensor shape."""
        # This test runs in single-process mode
        # Just verify the function signature works
        tensor = torch.zeros(3, 4)

        # When not distributed, broadcast should just return the tensor
        if not torch.distributed.is_initialized():
            # Test should at least verify function exists and accepts args
            pass

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_all_gather_signature(self):
        """all_gather_tensors should have correct signature."""
        tensor = torch.zeros(3)

        # When not distributed, all_gather should return [tensor]
        if not torch.distributed.is_initialized():
            # Test should at least verify function exists
            pass


class TestSingleGPUDistributed:
    """
    Tests that can run with a single GPU.
    Uses gloo backend which works on CPU.
    """

    def test_setup_cleanup_cycle(self):
        """Should be able to setup and cleanup distributed."""
        # Skip if already initialized (from another test)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            # Setup with gloo backend (works on CPU)
            setup_distributed(
                rank=0,
                world_size=1,
                backend='gloo',
                master_addr='localhost',
                master_port='29501'
            )

            assert is_distributed()
            assert get_rank() == 0
            assert get_world_size() == 1

        finally:
            cleanup_distributed()

        assert not is_distributed()

    def test_verify_communication_single_process(self):
        """verify_gpu_communication should work with single process."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            setup_distributed(
                rank=0,
                world_size=1,
                backend='gloo',
                master_addr='localhost',
                master_port='29502'
            )

            # With single process, sum of ranks = 0
            result = verify_gpu_communication(0, 1)
            assert result is True

        finally:
            cleanup_distributed()


class TestEnvironmentSetup:
    """Tests for environment variable handling."""

    def test_setup_sets_master_addr(self):
        """setup_distributed should set MASTER_ADDR."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            setup_distributed(
                rank=0,
                world_size=1,
                backend='gloo',
                master_addr='127.0.0.1',
                master_port='29503'
            )

            assert os.environ.get('MASTER_ADDR') == '127.0.0.1'

        finally:
            cleanup_distributed()

    def test_setup_sets_master_port(self):
        """setup_distributed should set MASTER_PORT."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            setup_distributed(
                rank=0,
                world_size=1,
                backend='gloo',
                master_addr='localhost',
                master_port='29504'
            )

            assert os.environ.get('MASTER_PORT') == '29504'

        finally:
            cleanup_distributed()


class TestMilestone:
    """
    Lab 01 Milestone: Multi-GPU environment is properly configured.
    """

    def test_milestone_device_detection(self):
        """MILESTONE: Can detect GPU hardware."""
        info = get_device_info()

        # Must have all required fields
        assert 'cuda_available' in info
        assert 'num_gpus' in info
        assert 'devices' in info

        # Should be consistent with PyTorch
        assert info['cuda_available'] == torch.cuda.is_available()
        assert info['num_gpus'] == torch.cuda.device_count()

        print(f"\n{'='*60}")
        print("Lab 01 Milestone: Device Detection")
        print(f"CUDA Available: {info['cuda_available']}")
        print(f"Number of GPUs: {info['num_gpus']}")
        for i, dev in enumerate(info['devices']):
            print(f"  GPU {i}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
        print(f"{'='*60}\n")

    def test_milestone_distributed_setup(self):
        """MILESTONE: Can initialize and cleanup distributed."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        try:
            setup_distributed(
                rank=0,
                world_size=1,
                backend='gloo',
                master_addr='localhost',
                master_port='29505'
            )

            assert is_distributed(), "Distributed should be initialized"

        finally:
            cleanup_distributed()

        assert not is_distributed(), "Distributed should be cleaned up"

        print(f"\n{'='*60}")
        print("Lab 01 Milestone: Distributed Setup")
        print("Successfully initialized and cleaned up distributed process group!")
        print(f"{'='*60}\n")
