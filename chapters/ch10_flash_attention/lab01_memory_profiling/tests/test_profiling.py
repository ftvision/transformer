"""Tests for Lab 01: Memory Profiling."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from profiling import (
    standard_attention,
    measure_attention_memory,
    profile_memory_scaling,
    estimate_attention_memory,
    analyze_scaling,
)


class TestStandardAttention:
    """Tests for standard attention implementation."""

    def test_output_shape(self):
        """Output should have correct shape."""
        seq_len, d_model = 32, 16
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        output, attn = standard_attention(Q, K, V)

        assert output.shape == (seq_len, d_model)
        assert attn.shape == (seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 along last axis."""
        Q = np.random.randn(16, 8).astype(np.float32)
        K = np.random.randn(16, 8).astype(np.float32)
        V = np.random.randn(16, 8).astype(np.float32)

        _, attn = standard_attention(Q, K, V)

        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(16), rtol=1e-5)

    def test_attention_weights_positive(self):
        """Attention weights should be non-negative."""
        Q = np.random.randn(16, 8).astype(np.float32)
        K = np.random.randn(16, 8).astype(np.float32)
        V = np.random.randn(16, 8).astype(np.float32)

        _, attn = standard_attention(Q, K, V)

        assert np.all(attn >= 0)

    def test_batched_attention(self):
        """Should work with batched inputs."""
        batch, seq_len, d_model = 2, 16, 8
        Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
        V = np.random.randn(batch, seq_len, d_model).astype(np.float32)

        output, attn = standard_attention(Q, K, V)

        assert output.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, seq_len, seq_len)

    def test_with_mask(self):
        """Masked positions should have zero attention weight."""
        seq_len, d_model = 8, 4
        Q = np.random.randn(seq_len, d_model).astype(np.float32)
        K = np.random.randn(seq_len, d_model).astype(np.float32)
        V = np.random.randn(seq_len, d_model).astype(np.float32)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        _, attn = standard_attention(Q, K, V, mask=mask)

        # Upper triangle should be ~0
        upper = np.triu(attn, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-6)

    def test_returns_attention_matrix(self):
        """Must return the full attention matrix (this is the point!)."""
        Q = np.random.randn(16, 8).astype(np.float32)
        K = np.random.randn(16, 8).astype(np.float32)
        V = np.random.randn(16, 8).astype(np.float32)

        output, attn = standard_attention(Q, K, V)

        # The attention matrix is N×N
        assert attn.shape == (16, 16)
        # And should be the full softmax output
        assert attn.dtype == np.float32


class TestMeasureMemory:
    """Tests for measure_attention_memory."""

    def test_returns_required_keys(self):
        """Should return all required memory statistics."""
        stats = measure_attention_memory(64, 16)

        assert 'input_memory' in stats
        assert 'attention_matrix_memory' in stats
        assert 'output_memory' in stats
        assert 'total_memory' in stats

    def test_attention_matrix_dominates(self):
        """For large seq_len, attention matrix should dominate memory."""
        stats = measure_attention_memory(512, 64)

        # Attention matrix: 512 * 512 * 4 = 1 MB
        # Input (Q, K, V): 3 * 512 * 64 * 4 = ~400 KB
        assert stats['attention_matrix_memory'] > stats['input_memory']

    def test_memory_values_positive(self):
        """All memory values should be positive."""
        stats = measure_attention_memory(32, 16)

        assert stats['input_memory'] > 0
        assert stats['attention_matrix_memory'] > 0
        assert stats['output_memory'] > 0
        assert stats['total_memory'] > 0

    def test_total_is_sum(self):
        """Total should equal sum of components."""
        stats = measure_attention_memory(64, 32)

        expected_total = (
            stats['input_memory'] +
            stats['attention_matrix_memory'] +
            stats['output_memory']
        )
        assert stats['total_memory'] == expected_total

    def test_correct_attention_matrix_size(self):
        """Attention matrix should be seq_len × seq_len × dtype_bytes."""
        seq_len = 128
        stats = measure_attention_memory(seq_len, 64)

        expected = seq_len * seq_len * 4  # float32
        assert stats['attention_matrix_memory'] == expected


class TestProfileScaling:
    """Tests for profile_memory_scaling."""

    def test_returns_all_seq_lengths(self):
        """Should return results for all requested seq_lengths."""
        seq_lengths = [32, 64, 128]
        results = profile_memory_scaling(seq_lengths, d_model=16)

        for sl in seq_lengths:
            assert sl in results

    def test_quadratic_scaling(self):
        """Attention matrix memory should scale quadratically."""
        results = profile_memory_scaling([64, 128], d_model=16)

        mem_64 = results[64]['attention_matrix_memory']
        mem_128 = results[128]['attention_matrix_memory']

        # When seq_len doubles, attention matrix should 4x
        ratio = mem_128 / mem_64
        np.testing.assert_allclose(ratio, 4.0, rtol=0.01)

    def test_contains_memory_stats(self):
        """Each result should contain memory statistics."""
        results = profile_memory_scaling([32], d_model=16)

        assert 'input_memory' in results[32]
        assert 'attention_matrix_memory' in results[32]


class TestEstimateMemory:
    """Tests for estimate_attention_memory."""

    def test_returns_required_keys(self):
        """Should return all required estimates."""
        est = estimate_attention_memory(64, 32)

        assert 'Q_memory' in est
        assert 'K_memory' in est
        assert 'V_memory' in est
        assert 'attention_matrix_memory' in est
        assert 'output_memory' in est
        assert 'total_memory' in est

    def test_correct_attention_estimate(self):
        """Attention matrix estimate should be N × N × dtype_bytes."""
        seq_len, d_model = 256, 64
        est = estimate_attention_memory(seq_len, d_model, dtype_bytes=4)

        expected = seq_len * seq_len * 4
        assert est['attention_matrix_memory'] == expected

    def test_correct_qkv_estimate(self):
        """Q, K, V estimates should be N × d × dtype_bytes each."""
        seq_len, d_model = 256, 64
        est = estimate_attention_memory(seq_len, d_model, dtype_bytes=4)

        expected_each = seq_len * d_model * 4
        assert est['Q_memory'] == expected_each
        assert est['K_memory'] == expected_each
        assert est['V_memory'] == expected_each

    def test_estimate_matches_actual(self):
        """Estimates should closely match actual measurements."""
        seq_len, d_model = 128, 32

        est = estimate_attention_memory(seq_len, d_model)
        actual = measure_attention_memory(seq_len, d_model)

        # Attention matrix memory should match exactly
        assert est['attention_matrix_memory'] == actual['attention_matrix_memory']

    def test_float16_estimate(self):
        """Should work with different dtype sizes."""
        seq_len = 256
        est_fp32 = estimate_attention_memory(seq_len, 64, dtype_bytes=4)
        est_fp16 = estimate_attention_memory(seq_len, 64, dtype_bytes=2)

        # FP16 should be half the memory
        assert est_fp16['attention_matrix_memory'] == est_fp32['attention_matrix_memory'] // 2


class TestAnalyzeScaling:
    """Tests for analyze_scaling."""

    def test_returns_required_keys(self):
        """Should return all required analysis."""
        results = profile_memory_scaling([32, 64], d_model=16)
        analysis = analyze_scaling(results)

        assert 'seq_lengths' in analysis
        assert 'attention_memory' in analysis
        assert 'total_memory' in analysis
        assert 'scaling_factor' in analysis

    def test_scaling_factor_approximately_4(self):
        """Scaling factor should be ~4 for quadratic attention."""
        results = profile_memory_scaling([64, 128, 256], d_model=16)
        analysis = analyze_scaling(results)

        # For O(N²), doubling N should 4x memory
        np.testing.assert_allclose(analysis['scaling_factor'], 4.0, rtol=0.1)

    def test_seq_lengths_preserved(self):
        """Should preserve the sequence lengths from input."""
        seq_lengths = [32, 64, 128]
        results = profile_memory_scaling(seq_lengths, d_model=16)
        analysis = analyze_scaling(results)

        assert set(analysis['seq_lengths']) == set(seq_lengths)


class TestMemoryBottleneck:
    """Tests demonstrating the memory bottleneck."""

    def test_attention_matrix_is_bottleneck(self):
        """Attention matrix should be the dominant memory consumer."""
        # For seq_len=1024, d_model=64:
        # Attention: 1024 * 1024 * 4 = 4 MB
        # Q, K, V each: 1024 * 64 * 4 = 256 KB
        stats = measure_attention_memory(1024, 64)

        attn_fraction = stats['attention_matrix_memory'] / stats['total_memory']

        # Attention matrix should be majority of memory
        assert attn_fraction > 0.7, (
            f"Attention matrix should dominate, but is only {attn_fraction:.1%}"
        )

    def test_scaling_demonstration(self):
        """Demonstrate quadratic scaling visually."""
        results = profile_memory_scaling([256, 512, 1024, 2048], d_model=64)

        print("\nMemory Scaling Demonstration:")
        print("-" * 50)
        print(f"{'Seq Length':<15} {'Attention (MB)':<20} {'Total (MB)':<15}")
        print("-" * 50)

        for sl in sorted(results.keys()):
            attn_mb = results[sl]['attention_matrix_memory'] / (1024 * 1024)
            total_mb = results[sl]['total_memory'] / (1024 * 1024)
            print(f"{sl:<15} {attn_mb:<20.2f} {total_mb:<15.2f}")

        print("-" * 50)
        print("Note: Attention matrix grows as O(N²)!")
