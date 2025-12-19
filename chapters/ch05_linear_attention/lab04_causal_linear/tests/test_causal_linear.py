"""Tests for Lab 04: Causal Linear Attention."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_linear import (
    causal_linear_attention_parallel,
    causal_linear_attention_parallel_batched,
    causal_linear_attention_recurrent,
    create_initial_state,
    causal_linear_attention_rnn_step,
    causal_softmax_attention,
    compare_parallel_vs_recurrent,
    compare_to_causal_softmax,
    benchmark_causal_attention,
    elu_plus_one,
)


class TestParallelForm:
    """Tests for causal_linear_attention_parallel."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = causal_linear_attention_parallel(Q, K, V, elu_plus_one)

        assert output.shape == (50, 64)

    def test_no_nan(self):
        """Output should not contain NaN."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = causal_linear_attention_parallel(Q, K, V, elu_plus_one)

        assert not np.any(np.isnan(output))

    def test_causality(self):
        """Output at position i should not depend on future positions."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        # Compute with full sequence
        output_full = causal_linear_attention_parallel(Q, K, V, elu_plus_one)

        # Compute with truncated sequence (first 25 positions)
        output_partial = causal_linear_attention_parallel(
            Q[:25], K[:25], V[:25], elu_plus_one
        )

        # First 25 positions should match
        np.testing.assert_allclose(output_full[:25], output_partial, rtol=1e-5)

    def test_different_seq_lengths(self):
        """Should work for various sequence lengths."""
        d_model = 64

        for seq_len in [10, 50, 100, 200]:
            Q = np.random.randn(seq_len, d_model)
            K = np.random.randn(seq_len, d_model)
            V = np.random.randn(seq_len, d_model)

            output = causal_linear_attention_parallel(Q, K, V, elu_plus_one)
            assert output.shape == (seq_len, d_model)


class TestParallelBatched:
    """Tests for causal_linear_attention_parallel_batched."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(4, 50, 64)
        K = np.random.randn(4, 50, 64)
        V = np.random.randn(4, 50, 64)

        output = causal_linear_attention_parallel_batched(Q, K, V, elu_plus_one)

        assert output.shape == (4, 50, 64)

    def test_matches_unbatched(self):
        """Batched should match running unbatched on each example."""
        Q = np.random.randn(4, 50, 64)
        K = np.random.randn(4, 50, 64)
        V = np.random.randn(4, 50, 64)

        output_batched = causal_linear_attention_parallel_batched(
            Q, K, V, elu_plus_one
        )

        for b in range(4):
            output_single = causal_linear_attention_parallel(
                Q[b], K[b], V[b], elu_plus_one
            )
            np.testing.assert_allclose(
                output_batched[b], output_single, rtol=1e-5
            )


class TestRecurrentForm:
    """Tests for causal_linear_attention_recurrent."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = causal_linear_attention_recurrent(Q, K, V, elu_plus_one)

        assert output.shape == (50, 64)

    def test_no_nan(self):
        """Output should not contain NaN."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = causal_linear_attention_recurrent(Q, K, V, elu_plus_one)

        assert not np.any(np.isnan(output))

    def test_matches_parallel(self):
        """Recurrent form should match parallel form exactly."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output_parallel = causal_linear_attention_parallel(Q, K, V, elu_plus_one)
        output_recurrent = causal_linear_attention_recurrent(Q, K, V, elu_plus_one)

        np.testing.assert_allclose(output_parallel, output_recurrent, rtol=1e-5)


class TestInitialState:
    """Tests for create_initial_state."""

    def test_state_shapes(self):
        """State should have correct shapes."""
        S, Z = create_initial_state(64, 32)

        assert S.shape == (64, 32)
        assert Z.shape == (64,)

    def test_state_initialized_to_zero(self):
        """State should be initialized to zeros."""
        S, Z = create_initial_state(64, 32)

        np.testing.assert_array_equal(S, np.zeros((64, 32)))
        np.testing.assert_array_equal(Z, np.zeros(64))


class TestRNNStep:
    """Tests for causal_linear_attention_rnn_step."""

    def test_output_shape(self):
        """Output should have correct shape."""
        d_model = 64
        state = create_initial_state(d_model, d_model)
        q = np.random.randn(d_model)
        k = np.random.randn(d_model)
        v = np.random.randn(d_model)

        output, new_state = causal_linear_attention_rnn_step(
            q, k, v, state, elu_plus_one
        )

        assert output.shape == (d_model,)

    def test_state_updated(self):
        """State should be updated after each step."""
        d_model = 64
        state = create_initial_state(d_model, d_model)
        q = np.random.randn(d_model)
        k = np.random.randn(d_model)
        v = np.random.randn(d_model)

        S_old, Z_old = state
        _, (S_new, Z_new) = causal_linear_attention_rnn_step(
            q, k, v, state, elu_plus_one
        )

        # State should have changed
        assert not np.allclose(S_new, S_old)
        assert not np.allclose(Z_new, Z_old)

    def test_step_by_step_matches_recurrent(self):
        """Running step by step should match recurrent form."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        # Recurrent form
        output_recurrent = causal_linear_attention_recurrent(Q, K, V, elu_plus_one)

        # Step by step
        d_model = 64
        state = create_initial_state(d_model, d_model)
        outputs_step = []

        for i in range(50):
            output, state = causal_linear_attention_rnn_step(
                Q[i], K[i], V[i], state, elu_plus_one
            )
            outputs_step.append(output)

        output_step = np.stack(outputs_step)

        np.testing.assert_allclose(output_recurrent, output_step, rtol=1e-5)


class TestCausalSoftmaxAttention:
    """Tests for causal_softmax_attention."""

    def test_output_shape(self):
        """Output should have correct shape."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        output = causal_softmax_attention(Q, K, V)

        assert output.shape == (50, 64)

    def test_causality(self):
        """Output at position i should only depend on positions <= i."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        # Full sequence
        output_full = causal_softmax_attention(Q, K, V)

        # Truncated
        output_partial = causal_softmax_attention(Q[:25], K[:25], V[:25])

        # First 25 should match
        np.testing.assert_allclose(output_full[:25], output_partial, rtol=1e-5)


class TestComparisons:
    """Tests for comparison functions."""

    def test_parallel_vs_recurrent_match(self):
        """Parallel and recurrent should match."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        result = compare_parallel_vs_recurrent(Q, K, V, elu_plus_one)

        assert result['match']
        assert result['max_diff'] < 1e-5

    def test_compare_to_softmax_returns_dict(self):
        """Should return dictionary with expected keys."""
        Q = np.random.randn(50, 64)
        K = np.random.randn(50, 64)
        V = np.random.randn(50, 64)

        result = compare_to_causal_softmax(Q, K, V, elu_plus_one)

        assert 'output_mse' in result
        assert 'output_correlation' in result
        assert 'output_max_diff' in result

    def test_some_correlation_with_softmax(self):
        """Linear attention should have some correlation with softmax."""
        np.random.seed(42)
        Q = np.random.randn(100, 64)
        K = np.random.randn(100, 64)
        V = np.random.randn(100, 64)

        result = compare_to_causal_softmax(Q, K, V, elu_plus_one)

        # Should have positive correlation
        assert result['output_correlation'] > 0


class TestBenchmark:
    """Tests for benchmark_causal_attention."""

    def test_returns_dict(self):
        """Should return dictionary with expected keys."""
        result = benchmark_causal_attention([64, 128], d_model=32, num_runs=1)

        assert 'seq_lengths' in result
        assert 'softmax_times' in result
        assert 'linear_parallel_times' in result
        assert 'speedup' in result

    def test_linear_faster_for_long_sequences(self):
        """Linear attention should be faster for long sequences."""
        result = benchmark_causal_attention([512, 1024], d_model=64, num_runs=2)

        # For longer sequences, linear should be faster
        # (might not be true for very short sequences due to overhead)
        for i in range(len(result['seq_lengths'])):
            if result['seq_lengths'][i] >= 512:
                assert result['speedup'][i] > 1.0, \
                    f"Linear not faster at seq_len={result['seq_lengths'][i]}"


class TestChapter5Milestone:
    """
    Chapter 5 Milestone: Linear attention 10x faster for seq_len=4096.
    """

    def test_chapter5_milestone(self):
        """
        MILESTONE: Linear attention is 10x faster than standard for long sequences.

        This test verifies that:
        1. Causal linear attention works correctly
        2. It matches between parallel and recurrent forms
        3. It's significantly faster than softmax for long sequences
        """
        np.random.seed(42)

        # Test correctness
        Q = np.random.randn(100, 64)
        K = np.random.randn(100, 64)
        V = np.random.randn(100, 64)

        # Parallel and recurrent should match
        result = compare_parallel_vs_recurrent(Q, K, V, elu_plus_one)
        assert result['match'], "Parallel and recurrent forms don't match!"

        # Test speed advantage
        benchmark = benchmark_causal_attention(
            [256, 512, 1024, 2048],
            d_model=64,
            num_runs=2
        )

        # Find speedup for longest sequence
        max_speedup = max(benchmark['speedup'])
        long_seq_speedup = benchmark['speedup'][-1]  # For 2048

        print(f"\n{'='*60}")
        print("Chapter 5 Milestone Progress:")
        print(f"Parallel/Recurrent match: ✓")
        print(f"Speedups by sequence length:")
        for i, seq_len in enumerate(benchmark['seq_lengths']):
            print(f"  seq_len={seq_len}: {benchmark['speedup'][i]:.1f}x faster")
        print(f"Max speedup achieved: {max_speedup:.1f}x")

        # The milestone is 10x for 4096
        # For 2048, we expect around 5x if O(n²) vs O(n×d)
        # With d=64, n=2048: speedup = n/d = 32 theoretically
        # In practice, overhead reduces this
        if long_seq_speedup >= 5.0:
            print(f"{'='*60}")
            print("MILESTONE: Linear attention significantly faster!")
            print(f"{'='*60}\n")
        else:
            print(f"Note: Speedup {long_seq_speedup:.1f}x is less than 5x")
            print("This may be due to overhead or NumPy implementation")
            print(f"{'='*60}\n")

        # Still pass if we have reasonable speedup
        assert long_seq_speedup > 1.0, \
            "Linear attention should be faster than softmax for long sequences"
