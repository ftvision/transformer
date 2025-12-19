"""Tests for Lab 05: Mixture of Experts."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moe import (
    Expert,
    Router,
    compute_aux_loss,
    compute_expert_usage,
    MixtureOfExperts,
    SparseMoE,
    analyze_moe_efficiency,
)


class TestExpert:
    """Tests for Expert class."""

    def test_init(self):
        """Should initialize with correct dimensions."""
        expert = Expert(d_model=512, d_ff=2048)
        assert expert.d_model == 512
        assert expert.d_ff == 2048

    def test_weight_shapes(self):
        """Weights should have correct shapes."""
        expert = Expert(d_model=512, d_ff=2048)
        assert expert.W1.shape == (512, 2048)
        assert expert.W2.shape == (2048, 512)

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        expert = Expert(d_model=512, d_ff=2048)
        x = np.random.randn(10, 512).astype(np.float32)

        output = expert(x)

        assert output.shape == x.shape

    def test_forward_batched(self):
        """Should handle batched input."""
        expert = Expert(d_model=512, d_ff=2048)
        x = np.random.randn(2, 10, 512).astype(np.float32)

        output = expert(x)

        assert output.shape == x.shape

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        expert = Expert(d_model=512, d_ff=2048)
        x1 = np.random.randn(10, 512).astype(np.float32)
        x2 = np.random.randn(10, 512).astype(np.float32)

        out1 = expert(x1)
        out2 = expert(x2)

        assert not np.allclose(out1, out2)


class TestRouter:
    """Tests for Router class."""

    def test_init(self):
        """Should initialize with correct dimensions."""
        router = Router(d_model=512, num_experts=8)
        assert router.d_model == 512
        assert router.num_experts == 8

    def test_weight_shape(self):
        """Router weights should have correct shape."""
        router = Router(d_model=512, num_experts=8)
        assert router.W_router.shape == (512, 8)

    def test_forward_shapes(self):
        """Forward should return correct shapes."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(10, 512).astype(np.float32)

        top_k_indices, top_k_weights, router_probs = router(x, top_k=2)

        assert top_k_indices.shape == (10, 2)
        assert top_k_weights.shape == (10, 2)
        assert router_probs.shape == (10, 8)

    def test_forward_batched(self):
        """Should handle batched input."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(2, 10, 512).astype(np.float32)

        top_k_indices, top_k_weights, router_probs = router(x, top_k=2)

        assert top_k_indices.shape == (2, 10, 2)
        assert top_k_weights.shape == (2, 10, 2)
        assert router_probs.shape == (2, 10, 8)

    def test_router_probs_sum_to_one(self):
        """Router probabilities should sum to 1."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(10, 512).astype(np.float32)

        _, _, router_probs = router(x, top_k=2)

        row_sums = router_probs.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_top_k_weights_sum_to_one(self):
        """Top-k weights should be normalized to sum to 1."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(10, 512).astype(np.float32)

        _, top_k_weights, _ = router(x, top_k=2)

        row_sums = top_k_weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_top_k_indices_valid(self):
        """Top-k indices should be valid expert indices."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(10, 512).astype(np.float32)

        top_k_indices, _, _ = router(x, top_k=2)

        assert np.all(top_k_indices >= 0)
        assert np.all(top_k_indices < 8)

    def test_top_k_indices_unique_per_token(self):
        """Top-k indices should be unique for each token."""
        router = Router(d_model=512, num_experts=8)
        x = np.random.randn(10, 512).astype(np.float32)

        top_k_indices, _, _ = router(x, top_k=3)

        for i in range(10):
            indices = top_k_indices[i]
            assert len(np.unique(indices)) == len(indices)


class TestAuxLoss:
    """Tests for compute_aux_loss function."""

    def test_returns_scalar(self):
        """Should return a scalar loss value."""
        router_probs = np.random.rand(10, 8)
        router_probs = router_probs / router_probs.sum(axis=-1, keepdims=True)
        top_k_indices = np.random.randint(0, 8, (10, 2))

        loss = compute_aux_loss(router_probs, top_k_indices, num_experts=8)

        assert isinstance(loss, (float, np.floating))

    def test_uniform_distribution_baseline(self):
        """Uniform routing should give loss ≈ 1.0."""
        num_experts = 8
        num_tokens = 100

        # Uniform routing probabilities
        router_probs = np.ones((num_tokens, num_experts)) / num_experts

        # Uniform expert assignment
        top_k_indices = np.zeros((num_tokens, 2), dtype=int)
        for i in range(num_tokens):
            top_k_indices[i] = [i % num_experts, (i + 1) % num_experts]

        loss = compute_aux_loss(router_probs, top_k_indices, num_experts)

        # Should be close to 1.0 for uniform distribution
        np.testing.assert_allclose(loss, 1.0, rtol=0.1)

    def test_imbalanced_increases_loss(self):
        """Imbalanced routing should increase loss."""
        num_experts = 8
        num_tokens = 100

        # Uniform baseline
        router_probs_uniform = np.ones((num_tokens, num_experts)) / num_experts
        top_k_uniform = np.zeros((num_tokens, 2), dtype=int)
        for i in range(num_tokens):
            top_k_uniform[i] = [i % num_experts, (i + 1) % num_experts]
        loss_uniform = compute_aux_loss(router_probs_uniform, top_k_uniform, num_experts)

        # Imbalanced: all tokens go to expert 0
        router_probs_imbal = np.zeros((num_tokens, num_experts))
        router_probs_imbal[:, 0] = 0.9
        router_probs_imbal[:, 1:] = 0.1 / (num_experts - 1)
        top_k_imbal = np.zeros((num_tokens, 2), dtype=int)
        top_k_imbal[:, 0] = 0
        top_k_imbal[:, 1] = 1
        loss_imbal = compute_aux_loss(router_probs_imbal, top_k_imbal, num_experts)

        # Imbalanced should have higher loss
        assert loss_imbal > loss_uniform


class TestExpertUsage:
    """Tests for compute_expert_usage function."""

    def test_returns_correct_shape(self):
        """Should return array of shape (num_experts,)."""
        top_k_indices = np.random.randint(0, 8, (10, 2))

        usage = compute_expert_usage(top_k_indices, num_experts=8)

        assert usage.shape == (8,)

    def test_usage_sums_correctly(self):
        """Usage should reflect assignment counts."""
        # All tokens go to expert 0 and 1
        top_k_indices = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])

        usage = compute_expert_usage(top_k_indices, num_experts=4)

        # Expert 0 and 1 should each have 50% of assignments
        assert usage[0] == 0.5
        assert usage[1] == 0.5
        assert usage[2] == 0.0
        assert usage[3] == 0.0

    def test_uniform_usage(self):
        """Uniform distribution should give equal usage."""
        num_experts = 4
        num_tokens = 100
        top_k = 1

        # Distribute evenly
        top_k_indices = np.zeros((num_tokens, top_k), dtype=int)
        for i in range(num_tokens):
            top_k_indices[i, 0] = i % num_experts

        usage = compute_expert_usage(top_k_indices, num_experts)

        expected = np.ones(num_experts) / num_experts
        np.testing.assert_allclose(usage, expected, rtol=1e-5)


class TestMixtureOfExperts:
    """Tests for MixtureOfExperts class."""

    def test_init(self):
        """Should initialize with correct parameters."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        assert moe.d_model == 512
        assert moe.num_experts == 8
        assert moe.top_k == 2

    def test_init_invalid_top_k(self):
        """Should raise error if top_k > num_experts."""
        with pytest.raises(ValueError):
            MixtureOfExperts(
                d_model=512,
                d_ff=2048,
                num_experts=4,
                top_k=5
            )

    def test_has_experts(self):
        """Should have correct number of experts."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        assert len(moe.experts) == 8

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(2, 10, 512).astype(np.float32)

        output, aux_loss = moe(x)

        assert output.shape == x.shape

    def test_forward_unbatched(self):
        """Should handle unbatched input."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(10, 512).astype(np.float32)

        output, aux_loss = moe(x)

        assert output.shape == x.shape

    def test_returns_aux_loss(self):
        """Should return auxiliary loss."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(10, 512).astype(np.float32)

        output, aux_loss = moe(x)

        assert aux_loss is not None
        assert isinstance(aux_loss, (float, np.floating))

    def test_can_skip_aux_loss(self):
        """Should be able to skip aux loss computation."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(10, 512).astype(np.float32)

        output, aux_loss = moe(x, return_aux_loss=False)

        assert aux_loss is None

    def test_get_expert_usage(self):
        """Should track expert usage."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(100, 512).astype(np.float32)

        _ = moe(x)
        usage = moe.get_expert_usage()

        assert usage.shape == (8,)
        assert np.all(usage >= 0)
        # Total should sum to 1 (each token contributes 1/num_tokens per expert selected)
        np.testing.assert_allclose(usage.sum(), 1.0, rtol=1e-5)

    def test_get_aux_loss(self):
        """Should store and return aux loss."""
        moe = MixtureOfExperts(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(10, 512).astype(np.float32)

        _, expected_loss = moe(x)
        stored_loss = moe.get_aux_loss()

        assert stored_loss == expected_loss

    def test_different_top_k(self):
        """Should work with different top_k values."""
        for top_k in [1, 2, 4]:
            moe = MixtureOfExperts(
                d_model=512,
                d_ff=2048,
                num_experts=8,
                top_k=top_k
            )
            x = np.random.randn(10, 512).astype(np.float32)

            output, _ = moe(x)

            assert output.shape == x.shape


class TestSparseMoE:
    """Tests for SparseMoE with capacity limiting."""

    def test_init(self):
        """Should initialize with capacity factor."""
        try:
            sparse_moe = SparseMoE(
                d_model=512,
                d_ff=2048,
                num_experts=8,
                top_k=2,
                capacity_factor=1.25
            )
            assert sparse_moe.capacity_factor == 1.25
        except NotImplementedError:
            pytest.skip("SparseMoE not implemented")

    def test_compute_capacity(self):
        """Should compute correct capacity."""
        try:
            sparse_moe = SparseMoE(
                d_model=512,
                d_ff=2048,
                num_experts=8,
                top_k=2,
                capacity_factor=1.25
            )

            capacity = sparse_moe.compute_capacity(num_tokens=100)

            # Expected: (100 / 8) * 1.25 = 15.625 → 16
            assert capacity == 16
        except NotImplementedError:
            pytest.skip("SparseMoE not implemented")

    def test_returns_drop_rate(self):
        """Should return token drop rate."""
        try:
            sparse_moe = SparseMoE(
                d_model=512,
                d_ff=2048,
                num_experts=8,
                top_k=2,
                capacity_factor=1.25
            )
            x = np.random.randn(100, 512).astype(np.float32)

            output, aux_loss, drop_rate = sparse_moe(x)

            assert isinstance(drop_rate, (float, np.floating))
            assert 0 <= drop_rate <= 1
        except NotImplementedError:
            pytest.skip("SparseMoE not implemented")


class TestAnalyzeEfficiency:
    """Tests for analyze_moe_efficiency function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = analyze_moe_efficiency(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2,
            seq_len=100,
            batch_size=2
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Should contain all required keys."""
        result = analyze_moe_efficiency(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2,
            seq_len=100,
            batch_size=2
        )

        assert 'total_params' in result
        assert 'active_params' in result
        assert 'dense_flops' in result
        assert 'moe_flops' in result
        assert 'param_efficiency' in result
        assert 'compute_efficiency' in result

    def test_param_efficiency(self):
        """MoE should have more total than active params."""
        result = analyze_moe_efficiency(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2,
            seq_len=100,
            batch_size=2
        )

        assert result['total_params'] > result['active_params']
        assert result['param_efficiency'] > 1

    def test_compute_efficiency(self):
        """MoE should use less compute than equivalent dense."""
        result = analyze_moe_efficiency(
            d_model=512,
            d_ff=2048,
            num_experts=8,
            top_k=2,
            seq_len=100,
            batch_size=2
        )

        # With top_k=2 and 8 experts, should use ~2/8 = 25% compute
        # But have 8x parameters
        assert result['moe_flops'] < result['dense_flops']


class TestMoEMilestone:
    """Milestone tests for Chapter 7 completion."""

    def test_routing_correctness(self):
        """Routing should select top-k experts correctly."""
        moe = MixtureOfExperts(
            d_model=64,
            d_ff=256,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(20, 64).astype(np.float32)

        # Run forward
        output, _ = moe(x)

        # Output should have correct shape
        assert output.shape == x.shape

        # Expert usage should reflect routing
        usage = moe.get_expert_usage()
        assert len(usage) == 8
        assert np.all(usage >= 0)

    def test_weighted_output(self):
        """Output should be weighted combination of expert outputs."""
        moe = MixtureOfExperts(
            d_model=64,
            d_ff=256,
            num_experts=4,
            top_k=2
        )

        # Use specific input to trace through
        x = np.random.randn(1, 64).astype(np.float32)
        output, _ = moe(x)

        # Output should not be zero (experts should contribute)
        assert not np.allclose(output, 0)

    def test_expert_usage_tracking(self):
        """Should track which experts are being used."""
        moe = MixtureOfExperts(
            d_model=64,
            d_ff=256,
            num_experts=8,
            top_k=2
        )
        x = np.random.randn(100, 64).astype(np.float32)

        _ = moe(x)
        usage = moe.get_expert_usage()

        # All experts should be used (with random routing)
        assert np.all(usage > 0), "Some experts never used"

        # Usage should sum to 1
        np.testing.assert_allclose(usage.sum(), 1.0, rtol=1e-5)

    def test_aux_loss_computed(self):
        """Auxiliary loss should be computed for load balancing."""
        moe = MixtureOfExperts(
            d_model=64,
            d_ff=256,
            num_experts=8,
            top_k=2,
            aux_loss_coef=0.01
        )
        x = np.random.randn(100, 64).astype(np.float32)

        _, aux_loss = moe(x)

        # Aux loss should be a reasonable value
        assert aux_loss is not None
        assert aux_loss > 0
        assert not np.isnan(aux_loss)
        assert not np.isinf(aux_loss)
