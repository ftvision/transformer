"""Tests for Lab 02: Gradient Visualization."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gradients import (
    compute_gradient_norm,
    compute_global_norm,
    compute_gradient_stats,
    detect_gradient_issues,
    clip_gradients,
    analyze_layer_gradients,
    track_gradient_history,
    create_gradient_visualization_data,
    simulate_gradient_flow,
)


class TestGradientNorm:
    """Tests for gradient norm computation."""

    def test_l2_norm_simple(self):
        """Test L2 norm with simple vector."""
        grad = np.array([3.0, 4.0])
        norm = compute_gradient_norm(grad, ord=2)

        np.testing.assert_allclose(norm, 5.0, rtol=1e-5)

    def test_l1_norm(self):
        """Test L1 norm."""
        grad = np.array([3.0, -4.0])
        norm = compute_gradient_norm(grad, ord=1)

        np.testing.assert_allclose(norm, 7.0, rtol=1e-5)

    def test_linf_norm(self):
        """Test Linf (max) norm."""
        grad = np.array([3.0, -4.0, 2.0])
        norm = compute_gradient_norm(grad, ord=np.inf)

        np.testing.assert_allclose(norm, 4.0, rtol=1e-5)

    def test_2d_array(self):
        """Should work for 2D arrays."""
        grad = np.array([[1.0, 2.0], [3.0, 4.0]])
        norm = compute_gradient_norm(grad, ord=2)

        # L2 norm of flattened: sqrt(1 + 4 + 9 + 16) = sqrt(30)
        expected = np.sqrt(30)
        np.testing.assert_allclose(norm, expected, rtol=1e-5)

    def test_zero_gradient(self):
        """Zero gradient should have zero norm."""
        grad = np.zeros((10, 10))
        norm = compute_gradient_norm(grad)

        assert norm == 0.0


class TestGlobalNorm:
    """Tests for global norm across multiple tensors."""

    def test_global_norm_simple(self):
        """Test global norm with two vectors."""
        g1 = np.array([3.0, 4.0])  # L2 norm = 5
        g2 = np.array([5.0, 12.0])  # L2 norm = 13

        global_norm = compute_global_norm([g1, g2])

        # sqrt(5^2 + 13^2) = sqrt(25 + 169) = sqrt(194)
        expected = np.sqrt(194)
        np.testing.assert_allclose(global_norm, expected, rtol=1e-5)

    def test_single_tensor(self):
        """Global norm of single tensor equals its L2 norm."""
        g = np.array([3.0, 4.0])

        global_norm = compute_global_norm([g])
        single_norm = compute_gradient_norm(g)

        np.testing.assert_allclose(global_norm, single_norm, rtol=1e-5)

    def test_different_shapes(self):
        """Should handle tensors of different shapes."""
        g1 = np.random.randn(10, 20)
        g2 = np.random.randn(5, 5, 5)
        g3 = np.random.randn(100)

        global_norm = compute_global_norm([g1, g2, g3])

        assert global_norm >= 0
        assert not np.isnan(global_norm)


class TestGradientStats:
    """Tests for gradient statistics."""

    def test_stats_keys(self):
        """Should return all required statistics."""
        grad = np.random.randn(100, 100)
        stats = compute_gradient_stats(grad)

        required_keys = ['norm', 'mean', 'std', 'min', 'max',
                         'abs_mean', 'near_zero_frac', 'large_frac']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_values_reasonable(self):
        """Statistics should be reasonable for standard normal gradients."""
        np.random.seed(42)
        grad = np.random.randn(1000, 1000) * 0.01  # Scale similar to real grads

        stats = compute_gradient_stats(grad)

        assert abs(stats['mean']) < 0.001  # Should be near 0
        assert 0.009 < stats['std'] < 0.011  # Should be near 0.01
        assert stats['near_zero_frac'] < 0.01  # Few near-zero values
        assert stats['large_frac'] == 0  # No large values

    def test_near_zero_detection(self):
        """Should correctly count near-zero gradients."""
        grad = np.array([1e-8, 1e-7, 1e-6, 0.0, 0.1])

        stats = compute_gradient_stats(grad)

        # Values < 1e-7: 1e-8 and 0.0 = 2 out of 5
        assert stats['near_zero_frac'] == pytest.approx(2/5, rel=1e-5)

    def test_large_detection(self):
        """Should correctly count large gradients."""
        grad = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

        stats = compute_gradient_stats(grad)

        # Values > 1e3: 10000.0 = 1 out of 5
        assert stats['large_frac'] == pytest.approx(1/5, rel=1e-5)


class TestDetectGradientIssues:
    """Tests for gradient issue detection."""

    def test_healthy_gradients(self):
        """Should detect healthy gradient flow."""
        norms = [0.1, 0.11, 0.09, 0.1, 0.12, 0.08]  # Relatively stable

        result = detect_gradient_issues(norms)

        assert result['status'] == 'healthy'

    def test_vanishing_gradients(self):
        """Should detect vanishing gradients."""
        norms = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # Decreasing

        result = detect_gradient_issues(norms)

        assert result['status'] == 'vanishing'
        assert 'vanish' in result['message'].lower()

    def test_exploding_gradients(self):
        """Should detect exploding gradients."""
        norms = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]  # Increasing

        result = detect_gradient_issues(norms)

        assert result['status'] == 'exploding'
        assert 'explod' in result['message'].lower()

    def test_already_vanished(self):
        """Should detect already-vanished gradients."""
        norms = [1e-8, 1e-9, 1e-10, 1e-11]  # All very small

        result = detect_gradient_issues(norms)

        assert result['status'] == 'vanishing'

    def test_already_exploded(self):
        """Should detect already-exploded gradients."""
        norms = [1e8, 1e9, 1e10]  # All very large

        result = detect_gradient_issues(norms)

        assert result['status'] == 'exploding'

    def test_returns_ratio(self):
        """Should return max/min ratio."""
        norms = [0.1, 0.05, 0.01]

        result = detect_gradient_issues(norms)

        expected_ratio = 0.1 / 0.01
        np.testing.assert_allclose(result['ratio'], expected_ratio, rtol=1e-5)


class TestClipGradients:
    """Tests for gradient clipping."""

    def test_no_clipping_needed(self):
        """Should not modify gradients below max_norm."""
        g1 = np.array([0.3, 0.4])  # norm = 0.5
        g2 = np.array([0.0, 0.0])

        clipped, orig_norm = clip_gradients([g1, g2], max_norm=1.0)

        np.testing.assert_allclose(clipped[0], g1, rtol=1e-5)
        np.testing.assert_allclose(orig_norm, 0.5, rtol=1e-5)

    def test_clipping_applied(self):
        """Should clip gradients above max_norm."""
        g1 = np.array([3.0, 4.0])  # norm = 5

        clipped, orig_norm = clip_gradients([g1], max_norm=2.5)

        np.testing.assert_allclose(orig_norm, 5.0, rtol=1e-5)

        clipped_norm = compute_global_norm(clipped)
        np.testing.assert_allclose(clipped_norm, 2.5, rtol=1e-5)

    def test_clipping_preserves_direction(self):
        """Clipping should preserve gradient direction."""
        g = np.array([3.0, 4.0])

        clipped, _ = clip_gradients([g], max_norm=2.5)

        # Direction should be same (ratio of components)
        original_ratio = g[0] / g[1]
        clipped_ratio = clipped[0][0] / clipped[0][1]
        np.testing.assert_allclose(clipped_ratio, original_ratio, rtol=1e-5)

    def test_clipping_multiple_tensors(self):
        """Should clip across multiple tensors correctly."""
        g1 = np.array([3.0, 4.0])  # norm = 5
        g2 = np.array([5.0, 12.0])  # norm = 13
        # Global norm = sqrt(25 + 169) = sqrt(194) ≈ 13.93

        clipped, orig_norm = clip_gradients([g1, g2], max_norm=1.0)

        clipped_norm = compute_global_norm(clipped)
        np.testing.assert_allclose(clipped_norm, 1.0, rtol=1e-5)

    def test_zero_gradients(self):
        """Should handle zero gradients gracefully."""
        g = np.zeros((10, 10))

        clipped, orig_norm = clip_gradients([g], max_norm=1.0)

        assert orig_norm == 0.0
        np.testing.assert_array_equal(clipped[0], g)


class TestAnalyzeLayerGradients:
    """Tests for layer-wise gradient analysis."""

    def test_analysis_structure(self):
        """Should return correct structure."""
        grads = {
            'layer_0': np.random.randn(10, 10) * 0.01,
            'layer_1': np.random.randn(10, 10) * 0.01,
        }

        analysis = analyze_layer_gradients(grads)

        assert 'layer_stats' in analysis
        assert 'layer_norms' in analysis
        assert 'global_norm' in analysis
        assert 'flow_status' in analysis
        assert 'recommendations' in analysis

    def test_layer_stats_present(self):
        """Should compute stats for each layer."""
        grads = {
            'embedding': np.random.randn(100, 64),
            'attention': np.random.randn(64, 64),
            'output': np.random.randn(64, 100),
        }

        analysis = analyze_layer_gradients(grads)

        assert len(analysis['layer_stats']) == 3
        assert 'embedding' in analysis['layer_stats']
        assert 'attention' in analysis['layer_stats']
        assert 'output' in analysis['layer_stats']

    def test_recommendations_for_issues(self):
        """Should provide recommendations when issues detected."""
        # Vanishing gradients
        grads = {
            'layer_0': np.random.randn(10, 10) * 1e-8,
            'layer_1': np.random.randn(10, 10) * 1e-10,
        }

        analysis = analyze_layer_gradients(grads)

        assert len(analysis['recommendations']) > 0


class TestTrackGradientHistory:
    """Tests for gradient history tracking."""

    def test_adds_to_empty_history(self):
        """Should initialize history for new layers."""
        history = {}
        grads = {'layer_0': np.array([3.0, 4.0])}

        updated = track_gradient_history(grads, history)

        assert 'layer_0' in updated
        assert len(updated['layer_0']) == 1
        np.testing.assert_allclose(updated['layer_0'][0], 5.0, rtol=1e-5)

    def test_appends_to_existing(self):
        """Should append to existing history."""
        history = {'layer_0': [1.0, 2.0]}
        grads = {'layer_0': np.array([3.0, 4.0])}

        updated = track_gradient_history(grads, history)

        assert len(updated['layer_0']) == 3
        np.testing.assert_allclose(updated['layer_0'][-1], 5.0, rtol=1e-5)

    def test_respects_max_history(self):
        """Should limit history length."""
        history = {'layer_0': list(range(100))}
        grads = {'layer_0': np.array([1.0])}

        updated = track_gradient_history(grads, history, max_history=50)

        assert len(updated['layer_0']) == 50

    def test_multiple_layers(self):
        """Should track multiple layers."""
        history = {}
        grads = {
            'layer_0': np.array([1.0]),
            'layer_1': np.array([2.0]),
        }

        updated = track_gradient_history(grads, history)

        assert 'layer_0' in updated
        assert 'layer_1' in updated


class TestVisualizationData:
    """Tests for visualization data creation."""

    def test_visualization_structure(self):
        """Should return correct structure."""
        norms = [('layer_0', 0.1), ('layer_1', 0.05)]

        viz = create_gradient_visualization_data(norms)

        assert 'layers' in viz
        assert 'max_norm' in viz
        assert 'min_norm' in viz
        assert len(viz['layers']) == 2

    def test_layer_data_structure(self):
        """Each layer should have required fields."""
        norms = [('layer_0', 0.1)]

        viz = create_gradient_visualization_data(norms)

        layer = viz['layers'][0]
        assert 'name' in layer
        assert 'norm' in layer
        assert 'bar_width' in layer
        assert 'bar_str' in layer
        assert 'norm_str' in layer

    def test_bar_width_scaling(self):
        """Bar width should scale with norm."""
        norms = [('layer_0', 1.0), ('layer_1', 0.5)]

        viz = create_gradient_visualization_data(norms, max_bar_width=50)

        # Largest norm should have max width
        assert viz['layers'][0]['bar_width'] == 50
        # Half norm should have half width
        assert viz['layers'][1]['bar_width'] == 25

    def test_bar_string_length(self):
        """Bar string should match width."""
        norms = [('layer_0', 1.0)]

        viz = create_gradient_visualization_data(norms, max_bar_width=10)

        assert len(viz['layers'][0]['bar_str']) == 10


class TestSimulateGradientFlow:
    """Tests for gradient flow simulation."""

    def test_vanishing_simulation(self):
        """Should simulate vanishing gradients."""
        norms = simulate_gradient_flow(10, gradient_scale=0.5, noise_scale=0.0)

        # Each layer should have roughly half the norm
        assert norms[0] > norms[-1]
        assert len(norms) == 10

    def test_exploding_simulation(self):
        """Should simulate exploding gradients."""
        norms = simulate_gradient_flow(10, gradient_scale=2.0, noise_scale=0.0)

        # Each layer should have roughly double the norm
        assert norms[0] < norms[-1]

    def test_stable_simulation(self):
        """Should simulate stable gradients with scale=1.0."""
        norms = simulate_gradient_flow(10, gradient_scale=1.0, noise_scale=0.0)

        # All norms should be roughly equal
        np.testing.assert_allclose(norms, norms[0] * np.ones(10), rtol=0.1)

    def test_initial_norm(self):
        """Should respect initial_norm parameter."""
        norms = simulate_gradient_flow(5, gradient_scale=1.0, noise_scale=0.0, initial_norm=10.0)

        np.testing.assert_allclose(norms[0], 10.0, rtol=0.1)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis workflow."""
        # Simulate a model with 6 layers
        np.random.seed(42)

        layer_gradients = {}
        for i in range(6):
            # Slight decay to simulate mild vanishing
            scale = 0.01 * (0.9 ** i)
            layer_gradients[f'layer_{i}'] = np.random.randn(100, 100) * scale

        # Run full analysis
        analysis = analyze_layer_gradients(layer_gradients)

        # Should detect the gradient flow status
        assert analysis['flow_status']['status'] in ['healthy', 'vanishing', 'exploding', 'unstable']

        # Should have stats for all layers
        assert len(analysis['layer_stats']) == 6

        # Global norm should be positive
        assert analysis['global_norm'] > 0

    def test_clip_and_analyze(self):
        """Test clipping followed by analysis."""
        grads = {
            'layer_0': np.random.randn(100, 100) * 10.0,  # Large
            'layer_1': np.random.randn(100, 100) * 10.0,
        }

        # Clip gradients
        grad_list = list(grads.values())
        clipped, orig_norm = clip_gradients(grad_list, max_norm=1.0)

        # Analyze clipped gradients
        clipped_grads = {f'layer_{i}': g for i, g in enumerate(clipped)}
        analysis = analyze_layer_gradients(clipped_grads)

        # Clipped global norm should be <= 1.0
        assert analysis['global_norm'] <= 1.0 + 1e-5
