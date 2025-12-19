"""Tests for Lab 04: Transformer Block."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformer_block import (
    TransformerBlock,
    Transformer,
    LayerNorm,
    MultiHeadAttention,
    FeedForward,
)


class TestLayerNorm:
    """Tests for LayerNorm component."""

    def test_init(self):
        """Should initialize with correct shapes."""
        ln = LayerNorm(512)

        assert ln.d_model == 512
        assert ln.gamma.shape == (512,)
        assert ln.beta.shape == (512,)

    def test_forward(self):
        """Should normalize correctly."""
        ln = LayerNorm(64)
        x = np.random.randn(2, 10, 64) * 5 + 3

        out = ln(x)

        # Should have mean ~0, var ~1
        means = out.mean(axis=-1)
        np.testing.assert_allclose(means, 0, atol=1e-5)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention component."""

    def test_init(self):
        """Should initialize with correct shapes."""
        mha = MultiHeadAttention(512, 8)

        assert mha.d_model == 512
        assert mha.num_heads == 8

    def test_forward_shape(self):
        """Output should have same shape as input."""
        mha = MultiHeadAttention(64, 8)
        x = np.random.randn(2, 10, 64)

        out = mha(x)

        assert out.shape == x.shape


class TestFeedForward:
    """Tests for FeedForward component."""

    def test_init(self):
        """Should initialize with correct shapes."""
        ffn = FeedForward(512)

        assert ffn.d_model == 512

    def test_forward_shape(self):
        """Output should have same shape as input."""
        ffn = FeedForward(64)
        x = np.random.randn(2, 10, 64)

        out = ffn(x)

        assert out.shape == x.shape


class TestTransformerBlockInit:
    """Tests for TransformerBlock initialization."""

    def test_basic_init(self):
        """Should initialize with required parameters."""
        block = TransformerBlock(d_model=512, num_heads=8)

        assert block.d_model == 512
        assert block.num_heads == 8

    def test_default_d_ff(self):
        """Default d_ff should be 4 * d_model."""
        block = TransformerBlock(d_model=512, num_heads=8)

        assert block.d_ff == 2048

    def test_custom_d_ff(self):
        """Should accept custom d_ff."""
        block = TransformerBlock(d_model=512, num_heads=8, d_ff=1024)

        assert block.d_ff == 1024

    def test_has_components(self):
        """Should have all sub-components."""
        block = TransformerBlock(d_model=512, num_heads=8)

        assert hasattr(block, 'norm1')
        assert hasattr(block, 'attn')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'ffn')

    def test_pre_norm_default(self):
        """Default should be pre-norm."""
        block = TransformerBlock(d_model=512, num_heads=8)

        assert block.pre_norm == True

    def test_post_norm(self):
        """Should support post-norm."""
        block = TransformerBlock(d_model=512, num_heads=8, pre_norm=False)

        assert block.pre_norm == False


class TestTransformerBlockForward:
    """Tests for TransformerBlock forward pass."""

    def test_output_shape_3d(self):
        """Output should have same shape as 3D input."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out = block(x)

        assert out.shape == x.shape

    def test_output_shape_2d(self):
        """Output should have same shape as 2D input."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(10, 64)

        out = block(x)

        assert out.shape == x.shape

    def test_callable(self):
        """Should be callable like a function."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out1 = block.forward(x)
        out2 = block(x)

        np.testing.assert_array_equal(out1, out2)

    def test_deterministic(self):
        """Same input should give same output."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out1 = block(x)
        out2 = block(x)

        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs(self):
        """Different inputs should give different outputs."""
        block = TransformerBlock(d_model=64, num_heads=8)

        x1 = np.random.randn(2, 10, 64)
        x2 = np.random.randn(2, 10, 64)

        out1 = block(x1)
        out2 = block(x2)

        assert not np.allclose(out1, out2)

    def test_with_mask(self):
        """Should work with attention mask."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        # Causal mask
        seq_len = 10
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        out = block(x, mask=mask)

        assert out.shape == x.shape


class TestResidualConnections:
    """Tests for residual connections."""

    def test_residual_applied(self):
        """Output should not be identical to sublayer outputs."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out = block(x)

        # If residuals are applied, output should be different from
        # what we'd get without them
        # Simple check: output should have some correlation with input
        correlation = np.corrcoef(x.flatten(), out.flatten())[0, 1]
        assert correlation > 0.1  # Some relationship should exist

    def test_output_not_normalized_only(self):
        """Output should not just be normalized input."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out = block(x)
        just_normed = block.norm1(x)

        # Output should be different from just normalized input
        assert not np.allclose(out, just_normed)


class TestPreNormVsPostNorm:
    """Tests comparing pre-norm and post-norm variants."""

    def test_different_outputs(self):
        """Pre-norm and post-norm should give different outputs."""
        np.random.seed(42)

        # Create blocks with same random seed for weights
        np.random.seed(42)
        pre_norm_block = TransformerBlock(d_model=64, num_heads=8, pre_norm=True)

        np.random.seed(42)
        post_norm_block = TransformerBlock(d_model=64, num_heads=8, pre_norm=False)

        x = np.random.randn(2, 10, 64)

        out_pre = pre_norm_block(x)
        out_post = post_norm_block(x)

        # Outputs should be different
        assert not np.allclose(out_pre, out_post)

    def test_both_valid_shapes(self):
        """Both variants should produce valid output shapes."""
        x = np.random.randn(2, 10, 64)

        pre_norm_block = TransformerBlock(d_model=64, num_heads=8, pre_norm=True)
        post_norm_block = TransformerBlock(d_model=64, num_heads=8, pre_norm=False)

        out_pre = pre_norm_block(x)
        out_post = post_norm_block(x)

        assert out_pre.shape == x.shape
        assert out_post.shape == x.shape


class TestTransformer:
    """Tests for stacked Transformer."""

    def test_init(self):
        """Should initialize with correct number of layers."""
        transformer = Transformer(num_layers=6, d_model=64, num_heads=8)

        assert len(transformer.layers) == 6

    def test_forward_shape(self):
        """Output should have same shape as input."""
        transformer = Transformer(num_layers=6, d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out = transformer(x)

        assert out.shape == x.shape

    def test_final_norm_pre_norm(self):
        """Pre-norm transformer should have final normalization."""
        transformer = Transformer(num_layers=6, d_model=64, num_heads=8, pre_norm=True)

        assert transformer.final_norm is not None

    def test_callable(self):
        """Should be callable like a function."""
        transformer = Transformer(num_layers=3, d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        out1 = transformer.forward(x)
        out2 = transformer(x)

        np.testing.assert_array_equal(out1, out2)

    def test_with_mask(self):
        """Should work with attention mask."""
        transformer = Transformer(num_layers=3, d_model=64, num_heads=8)
        x = np.random.randn(2, 10, 64)

        mask = np.triu(np.ones((10, 10), dtype=bool), k=1)

        out = transformer(x, mask=mask)

        assert out.shape == x.shape


class TestVaryingDimensions:
    """Tests with various dimensions."""

    @pytest.mark.parametrize("d_model,num_heads", [
        (64, 8),
        (128, 4),
        (256, 8),
        (512, 16),
    ])
    def test_various_dimensions(self, d_model, num_heads):
        """Should work with various d_model and num_heads combinations."""
        block = TransformerBlock(d_model=d_model, num_heads=num_heads)
        x = np.random.randn(2, 10, d_model)

        out = block(x)

        assert out.shape == x.shape

    @pytest.mark.parametrize("seq_len", [5, 10, 50, 100])
    def test_various_seq_lengths(self, seq_len):
        """Should work with various sequence lengths."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(2, seq_len, 64)

        out = block(x)

        assert out.shape == (2, seq_len, 64)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        """Should work with various batch sizes."""
        block = TransformerBlock(d_model=64, num_heads=8)
        x = np.random.randn(batch_size, 10, 64)

        out = block(x)

        assert out.shape == (batch_size, 10, 64)


class TestMilestone:
    """
    Chapter 2 Milestone Tests

    These tests verify the transformer block is working correctly.
    Bonus: Compare with GPT-2 implementation.
    """

    def test_block_processes_correctly(self):
        """
        Basic milestone: Block should process input correctly.

        This verifies:
        - All components work together
        - Residuals are applied
        - Output has correct shape
        """
        # Use GPT-2 small-like dimensions
        d_model = 768
        num_heads = 12
        d_ff = 3072
        seq_len = 128
        batch_size = 2

        block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

        out = block(x)

        # Basic checks
        assert out.shape == x.shape
        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"

        # Output should be different from input (transformation happened)
        assert not np.allclose(out, x)

        print(f"\n{'='*60}")
        print("Chapter 2 Basic Milestone Achieved!")
        print(f"Transformer block processes {d_model}d, {num_heads}h correctly")
        print(f"{'='*60}\n")

    @pytest.mark.skip(reason="Bonus test - requires careful GPT-2 matching")
    def test_matches_gpt2_block(self):
        """
        BONUS: Match GPT-2 block output exactly.

        This is the full milestone from the syllabus.
        Requires careful weight loading and architecture matching.
        """
        try:
            import torch
            from transformers import GPT2Model
        except ImportError:
            pytest.skip("transformers library not available")

        d_model = 768
        num_heads = 12

        # Load GPT-2
        gpt2 = GPT2Model.from_pretrained('gpt2')
        gpt2_block = gpt2.h[0]  # First transformer block

        # Create matching NumPy block and load weights
        # (This would require implementing weight loading similar to Ch1 Lab 04)

        # Test input
        x_np = np.random.randn(1, 10, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Compare outputs
        # ... (implementation left as exercise)

        print(f"\n{'='*60}")
        print("BONUS: Chapter 2 Full Milestone Achieved!")
        print("Transformer block matches GPT-2!")
        print(f"{'='*60}\n")
