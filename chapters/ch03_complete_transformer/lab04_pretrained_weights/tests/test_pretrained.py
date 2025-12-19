"""Tests for Lab 04: Load Pretrained Weights."""

import numpy as np
import pytest
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pretrained import (
    GPTModel,
    get_gpt2_config,
    create_gpt2_model,
    load_gpt2_weights,
    compare_outputs,
    analyze_weight_differences,
)


class TestGPT2Config:
    """Tests for GPT-2 configuration."""

    def test_gpt2_config(self):
        """Should return correct GPT-2 base config."""
        config = get_gpt2_config('gpt2')

        assert config['vocab_size'] == 50257
        assert config['d_model'] == 768
        assert config['num_layers'] == 12
        assert config['num_heads'] == 12
        assert config['d_ff'] == 3072

    def test_gpt2_medium_config(self):
        """Should return correct GPT-2 medium config."""
        config = get_gpt2_config('gpt2-medium')

        assert config['d_model'] == 1024
        assert config['num_layers'] == 24
        assert config['num_heads'] == 16


class TestCreateModel:
    """Tests for creating GPT-2 model."""

    def test_create_gpt2_model(self):
        """Should create model with correct architecture."""
        model = create_gpt2_model('gpt2')

        assert model.vocab_size == 50257
        assert model.d_model == 768
        assert model.num_layers == 12
        assert len(model.blocks) == 12

    def test_model_has_required_components(self):
        """Model should have all required components."""
        model = create_gpt2_model('gpt2')

        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'pos_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'ln_f')
        assert hasattr(model, 'lm_head')


class TestLoadWeights:
    """Tests for weight loading."""

    @pytest.fixture(scope="class")
    def loaded_models(self):
        """Load models once for all tests in this class."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')
        return your_model, hf_model

    def test_token_embedding_loaded(self, loaded_models):
        """Token embedding should match HuggingFace."""
        your_model, hf_model = loaded_models

        hf_weight = hf_model.transformer.wte.weight.detach().numpy()
        your_weight = your_model.token_embedding.weight

        np.testing.assert_allclose(your_weight, hf_weight, rtol=1e-5)

    def test_position_embedding_loaded(self, loaded_models):
        """Position embedding should match HuggingFace."""
        your_model, hf_model = loaded_models

        hf_weight = hf_model.transformer.wpe.weight.detach().numpy()
        your_weight = your_model.pos_embedding.weight

        np.testing.assert_allclose(your_weight, hf_weight, rtol=1e-5)

    def test_layer_norm_loaded(self, loaded_models):
        """Layer norm weights should match HuggingFace."""
        your_model, hf_model = loaded_models

        # Check first block's first layer norm
        hf_gamma = hf_model.transformer.h[0].ln_1.weight.detach().numpy()
        hf_beta = hf_model.transformer.h[0].ln_1.bias.detach().numpy()

        np.testing.assert_allclose(your_model.blocks[0].ln1.gamma, hf_gamma, rtol=1e-5)
        np.testing.assert_allclose(your_model.blocks[0].ln1.beta, hf_beta, rtol=1e-5)

    def test_attention_weights_loaded(self, loaded_models):
        """Attention weights should match HuggingFace."""
        your_model, hf_model = loaded_models
        d_model = 768

        # Get HuggingFace combined QKV weight
        hf_c_attn = hf_model.transformer.h[0].attn.c_attn.weight.detach().numpy()

        # Split into Q, K, V
        hf_W_Q = hf_c_attn[:, :d_model]
        hf_W_K = hf_c_attn[:, d_model:2*d_model]
        hf_W_V = hf_c_attn[:, 2*d_model:]

        np.testing.assert_allclose(your_model.blocks[0].attn.W_Q, hf_W_Q, rtol=1e-5)
        np.testing.assert_allclose(your_model.blocks[0].attn.W_K, hf_W_K, rtol=1e-5)
        np.testing.assert_allclose(your_model.blocks[0].attn.W_V, hf_W_V, rtol=1e-5)

    def test_ffn_weights_loaded(self, loaded_models):
        """FFN weights should match HuggingFace."""
        your_model, hf_model = loaded_models

        hf_W1 = hf_model.transformer.h[0].mlp.c_fc.weight.detach().numpy()
        hf_b1 = hf_model.transformer.h[0].mlp.c_fc.bias.detach().numpy()

        np.testing.assert_allclose(your_model.blocks[0].ffn.W1, hf_W1, rtol=1e-5)
        np.testing.assert_allclose(your_model.blocks[0].ffn.b1, hf_b1, rtol=1e-5)

    def test_final_layer_norm_loaded(self, loaded_models):
        """Final layer norm should match HuggingFace."""
        your_model, hf_model = loaded_models

        hf_gamma = hf_model.transformer.ln_f.weight.detach().numpy()
        hf_beta = hf_model.transformer.ln_f.bias.detach().numpy()

        np.testing.assert_allclose(your_model.ln_f.gamma, hf_gamma, rtol=1e-5)
        np.testing.assert_allclose(your_model.ln_f.beta, hf_beta, rtol=1e-5)


class TestCompareOutputs:
    """Tests for output comparison."""

    @pytest.fixture(scope="class")
    def models_and_tokenizer(self):
        """Load models and tokenizer once."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')
        return your_model, hf_model, tokenizer

    def test_compare_simple_text(self, models_and_tokenizer):
        """Outputs should match for simple text."""
        your_model, hf_model, tokenizer = models_and_tokenizer

        match, max_diff = compare_outputs(
            your_model, hf_model, tokenizer,
            "Hello, world!"
        )

        assert match, f"Outputs don't match! Max diff: {max_diff:.2e}"
        assert max_diff < 1e-4

    def test_compare_longer_text(self, models_and_tokenizer):
        """Outputs should match for longer text."""
        your_model, hf_model, tokenizer = models_and_tokenizer

        match, max_diff = compare_outputs(
            your_model, hf_model, tokenizer,
            "The quick brown fox jumps over the lazy dog."
        )

        assert match, f"Outputs don't match! Max diff: {max_diff:.2e}"

    def test_compare_multiple_sentences(self, models_and_tokenizer):
        """Outputs should match for various inputs."""
        your_model, hf_model, tokenizer = models_and_tokenizer

        test_texts = [
            "Hello",
            "The cat sat on the mat.",
            "In a hole in the ground there lived a hobbit.",
        ]

        for text in test_texts:
            match, max_diff = compare_outputs(
                your_model, hf_model, tokenizer, text
            )
            assert match, f"Failed for '{text}'. Max diff: {max_diff:.2e}"


class TestAnalyzeWeights:
    """Tests for weight analysis."""

    @pytest.fixture(scope="class")
    def loaded_models(self):
        """Load models once."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')
        return your_model, hf_model

    def test_analyze_returns_dict(self, loaded_models):
        """Should return dictionary with weight analysis."""
        your_model, hf_model = loaded_models

        analysis = analyze_weight_differences(your_model, hf_model)

        assert isinstance(analysis, dict)
        assert 'token_embedding' in analysis or len(analysis) > 0

    def test_weights_match_after_loading(self, loaded_models):
        """All weights should have zero or near-zero difference."""
        your_model, hf_model = loaded_models

        analysis = analyze_weight_differences(your_model, hf_model)

        for key, stats in analysis.items():
            assert stats['max_diff'] < 1e-5, f"Weight {key} doesn't match!"


class TestChapter3Milestone:
    """
    Chapter 3 Milestone Test

    Your implementation should generate the same logits as
    HuggingFace GPT-2 for the same input.
    """

    @pytest.fixture(scope="class")
    def models_and_tokenizer(self):
        """Load models and tokenizer once."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')
        return your_model, hf_model, tokenizer

    def test_chapter_3_milestone(self, models_and_tokenizer):
        """
        MILESTONE: Output logits match HuggingFace GPT-2.

        This is the final test for Chapter 3.
        Passing this means you've correctly implemented a complete transformer!
        """
        your_model, hf_model, tokenizer = models_and_tokenizer

        test_texts = [
            "Hello, my name is",
            "The meaning of life is",
            "Once upon a time",
        ]

        all_match = True
        max_diffs = []

        for text in test_texts:
            match, max_diff = compare_outputs(
                your_model, hf_model, tokenizer, text,
                atol=1e-4
            )
            max_diffs.append(max_diff)
            if not match:
                all_match = False

        overall_max_diff = max(max_diffs)

        assert all_match, (
            f"MILESTONE NOT MET: Max difference is {overall_max_diff:.2e}, "
            f"expected < 1e-4"
        )

        print(f"\n{'='*60}")
        print("CONGRATULATIONS! Chapter 3 Milestone Achieved!")
        print(f"Max difference: {overall_max_diff:.2e} (< 1e-4)")
        print("Your transformer matches HuggingFace GPT-2!")
        print(f"{'='*60}\n")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_token(self):
        """Should work with single token input."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')

        match, max_diff = compare_outputs(
            your_model, hf_model, tokenizer, "Hi"
        )

        assert match, f"Failed for single token. Max diff: {max_diff:.2e}"

    def test_special_characters(self):
        """Should handle special characters."""
        your_model = create_gpt2_model('gpt2')
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        load_gpt2_weights(your_model, 'gpt2')

        match, max_diff = compare_outputs(
            your_model, hf_model, tokenizer,
            "Hello! How are you? I'm fine, thanks."
        )

        assert match, f"Failed for special characters. Max diff: {max_diff:.2e}"
