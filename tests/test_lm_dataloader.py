"""Tests for fastai.text.data module.

Covers LMDataLoader backward language model support, ensuring that
text sequences are properly reversed when backwards=True.
"""
import pytest

import torch

from fastai.text.data import LMDataLoader, reverse_text, LMTensorText


# ============================================================
# Tests for reverse_text utility
# ============================================================

class TestReverseText:
    """Tests for the reverse_text function."""

    def test_reverse_text_flips_1d_tensor(self):
        x = torch.arange(1, 6)  # tensor([1, 2, 3, 4, 5])
        result = reverse_text(x)
        expected = torch.arange(5, 0, -1)  # tensor([5, 4, 3, 2, 1])
        assert torch.equal(result, expected)

    def test_reverse_text_single_element(self):
        x = torch.arange(42, 43)  # tensor([42])
        result = reverse_text(x)
        assert torch.equal(result, x)

    def test_reverse_text_preserves_length(self):
        x = torch.arange(10)
        result = reverse_text(x)
        assert result.shape == x.shape
        assert result[0] == x[-1]
        assert result[-1] == x[0]


# ============================================================
# Tests for LMDataLoader backward support
# ============================================================

class TestLMDataLoaderBackward:
    """Tests for LMDataLoader with backwards=True."""

    @pytest.fixture
    def sample_data(self):
        """Create a simple token sequence dataset for testing."""
        tokens = torch.arange(1, 65)  # 64 tokens: [1, 2, ..., 64]
        return [tokens]

    @pytest.fixture
    def sample_lens(self):
        return [64]

    def test_default_backwards_is_false(self, sample_data, sample_lens):
        """LMDataLoader defaults to forwards (backwards=False)."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=2, seq_len=4)
        assert dl.backwards is False

    def test_backwards_parameter_stored(self, sample_data, sample_lens):
        """LMDataLoader stores the backwards flag."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=2, seq_len=4, backwards=True)
        assert dl.backwards is True

    def test_forward_create_item(self, sample_data, sample_lens):
        """Forward LMDataLoader returns unreversed sequences."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=False)
        x, y = dl.create_item(0)
        # In forward mode, x and y should be sequential
        assert x[1] == x[0] + 1  # tokens increase by 1
        assert y[0] == x[1]  # target is shifted by 1

    def test_backward_create_item_reverses_sequence(self, sample_data, sample_lens):
        """Backward LMDataLoader reverses the text chunk before splitting."""
        dl_fwd = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=False)
        dl_bwd = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)

        x_fwd, y_fwd = dl_fwd.create_item(0)
        x_bwd, y_bwd = dl_bwd.create_item(0)

        # The backward version should be the reverse of the forward chunk
        # Forward: txt = [a, b, c, d, e] -> x=[a,b,c,d], y=[b,c,d,e]
        # Backward: txt_rev = [e, d, c, b, a] -> x=[e,d,c,b], y=[d,c,b,a]
        full_fwd = torch.cat([x_fwd, y_fwd[-1:]])
        full_bwd = torch.cat([x_bwd, y_bwd[-1:]])
        assert torch.equal(full_bwd, reverse_text(full_fwd))

    def test_backward_input_target_relationship(self, sample_data, sample_lens):
        """In backward mode, target[i] should be the token before input[i] in original order."""
        dl_bwd = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)
        x, y = dl_bwd.create_item(0)

        # In reversed text, each target is the next token in reverse direction
        # i.e., x[i+1] == y[i]
        for i in range(len(x) - 1):
            assert x[i + 1] == y[i], f"Mismatch at position {i}: x[{i+1}]={x[i+1]}, y[{i}]={y[i]}"

    def test_backward_tokens_decrease(self, sample_data, sample_lens):
        """In backward mode with sequential tokens, values should decrease."""
        dl_bwd = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)
        x, y = dl_bwd.create_item(0)
        # Since original tokens are increasing, reversed tokens should be decreasing
        for i in range(len(x) - 1):
            assert x[i] > x[i + 1], f"Tokens not decreasing at position {i}"

    def test_backward_create_item_returns_lm_tensor_text(self, sample_data, sample_lens):
        """Backward LMDataLoader still returns LMTensorText for inputs."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)
        x, y = dl.create_item(0)
        assert isinstance(x, LMTensorText)

    def test_new_propagates_backwards(self, sample_data, sample_lens):
        """The new() method should propagate the backwards setting."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)
        dl_new = dl.new(dataset=sample_data)
        assert dl_new.backwards is True

    def test_new_propagates_backwards_false(self, sample_data, sample_lens):
        """The new() method should propagate backwards=False as well."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=False)
        dl_new = dl.new(dataset=sample_data)
        assert dl_new.backwards is False

    def test_new_can_override_backwards(self, sample_data, sample_lens):
        """The new() method allows overriding the backwards parameter."""
        dl = LMDataLoader(sample_data, lens=sample_lens, bs=1, seq_len=4, backwards=True)
        dl_new = dl.new(dataset=sample_data, backwards=False)
        assert dl_new.backwards is False

    def test_forward_and_backward_same_length(self, sample_data, sample_lens):
        """Forward and backward data loaders produce items of the same length."""
        dl_fwd = LMDataLoader(sample_data, lens=sample_lens, bs=2, seq_len=4, backwards=False)
        dl_bwd = LMDataLoader(sample_data, lens=sample_lens, bs=2, seq_len=4, backwards=True)

        x_fwd, y_fwd = dl_fwd.create_item(0)
        x_bwd, y_bwd = dl_bwd.create_item(0)
        assert len(x_fwd) == len(x_bwd)
        assert len(y_fwd) == len(y_bwd)
