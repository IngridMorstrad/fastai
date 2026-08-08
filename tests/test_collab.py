"""Tests for fastai.collab module.

Covers EmbeddingDotBias: __init__, forward, from_classes, bias, weight, _get_idx.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.collab import EmbeddingDotBias


# ============================================================
# Tests for EmbeddingDotBias.__init__
# ============================================================

class TestEmbeddingDotBiasInit:
    """Tests for EmbeddingDotBias.__init__ - verify embeddings are created with correct dimensions."""

    def test_creates_user_weight_embedding(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        assert model.u_weight.weight.shape == (10, 5)

    def test_creates_item_weight_embedding(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        assert model.i_weight.weight.shape == (8, 5)

    def test_creates_user_bias_embedding(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        assert model.u_bias.weight.shape == (10, 1)

    def test_creates_item_bias_embedding(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        assert model.i_bias.weight.shape == (8, 1)

    def test_y_range_stored_when_provided(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8, y_range=(1.0, 5.0))
        assert model.y_range == (1.0, 5.0)

    def test_y_range_none_by_default(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        assert model.y_range is None


# ============================================================
# Tests for EmbeddingDotBias.forward
# ============================================================

class TestEmbeddingDotBiasForward:
    """Tests for EmbeddingDotBias.forward - verify output shape with and without y_range."""

    def test_forward_output_shape_single_sample(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        x = torch.tensor([[0, 0]])
        out = model(x)
        assert out.shape == (1,)

    def test_forward_output_shape_batch(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        x = torch.tensor([[0, 1], [2, 3], [4, 5]])
        out = model(x)
        assert out.shape == (3,)

    def test_forward_without_y_range_returns_unbounded(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8)
        x = torch.tensor([[0, 0], [1, 1]])
        out = model(x)
        # Without y_range, output is unbounded (can be any float)
        assert out.dtype == torch.float32 or out.dtype == torch.float64

    def test_forward_with_y_range_bounds_output(self):
        y_range = (1.0, 5.0)
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8, y_range=y_range)
        x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        out = model(x)
        # Output should be within y_range (sigmoid maps to (0,1) so strictly inside range)
        assert (out > y_range[0]).all()
        assert (out < y_range[1]).all()

    def test_forward_with_y_range_output_shape(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=8, y_range=(0.0, 10.0))
        x = torch.tensor([[0, 0], [1, 2], [3, 4]])
        out = model(x)
        assert out.shape == (3,)


# ============================================================
# Tests for EmbeddingDotBias.from_classes
# ============================================================

class TestEmbeddingDotBiasFromClasses:
    """Tests for EmbeddingDotBias.from_classes - verify classmethod infers dimensions."""

    def test_infers_n_users_and_n_items(self):
        classes = {'user': ['Alice', 'Bob', 'Carol'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        assert model.u_weight.weight.shape == (3, 4)
        assert model.i_weight.weight.shape == (2, 4)

    def test_uses_explicit_user_and_item_keys(self):
        classes = {'viewers': ['A', 'B'], 'movies': ['M1', 'M2', 'M3', 'M4']}
        model = EmbeddingDotBias.from_classes(
            n_factors=3, classes=classes, user='viewers', item='movies'
        )
        assert model.u_weight.weight.shape == (2, 3)
        assert model.i_weight.weight.shape == (4, 3)

    def test_defaults_to_first_two_keys(self):
        classes = {'users': ['U1', 'U2', 'U3', 'U4'], 'items': ['I1', 'I2', 'I3']}
        model = EmbeddingDotBias.from_classes(n_factors=6, classes=classes)
        assert model.u_weight.weight.shape == (4, 6)
        assert model.i_weight.weight.shape == (3, 6)

    def test_stores_classes_attribute(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        assert model.classes is classes

    def test_stores_user_and_item_keys(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        assert model.user == 'user'
        assert model.item == 'item'

    def test_y_range_passed_through(self):
        classes = {'user': ['A'], 'item': ['B']}
        model = EmbeddingDotBias.from_classes(
            n_factors=2, classes=classes, y_range=(0.5, 4.5)
        )
        assert model.y_range == (0.5, 4.5)


# ============================================================
# Tests for EmbeddingDotBias.bias and EmbeddingDotBias.weight
# ============================================================

class TestEmbeddingDotBiasBiasAndWeight:
    """Tests for bias() and weight() - verify they return correct shapes."""

    @pytest.fixture
    def model(self):
        classes = {'user': ['Alice', 'Bob', 'Carol'], 'item': ['X', 'Y', 'Z', 'W']}
        return EmbeddingDotBias.from_classes(n_factors=5, classes=classes)

    def test_bias_item_shape(self, model):
        result = model.bias(['X', 'Y'], is_item=True)
        assert result.shape == (2,)

    def test_bias_user_shape(self, model):
        result = model.bias(['Alice', 'Bob', 'Carol'], is_item=False)
        assert result.shape == (3,)

    def test_weight_item_shape(self, model):
        result = model.weight(['X', 'Y', 'Z'], is_item=True)
        assert result.shape == (3, 5)

    def test_weight_user_shape(self, model):
        result = model.weight(['Alice'], is_item=False)
        assert result.shape == (1, 5)

    def test_bias_single_item(self, model):
        result = model.bias(['W'], is_item=True)
        # squeeze() on a (1,) tensor produces a scalar (0-d tensor)
        assert result.shape == torch.Size([])

    def test_weight_single_user(self, model):
        result = model.weight(['Bob'], is_item=False)
        assert result.shape == (1, 5)


# ============================================================
# Tests for EmbeddingDotBias._get_idx
# ============================================================

class TestEmbeddingDotBiasGetIdx:
    """Tests for _get_idx - verify error handling for unknown items."""

    def test_raises_error_for_unknown_item(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        with pytest.raises(Exception):
            model._get_idx(['UNKNOWN'], is_item=True)

    def test_raises_error_for_unknown_user(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        with pytest.raises(Exception):
            model._get_idx(['UNKNOWN'], is_item=False)

    def test_returns_correct_indices_for_known_items(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        idx = model._get_idx(['Y', 'X', 'Z'], is_item=True)
        assert idx.tolist() == [1, 0, 2]

    def test_returns_correct_indices_for_known_users(self):
        classes = {'user': ['Alice', 'Bob', 'Carol'], 'item': ['X']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        idx = model._get_idx(['Carol', 'Alice'], is_item=False)
        assert idx.tolist() == [2, 0]

    def test_raises_assertion_without_from_classes(self):
        model = EmbeddingDotBias(n_factors=3, n_users=5, n_items=5)
        with pytest.raises(AssertionError):
            model._get_idx(['something'], is_item=True)
