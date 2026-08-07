"""Tests for fastai.collab module.

Covers: EmbeddingDotBias (init, forward, from_classes, bias, weight methods),
including y_range sigmoid scaling behavior.
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
    """Tests for EmbeddingDotBias initialization."""

    def test_creates_correct_embedding_dimensions(self):
        n_factors, n_users, n_items = 5, 10, 20
        model = EmbeddingDotBias(n_factors, n_users, n_items)
        assert model.u_weight.weight.shape == (n_users, n_factors)
        assert model.i_weight.weight.shape == (n_items, n_factors)
        assert model.u_bias.weight.shape == (n_users, 1)
        assert model.i_bias.weight.shape == (n_items, 1)

    def test_y_range_none_by_default(self):
        model = EmbeddingDotBias(5, 10, 20)
        assert model.y_range is None

    def test_y_range_stored(self):
        model = EmbeddingDotBias(5, 10, 20, y_range=(1.0, 5.0))
        assert model.y_range == (1.0, 5.0)

    def test_single_factor(self):
        model = EmbeddingDotBias(1, 3, 4)
        assert model.u_weight.weight.shape == (3, 1)
        assert model.i_weight.weight.shape == (4, 1)

    def test_model_has_parameters(self):
        model = EmbeddingDotBias(5, 10, 20)
        params = list(model.parameters())
        assert len(params) == 4  # u_weight, i_weight, u_bias, i_bias


# ============================================================
# Tests for EmbeddingDotBias.forward
# ============================================================

class TestEmbeddingDotBiasForward:
    """Tests for the forward pass of EmbeddingDotBias."""

    def test_output_shape_single_sample(self):
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 0]])
        out = model(x)
        assert out.shape == (1,)

    def test_output_shape_batch(self):
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 0], [1, 1], [2, 2]])
        out = model(x)
        assert out.shape == (3,)

    def test_forward_produces_finite_values(self):
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 0], [3, 5], [9, 19]])
        out = model(x)
        assert torch.isfinite(out).all()

    def test_forward_without_y_range(self):
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 0], [1, 1]])
        out = model(x)
        # Without y_range, output is unbounded (dot product + biases)
        assert out.shape == (2,)

    def test_forward_with_y_range_bounds_output(self):
        model = EmbeddingDotBias(5, 10, 20, y_range=(1.0, 5.0))
        x = torch.tensor([[0, 0], [1, 1], [2, 3], [5, 10]])
        out = model(x)
        # sigmoid maps to (0,1), scaled to (y_range[0], y_range[1])
        assert (out >= 1.0).all()
        assert (out <= 5.0).all()

    def test_forward_with_y_range_extreme_inputs(self):
        """Large weights should still be clamped by sigmoid + y_range."""
        model = EmbeddingDotBias(5, 10, 20, y_range=(0.0, 10.0))
        # Set large weights to force extreme dot products
        with torch.no_grad():
            model.u_weight.weight.fill_(100.0)
            model.i_weight.weight.fill_(100.0)
            model.u_bias.weight.fill_(100.0)
            model.i_bias.weight.fill_(100.0)
        x = torch.tensor([[0, 0]])
        out = model(x)
        # Should be close to upper bound (sigmoid(large) -> 1)
        assert out.item() <= 10.0
        assert out.item() >= 0.0

    def test_different_users_same_item_different_output(self):
        torch.manual_seed(42)
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 5], [1, 5]])
        out = model(x)
        # Different users should generally produce different scores
        # (unless embeddings happen to be identical, very unlikely with random init)
        assert out[0].item() != out[1].item()

    def test_forward_deterministic(self):
        model = EmbeddingDotBias(5, 10, 20)
        model.eval()
        x = torch.tensor([[2, 3]])
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_forward_manual_computation(self):
        """Verify forward matches manual dot product + bias computation."""
        model = EmbeddingDotBias(2, 3, 4)
        with torch.no_grad():
            model.u_weight.weight.fill_(1.0)
            model.i_weight.weight.fill_(2.0)
            model.u_bias.weight.fill_(0.5)
            model.i_bias.weight.fill_(0.25)
        x = torch.tensor([[0, 0]])
        out = model(x)
        # dot = u_weight[0] * i_weight[0] = [1,1] * [2,2] = [2,2], sum = 4
        # u_bias[0] = 0.5, i_bias[0] = 0.25
        # result = 4 + 0.5 + 0.25 = 4.75
        assert abs(out.item() - 4.75) < 1e-5


# ============================================================
# Tests for EmbeddingDotBias.from_classes
# ============================================================

class TestEmbeddingDotBiasFromClasses:
    """Tests for building EmbeddingDotBias from class dictionaries."""

    def test_basic_from_classes(self):
        classes = {'user': ['Alice', 'Bob', 'Charlie'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        assert model.u_weight.weight.shape == (3, 5)
        assert model.i_weight.weight.shape == (2, 5)

    def test_from_classes_stores_metadata(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(5, classes)
        assert model.classes == classes
        assert model.user == 'user'
        assert model.item == 'item'

    def test_from_classes_custom_user_item_keys(self):
        classes = {'viewer': ['A', 'B'], 'movie': ['M1', 'M2', 'M3']}
        model = EmbeddingDotBias.from_classes(5, classes, user='viewer', item='movie')
        assert model.user == 'viewer'
        assert model.item == 'movie'
        assert model.u_weight.weight.shape == (2, 5)
        assert model.i_weight.weight.shape == (3, 5)

    def test_from_classes_with_y_range(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes, y_range=(1.0, 5.0))
        assert model.y_range == (1.0, 5.0)

    def test_from_classes_defaults_to_first_two_keys(self):
        classes = {'customer': ['C1', 'C2'], 'product': ['P1', 'P2', 'P3']}
        model = EmbeddingDotBias.from_classes(5, classes)
        assert model.user == 'customer'
        assert model.item == 'product'

    def test_from_classes_model_is_functional(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        x = torch.tensor([[0, 0], [1, 1], [2, 0]])
        out = model(x)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()


# ============================================================
# Tests for EmbeddingDotBias.bias
# ============================================================

class TestEmbeddingDotBiasBias:
    """Tests for the bias retrieval method."""

    def test_item_bias_shape(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(5, classes)
        biases = model.bias(['X', 'Y'], is_item=True)
        assert biases.shape == (2,)

    def test_user_bias_shape(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(5, classes)
        biases = model.bias(['A', 'B'], is_item=False)
        assert biases.shape == (2,)

    def test_bias_returns_detached_tensor(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        biases = model.bias(['X'], is_item=True)
        assert not biases.requires_grad

    def test_bias_single_item(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        biases = model.bias(['X'], is_item=True)
        # Single item bias is squeezed to a scalar
        assert biases.ndim == 0

    def test_bias_invalid_class_raises_error(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        with pytest.raises(KeyError):
            model.bias(['NONEXISTENT'], is_item=True)

    def test_bias_requires_from_classes(self):
        model = EmbeddingDotBias(5, 10, 20)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.bias(['X'], is_item=True)


# ============================================================
# Tests for EmbeddingDotBias.weight
# ============================================================

class TestEmbeddingDotBiasWeight:
    """Tests for the weight retrieval method."""

    def test_item_weight_shape(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(5, classes)
        weights = model.weight(['X', 'Y'], is_item=True)
        assert weights.shape == (2, 5)

    def test_user_weight_shape(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(5, classes)
        weights = model.weight(['A', 'B'], is_item=False)
        assert weights.shape == (2, 5)

    def test_weight_returns_detached_tensor(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        weights = model.weight(['X'], is_item=True)
        assert not weights.requires_grad

    def test_weight_single_item(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        weights = model.weight(['Y'], is_item=True)
        assert weights.shape == (1, 5)

    def test_weight_invalid_class_raises_error(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        with pytest.raises(KeyError):
            model.weight(['NONEXISTENT'], is_item=True)

    def test_weight_requires_from_classes(self):
        model = EmbeddingDotBias(5, 10, 20)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.weight(['X'], is_item=True)

    def test_weight_consistent_with_embedding(self):
        """Weight retrieved for an item should match the embedding layer."""
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(5, classes)
        model.eval()
        weights = model.weight(['X'], is_item=True)
        # Directly access the embedding layer for item index 0
        direct = model.i_weight.weight[0].detach()
        assert torch.allclose(weights.squeeze(), direct)


# ============================================================
# Tests for EmbeddingDotBias gradient flow
# ============================================================

class TestEmbeddingDotBiasGradient:
    """Tests for gradient computation through the model."""

    def test_gradients_flow(self):
        model = EmbeddingDotBias(5, 10, 20)
        x = torch.tensor([[0, 0], [1, 1]])
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for all embedding parameters
        assert model.u_weight.weight.grad is not None
        assert model.i_weight.weight.grad is not None
        assert model.u_bias.weight.grad is not None
        assert model.i_bias.weight.grad is not None

    def test_gradients_flow_with_y_range(self):
        model = EmbeddingDotBias(5, 10, 20, y_range=(1.0, 5.0))
        x = torch.tensor([[0, 0], [1, 1]])
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.u_weight.weight.grad is not None
        assert model.i_weight.weight.grad is not None
