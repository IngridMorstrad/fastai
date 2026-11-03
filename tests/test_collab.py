"""Tests for fastai.collab module.

Covers the EmbeddingDotBias model: creation, forward pass, y_range clamping,
from_classes factory method, and bias/weight retrieval.
"""
import pytest
import torch

from fastai.collab import EmbeddingDotBias


# ============================================================
# Tests for EmbeddingDotBias creation
# ============================================================

class TestEmbeddingDotBiasCreation:
    """Tests for EmbeddingDotBias initialization."""

    def test_basic_creation(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        assert model is not None

    def test_embedding_dimensions(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        assert model.u_weight.weight.shape == (50, 10)
        assert model.i_weight.weight.shape == (100, 10)

    def test_bias_dimensions(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        assert model.u_bias.weight.shape == (50, 1)
        assert model.i_bias.weight.shape == (100, 1)

    def test_y_range_stored(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100, y_range=(0.5, 5.5))
        assert model.y_range == (0.5, 5.5)

    def test_y_range_none_by_default(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        assert model.y_range is None

    def test_single_factor(self):
        model = EmbeddingDotBias(n_factors=1, n_users=5, n_items=3)
        assert model.u_weight.weight.shape == (5, 1)
        assert model.i_weight.weight.shape == (3, 1)

    def test_large_factors(self):
        model = EmbeddingDotBias(n_factors=200, n_users=1000, n_items=5000)
        assert model.u_weight.weight.shape == (1000, 200)
        assert model.i_weight.weight.shape == (5000, 200)


# ============================================================
# Tests for EmbeddingDotBias forward pass
# ============================================================

class TestEmbeddingDotBiasForward:
    """Tests for EmbeddingDotBias forward method."""

    def test_forward_shape_single_sample(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        x = torch.tensor([[0, 0]])
        result = model(x)
        assert result.shape == (1,)

    def test_forward_shape_batch(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        x = torch.tensor([[0, 0], [1, 2], [3, 5], [10, 20]])
        result = model(x)
        assert result.shape == (4,)

    def test_forward_returns_float(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        x = torch.tensor([[0, 0]])
        result = model(x)
        assert result.dtype == torch.float32

    def test_forward_differentiable(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        x = torch.tensor([[0, 0]])
        result = model(x)
        result.backward()
        # Gradients should exist after backward
        assert model.u_weight.weight.grad is not None

    def test_forward_valid_user_item_indices(self):
        model = EmbeddingDotBias(n_factors=10, n_users=5, n_items=3)
        # All valid combos of user (0-4) and item (0-2)
        x = torch.tensor([[0, 0], [4, 2], [2, 1]])
        result = model(x)
        assert result.shape == (3,)
        assert not torch.any(torch.isnan(result))

    def test_forward_computation_correctness(self):
        """Manually verify the forward computation: dot product + biases."""
        model = EmbeddingDotBias(n_factors=2, n_users=3, n_items=3)
        # Set known values for embeddings
        with torch.no_grad():
            model.u_weight.weight.data = torch.tensor([
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ])
            model.i_weight.weight.data = torch.tensor([
                [0.5, 0.5],
                [1.0, 1.0],
                [2.0, 2.0],
            ])
            model.u_bias.weight.data = torch.tensor([[0.1], [0.2], [0.3]])
            model.i_bias.weight.data = torch.tensor([[0.01], [0.02], [0.03]])

        # user=0, item=0: dot([1,2],[0.5,0.5]) + 0.1 + 0.01 = 1.5 + 0.11 = 1.61
        x = torch.tensor([[0, 0]])
        result = model(x)
        assert abs(result.item() - 1.61) < 1e-5

        # user=1, item=1: dot([3,4],[1,1]) + 0.2 + 0.02 = 7 + 0.22 = 7.22
        x = torch.tensor([[1, 1]])
        result = model(x)
        assert abs(result.item() - 7.22) < 1e-5

        # user=2, item=2: dot([5,6],[2,2]) + 0.3 + 0.03 = 22 + 0.33 = 22.33
        x = torch.tensor([[2, 2]])
        result = model(x)
        assert abs(result.item() - 22.33) < 1e-5


# ============================================================
# Tests for EmbeddingDotBias y_range
# ============================================================

class TestEmbeddingDotBiasYRange:
    """Tests for EmbeddingDotBias output clamping via y_range."""

    def test_y_range_clamps_output(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100, y_range=(1.0, 5.0))
        x = torch.randint(0, 50, (100, 1))
        items = torch.randint(0, 100, (100, 1))
        x = torch.cat([x, items], dim=1)
        result = model(x)
        # sigmoid maps to (0,1), scaled to (1,5) -> all values in (1, 5)
        assert torch.all(result > 1.0)
        assert torch.all(result < 5.0)

    def test_no_y_range_allows_unbounded(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        # Set large weights to get large outputs
        with torch.no_grad():
            model.u_weight.weight.data.fill_(10.0)
            model.i_weight.weight.data.fill_(10.0)
            model.u_bias.weight.data.fill_(100.0)
            model.i_bias.weight.data.fill_(100.0)
        x = torch.tensor([[0, 0]])
        result = model(x)
        # Without y_range, output can exceed any fixed range
        assert result.item() > 5.0

    def test_y_range_with_sigmoid_midpoint(self):
        """When dot product + bias = 0, sigmoid is 0.5, output is midpoint of range."""
        model = EmbeddingDotBias(n_factors=2, n_users=3, n_items=3, y_range=(0.0, 10.0))
        with torch.no_grad():
            model.u_weight.weight.data.zero_()
            model.i_weight.weight.data.zero_()
            model.u_bias.weight.data.zero_()
            model.i_bias.weight.data.zero_()
        x = torch.tensor([[0, 0]])
        result = model(x)
        # sigmoid(0) = 0.5, so result = 0.5 * (10-0) + 0 = 5.0
        assert abs(result.item() - 5.0) < 1e-5

    def test_y_range_negative_range(self):
        model = EmbeddingDotBias(n_factors=10, n_users=20, n_items=30, y_range=(-3.0, 3.0))
        x = torch.randint(0, 20, (50, 1))
        items = torch.randint(0, 30, (50, 1))
        x = torch.cat([x, items], dim=1)
        result = model(x)
        assert torch.all(result > -3.0)
        assert torch.all(result < 3.0)


# ============================================================
# Tests for EmbeddingDotBias.from_classes
# ============================================================

class TestEmbeddingDotBiasFromClasses:
    """Tests for the from_classes class method."""

    def test_from_classes_basic(self):
        classes = {'user': ['Alice', 'Bob', 'Charlie'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        assert model.u_weight.weight.shape == (3, 5)
        assert model.i_weight.weight.shape == (2, 5)

    def test_from_classes_stores_classes(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2', 'Movie3']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        assert model.classes == classes

    def test_from_classes_stores_user_item_keys(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        assert model.user == 'user'
        assert model.item == 'item'

    def test_from_classes_custom_keys(self):
        classes = {'viewer': ['Alice', 'Bob'], 'movie': ['M1', 'M2', 'M3']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes, user='viewer', item='movie')
        assert model.user == 'viewer'
        assert model.item == 'movie'
        assert model.u_weight.weight.shape == (2, 5)
        assert model.i_weight.weight.shape == (3, 5)

    def test_from_classes_with_y_range(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes, y_range=(1.0, 5.0))
        assert model.y_range == (1.0, 5.0)

    def test_from_classes_infers_first_key_as_user(self):
        """When user/item not specified, uses first and second keys."""
        classes = {'customers': ['C1', 'C2', 'C3'], 'products': ['P1', 'P2']}
        model = EmbeddingDotBias.from_classes(n_factors=8, classes=classes)
        assert model.user == 'customers'
        assert model.item == 'products'
        assert model.u_weight.weight.shape == (3, 8)
        assert model.i_weight.weight.shape == (2, 8)

    def test_from_classes_forward_works(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        x = torch.tensor([[0, 0], [1, 1], [2, 0]])
        result = model(x)
        assert result.shape == (3,)


# ============================================================
# Tests for EmbeddingDotBias.bias method
# ============================================================

class TestEmbeddingDotBiasBiasMethod:
    """Tests for the bias retrieval method."""

    def test_item_bias(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2', 'Movie3']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.bias(['Movie1', 'Movie2'], is_item=True)
        assert result.shape == (2,)

    def test_user_bias(self):
        classes = {'user': ['Alice', 'Bob', 'Charlie'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.bias(['Alice', 'Charlie'], is_item=False)
        assert result.shape == (2,)

    def test_bias_single_item(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.bias(['Movie1'], is_item=True)
        # Single item squeeze results in a scalar tensor
        assert result.shape == torch.Size([])

    def test_bias_returns_detached_tensor(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.bias(['Movie1'], is_item=True)
        assert not result.requires_grad

    def test_bias_without_from_classes_raises(self):
        """Model built without from_classes should raise on bias call."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.bias(['something'], is_item=True)

    def test_bias_invalid_item_raises(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        with pytest.raises(Exception):
            model.bias(['NonExistentMovie'], is_item=True)


# ============================================================
# Tests for EmbeddingDotBias.weight method
# ============================================================

class TestEmbeddingDotBiasWeightMethod:
    """Tests for the weight retrieval method."""

    def test_item_weight(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2', 'Movie3']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.weight(['Movie1', 'Movie2'], is_item=True)
        assert result.shape == (2, 5)

    def test_user_weight(self):
        classes = {'user': ['Alice', 'Bob', 'Charlie'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.weight(['Alice', 'Charlie'], is_item=False)
        assert result.shape == (2, 5)

    def test_weight_single_item(self):
        classes = {'user': ['Alice'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=8, classes=classes)
        result = model.weight(['Movie2'], is_item=True)
        assert result.shape == (1, 8)

    def test_weight_returns_detached_tensor(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        result = model.weight(['Movie1'], is_item=True)
        assert not result.requires_grad

    def test_weight_without_from_classes_raises(self):
        """Model built without from_classes should raise on weight call."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.weight(['something'], is_item=True)

    def test_weight_invalid_user_raises(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        with pytest.raises(Exception):
            model.weight(['NonExistentUser'], is_item=False)

    def test_weight_all_items(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['Movie1', 'Movie2', 'Movie3']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        result = model.weight(['Movie1', 'Movie2', 'Movie3'], is_item=True)
        assert result.shape == (3, 4)


# ============================================================
# Tests for EmbeddingDotBias as nn.Module
# ============================================================

class TestEmbeddingDotBiasModule:
    """Tests for EmbeddingDotBias behavior as a proper nn.Module."""

    def test_parameters_count(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        params = list(model.parameters())
        # 4 embedding layers: u_weight, i_weight, u_bias, i_bias
        assert len(params) == 4

    def test_total_parameter_size(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        total = sum(p.numel() for p in model.parameters())
        # u_weight: 50*10=500, i_weight: 100*10=1000, u_bias: 50*1=50, i_bias: 100*1=100
        assert total == 500 + 1000 + 50 + 100

    def test_eval_mode(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        model.eval()
        x = torch.tensor([[0, 0], [1, 1]])
        with torch.no_grad():
            result = model(x)
        assert result.shape == (2,)

    def test_train_mode(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        model.train()
        assert model.training is True

    def test_to_device(self):
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=100)
        model = model.cpu()
        x = torch.tensor([[0, 0]])
        result = model(x)
        assert result.device == torch.device('cpu')
