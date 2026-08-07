"""Tests for fastai.collab module.

Covers the EmbeddingDotBias collaborative filtering model:
- Constructor and embedding dimensions
- Forward pass computation (dot product + biases)
- y_range sigmoid clamping behavior
- from_classes class method
- bias() and weight() extraction methods
- Error handling for missing classes
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.collab import EmbeddingDotBias


# ============================================================
# Tests for EmbeddingDotBias construction
# ============================================================

class TestEmbeddingDotBiasConstruction:
    """Tests for EmbeddingDotBias initialization."""

    def test_basic_construction(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=20)
        assert model.y_range is None

    def test_embedding_dimensions(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=20)
        # User weight embedding: (n_users, n_factors)
        assert model.u_weight.weight.shape == (10, 5)
        # Item weight embedding: (n_items, n_factors)
        assert model.i_weight.weight.shape == (20, 5)
        # User bias embedding: (n_users, 1)
        assert model.u_bias.weight.shape == (10, 1)
        # Item bias embedding: (n_items, 1)
        assert model.i_bias.weight.shape == (20, 1)

    def test_y_range_stored(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=20, y_range=(1.0, 5.0))
        assert model.y_range == (1.0, 5.0)

    def test_single_factor(self):
        model = EmbeddingDotBias(n_factors=1, n_users=3, n_items=4)
        assert model.u_weight.weight.shape == (3, 1)
        assert model.i_weight.weight.shape == (4, 1)


# ============================================================
# Tests for EmbeddingDotBias forward pass
# ============================================================

class TestEmbeddingDotBiasForward:
    """Tests for EmbeddingDotBias forward pass."""

    def test_forward_output_shape(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=20)
        # Input: batch of (user_idx, item_idx) pairs
        x = torch.tensor([[0, 0], [1, 2], [3, 5]])
        out = model(x)
        assert out.shape == (3,)

    def test_forward_single_sample(self):
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=20)
        x = torch.tensor([[2, 3]])
        out = model(x)
        assert out.shape == (1,)
        assert out.dtype == torch.float32

    def test_forward_computation_correctness(self):
        """Verify forward pass computes dot product + biases correctly."""
        torch.manual_seed(42)
        model = EmbeddingDotBias(n_factors=3, n_users=5, n_items=5)

        # Get the raw embeddings for user 0 and item 1
        user_idx = torch.tensor([0])
        item_idx = torch.tensor([1])

        u_w = model.u_weight(user_idx)  # shape (1, 3)
        i_w = model.i_weight(item_idx)  # shape (1, 3)
        u_b = model.u_bias(user_idx)    # shape (1, 1)
        i_b = model.i_bias(item_idx)    # shape (1, 1)

        expected = (u_w * i_w).sum(1) + u_b.squeeze() + i_b.squeeze()

        x = torch.tensor([[0, 1]])
        actual = model(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_forward_no_y_range(self):
        """Without y_range, output is unbounded."""
        torch.manual_seed(0)
        model = EmbeddingDotBias(n_factors=10, n_users=5, n_items=5)
        x = torch.tensor([[0, 0], [1, 1], [2, 2]])
        out = model(x)
        # Output should be a raw scalar (no clamping), can be any value
        assert out.shape == (3,)

    def test_forward_with_y_range(self):
        """With y_range, output should be clamped between y_range[0] and y_range[1]."""
        model = EmbeddingDotBias(n_factors=10, n_users=50, n_items=50, y_range=(1.0, 5.0))
        # Use multiple samples to ensure range is respected
        x = torch.tensor([[i, j] for i in range(10) for j in range(10)])
        out = model(x)
        assert torch.all(out >= 1.0)
        assert torch.all(out <= 5.0)

    def test_forward_y_range_sigmoid_mapping(self):
        """y_range should map sigmoid output to [low, high] range."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10, y_range=(2.0, 8.0))
        x = torch.tensor([[0, 0]])
        out = model(x)
        # Output must be in [2.0, 8.0]
        assert out.item() >= 2.0
        assert out.item() <= 8.0

    def test_forward_batch_independence(self):
        """Each sample in a batch should be computed independently."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        model.eval()

        x_single_0 = torch.tensor([[2, 3]])
        x_single_1 = torch.tensor([[4, 5]])
        x_batch = torch.tensor([[2, 3], [4, 5]])

        out_single_0 = model(x_single_0)
        out_single_1 = model(x_single_1)
        out_batch = model(x_batch)

        assert torch.allclose(out_batch[0], out_single_0[0], atol=1e-6)
        assert torch.allclose(out_batch[1], out_single_1[0], atol=1e-6)


# ============================================================
# Tests for EmbeddingDotBias.from_classes
# ============================================================

class TestEmbeddingDotBiasFromClasses:
    """Tests for the from_classes class method."""

    def test_from_classes_basic(self):
        classes = {'user': ['Alice', 'Bob', 'Charlie'], 'item': ['Movie1', 'Movie2']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        # n_users=3, n_items=2
        assert model.u_weight.weight.shape == (3, 5)
        assert model.i_weight.weight.shape == (2, 5)

    def test_from_classes_stores_metadata(self):
        classes = {'user': ['Alice', 'Bob'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        assert model.classes == classes
        assert model.user == 'user'
        assert model.item == 'item'

    def test_from_classes_explicit_user_item(self):
        classes = {'viewers': ['A', 'B'], 'movies': ['M1', 'M2', 'M3']}
        model = EmbeddingDotBias.from_classes(
            n_factors=3, classes=classes, user='viewers', item='movies'
        )
        assert model.user == 'viewers'
        assert model.item == 'movies'
        assert model.u_weight.weight.shape == (2, 3)
        assert model.i_weight.weight.shape == (3, 3)

    def test_from_classes_with_y_range(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes, y_range=(0.5, 5.5))
        assert model.y_range == (0.5, 5.5)

    def test_from_classes_uses_first_two_keys(self):
        """When user/item not specified, uses first two keys of classes dict."""
        classes = {'people': ['P1', 'P2', 'P3'], 'things': ['T1', 'T2']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        assert model.user == 'people'
        assert model.item == 'things'

    def test_from_classes_forward_works(self):
        """Model built from from_classes should produce valid forward pass."""
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes, y_range=(1.0, 5.0))
        # user index 0, item index 1
        x = torch.tensor([[0, 1], [2, 0]])
        out = model(x)
        assert out.shape == (2,)
        assert torch.all(out >= 1.0)
        assert torch.all(out <= 5.0)


# ============================================================
# Tests for EmbeddingDotBias.bias
# ============================================================

class TestEmbeddingDotBiasBias:
    """Tests for the bias() extraction method."""

    def test_bias_item(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        bias = model.bias(['X', 'Y'], is_item=True)
        assert bias.shape == (2,)

    def test_bias_user(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        bias = model.bias(['A', 'B', 'C'], is_item=False)
        assert bias.shape == (3,)

    def test_bias_single_item(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        bias = model.bias(['X'], is_item=True)
        # Single item bias is squeezed to a scalar
        assert bias.ndim == 0

    def test_bias_returns_detached_tensor(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        bias = model.bias(['X'], is_item=True)
        assert not bias.requires_grad

    def test_bias_consistency(self):
        """Bias values should be consistent with the embedding layer."""
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        model.eval()

        bias = model.bias(['X', 'Y'], is_item=True)
        # Directly access the embedding
        idx = torch.tensor([0, 1])
        direct = model.i_bias(idx).squeeze().detach()
        assert torch.allclose(bias, direct, atol=1e-6)

    def test_bias_without_classes_raises(self):
        """Using bias without from_classes should raise an assertion error."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.bias(['something'], is_item=True)

    def test_bias_invalid_item_raises(self):
        """Requesting bias for a non-existent item should raise KeyError."""
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        with pytest.raises(KeyError):
            model.bias(['NonExistent'], is_item=True)


# ============================================================
# Tests for EmbeddingDotBias.weight
# ============================================================

class TestEmbeddingDotBiasWeight:
    """Tests for the weight() extraction method."""

    def test_weight_item(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        weight = model.weight(['X', 'Y'], is_item=True)
        assert weight.shape == (2, 5)

    def test_weight_user(self):
        classes = {'user': ['A', 'B', 'C'], 'item': ['X', 'Y', 'Z']}
        model = EmbeddingDotBias.from_classes(n_factors=5, classes=classes)
        weight = model.weight(['A', 'B', 'C'], is_item=False)
        assert weight.shape == (3, 5)

    def test_weight_single_item(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        weight = model.weight(['Y'], is_item=True)
        assert weight.shape == (1, 4)

    def test_weight_returns_detached_tensor(self):
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        weight = model.weight(['X'], is_item=True)
        assert not weight.requires_grad

    def test_weight_consistency(self):
        """Weight values should be consistent with the embedding layer."""
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=4, classes=classes)
        model.eval()

        weight = model.weight(['X', 'Y'], is_item=True)
        idx = torch.tensor([0, 1])
        direct = model.i_weight(idx).detach()
        assert torch.allclose(weight, direct, atol=1e-6)

    def test_weight_without_classes_raises(self):
        """Using weight without from_classes should raise an assertion error."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        with pytest.raises(AssertionError, match="Build your model with"):
            model.weight(['something'], is_item=True)

    def test_weight_invalid_user_raises(self):
        """Requesting weight for a non-existent user should raise KeyError."""
        classes = {'user': ['A', 'B'], 'item': ['X', 'Y']}
        model = EmbeddingDotBias.from_classes(n_factors=3, classes=classes)
        with pytest.raises(KeyError):
            model.weight(['NonExistent'], is_item=False)


# ============================================================
# Tests for EmbeddingDotBias gradient flow
# ============================================================

class TestEmbeddingDotBiasGradient:
    """Tests verifying gradient flow through the model."""

    def test_gradients_flow(self):
        """Verify gradients flow through all parameters during backprop."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10)
        x = torch.tensor([[0, 1], [2, 3]])
        target = torch.tensor([3.5, 4.0])

        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # Check that gradients exist for embeddings that were used
        assert model.u_weight.weight.grad is not None
        assert model.i_weight.weight.grad is not None
        assert model.u_bias.weight.grad is not None
        assert model.i_bias.weight.grad is not None

    def test_gradients_with_y_range(self):
        """Gradients should still flow when y_range is used (through sigmoid)."""
        model = EmbeddingDotBias(n_factors=5, n_users=10, n_items=10, y_range=(1.0, 5.0))
        x = torch.tensor([[0, 1], [2, 3]])
        target = torch.tensor([3.5, 4.0])

        out = model(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        assert model.u_weight.weight.grad is not None
        assert model.i_weight.weight.grad is not None
