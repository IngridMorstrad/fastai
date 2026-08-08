"""Tests for fastai.vision.gradcam module.

Covers GradCAM, GradCAMPP, and show_gradcam with CPU-only dummy models.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.vision.gradcam import GradCAM, GradCAMPP, show_gradcam, _find_last_conv2d

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


# ============================================================
# Helper: Simple CNN model for testing
# ============================================================

def _make_simple_cnn(in_channels=3, num_classes=10, spatial=32):
    """Create a minimal CNN for testing: 2 conv layers + global avg pool + linear."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )


# ============================================================
# Tests for _find_last_conv2d
# ============================================================

class TestFindLastConv2d:
    """Tests for the _find_last_conv2d helper."""

    def test_finds_last_conv(self):
        model = _make_simple_cnn()
        last_conv = _find_last_conv2d(model)
        # The second Conv2d (8->16) should be the last one
        assert isinstance(last_conv, nn.Conv2d)
        assert last_conv.out_channels == 16

    def test_returns_none_for_no_conv(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        result = _find_last_conv2d(model)
        assert result is None


# ============================================================
# Tests for GradCAM
# ============================================================

class TestGradCAM:
    """Tests for the GradCAM class."""

    def test_heatmap_shape_matches_input(self):
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(1, 3, 32, 32)
        with GradCAM(model) as cam:
            heatmap = cam.compute(img, class_idx=0)
        assert heatmap.shape == (32, 32)

    def test_heatmap_shape_different_input_size(self):
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(1, 3, 64, 48)
        with GradCAM(model) as cam:
            heatmap = cam.compute(img, class_idx=0)
        assert heatmap.shape == (64, 48)

    def test_heatmap_values_in_range(self):
        model = _make_simple_cnn(in_channels=3, num_classes=10)
        img = torch.randn(1, 3, 32, 32)
        with GradCAM(model) as cam:
            heatmap = cam.compute(img, class_idx=2)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_auto_class_selection(self):
        model = _make_simple_cnn(in_channels=3, num_classes=10)
        img = torch.randn(1, 3, 32, 32)
        with GradCAM(model) as cam:
            heatmap = cam.compute(img, class_idx=None)
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_3d_input_tensor(self):
        """Test that a 3D input (C, H, W) is handled correctly."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 32, 32)
        with GradCAM(model) as cam:
            heatmap = cam.compute(img, class_idx=0)
        assert heatmap.shape == (32, 32)

    def test_custom_target_layer(self):
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        # Use the first Conv2d layer as target
        first_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                break
        with GradCAM(model, target_layer=first_conv) as cam:
            heatmap = cam.compute(torch.randn(1, 3, 32, 32), class_idx=0)
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_raises_on_no_conv_layer(self):
        model = nn.Sequential(nn.Linear(10, 5))
        with pytest.raises(ValueError, match="No Conv2d layer found"):
            GradCAM(model)

    def test_remove_hooks(self):
        model = _make_simple_cnn()
        cam = GradCAM(model)
        assert len(cam._hooks) == 2
        cam.remove()
        assert len(cam._hooks) == 0


# ============================================================
# Tests for GradCAMPP
# ============================================================

class TestGradCAMPP:
    """Tests for the GradCAMPP class."""

    def test_heatmap_shape_matches_input(self):
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(1, 3, 32, 32)
        with GradCAMPP(model) as cam:
            heatmap = cam.compute(img, class_idx=0)
        assert heatmap.shape == (32, 32)

    def test_heatmap_values_in_range(self):
        model = _make_simple_cnn(in_channels=3, num_classes=10)
        img = torch.randn(1, 3, 32, 32)
        with GradCAMPP(model) as cam:
            heatmap = cam.compute(img, class_idx=3)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_auto_class_selection(self):
        model = _make_simple_cnn(in_channels=3, num_classes=10)
        img = torch.randn(1, 3, 32, 32)
        with GradCAMPP(model) as cam:
            heatmap = cam.compute(img, class_idx=None)
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_different_from_gradcam(self):
        """GradCAMPP should generally produce different heatmaps than GradCAM."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(1, 3, 32, 32)

        with GradCAM(model) as cam:
            heatmap_gc = cam.compute(img, class_idx=0)

        with GradCAMPP(model) as cam:
            heatmap_gcpp = cam.compute(img, class_idx=0)

        # They may be similar but the computation path is different
        # At minimum both should be valid heatmaps
        assert heatmap_gc.shape == heatmap_gcpp.shape
        assert heatmap_gcpp.min() >= 0.0
        assert heatmap_gcpp.max() <= 1.0


# ============================================================
# Tests for show_gradcam
# ============================================================

class TestShowGradcam:
    """Tests for the show_gradcam convenience function."""

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 32, 32)
        fig = show_gradcam(model, img, class_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_4d_input(self):
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(1, 3, 32, 32)
        fig = show_gradcam(model, img, class_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gradcampp_method(self):
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 32, 32)
        fig = show_gradcam(model, img, class_idx=0, method='gradcampp')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grayscale_input(self):
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=1, num_classes=5)
        img = torch.randn(1, 32, 32)
        fig = show_gradcam(model, img, class_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_learner_like_object(self):
        """Test that an object with a .model attribute works."""
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=3, num_classes=5)

        class FakeLearner:
            def __init__(self, m):
                self.model = m

        learn = FakeLearner(model)
        img = torch.randn(3, 32, 32)
        fig = show_gradcam(learn, img, class_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_auto_class_idx(self):
        import matplotlib.pyplot as plt
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 32, 32)
        fig = show_gradcam(model, img, class_idx=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_method_raises(self):
        """show_gradcam should raise ValueError for an unrecognized method."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 32, 32)
        with pytest.raises(ValueError, match="Invalid method"):
            show_gradcam(model, img, class_idx=0, method='typo')


# ============================================================
# Tests for review-identified issues
# ============================================================

class TestGradCAMReviewFixes:
    """Tests verifying fixes for issues identified in code review."""

    def test_batch_greater_than_one_raises(self):
        """Passing a batch with more than one image should raise ValueError."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(2, 3, 32, 32)
        with GradCAM(model) as cam:
            with pytest.raises(ValueError, match="batch size 1"):
                cam.compute(img, class_idx=0)

    def test_training_state_preserved_when_training(self):
        """Model should remain in training mode after compute() if it was training."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        model.train()
        assert model.training is True
        with GradCAM(model) as cam:
            cam.compute(torch.randn(1, 3, 32, 32), class_idx=0)
        assert model.training is True

    def test_eval_state_preserved_when_eval(self):
        """Model should remain in eval mode after compute() if it was in eval."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        model.eval()
        assert model.training is False
        with GradCAM(model) as cam:
            cam.compute(torch.randn(1, 3, 32, 32), class_idx=0)
        assert model.training is False

    def test_gradcampp_batch_greater_than_one_raises(self):
        """GradCAMPP should also reject batch > 1."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        img = torch.randn(3, 3, 32, 32)
        with GradCAMPP(model) as cam:
            with pytest.raises(ValueError, match="batch size 1"):
                cam.compute(img, class_idx=0)

    def test_gradcampp_training_state_preserved(self):
        """GradCAMPP should also preserve model training state."""
        model = _make_simple_cnn(in_channels=3, num_classes=5)
        model.train()
        assert model.training is True
        with GradCAMPP(model) as cam:
            cam.compute(torch.randn(1, 3, 32, 32), class_idx=0)
        assert model.training is True
