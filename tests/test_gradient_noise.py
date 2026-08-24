"""Tests for GradientNoiseCallback."""

import pytest
import torch
import torch.nn as nn

from fastai.callback.gradient_noise import GradientNoiseCallback


class TestGradientNoiseCallback:
    """Unit tests for GradientNoiseCallback."""

    def test_default_parameters(self):
        """Callback initializes with correct default parameters."""
        cb = GradientNoiseCallback()
        assert cb.noise_scale == 0.1
        assert cb.decay_rate == 0.55
        assert cb.order == 10

    def test_custom_parameters(self):
        """Callback accepts custom noise_scale and decay_rate."""
        cb = GradientNoiseCallback(noise_scale=0.5, decay_rate=0.75)
        assert cb.noise_scale == 0.5
        assert cb.decay_rate == 0.75

    def test_negative_noise_scale_raises(self):
        """Negative noise_scale raises ValueError."""
        with pytest.raises(ValueError, match="noise_scale must be non-negative"):
            GradientNoiseCallback(noise_scale=-0.1)

    def test_negative_decay_rate_raises(self):
        """Negative decay_rate raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be non-negative"):
            GradientNoiseCallback(decay_rate=-0.5)

    def test_zero_noise_scale_no_modification(self):
        """With noise_scale=0, gradients should remain unchanged."""
        cb = GradientNoiseCallback(noise_scale=0.0)
        model = nn.Linear(4, 2)
        # Simulate a backward pass
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        # Store original gradients
        original_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        # Mock the learn object
        cb.learn = _MockLearn(model)
        cb.after_backward()
        # Gradients should be unchanged (noise std = sqrt(0) = 0)
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, original_grads[name]), \
                    f"Gradient for {name} changed with noise_scale=0"

    def test_noise_modifies_gradients(self):
        """With non-zero noise_scale, gradients should be modified."""
        torch.manual_seed(42)
        cb = GradientNoiseCallback(noise_scale=1.0)
        model = nn.Linear(4, 2)
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        original_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        cb.learn = _MockLearn(model)
        cb.after_backward()
        # At least one gradient should differ
        any_changed = False
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.allclose(p.grad, original_grads[name]):
                any_changed = True
                break
        assert any_changed, "No gradients were modified despite non-zero noise_scale"

    def test_step_counter_increments(self):
        """Step counter increments after each after_backward call."""
        cb = GradientNoiseCallback()
        model = nn.Linear(2, 1)
        cb.learn = _MockLearn(model)
        assert cb._step == 0
        # Simulate multiple backward passes
        for i in range(5):
            x = torch.randn(1, 2)
            loss = model(x).sum()
            loss.backward()
            cb.after_backward()
            assert cb._step == i + 1

    def test_before_fit_resets_step(self):
        """before_fit resets the step counter to 0."""
        cb = GradientNoiseCallback()
        cb._step = 100
        cb.before_fit()
        assert cb._step == 0

    def test_noise_variance_decays(self):
        """Noise variance decreases as training progresses."""
        cb = GradientNoiseCallback(noise_scale=1.0, decay_rate=0.55)
        variances = []
        for step in range(10):
            cb._step = step
            variances.append(cb.current_variance)
        # Each variance should be less than or equal to the previous
        for i in range(1, len(variances)):
            assert variances[i] <= variances[i - 1], \
                f"Variance did not decay: step {i-1}={variances[i-1]}, step {i}={variances[i]}"

    def test_current_variance_formula(self):
        """current_variance property follows the expected formula."""
        cb = GradientNoiseCallback(noise_scale=0.3, decay_rate=0.7)
        cb._step = 5
        expected = 0.3 / (1 + 5) ** 0.7
        assert abs(cb.current_variance - expected) < 1e-10

    def test_skips_params_without_grad(self):
        """Parameters without gradients are skipped without error."""
        cb = GradientNoiseCallback(noise_scale=0.1)
        model = nn.Linear(4, 2)
        # Don't do backward - no grads exist
        cb.learn = _MockLearn(model)
        # Should not raise
        cb.after_backward()

    def test_repr(self):
        """Callback repr returns the class name."""
        cb = GradientNoiseCallback()
        assert repr(cb) == "GradientNoiseCallback"


class _MockLearn:
    """Minimal mock of a Learner for testing callbacks."""

    def __init__(self, model):
        self.model = model
