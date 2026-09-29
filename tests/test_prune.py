"""Tests for structured pruning (fastai/prune.py)."""

import pytest
import torch
import torch.nn as nn

from fastai.prune import structured_prune_model, prune


class TestStructuredPruneModel:
    """Unit tests for the pure structured_prune_model function."""

    def test_linear_zeroes_output_neurons(self):
        """Pruning an nn.Linear zeroes the requested fraction of output neurons."""
        torch.manual_seed(0)
        m = nn.Linear(8, 6)
        structured_prune_model(m, amount=0.5, n=2, dim=0)
        # dim=0 prunes output neurons (rows of the weight matrix).
        zero_rows = int((m.weight.abs().sum(dim=1) == 0).sum())
        assert zero_rows == round(6 * 0.5)
        # Mask baked in permanently: plain Parameter, no pruning reparam left.
        assert isinstance(m.weight, nn.Parameter)
        assert not hasattr(m, 'weight_orig')
        assert not hasattr(m, 'weight_mask')

    def test_amount_zero_is_noop(self):
        """amount=0.0 validates and leaves the weights unchanged."""
        torch.manual_seed(1)
        m = nn.Linear(8, 6)
        original = m.weight.detach().clone()
        structured_prune_model(m, amount=0.0)
        assert torch.allclose(m.weight, original)
        assert not hasattr(m, 'weight_orig')

    def test_conv2d_zeroes_output_filters(self):
        """Pruning an nn.Conv2d zeroes the requested fraction of output filters."""
        torch.manual_seed(2)
        m = nn.Conv2d(4, 8, 3)
        structured_prune_model(m, amount=0.25, n=2, dim=0)
        # dim=0 prunes output filters; a filter is zero when all its weights are zero.
        zero_filters = int((m.weight.abs().sum(dim=(1, 2, 3)) == 0).sum())
        assert zero_filters == round(8 * 0.25)
        assert isinstance(m.weight, nn.Parameter)
        assert not hasattr(m, 'weight_orig')

    def test_amount_one_raises(self):
        """amount >= 1 raises ValueError."""
        with pytest.raises(ValueError, match=r"amount must be in \[0, 1\)"):
            structured_prune_model(nn.Linear(8, 6), amount=1.0)

    def test_amount_negative_raises(self):
        """amount < 0 raises ValueError."""
        with pytest.raises(ValueError, match=r"amount must be in \[0, 1\)"):
            structured_prune_model(nn.Linear(8, 6), amount=-0.1)

    def test_non_positive_n_raises(self):
        """n <= 0 raises ValueError with a message matching the actual contract."""
        with pytest.raises(ValueError, match="n must be a positive number"):
            structured_prune_model(nn.Linear(8, 6), amount=0.5, n=0)

    def test_float_n_is_accepted(self):
        """A positive float n (e.g. 2.5) is a valid L-n order and prunes as usual."""
        torch.manual_seed(4)
        m = nn.Linear(8, 6)
        structured_prune_model(m, amount=0.5, n=2.5, dim=0)
        zero_rows = int((m.weight.abs().sum(dim=1) == 0).sum())
        assert zero_rows == round(6 * 0.5)

    def test_out_of_range_dim_raises_before_mutating(self):
        """An out-of-range dim raises ValueError before any module is pruned."""
        torch.manual_seed(5)
        # Sequential so there is an earlier eligible module (index 0) that must stay
        # untouched if validation happens before mutation.
        m = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
        first_before = m[0].weight.detach().clone()
        # Linear weight has 2 dims; dim=2 is out of range.
        with pytest.raises(ValueError, match="dim 2 is out of range"):
            structured_prune_model(m, amount=0.5, dim=2)
        # The earlier module must not have been partially pruned.
        assert torch.allclose(m[0].weight, first_before)

    def test_exclude_instance_protects_module(self):
        """A module passed via exclude is left untouched while others are pruned."""
        torch.manual_seed(6)
        head = nn.Linear(6, 4)
        m = nn.Sequential(nn.Linear(8, 6), head)
        head_before = head.weight.detach().clone()
        structured_prune_model(m, amount=0.5, dim=0, exclude=[head])
        # The excluded head is unchanged.
        assert torch.allclose(head.weight, head_before)
        # The other Linear was pruned.
        zero_rows = int((m[0].weight.abs().sum(dim=1) == 0).sum())
        assert zero_rows == round(6 * 0.5)

    def test_exclude_type_protects_all_of_that_type(self):
        """Passing a type to exclude skips every module of that type."""
        torch.manual_seed(7)
        m = nn.Sequential(nn.Conv2d(4, 8, 3), nn.Flatten(), nn.Linear(8, 4))
        lin_before = m[2].weight.detach().clone()
        # Exclude all Linear layers; only the Conv2d should be pruned.
        structured_prune_model(m, amount=0.25, dim=0, exclude=(nn.Linear,))
        assert torch.allclose(m[2].weight, lin_before)
        zero_filters = int((m[0].weight.abs().sum(dim=(1, 2, 3)) == 0).sum())
        assert zero_filters == round(8 * 0.25)

    def test_default_exclude_is_none_prunes_everything(self):
        """Default (exclude=None) still prunes every eligible module including the head."""
        torch.manual_seed(8)
        m = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
        structured_prune_model(m, amount=0.5, dim=0)
        assert int((m[0].weight.abs().sum(dim=1) == 0).sum()) == round(6 * 0.5)
        assert int((m[1].weight.abs().sum(dim=1) == 0).sum()) == round(4 * 0.5)


class _FakeLearner:
    """Minimal Learner stand-in exposing the attributes prune() touches.

    Building a real Learner drags in the full fastai stack (matplotlib, requests,
    scipy, torchvision, ...) which is not present in the minimal CI test env. The
    feature spec explicitly allows exercising the patched prune() via a lightweight
    object with a ``model`` attr, so we call the exported ``prune`` function directly.
    """

    def __init__(self, model):
        self.model = model
        self.fine_tuned_with = None

    def fine_tune(self, epochs, base_lr=None, **kwargs):
        self.fine_tuned_with = (epochs, base_lr, kwargs)


class _FakeLearnerNoFineTune:
    """Learner stand-in WITHOUT ``fine_tune``, to exercise the ``self.fit`` fallback.

    ``prune`` prefers ``self.fine_tune`` and falls back to ``self.fit``; a stub that
    lacks ``fine_tune`` is the only way to drive that branch without the full stack.
    """

    def __init__(self, model):
        self.model = model
        self.fit_with = None

    def fit(self, epochs, lr=None, **kwargs):
        self.fit_with = (epochs, lr, kwargs)


class TestLearnerPrune:
    """Tests for the @patch-ed Learner.prune method (via the exported function)."""

    def test_unsupported_method_raises(self):
        """An unsupported method raises ValueError."""
        learn = _FakeLearner(nn.Linear(1, 8))
        with pytest.raises(ValueError, match="Unsupported prune method"):
            prune(learn, amount=0.5, method='unstructured', fine_tune_epochs=0)

    def test_prune_without_fine_tune_returns_learner(self):
        """prune with fine_tune_epochs=0 prunes in place, returns self, skips training."""
        torch.manual_seed(3)
        learn = _FakeLearner(nn.Linear(1, 8))
        before = learn.model.weight.detach().clone()
        result = prune(learn, amount=0.5, fine_tune_epochs=0)
        assert result is learn
        # dim=0 prunes output neurons: half the rows become zero.
        zero_rows = int((learn.model.weight.abs().sum(dim=1) == 0).sum())
        assert zero_rows == round(8 * 0.5)
        assert not torch.allclose(learn.model.weight, before)
        assert not hasattr(learn.model, 'weight_orig')
        # fine_tune_epochs=0 must not trigger any training.
        assert learn.fine_tuned_with is None

    def test_prune_runs_fine_tune_when_epochs_positive(self):
        """prune calls fine_tune with the requested epochs/base_lr when epochs>0."""
        learn = _FakeLearner(nn.Linear(1, 8))
        result = prune(learn, amount=0.25, fine_tune_epochs=2, base_lr=1e-2)
        assert result is learn
        assert learn.fine_tuned_with == (2, 1e-2, {})

    def test_prune_falls_back_to_fit_without_fine_tune(self):
        """When the learner has no fine_tune, prune falls back to self.fit(lr=base_lr)."""
        learn = _FakeLearnerNoFineTune(nn.Linear(1, 8))
        result = prune(learn, amount=0.25, fine_tune_epochs=3, base_lr=5e-3)
        assert result is learn
        # base_lr is mapped to lr= for the fit fallback.
        assert learn.fit_with == (3, 5e-3, {})

    def test_prune_exclude_protects_module(self):
        """prune forwards exclude so a protected module is left unpruned."""
        torch.manual_seed(9)
        head = nn.Linear(6, 4)
        model = nn.Sequential(nn.Linear(8, 6), head)
        learn = _FakeLearner(model)
        head_before = head.weight.detach().clone()
        prune(learn, amount=0.5, fine_tune_epochs=0, exclude=[head])
        assert torch.allclose(head.weight, head_before)
        assert int((model[0].weight.abs().sum(dim=1) == 0).sum()) == round(6 * 0.5)
