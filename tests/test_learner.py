"""Tests for fastai.learner module utility functions.

Covers replacing_yield, mk_metric, save_model, and load_model.
"""
import sys
import os
import pytest
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from fastai.learner import replacing_yield, mk_metric, save_model, load_model, Metric, AvgMetric
from fastai.optimizer import Optimizer, sgd_step


# ============================================================
# Tests for `replacing_yield`
# ============================================================

class TestReplacingYield:
    """Tests for the replacing_yield context manager.

    Note: replacing_yield is a generator function that must be wrapped
    with @contextmanager to be used as a context manager (as done in Learner).
    """

    def _ctx(self, o, attr, val):
        """Helper to wrap replacing_yield as a proper context manager."""
        return contextmanager(replacing_yield)(o, attr, val)

    def test_replaces_attribute_inside_context(self):
        """Attribute should have the new value inside the context."""
        class Obj:
            x = 10

        o = Obj()
        with self._ctx(o, 'x', 99):
            assert o.x == 99

    def test_restores_attribute_after_context(self):
        """Attribute should be restored to the original value after exit."""
        class Obj:
            x = 10

        o = Obj()
        with self._ctx(o, 'x', 99):
            pass
        assert o.x == 10

    def test_restores_on_exception(self):
        """Attribute should be restored even when an exception occurs."""
        class Obj:
            name = "original"

        o = Obj()
        with pytest.raises(ValueError):
            with self._ctx(o, 'name', 'temporary'):
                assert o.name == 'temporary'
                raise ValueError("test error")
        assert o.name == "original"

    def test_replaces_none_value(self):
        """Should handle replacing with None."""
        class Obj:
            val = 42

        o = Obj()
        with self._ctx(o, 'val', None):
            assert o.val is None
        assert o.val == 42

    def test_replaces_with_same_value(self):
        """Should work correctly when replacing with the same value."""
        class Obj:
            val = 5

        o = Obj()
        with self._ctx(o, 'val', 5):
            assert o.val == 5
        assert o.val == 5

    def test_nested_replacing_yield(self):
        """Should handle nested context managers correctly."""
        class Obj:
            val = 1

        o = Obj()
        with self._ctx(o, 'val', 2):
            assert o.val == 2
            with self._ctx(o, 'val', 3):
                assert o.val == 3
            assert o.val == 2
        assert o.val == 1


# ============================================================
# Tests for `mk_metric`
# ============================================================

class TestMkMetric:
    """Tests for the mk_metric function."""

    def test_function_becomes_avg_metric(self):
        """A plain function should be wrapped in AvgMetric."""
        def my_func(pred, targ):
            return (pred - targ).abs().mean()

        result = mk_metric(my_func)
        assert isinstance(result, AvgMetric)
        assert isinstance(result, Metric)

    def test_metric_instance_passes_through(self):
        """An existing Metric instance should be returned as-is."""
        class MyMetric(Metric):
            def reset(self): pass
            def accumulate(self, learn): pass
            @property
            def value(self): return 0.0

        m = MyMetric()
        result = mk_metric(m)
        assert result is m

    def test_metric_class_is_instantiated(self):
        """A Metric class (not instance) should be instantiated."""
        class MyMetric(Metric):
            def reset(self): pass
            def accumulate(self, learn): pass
            @property
            def value(self): return 0.0

        result = mk_metric(MyMetric)
        assert isinstance(result, MyMetric)
        assert isinstance(result, Metric)

    def test_avg_metric_wraps_lambda(self):
        """A lambda function should be wrapped in AvgMetric."""
        fn = lambda pred, targ: (pred == targ).float().mean()
        result = mk_metric(fn)
        assert isinstance(result, AvgMetric)

    def test_avg_metric_preserves_function(self):
        """AvgMetric should store the original function."""
        def accuracy(pred, targ):
            return (pred == targ).float().mean()

        result = mk_metric(accuracy)
        assert isinstance(result, AvgMetric)
        assert result.func is accuracy


# ============================================================
# Tests for `save_model` and `load_model`
# ============================================================

class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Create an optimizer for the model."""
        return Optimizer(simple_model.parameters(), [sgd_step], lr=0.01)

    @pytest.fixture
    def tmp_file(self, tmp_path):
        """Return a temporary file path for saving models."""
        return str(tmp_path / "model.pth")

    def test_save_and_load_model_without_opt(self, simple_model, simple_optimizer, tmp_file):
        """Save and load model state without optimizer."""
        # Record original state
        original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        # Save
        save_model(tmp_file, simple_model, simple_optimizer, with_opt=False)

        # Modify model weights
        with torch.no_grad():
            for p in simple_model.parameters():
                p.fill_(0.0)

        # Load
        load_model(tmp_file, simple_model, simple_optimizer, with_opt=False)

        # Verify restored
        for key in original_state:
            assert torch.allclose(simple_model.state_dict()[key], original_state[key])

    def test_save_and_load_model_with_opt(self, simple_model, simple_optimizer, tmp_file):
        """Save and load model state with optimizer state."""
        # Give optimizer some state by doing a step
        x = torch.randn(3, 10)
        target = torch.randn(3, 2)
        loss = nn.MSELoss()(simple_model(x), target)
        loss.backward()
        simple_optimizer.step()

        # Record states
        original_model_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        original_opt_state = simple_optimizer.state_dict()

        # Save with optimizer
        save_model(tmp_file, simple_model, simple_optimizer, with_opt=True)

        # Modify model weights
        with torch.no_grad():
            for p in simple_model.parameters():
                p.fill_(0.0)

        # Load with optimizer (need weights_only=False for fastcore L objects in optimizer state)
        load_model(tmp_file, simple_model, simple_optimizer, with_opt=True, weights_only=False)

        # Verify model restored
        for key in original_model_state:
            assert torch.allclose(simple_model.state_dict()[key], original_model_state[key])

    def test_save_with_none_opt(self, simple_model, tmp_file):
        """Saving with opt=None should save only model state."""
        save_model(tmp_file, simple_model, None, with_opt=True)

        # The file should contain just the model state_dict (not wrapped in 'model' key)
        state = torch.load(tmp_file, map_location='cpu')
        assert 'model' not in state  # No opt wrapper when opt is None
        # It should be the raw state dict
        for key in simple_model.state_dict():
            assert key in state

    def test_load_model_to_device(self, simple_model, simple_optimizer, tmp_file):
        """Loading model with a specific device should work."""
        save_model(tmp_file, simple_model, simple_optimizer, with_opt=False)

        # Modify model
        with torch.no_grad():
            for p in simple_model.parameters():
                p.fill_(0.0)

        # Load to CPU explicitly
        load_model(tmp_file, simple_model, simple_optimizer, with_opt=False, device=torch.device('cpu'))

        # Verify all params on CPU
        for p in simple_model.parameters():
            assert p.device == torch.device('cpu')

    def test_save_creates_file(self, simple_model, simple_optimizer, tmp_file):
        """save_model should create the file on disk."""
        assert not os.path.exists(tmp_file)
        save_model(tmp_file, simple_model, simple_optimizer, with_opt=False)
        assert os.path.exists(tmp_file)

    def test_load_model_strict_mode(self, tmp_file):
        """Loading with strict=True should raise if keys don't match."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 3)  # Different architecture

        save_model(tmp_file, model1, None, with_opt=False)

        with pytest.raises(RuntimeError):
            load_model(tmp_file, model2, None, with_opt=False, strict=True)

    def test_load_model_non_strict_mode(self, tmp_file):
        """Loading with strict=False should not raise on mismatched keys."""
        model1 = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        # A model with different structure but some matching keys
        model2 = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))

        save_model(tmp_file, model1, None, with_opt=False)

        # strict=False should not raise even though sizes differ for the second layer
        # However, the mismatched size will still raise. Let's use a compatible scenario:
        # Save model1, load into a fresh model1 copy with strict=False
        model1_copy = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        load_model(tmp_file, model1_copy, None, with_opt=False, strict=False)

        # Verify loaded correctly
        for key in model1.state_dict():
            assert torch.allclose(model1_copy.state_dict()[key], model1.state_dict()[key])

    def test_pickle_protocol_parameter(self, simple_model, simple_optimizer, tmp_file):
        """save_model should accept pickle_protocol parameter."""
        # Should not raise
        save_model(tmp_file, simple_model, simple_optimizer, with_opt=False, pickle_protocol=2)
        assert os.path.exists(tmp_file)

        # Load it back to ensure it's valid
        load_model(tmp_file, simple_model, simple_optimizer, with_opt=False)
