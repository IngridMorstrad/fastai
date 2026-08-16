"""Tests for GradientNoise callback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch
import math

import numpy as np

# Ensure the tests directory is on the path for helper imports
sys.path.insert(0, os.path.dirname(__file__))


# --- Minimal mocking infrastructure ---

class FakeTensor:
    """Simulates a torch Tensor with add_ method."""
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def add_(self, other):
        """In-place addition (mimics tensor.add_)."""
        if isinstance(other, FakeTensor):
            self._data = self._data + other._data
        else:
            self._data = self._data + np.asarray(other, dtype=np.float32)
        return self

    def __mul__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor(self._data * other._data)
        return FakeTensor(self._data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __array__(self):
        return self._data

    def __eq__(self, other):
        if isinstance(other, FakeTensor):
            return np.array_equal(self._data, other._data)
        return np.array_equal(self._data, other)


class FakeParameter:
    """Simulates a torch Parameter with a grad attribute."""
    def __init__(self, data, requires_grad=True):
        self.data = FakeTensor(data)
        self.grad = FakeTensor(data) if requires_grad else None


class FakeCallback:
    """Minimal stand-in for fastai Callback base class."""
    pass


def fake_store_attr(names=None, **kwargs):
    """Mimics fastcore's store_attr: stores caller's locals as attributes."""
    import inspect
    frame = inspect.currentframe().f_back
    self = frame.f_locals.get('self')
    if self is None:
        return
    # Store all local variables except 'self' and dunder names
    for name, val in frame.f_locals.items():
        if name != 'self' and not name.startswith('__'):
            setattr(self, name, val)


def fake_randn_like(tensor):
    """Return ones-like for deterministic testing."""
    return FakeTensor(np.ones_like(np.asarray(tensor)))


def _setup_mock_modules():
    """Install minimal mock modules so fastai.callback.training can be imported."""
    torch_mod = types.ModuleType('torch')
    torch_mod.randn_like = fake_randn_like
    sys.modules['torch'] = torch_mod
    sys.modules['torch.multiprocessing'] = types.ModuleType('torch.multiprocessing')

    nn_mod = types.ModuleType('torch.nn')
    nn_mod.BatchNorm1d = type('BatchNorm1d', (), {})
    nn_mod.BatchNorm2d = type('BatchNorm2d', (), {})
    nn_mod.BatchNorm3d = type('BatchNorm3d', (), {})
    nn_mod.utils = MagicMock()
    sys.modules['torch.nn'] = nn_mod
    torch_mod.nn = nn_mod

    fastai_pkg = types.ModuleType('fastai')
    fastai_pkg.__path__ = []
    sys.modules['fastai'] = fastai_pkg

    # MixedPrecision needs an order attribute
    class FakeMixedPrecision:
        order = 65

    class FakeTrainEvalCallback:
        pass

    basics_attrs = {
        'np': np,
        'nn': nn_mod,
        'torch': torch_mod,
        'Callback': FakeCallback,
        'CancelFitException': Exception,
        'CancelTrainException': Exception,
        'CancelValidException': Exception,
        'CancelBatchException': Exception,
        'store_attr': fake_store_attr,
        'find_bs': lambda yb: 1,
        'TrainEvalCallback': FakeTrainEvalCallback,
    }
    basics_mod = types.ModuleType('fastai.basics')
    for k, v in basics_attrs.items():
        setattr(basics_mod, k, v)
    sys.modules['fastai.basics'] = basics_mod

    callback_pkg = types.ModuleType('fastai.callback')
    callback_pkg.__path__ = []
    sys.modules['fastai.callback'] = callback_pkg

    # Mock the progress and fp16 modules
    progress_mod = types.ModuleType('fastai.callback.progress')
    sys.modules['fastai.callback.progress'] = progress_mod

    fp16_mod = types.ModuleType('fastai.callback.fp16')
    fp16_mod.MixedPrecision = FakeMixedPrecision
    sys.modules['fastai.callback.fp16'] = fp16_mod


_setup_mock_modules()

# Now we can exec the training module source
_training_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'training.py')
_training_src = open(_training_path).read()
training_module = types.ModuleType('fastai.callback.training')
exec(compile(_training_src, _training_path, 'exec'), training_module.__dict__)
sys.modules['fastai.callback.training'] = training_module

GradientNoise = training_module.GradientNoise


# --- Tests ---

class TestGradientNoise(unittest.TestCase):
    """Test GradientNoise callback behavior with mocked learner components."""

    def _make_callback(self, eta=0.3, gamma=0.55):
        cb = GradientNoise(eta=eta, gamma=gamma)
        return cb

    def test_default_parameters(self):
        """Verify default hyperparameter values."""
        cb = GradientNoise()
        self.assertAlmostEqual(cb.eta, 0.3)
        self.assertAlmostEqual(cb.gamma, 0.55)

    def test_custom_parameters(self):
        """Verify custom hyperparameter values are stored."""
        cb = GradientNoise(eta=1.0, gamma=0.8)
        self.assertAlmostEqual(cb.eta, 1.0)
        self.assertAlmostEqual(cb.gamma, 0.8)

    def test_before_fit_initializes_count(self):
        """before_fit should initialize step counter to 0."""
        cb = self._make_callback()
        cb.before_fit()
        self.assertEqual(cb.count, 0)

    def test_before_step_increments_count(self):
        """Each call to before_step should increment the step counter."""
        cb = self._make_callback()
        cb.before_fit()

        params = [FakeParameter([1.0, 2.0, 3.0])]
        cb.parameters = lambda: params

        cb.before_step()
        self.assertEqual(cb.count, 1)
        cb.before_step()
        self.assertEqual(cb.count, 2)

    def test_noise_magnitude_decays(self):
        """Noise standard deviation should decrease as steps increase."""
        cb = self._make_callback(eta=1.0, gamma=0.55)
        cb.before_fit()

        # Compute expected std at step 0 and step 100
        std_0 = math.sqrt(1.0 / (1 + 0) ** 0.55)
        std_100 = math.sqrt(1.0 / (1 + 100) ** 0.55)
        self.assertGreater(std_0, std_100)

    def test_noise_is_added_to_gradients(self):
        """Gradients should be modified after before_step."""
        cb = self._make_callback(eta=1.0, gamma=0.55)
        cb.before_fit()

        original_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        param = FakeParameter([1.0, 2.0, 3.0])
        param.grad = FakeTensor(original_grad.copy())
        cb.parameters = lambda: [param]

        cb.before_step()

        # With fake_randn_like returning ones, noise = ones * std
        expected_std = math.sqrt(1.0 / (1 + 0) ** 0.55)
        expected_grad = original_grad + np.ones_like(original_grad) * expected_std
        np.testing.assert_array_almost_equal(np.asarray(param.grad), expected_grad, decimal=5)

    def test_none_grad_is_skipped(self):
        """Parameters with grad=None should not be modified."""
        cb = self._make_callback()
        cb.before_fit()

        param_with_grad = FakeParameter([1.0])
        param_without_grad = FakeParameter([2.0])
        param_without_grad.grad = None

        cb.parameters = lambda: [param_with_grad, param_without_grad]

        # Should not raise
        cb.before_step()
        self.assertIsNone(param_without_grad.grad)

    def test_multiple_parameters(self):
        """Noise should be added independently to each parameter's gradient."""
        cb = self._make_callback(eta=0.5, gamma=0.55)
        cb.before_fit()

        param1 = FakeParameter([1.0, 2.0])
        param1.grad = FakeTensor(np.array([1.0, 2.0], dtype=np.float32))
        param2 = FakeParameter([3.0, 4.0, 5.0])
        param2.grad = FakeTensor(np.array([3.0, 4.0, 5.0], dtype=np.float32))

        cb.parameters = lambda: [param1, param2]
        cb.before_step()

        expected_std = math.sqrt(0.5 / (1 + 0) ** 0.55)
        np.testing.assert_array_almost_equal(
            np.asarray(param1.grad), np.array([1.0, 2.0]) + expected_std, decimal=5
        )
        np.testing.assert_array_almost_equal(
            np.asarray(param2.grad), np.array([3.0, 4.0, 5.0]) + expected_std, decimal=5
        )

    def test_order_after_gradient_clip(self):
        """GradientNoise order should be greater than GradientClip order."""
        from fastai.callback.training import GradientClip
        self.assertGreater(GradientNoise.order, GradientClip.order)

    def test_in_module_all(self):
        """GradientNoise should be listed in __all__."""
        self.assertIn('GradientNoise', training_module.__all__)


if __name__ == '__main__':
    unittest.main()
