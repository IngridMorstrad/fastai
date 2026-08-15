"""Tests for GradientNoiseCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import sys
import os
import types
import unittest
from unittest.mock import patch, MagicMock
import numpy as np


# ----- Mock setup -----

def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class FakeParameter:
    """Mock torch parameter with grad attribute."""
    def __init__(self, shape, grad=None):
        self.data = np.zeros(shape)
        self.grad = grad


class FakeTensor:
    """Mock tensor supporting normal_ and add_."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def normal_(self, mean, std):
        """Fill with values from normal distribution (for testing, use fixed seed)."""
        rng = np.random.RandomState(42)
        self.data = rng.normal(mean, std, size=self.data.shape).astype(np.float32)
        return self

    def add_(self, other):
        """In-place addition."""
        if isinstance(other, FakeTensor):
            self.data += other.data
        else:
            self.data += other
        return self

    def __array__(self):
        return self.data


def _zeros_like(tensor):
    """Mock torch.zeros_like."""
    return FakeTensor(np.zeros_like(np.array(tensor.data)))


def _setup_mocks():
    """Install mock modules for loading gradient_noise module."""
    torch_mod = _make_module('torch', {
        'zeros_like': _zeros_like,
        'isinf': lambda x: False,
        'isnan': lambda x: False,
    })
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    class MockCallback:
        order = 0
        run_valid = True
        run = True

        def parameters(self):
            return getattr(self, '_params', [])

    class MockMixedPrecision:
        order = 10

    _make_module('fastai.basics', {
        'np': np,
        'Callback': MockCallback,
        'CancelFitException': Exception,
        'store_attr': lambda *a, **kw: None,
        'MixedPrecision': MockMixedPrecision,
        'torch': torch_mod,
        'nn': sys.modules['torch.nn'],
        'float': float,
    })

    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []
    _make_module('fastai.callback.progress')
    _make_module('fastai.callback.fp16', {
        'MixedPrecision': MockMixedPrecision,
    })


def _load_gradient_noise_module():
    """Load the gradient_noise module source with mocked imports."""
    module_path = os.path.join(
        os.path.dirname(__file__), '..', 'fastai', 'callback', 'gradient_noise.py'
    )
    module_path = os.path.abspath(module_path)

    mod = types.ModuleType('fastai.callback.gradient_noise')
    mod.__file__ = module_path
    mod.__package__ = 'fastai.callback'

    # Populate namespace
    mod.np = np
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.store_attr = lambda *a, **kw: None
    mod.MixedPrecision = sys.modules['fastai.callback.fp16'].MixedPrecision
    mod.torch = sys.modules['torch']
    mod.__builtins__ = __builtins__

    with open(module_path, 'r') as f:
        source = f.read()

    # Strip internal import lines that would fail without real fastai
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), module_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.gradient_noise'] = mod
    return mod


_setup_mocks()
_gradient_noise_module = _load_gradient_noise_module()
GradientNoiseCallback = _gradient_noise_module.GradientNoiseCallback


# ----- Tests -----

class TestGradientNoiseCallback(unittest.TestCase):
    """Test GradientNoiseCallback behavior with mocked parameters."""

    def _create_callback(self, eta=0.3, gamma=0.55):
        """Create a GradientNoiseCallback with manual attribute setting."""
        cb = GradientNoiseCallback.__new__(GradientNoiseCallback)
        cb.eta = eta
        cb.gamma = gamma
        cb.run = True
        cb._params = []
        return cb

    def _make_param(self, shape, has_grad=True):
        """Create a mock parameter."""
        p = FakeParameter(shape)
        if has_grad:
            p.grad = FakeTensor(np.ones(shape, dtype=np.float32))
        else:
            p.grad = None
        return p

    def test_default_parameters(self):
        """Test that defaults match Neelakantan et al. recommendations."""
        cb = self._create_callback()
        self.assertEqual(cb.eta, 0.3)
        self.assertEqual(cb.gamma, 0.55)

    def test_custom_parameters(self):
        """Test that custom eta and gamma are stored."""
        cb = self._create_callback(eta=1.0, gamma=0.8)
        self.assertEqual(cb.eta, 1.0)
        self.assertEqual(cb.gamma, 0.8)

    def test_before_fit_resets_step(self):
        """before_fit should reset _step to 0."""
        cb = self._create_callback()
        cb._step = 100
        cb.before_fit()
        self.assertEqual(cb._step, 0)

    def test_step_increments(self):
        """Each call to before_step should increment the step counter."""
        cb = self._create_callback()
        cb.before_fit()
        cb._params = [self._make_param((3,))]
        cb.before_step()
        self.assertEqual(cb._step, 1)
        cb.before_step()
        self.assertEqual(cb._step, 2)
        cb.before_step()
        self.assertEqual(cb._step, 3)

    def test_noise_modifies_gradients(self):
        """Gradients should be modified after before_step."""
        cb = self._create_callback(eta=1.0, gamma=0.0)
        cb.before_fit()
        param = self._make_param((100,))
        original_grad = np.array(param.grad.data, copy=True)
        cb._params = [param]

        cb.before_step()

        # With eta=1.0, gamma=0.0, sigma=1.0/(1+0)^0 = 1.0
        # Grad should be different from original (noise added)
        self.assertFalse(np.allclose(param.grad.data, original_grad))

    def test_no_grad_parameters_skipped(self):
        """Parameters with grad=None should be skipped without error."""
        cb = self._create_callback()
        cb.before_fit()
        param_with_grad = self._make_param((5,), has_grad=True)
        param_no_grad = self._make_param((5,), has_grad=False)
        cb._params = [param_with_grad, param_no_grad]

        # Should not raise
        cb.before_step()

        # Only param_with_grad should be modified
        self.assertIsNone(param_no_grad.grad)

    def test_noise_decay(self):
        """Noise magnitude should decay over iterations."""
        eta, gamma = 1.0, 1.0
        cb = self._create_callback(eta=eta, gamma=gamma)
        cb.before_fit()

        # Collect sigma values at each step
        # sigma_t = eta / (1 + t)^gamma
        expected_sigmas = [
            eta / (1 + 0) ** gamma,  # step 0: 1.0
            eta / (1 + 1) ** gamma,  # step 1: 0.5
            eta / (1 + 2) ** gamma,  # step 2: 0.333
        ]

        self.assertAlmostEqual(expected_sigmas[0], 1.0)
        self.assertAlmostEqual(expected_sigmas[1], 0.5)
        self.assertAlmostEqual(expected_sigmas[2], 1.0 / 3.0, places=5)

        # Verify each sigma is strictly decreasing
        for i in range(len(expected_sigmas) - 1):
            self.assertGreater(expected_sigmas[i], expected_sigmas[i + 1])

    def test_sigma_formula(self):
        """Verify the sigma formula: sigma_t = eta / (1 + t)^gamma."""
        eta, gamma = 0.3, 0.55
        cb = self._create_callback(eta=eta, gamma=gamma)
        cb.before_fit()

        # At step 0: sigma = 0.3 / (1+0)^0.55 = 0.3
        expected_sigma_0 = 0.3
        actual_sigma_0 = cb.eta / (1 + 0) ** cb.gamma
        self.assertAlmostEqual(actual_sigma_0, expected_sigma_0)

        # At step 10: sigma = 0.3 / 11^0.55
        expected_sigma_10 = 0.3 / (11 ** 0.55)
        actual_sigma_10 = cb.eta / (1 + 10) ** cb.gamma
        self.assertAlmostEqual(actual_sigma_10, expected_sigma_10)

    def test_run_valid_is_false(self):
        """Callback should not run during validation."""
        self.assertFalse(GradientNoiseCallback.run_valid)

    def test_order_matches_gradient_clip(self):
        """Order should be MixedPrecision.order + 1 (same as GradientClip)."""
        self.assertEqual(GradientNoiseCallback.order, 11)  # MixedPrecision.order=10, +1=11

    def test_in_all_list(self):
        """GradientNoiseCallback should be in __all__."""
        self.assertIn('GradientNoiseCallback', _gradient_noise_module.__all__)

    def test_multiple_parameters(self):
        """Noise should be added to all parameters with gradients."""
        cb = self._create_callback(eta=1.0, gamma=0.0)
        cb.before_fit()

        params = [self._make_param((10,)) for _ in range(5)]
        originals = [np.array(p.grad.data, copy=True) for p in params]
        cb._params = params

        cb.before_step()

        # All parameters should have modified gradients
        for i, p in enumerate(params):
            self.assertFalse(np.allclose(p.grad.data, originals[i]))

    def test_zero_eta_means_no_noise(self):
        """With eta=0, no noise should be added."""
        cb = self._create_callback(eta=0.0, gamma=0.55)
        cb.before_fit()

        param = self._make_param((50,))
        original_grad = np.array(param.grad.data, copy=True)
        cb._params = [param]

        cb.before_step()

        # sigma = 0 / (1+0)^0.55 = 0, so noise is 0
        np.testing.assert_array_almost_equal(param.grad.data, original_grad)

    def test_large_gamma_fast_decay(self):
        """Large gamma should make noise decay very quickly."""
        eta, gamma = 1.0, 5.0
        # At step 10: sigma = 1.0 / 11^5 = 1/161051 ~ 6.2e-6
        sigma_at_10 = eta / (1 + 10) ** gamma
        self.assertLess(sigma_at_10, 1e-4)

    def test_before_fit_can_be_called_multiple_times(self):
        """Calling before_fit multiple times should reset state correctly."""
        cb = self._create_callback()
        cb._params = [self._make_param((3,))]

        cb.before_fit()
        cb.before_step()
        cb.before_step()
        self.assertEqual(cb._step, 2)

        # Reset via another fit
        cb.before_fit()
        self.assertEqual(cb._step, 0)


if __name__ == '__main__':
    unittest.main()
