"""Tests for GradientNoiseCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
import sys
import os
import types
import math


# ---------------------------------------------------------------------------
# Minimal mock infrastructure (self-contained for this test file)
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Minimal tensor mock supporting .grad, .shape, .device, .dtype, and add_."""
    def __init__(self, shape=(10,), device='cpu', dtype='float32', requires_grad=True):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = _FakeGrad(shape, device, dtype) if requires_grad else None


class _FakeGrad:
    """Minimal gradient tensor mock that records add_ calls."""
    def __init__(self, shape, device, dtype):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.noise_added = []

    def add_(self, noise):
        self.noise_added.append(noise)


class _FakeNoise:
    """Represents noise returned by torch.normal."""
    def __init__(self, mean, std, size, device, dtype):
        self.mean = mean
        self.std = std
        self.size = size
        self.device = device
        self.dtype = dtype


_torch_normal_calls = []


def _mock_torch_normal(mean, std, size, device, dtype):
    """Mock torch.normal that records calls and returns a _FakeNoise."""
    noise = _FakeNoise(mean=mean, std=std, size=size, device=device, dtype=dtype)
    _torch_normal_calls.append(noise)
    return noise


def _setup_mocks():
    """Install minimal mock modules for fastai.callback.gradient_noise."""
    torch_mod = types.ModuleType('torch')
    torch_mod.normal = _mock_torch_normal
    sys.modules['torch'] = torch_mod
    sys.modules['torch.multiprocessing'] = types.ModuleType('torch.multiprocessing')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')

    fastai_pkg = types.ModuleType('fastai')
    fastai_pkg.__path__ = []
    sys.modules['fastai'] = fastai_pkg

    class _Callback:
        """Minimal Callback base that provides the interface we need."""
        order = 0
        run_valid = True

    sys.modules['fastai.basics'] = types.ModuleType('fastai.basics')
    sys.modules['fastai.basics'].Callback = _Callback
    sys.modules['fastai.basics'].torch = torch_mod

    callback_pkg = types.ModuleType('fastai.callback')
    callback_pkg.__path__ = []
    sys.modules['fastai.callback'] = callback_pkg


def _load_gradient_noise_module():
    """Load the gradient_noise module with mocked dependencies."""
    import importlib
    mod_name = 'fastai.callback.gradient_noise'
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    src_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'gradient_noise.py')
    src_path = os.path.abspath(src_path)

    mod = types.ModuleType(mod_name)
    mod.__file__ = src_path
    mod.__package__ = 'fastai.callback'

    import torch as torch_mod
    mod.torch = torch_mod
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.__builtins__ = __builtins__

    with open(src_path, 'r') as f:
        source = f.read()

    # Strip relative imports that would fail without full fastai
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), src_path, 'exec'), mod.__dict__)
    sys.modules[mod_name] = mod
    return mod


_setup_mocks()
_gn_module = _load_gradient_noise_module()
GradientNoiseCallback = _gn_module.GradientNoiseCallback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGradientNoiseCallback(unittest.TestCase):
    """Test GradientNoiseCallback behavior with mocked learner components."""

    def setUp(self):
        _torch_normal_calls.clear()

    def _make_cb(self, eta=0.3, gamma=0.55):
        cb = GradientNoiseCallback(eta=eta, gamma=gamma)
        # Simulate learner binding
        params = [_FakeTensor(shape=(10, 5)), _FakeTensor(shape=(20,))]
        cb.parameters = lambda: params
        cb._params = params
        return cb

    def test_default_parameters(self):
        """Default eta=0.3, gamma=0.55."""
        cb = GradientNoiseCallback()
        self.assertEqual(cb.eta, 0.3)
        self.assertEqual(cb.gamma, 0.55)

    def test_custom_parameters(self):
        """Custom eta and gamma are stored correctly."""
        cb = GradientNoiseCallback(eta=1.0, gamma=0.8)
        self.assertEqual(cb.eta, 1.0)
        self.assertEqual(cb.gamma, 0.8)

    def test_before_fit_resets_step_count(self):
        """before_fit initializes step_count to 0."""
        cb = self._make_cb()
        cb.step_count = 42
        cb.before_fit()
        self.assertEqual(cb.step_count, 0)

    def test_before_step_increments_step_count(self):
        """Each call to before_step increments step_count."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()
        self.assertEqual(cb.step_count, 1)
        cb.before_step()
        self.assertEqual(cb.step_count, 2)
        cb.before_step()
        self.assertEqual(cb.step_count, 3)

    def test_noise_sigma_formula(self):
        """Noise std follows sigma = eta / (1 + t)^gamma."""
        eta, gamma = 0.3, 0.55
        cb = self._make_cb(eta=eta, gamma=gamma)
        cb.before_fit()

        cb.before_step()
        expected_sigma = eta / (1 + 1) ** gamma
        # Two parameters, so two noise calls
        self.assertEqual(len(_torch_normal_calls), 2)
        self.assertAlmostEqual(_torch_normal_calls[0].std, expected_sigma, places=10)
        self.assertAlmostEqual(_torch_normal_calls[1].std, expected_sigma, places=10)

    def test_noise_sigma_decays_over_steps(self):
        """Sigma decreases as step count increases."""
        eta, gamma = 1.0, 0.55
        cb = self._make_cb(eta=eta, gamma=gamma)
        cb.before_fit()

        sigmas = []
        for _ in range(5):
            _torch_normal_calls.clear()
            cb.before_step()
            sigmas.append(_torch_normal_calls[0].std)

        # Each sigma should be strictly less than the previous
        for i in range(1, len(sigmas)):
            self.assertLess(sigmas[i], sigmas[i - 1])

    def test_noise_shape_matches_grad(self):
        """Noise tensor shape matches the parameter gradient shape."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()

        self.assertEqual(_torch_normal_calls[0].size, (10, 5))
        self.assertEqual(_torch_normal_calls[1].size, (20,))

    def test_noise_device_and_dtype_match_grad(self):
        """Noise device and dtype match the gradient's."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()

        for call in _torch_normal_calls:
            self.assertEqual(call.device, 'cpu')
            self.assertEqual(call.dtype, 'float32')

    def test_noise_mean_is_zero(self):
        """Noise mean is always 0."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()

        for call in _torch_normal_calls:
            self.assertEqual(call.mean, 0.0)

    def test_noise_added_to_grad(self):
        """Noise is actually added to each parameter's grad via add_."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()

        for p in cb._params:
            self.assertEqual(len(p.grad.noise_added), 1)

    def test_skips_params_without_grad(self):
        """Parameters with grad=None are skipped."""
        cb = GradientNoiseCallback(eta=0.3, gamma=0.55)
        params = [
            _FakeTensor(shape=(10,), requires_grad=True),
            _FakeTensor(shape=(5,), requires_grad=False),  # grad is None
        ]
        cb.parameters = lambda: params
        cb.before_fit()

        _torch_normal_calls.clear()
        cb.before_step()

        # Only 1 noise call (the param with grad)
        self.assertEqual(len(_torch_normal_calls), 1)
        self.assertEqual(_torch_normal_calls[0].size, (10,))

    def test_run_valid_is_false(self):
        """Callback should not run during validation."""
        cb = GradientNoiseCallback()
        self.assertFalse(cb.run_valid)

    def test_order_attribute(self):
        """Callback has a defined order."""
        cb = GradientNoiseCallback()
        self.assertEqual(cb.order, 60)

    def test_sigma_with_gamma_zero_is_constant(self):
        """When gamma=0, sigma = eta / 1 = eta (no decay)."""
        eta = 0.5
        cb = self._make_cb(eta=eta, gamma=0.0)
        cb.before_fit()

        for step in range(1, 4):
            _torch_normal_calls.clear()
            cb.before_step()
            expected = eta / (1 + step) ** 0.0  # = eta / 1 = eta
            self.assertAlmostEqual(_torch_normal_calls[0].std, eta, places=10)

    def test_multiple_fits_reset_step_count(self):
        """Calling before_fit again resets step_count."""
        cb = self._make_cb()
        cb.before_fit()
        cb.before_step()
        cb.before_step()
        self.assertEqual(cb.step_count, 2)

        cb.before_fit()
        self.assertEqual(cb.step_count, 0)


if __name__ == '__main__':
    unittest.main()
