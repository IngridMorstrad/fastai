"""Tests for StochasticWeightAveraging callback.

These tests mock the heavy fastai/torch dependencies so the callback logic can be
validated without installing PyTorch or the full fastai stack. A mock ``torch`` is
installed in ``sys.modules`` BEFORE the module under test is imported, because that
module does ``import torch`` at the top. The EMA math is driven entirely by plain
numpy arrays held in the mocked model's state, so the mock ``torch`` only needs to
exist, not do any heavy lifting.
"""
import os
import sys
import types
import unittest

import numpy as np


# ----- Install a mock ``torch`` before importing the module under test -----

if 'torch' not in sys.modules:
    sys.modules['torch'] = types.ModuleType('torch')


def _load_swa_module():
    """Load the standalone SWA module with the real fastai stack absent.

    The module's ``try: from fastai.callback.core import Callback`` import is expected
    to fail here, exercising the ``_BaseCallback`` fallback shim.
    """
    module_name = 'fastai.callback.stochastic_weight_averaging'
    if module_name in sys.modules:
        del sys.modules[module_name]
    path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback',
                        'stochastic_weight_averaging.py')
    path = os.path.abspath(path)
    mod = types.ModuleType(module_name)
    mod.__file__ = path
    with open(path, 'r') as f:
        source = f.read()
    exec(compile(source, path, 'exec'), mod.__dict__)
    sys.modules[module_name] = mod
    return mod


swa_module = _load_swa_module()
StochasticWeightAveraging = swa_module.StochasticWeightAveraging


# ----- Mock model / learner -----

class FakeModel:
    """Mock model backed by numpy arrays.

    ``named_parameters`` yields only the trainable parameters (used to drive the EMA),
    while ``state_dict``/``load_state_dict`` expose the full state including buffers.
    """
    def __init__(self, params=None, buffers=None):
        # params: trainable parameters (averaged); buffers: extra state (e.g. int buffers)
        self._params = params or {'layer1.weight': np.array([1.0, 2.0]),
                                  'layer1.bias': np.array([0.5])}
        self._buffers = buffers or {}

    def named_parameters(self):
        for k, v in self._params.items():
            yield k, v

    def parameters(self):
        for _, v in self._params.items():
            yield v

    def state_dict(self):
        d = {}
        for k, v in self._params.items():
            d[k] = v
        for k, v in self._buffers.items():
            d[k] = v
        return d

    def load_state_dict(self, state_dict):
        for k in self._params:
            if k in state_dict:
                self._params[k] = state_dict[k]
        for k in self._buffers:
            if k in state_dict:
                self._buffers[k] = state_dict[k]


class FakeLearner:
    """Mock learner holding a model."""
    def __init__(self, model=None):
        self.model = model or FakeModel()


def _setup(cb, model):
    """Attach a mocked learner/model and default epoch to a callback."""
    cb.learn = FakeLearner(model)
    cb.epoch = 0
    return cb


# ----- Tests -----

class TestStochasticWeightAveraging(unittest.TestCase):

    def test_default_parameters(self):
        cb = StochasticWeightAveraging()
        self.assertEqual(cb.decay, 0.99)
        self.assertEqual(cb.start_epoch, 0)
        self.assertEqual(cb.ema_state, {})

    def test_custom_parameters(self):
        cb = StochasticWeightAveraging(decay=0.5, start_epoch=3)
        self.assertEqual(cb.decay, 0.5)
        self.assertEqual(cb.start_epoch, 3)

    def test_order(self):
        self.assertEqual(StochasticWeightAveraging().order, 20)

    def test_repr(self):
        self.assertEqual(repr(StochasticWeightAveraging()), 'StochasticWeightAveraging')

    def test_invalid_decay_negative_raises(self):
        with self.assertRaises(ValueError):
            StochasticWeightAveraging(decay=-0.1)

    def test_invalid_decay_above_one_raises(self):
        with self.assertRaises(ValueError):
            StochasticWeightAveraging(decay=1.1)

    def test_decay_bounds_allowed(self):
        # The closed interval [0, 1] is valid.
        self.assertEqual(StochasticWeightAveraging(decay=0.0).decay, 0.0)
        self.assertEqual(StochasticWeightAveraging(decay=1.0).decay, 1.0)

    def test_invalid_start_epoch_raises(self):
        with self.assertRaises(ValueError):
            StochasticWeightAveraging(start_epoch=-1)

    def test_in_all_list(self):
        self.assertIn('StochasticWeightAveraging', swa_module.__all__)

    def test_ema_initialized_lazily_from_first_weights(self):
        cb = StochasticWeightAveraging(decay=0.9)
        model = FakeModel(params={'w': np.array([1.0, 2.0])})
        _setup(cb, model)
        cb.before_fit()
        self.assertEqual(cb.ema_state, {})
        cb.after_step()
        # First observed weights become the EMA seed (independent copy).
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0, 2.0]))
        # Mutating the model's live weights must not change the stashed EMA buffer.
        model._params['w'] = np.array([99.0, 99.0])
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0, 2.0]))

    def test_ema_update_math_is_correct(self):
        decay = 0.8
        cb = StochasticWeightAveraging(decay=decay)
        model = FakeModel(params={'w': np.array([0.0])})
        _setup(cb, model)
        cb.before_fit()

        weight_sequence = [np.array([0.0]), np.array([10.0]),
                           np.array([20.0]), np.array([30.0])]
        expected = None
        for w in weight_sequence:
            model._params['w'] = w.copy()
            cb.after_step()
            if expected is None:
                expected = w.copy()
            else:
                expected = decay * expected + (1. - decay) * w
            np.testing.assert_array_almost_equal(cb.ema_state['w'], expected)

    def test_averaged_state_property(self):
        cb = StochasticWeightAveraging()
        model = FakeModel(params={'w': np.array([1.0])})
        _setup(cb, model)
        cb.before_fit()
        cb.after_step()
        self.assertIs(cb.averaged_state, cb.ema_state)
        np.testing.assert_array_almost_equal(cb.averaged_state['w'], np.array([1.0]))

    def test_before_validate_swaps_and_after_validate_restores(self):
        cb = StochasticWeightAveraging(decay=0.5)
        model = FakeModel(params={'w': np.array([2.0, 4.0])})
        _setup(cb, model)
        cb.before_fit()

        # Seed EMA then take a step so EMA diverges from live weights.
        cb.after_step()                      # ema = [2, 4]
        model._params['w'] = np.array([6.0, 8.0])
        cb.after_step()                      # ema = 0.5*[2,4] + 0.5*[6,8] = [4, 6]
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([4.0, 6.0]))

        live_before = model._params['w'].copy()  # [6, 8]
        cb.before_validate()
        # Model now holds the averaged weights.
        np.testing.assert_array_almost_equal(model._params['w'], np.array([4.0, 6.0]))

        cb.after_validate()
        # Live weights restored exactly.
        np.testing.assert_array_almost_equal(model._params['w'], live_before)

    def test_before_validate_noop_when_ema_uninitialized(self):
        cb = StochasticWeightAveraging()
        model = FakeModel(params={'w': np.array([5.0])})
        _setup(cb, model)
        cb.before_fit()
        # No steps taken -> EMA empty -> validate should not touch weights.
        cb.before_validate()
        np.testing.assert_array_almost_equal(model._params['w'], np.array([5.0]))
        cb.after_validate()
        np.testing.assert_array_almost_equal(model._params['w'], np.array([5.0]))

    def test_integer_buffer_not_averaged(self):
        cb = StochasticWeightAveraging(decay=0.5)
        model = FakeModel(
            params={'bn.weight': np.array([1.0, 2.0])},
            buffers={'bn.num_batches_tracked': np.array([100], dtype=np.int64)},
        )
        _setup(cb, model)
        cb.before_fit()

        cb.after_step()                                 # seed ema weight
        model._params['bn.weight'] = np.array([3.0, 4.0])
        cb.after_step()                                 # ema = [2, 3]

        # Integer buffer must never enter the EMA state.
        self.assertNotIn('bn.num_batches_tracked', cb.ema_state)

        model._buffers['bn.num_batches_tracked'] = np.array([200], dtype=np.int64)
        cb.before_validate()
        # Float weight is averaged...
        np.testing.assert_array_almost_equal(model._params['bn.weight'], np.array([2.0, 3.0]))
        # ...but the integer buffer keeps its live value (not averaged, not clobbered).
        np.testing.assert_array_equal(model._buffers['bn.num_batches_tracked'], np.array([200]))
        self.assertEqual(model._buffers['bn.num_batches_tracked'].dtype, np.int64)

        cb.after_validate()
        np.testing.assert_array_almost_equal(model._params['bn.weight'], np.array([3.0, 4.0]))

    def test_run_false_prevents_accumulation_and_swap(self):
        cb = StochasticWeightAveraging(decay=0.5)
        model = FakeModel(params={'w': np.array([7.0])})
        _setup(cb, model)
        cb.run = False

        cb.before_fit()
        cb.after_step()
        self.assertEqual(cb.ema_state, {})

        cb.before_validate()
        np.testing.assert_array_almost_equal(model._params['w'], np.array([7.0]))
        cb.after_validate()
        np.testing.assert_array_almost_equal(model._params['w'], np.array([7.0]))

    def test_start_epoch_delays_accumulation(self):
        cb = StochasticWeightAveraging(decay=0.5, start_epoch=2)
        model = FakeModel(params={'w': np.array([1.0])})
        _setup(cb, model)
        cb.before_fit()

        # Epochs before start_epoch do not accumulate.
        cb.epoch = 0
        cb.after_step()
        cb.epoch = 1
        cb.after_step()
        self.assertEqual(cb.ema_state, {})

        # From start_epoch onwards accumulation begins.
        cb.epoch = 2
        model._params['w'] = np.array([10.0])
        cb.after_step()
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([10.0]))


if __name__ == '__main__':
    unittest.main()
