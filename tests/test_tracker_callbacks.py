"""Tests for fastai/callback/tracker.py callbacks.

Tests cover: TerminateOnNaNCallback, TrackerCallback, EarlyStoppingCallback,
SaveModelCallback, and ReduceLROnPlateau.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without running full training loops.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock
import numpy as np
import math
import os


# ----- Mock Setup -----
# We need to mock the entire import chain so tracker.py can be imported
# without torch or full fastai dependencies.

class _CancelFitException(Exception):
    pass


def _make_module(name, attrs=None):
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeTensor:
    """A minimal fake tensor that supports isinf/isnan checks."""
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def _fake_isinf(x):
    """Check if a FakeTensor value is infinite."""
    if isinstance(x, _FakeTensor):
        return math.isinf(x.value)
    return math.isinf(x)


def _fake_isnan(x):
    """Check if a FakeTensor value is NaN."""
    if isinstance(x, _FakeTensor):
        return math.isnan(x.value)
    return math.isnan(x)


# Mock torch module with our fake implementations
_mock_torch = _make_module('torch', {
    'isinf': _fake_isinf,
    'isnan': _fake_isnan,
})
_make_module('torch.multiprocessing')
_make_module('torch.nn')


class _Callback:
    """Minimal Callback base class for testing."""
    order = 0


# Create fastai package hierarchy
fastai_pkg = _make_module('fastai')
fastai_pkg.__path__ = []

basics_mod = _make_module('fastai.basics', {
    'np': np,
    'Callback': _Callback,
    'CancelFitException': _CancelFitException,
    'store_attr': lambda *a, **kw: None,
    'float': float,
})

callback_pkg = _make_module('fastai.callback')
callback_pkg.__path__ = []

_make_module('fastai.callback.progress')

fp16_mod = _make_module('fastai.callback.fp16', {
    'MixedPrecision': type('MixedPrecision', (object,), {})
})

# Remove any cached tracker module
if 'fastai.callback.tracker' in sys.modules:
    del sys.modules['fastai.callback.tracker']

# Load and exec the tracker source with our mock namespace
_tracker_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'tracker.py')
_tracker_path = os.path.abspath(_tracker_path)

tracker_module = types.ModuleType('fastai.callback.tracker')
tracker_module.__file__ = _tracker_path
tracker_module.__package__ = 'fastai.callback'

# Populate namespace with what `from ..basics import *` would provide
tracker_module.np = np
tracker_module.Callback = _Callback
tracker_module.CancelFitException = _CancelFitException
tracker_module.store_attr = lambda *a, **kw: None
tracker_module.MixedPrecision = fp16_mod.MixedPrecision
tracker_module.torch = _mock_torch
tracker_module.__builtins__ = __builtins__

# Execute the tracker source, skipping the internal import lines
with open(_tracker_path, 'r') as f:
    source = f.read()

lines = source.split('\n')
filtered_lines = []
for line in lines:
    if line.startswith('from __future__'):
        filtered_lines.append(line)
    elif line.startswith('from ..') or line.startswith('from .'):
        filtered_lines.append('pass  # skipped import')
    else:
        filtered_lines.append(line)

exec(compile('\n'.join(filtered_lines), _tracker_path, 'exec'), tracker_module.__dict__)
sys.modules['fastai.callback.tracker'] = tracker_module


# ----- Helpers -----

class FakeRecorder:
    """Mock recorder that simulates metric tracking."""
    def __init__(self, metric_names, values=None):
        self.metric_names = ['epoch'] + list(metric_names)
        self.values = values if values is not None else []


class FakeOptimizer:
    """Mock optimizer with hypers list."""
    def __init__(self, lr=0.01, n_groups=1):
        self.hypers = [{'lr': lr} for _ in range(n_groups)]


class FakeLearner:
    """Mock learner for SaveModelCallback tests."""
    def __init__(self):
        self.saved = []
        self.loaded = []

    def save(self, name, with_opt=False):
        self.saved.append((name, with_opt))
        return f'/path/to/{name}.pth'

    def load(self, name, with_opt=False):
        self.loaded.append((name, with_opt))


# ----- TerminateOnNaNCallback Tests -----

class TestTerminateOnNaNCallback(unittest.TestCase):
    """Test TerminateOnNaNCallback behavior."""

    def _create_callback(self):
        cb = tracker_module.TerminateOnNaNCallback()
        return cb

    def test_order_is_negative(self):
        """TerminateOnNaN should run early (negative order)."""
        cb = self._create_callback()
        self.assertEqual(cb.order, -9)

    def test_nan_loss_raises_cancel_fit(self):
        """NaN loss should raise CancelFitException."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(float('nan'))
        with self.assertRaises(_CancelFitException):
            cb.after_batch()

    def test_inf_loss_raises_cancel_fit(self):
        """Infinite loss should raise CancelFitException."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(float('inf'))
        with self.assertRaises(_CancelFitException):
            cb.after_batch()

    def test_negative_inf_loss_raises_cancel_fit(self):
        """Negative infinite loss should raise CancelFitException."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(float('-inf'))
        with self.assertRaises(_CancelFitException):
            cb.after_batch()

    def test_normal_loss_does_not_raise(self):
        """Normal loss values should not raise."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(0.5)
        # Should not raise
        cb.after_batch()

    def test_zero_loss_does_not_raise(self):
        """Zero loss should not raise."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(0.0)
        cb.after_batch()

    def test_negative_loss_does_not_raise(self):
        """Negative (but finite) loss should not raise."""
        cb = self._create_callback()
        cb.loss = _FakeTensor(-1.5)
        cb.after_batch()


# ----- TrackerCallback Tests -----

class TestTrackerCallback(unittest.TestCase):
    """Test TrackerCallback behavior."""

    def _create_callback(self, monitor='valid_loss', comp=None, min_delta=0., reset_on_fit=True):
        cb = tracker_module.TrackerCallback(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        return cb

    def test_default_comp_for_loss(self):
        """Monitor with 'loss' should default to np.less."""
        cb = self._create_callback(monitor='valid_loss')
        self.assertEqual(cb.comp, np.less)

    def test_default_comp_for_error(self):
        """Monitor with 'error' should default to np.less."""
        cb = self._create_callback(monitor='error_rate')
        self.assertEqual(cb.comp, np.less)

    def test_default_comp_for_metric(self):
        """Monitor without 'loss'/'error' should default to np.greater."""
        cb = self._create_callback(monitor='accuracy')
        self.assertEqual(cb.comp, np.greater)

    def test_custom_comp(self):
        """Explicit comp should override the default."""
        cb = self._create_callback(monitor='valid_loss', comp=np.greater)
        self.assertEqual(cb.comp, np.greater)

    def test_min_delta_negated_for_less(self):
        """min_delta should be negated when comp is np.less."""
        cb = self._create_callback(monitor='valid_loss', min_delta=0.01)
        self.assertEqual(cb.min_delta, -0.01)

    def test_min_delta_unchanged_for_greater(self):
        """min_delta should stay positive when comp is np.greater."""
        cb = self._create_callback(monitor='accuracy', min_delta=0.01)
        self.assertEqual(cb.min_delta, 0.01)

    def test_order(self):
        """TrackerCallback order should be 60."""
        cb = self._create_callback()
        self.assertEqual(cb.order, 60)

    def test_before_fit_sets_best_inf_for_less(self):
        """For np.less, best should start at +inf."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, float('inf'))

    def test_before_fit_sets_best_neg_inf_for_greater(self):
        """For np.greater, best should start at -inf."""
        cb = self._create_callback(monitor='accuracy')
        cb.recorder = FakeRecorder(['accuracy'])
        cb.before_fit()
        self.assertEqual(cb.best, -float('inf'))

    def test_before_fit_asserts_monitor_exists(self):
        """before_fit should raise if monitor is not in recorder metrics."""
        cb = self._create_callback(monitor='nonexistent')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        with self.assertRaises(AssertionError):
            cb.before_fit()

    def test_before_fit_skips_during_lr_finder(self):
        """If lr_finder attr exists, run should be False."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.lr_finder = True
        cb.before_fit()
        self.assertFalse(cb.run)

    def test_before_fit_skips_during_gather_preds(self):
        """If gather_preds attr exists, run should be False."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.gather_preds = True
        cb.before_fit()
        self.assertFalse(cb.run)

    def test_after_epoch_new_best_when_improving(self):
        """After epoch with improvement, new_best should be True."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.5])
        cb.after_epoch()
        self.assertTrue(cb.new_best)
        self.assertEqual(cb.best, 0.5)

    def test_after_epoch_not_new_best_when_worsening(self):
        """After epoch without improvement, new_best should be False."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # First epoch improves
        cb.recorder.values.append([0.5])
        cb.after_epoch()

        # Second epoch worsens
        cb.recorder.values.append([0.6])
        cb.after_epoch()
        self.assertFalse(cb.new_best)
        self.assertEqual(cb.best, 0.5)

    def test_after_epoch_with_min_delta(self):
        """Improvement must exceed min_delta to count."""
        cb = self._create_callback(monitor='valid_loss', min_delta=0.1)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # First epoch
        cb.recorder.values.append([0.5])
        cb.after_epoch()
        self.assertTrue(cb.new_best)

        # Slight improvement (less than delta) should NOT count
        cb.recorder.values.append([0.45])
        cb.after_epoch()
        self.assertFalse(cb.new_best)

        # Significant improvement should count
        cb.recorder.values.append([0.35])
        cb.after_epoch()
        self.assertTrue(cb.new_best)

    def test_reset_on_fit_true_resets_best(self):
        """With reset_on_fit=True, best resets on each fit call."""
        cb = self._create_callback(monitor='valid_loss', reset_on_fit=True)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.3])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.3)

        # Second fit call should reset
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, float('inf'))

    def test_reset_on_fit_false_preserves_best(self):
        """With reset_on_fit=False, best persists across fit calls."""
        cb = self._create_callback(monitor='valid_loss', reset_on_fit=False)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.3])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.3)

        # Second fit call should preserve
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, 0.3)

    def test_after_fit_sets_run_true(self):
        """after_fit should set run=True."""
        cb = self._create_callback(monitor='valid_loss')
        cb.run = False
        cb.after_fit()
        self.assertTrue(cb.run)

    def test_tracks_accuracy_correctly(self):
        """TrackerCallback should track a metric that uses np.greater."""
        cb = self._create_callback(monitor='accuracy')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # First epoch - accuracy 0.8
        cb.recorder.values.append([0.5, 0.8])
        cb.after_epoch()
        self.assertTrue(cb.new_best)
        self.assertEqual(cb.best, 0.8)

        # Second epoch - accuracy lower
        cb.recorder.values.append([0.4, 0.75])
        cb.after_epoch()
        self.assertFalse(cb.new_best)

        # Third epoch - accuracy higher
        cb.recorder.values.append([0.3, 0.85])
        cb.after_epoch()
        self.assertTrue(cb.new_best)
        self.assertEqual(cb.best, 0.85)


# ----- EarlyStoppingCallback Tests -----

class TestEarlyStoppingCallback(unittest.TestCase):
    """Test EarlyStoppingCallback behavior."""

    def _create_callback(self, monitor='valid_loss', patience=2, min_delta=0., comp=None):
        cb = tracker_module.EarlyStoppingCallback(
            monitor=monitor, comp=comp, min_delta=min_delta, patience=patience
        )
        return cb

    def test_order_is_tracker_plus_three(self):
        """EarlyStopping order should be TrackerCallback.order + 3."""
        cb = self._create_callback()
        self.assertEqual(cb.order, 63)

    def test_stops_after_patience_exhausted(self):
        """Should raise CancelFitException after patience epochs without improvement."""
        cb = self._create_callback(patience=2)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: worsens (wait=1)
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: worsens again (wait=2 >= patience=2)
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_does_not_stop_before_patience(self):
        """Should not stop if patience has not been exhausted."""
        cb = self._create_callback(patience=3)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: worsens (wait=1)
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: worsens again (wait=2, still < patience=3)
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()  # Should NOT raise

    def test_wait_resets_on_improvement(self):
        """Wait counter should reset when metric improves."""
        cb = self._create_callback(patience=2)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

        # Epoch 2: worsens
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.wait, 1)

        # Epoch 3: improves again (resets wait)
        cb.recorder.values.append([0.4])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

    def test_patience_one_stops_immediately_on_no_improvement(self):
        """With patience=1, should stop after first non-improving epoch."""
        cb = self._create_callback(patience=1)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: worsens (patience=1 exhausted)
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_before_fit_resets_wait(self):
        """before_fit should reset wait counter."""
        cb = self._create_callback(patience=2)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.wait = 5  # Simulate leftover state
        cb.before_fit()
        self.assertEqual(cb.wait, 0)

    def test_with_min_delta(self):
        """Improvement must exceed min_delta."""
        cb = self._create_callback(patience=2, min_delta=0.1)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: significant improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: slight improvement (not enough)
        cb.recorder.values.append([0.48])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.wait, 1)

        # Epoch 3: slight improvement again (still not enough)
        cb.recorder.values.append([0.46])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_monitoring_accuracy(self):
        """Should work correctly when monitoring accuracy (np.greater)."""
        cb = self._create_callback(monitor='accuracy', patience=2)
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # Epoch 1: accuracy improves
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: accuracy decreases
        cb.recorder.values.append([0.4, 0.75])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: accuracy decreases again
        cb.recorder.values.append([0.3, 0.70])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()


# ----- SaveModelCallback Tests -----

class TestSaveModelCallback(unittest.TestCase):
    """Test SaveModelCallback behavior."""

    def _create_callback(self, monitor='valid_loss', fname='model',
                         every_epoch=False, at_end=False, with_opt=False):
        cb = tracker_module.SaveModelCallback(
            monitor=monitor, fname=fname, every_epoch=every_epoch,
            at_end=at_end, with_opt=with_opt
        )
        # Manually set attributes since store_attr is mocked
        cb.fname = fname
        cb.every_epoch = every_epoch
        cb.at_end = at_end
        cb.with_opt = with_opt
        return cb

    def test_order_is_tracker_plus_one(self):
        """SaveModelCallback order should be TrackerCallback.order + 1."""
        cb = self._create_callback()
        self.assertEqual(cb.order, 61)

    def test_cannot_set_every_epoch_and_at_end(self):
        """Setting both every_epoch and at_end should raise."""
        with self.assertRaises(AssertionError):
            tracker_module.SaveModelCallback(
                every_epoch=True, at_end=True
            )

    def test_saves_on_improvement(self):
        """Should save model when metric improves."""
        cb = self._create_callback()
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()
        self.assertEqual(learner.saved, [('model', False)])
        self.assertEqual(cb.last_saved_path, '/path/to/model.pth')

    def test_does_not_save_without_improvement(self):
        """Should not save when metric does not improve."""
        cb = self._create_callback()
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()

        # Epoch 2: no improvement
        cb.recorder.values.append([0.6])
        cb.epoch = 1
        cb.after_epoch()
        # Should still only have one save
        self.assertEqual(len(learner.saved), 1)

    def test_every_epoch_saves_each_epoch(self):
        """With every_epoch=1, should save at every epoch."""
        cb = self._create_callback(every_epoch=1)
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()

        cb.recorder.values.append([0.6])
        cb.epoch = 1
        cb.after_epoch()

        cb.recorder.values.append([0.7])
        cb.epoch = 2
        cb.after_epoch()

        self.assertEqual(learner.saved, [
            ('model_0', False),
            ('model_1', False),
            ('model_2', False),
        ])

    def test_every_epoch_with_interval(self):
        """With every_epoch=2, should save every 2 epochs."""
        cb = self._create_callback(every_epoch=2)
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        for epoch in range(5):
            cb.recorder.values.append([0.5])
            cb.epoch = epoch
            cb.after_epoch()

        # Epochs 0, 2, 4 should save (every_epoch checks epoch % every_epoch == 0)
        self.assertEqual(learner.saved, [
            ('model_0', False),
            ('model_2', False),
            ('model_4', False),
        ])

    def test_at_end_saves_on_fit_end(self):
        """With at_end=True, should save at end of training."""
        cb = self._create_callback(at_end=True)
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.after_fit()
        self.assertEqual(learner.saved, [('model', False)])

    def test_loads_best_model_at_end_default(self):
        """Without at_end or every_epoch, should load best model after fit."""
        cb = self._create_callback()
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.after_fit()
        self.assertEqual(learner.loaded, [('model', False)])

    def test_with_opt_saves_optimizer(self):
        """With with_opt=True, should pass with_opt to save."""
        cb = self._create_callback(with_opt=True)
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()
        self.assertEqual(learner.saved, [('model', True)])

    def test_custom_fname(self):
        """Custom fname should be used when saving."""
        cb = self._create_callback(fname='best_classifier')
        learner = FakeLearner()
        cb.learn = learner
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()
        self.assertEqual(learner.saved, [('best_classifier', False)])


# ----- ReduceLROnPlateau Tests -----

class TestReduceLROnPlateau(unittest.TestCase):
    """Test ReduceLROnPlateau behavior."""

    def _create_callback(self, monitor='valid_loss', patience=2, factor=10., min_lr=0):
        cb = tracker_module.ReduceLROnPlateau(
            monitor=monitor, patience=patience, factor=factor, min_lr=min_lr
        )
        return cb

    def test_order_is_tracker_plus_two(self):
        """ReduceLROnPlateau order should be TrackerCallback.order + 2."""
        cb = self._create_callback()
        self.assertEqual(cb.order, 62)

    def test_reduces_lr_after_patience(self):
        """LR should be reduced after patience epochs without improvement."""
        cb = self._create_callback(patience=2, factor=10.)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.01)
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: no improvement (wait=1)
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01)

        # Epoch 3: no improvement (wait=2 >= patience=2 -> reduce)
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.001)

    def test_does_not_reduce_below_min_lr(self):
        """LR should not go below min_lr."""
        cb = self._create_callback(patience=1, factor=10., min_lr=0.005)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.01)
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: no improvement -> would reduce to 0.001 but capped at 0.005
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.005)

    def test_wait_resets_after_reduction(self):
        """Wait counter should reset after LR is reduced."""
        cb = self._create_callback(patience=1, factor=2.)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.1)
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

        # Epoch 2: no improvement -> reduce
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)
        self.assertEqual(cb.wait, 0)  # wait resets after reduction

    def test_wait_resets_on_improvement(self):
        """Wait counter should reset when metric improves."""
        cb = self._create_callback(patience=3)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.01)
        cb.before_fit()

        # Epoch 1
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: no improvement
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.wait, 1)

        # Epoch 3: improvement -> reset
        cb.recorder.values.append([0.4])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

    def test_multiple_reductions(self):
        """LR can be reduced multiple times."""
        cb = self._create_callback(patience=1, factor=2.)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.1)
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: no improvement -> reduce to 0.05
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)

        # Epoch 3: no improvement -> reduce to 0.025
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.025)

    def test_multiple_param_groups(self):
        """All parameter groups should have their LR reduced."""
        cb = self._create_callback(patience=1, factor=2.)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.1, n_groups=3)
        cb.before_fit()

        # Epoch 1: improvement
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: no improvement -> reduce
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()

        for h in cb.opt.hypers:
            self.assertAlmostEqual(h['lr'], 0.05)

    def test_before_fit_resets_wait(self):
        """before_fit should reset wait counter."""
        cb = self._create_callback(patience=2)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.opt = FakeOptimizer(lr=0.01)
        cb.wait = 5
        cb.before_fit()
        self.assertEqual(cb.wait, 0)

    def test_monitoring_accuracy(self):
        """Should work correctly monitoring accuracy (np.greater)."""
        cb = self._create_callback(monitor='accuracy', patience=1, factor=2.)
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.opt = FakeOptimizer(lr=0.1)
        cb.before_fit()

        # Epoch 1: accuracy improves
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: accuracy declines -> reduce
        cb.recorder.values.append([0.4, 0.75])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)


if __name__ == '__main__':
    unittest.main()
