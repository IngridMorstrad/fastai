"""Tests for tracker callbacks: TerminateOnNaNCallback, TrackerCallback,
EarlyStoppingCallback, SaveModelCallback, and ReduceLROnPlateau.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import os


# ----- Mock Setup -----
# We need to mock the entire import chain so tracker.py can be imported
# without torch or full fastai dependencies.

class _CancelFitException(Exception):
    pass


# Create mock modules as proper module objects (not MagicMock)
# so Python's import system treats them as real packages.
def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Mock torch and its submodules
_torch_mod = _make_module('torch', {
    'isinf': lambda x: getattr(x, '_is_inf', False),
    'isnan': lambda x: getattr(x, '_is_nan', False),
})
_make_module('torch.multiprocessing')
_make_module('torch.nn')

# Create the fastai package hierarchy
fastai_pkg = _make_module('fastai')
fastai_pkg.__path__ = []  # Mark as package

# fastai.basics - provides Callback, np, CancelFitException, store_attr, etc.
basics_mod = _make_module('fastai.basics', {
    'np': np,
    'Callback': type('Callback', (object,), {}),
    'CancelFitException': _CancelFitException,
    'store_attr': lambda *a, **kw: None,
    'float': float,
})

# fastai.callback package
callback_pkg = _make_module('fastai.callback')
callback_pkg.__path__ = []  # Mark as package

# fastai.callback.progress
_make_module('fastai.callback.progress')

# fastai.callback.fp16
fp16_mod = _make_module('fastai.callback.fp16', {'MixedPrecision': type('MixedPrecision', (object,), {})})

# Now import the tracker module by executing its source with our mocked namespace
if 'fastai.callback.tracker' in sys.modules:
    del sys.modules['fastai.callback.tracker']

_tracker_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'tracker.py')
_tracker_path = os.path.abspath(_tracker_path)

# Create the tracker module
tracker_module = types.ModuleType('fastai.callback.tracker')
tracker_module.__file__ = _tracker_path
tracker_module.__package__ = 'fastai.callback'

# Populate namespace with what `from ..basics import *` would provide
tracker_module.np = np
tracker_module.Callback = basics_mod.Callback
tracker_module.CancelFitException = _CancelFitException
tracker_module.store_attr = lambda self, *a, **kw: None
tracker_module.MixedPrecision = fp16_mod.MixedPrecision
tracker_module.torch = _torch_mod
tracker_module.__builtins__ = __builtins__

# Execute the tracker source, skipping the import lines
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


class FakeLoss:
    """Mock tensor-like loss value that can signal NaN or Inf."""
    def __init__(self, value=0.5, is_nan=False, is_inf=False):
        self.value = value
        self._is_nan = is_nan
        self._is_inf = is_inf


class FakeOptimizer:
    """Mock optimizer with hypers list for ReduceLROnPlateau tests."""
    def __init__(self, lr=0.01, n_groups=1):
        self.hypers = [{'lr': lr} for _ in range(n_groups)]


# ----- Tests for TerminateOnNaNCallback -----

class TestTerminateOnNaNCallback(unittest.TestCase):
    """Test TerminateOnNaNCallback behavior."""

    def _create_callback(self):
        cb = tracker_module.TerminateOnNaNCallback()
        return cb

    def test_normal_loss_continues(self):
        """Training should continue when loss is a normal value."""
        cb = self._create_callback()
        cb.loss = FakeLoss(value=0.5)
        # Should not raise
        cb.after_batch()

    def test_nan_loss_raises(self):
        """Training should terminate when loss is NaN."""
        cb = self._create_callback()
        cb.loss = FakeLoss(is_nan=True)
        with self.assertRaises(_CancelFitException):
            cb.after_batch()

    def test_inf_loss_raises(self):
        """Training should terminate when loss is Inf."""
        cb = self._create_callback()
        cb.loss = FakeLoss(is_inf=True)
        with self.assertRaises(_CancelFitException):
            cb.after_batch()

    def test_order_is_negative(self):
        """TerminateOnNaNCallback should have a negative order (runs early)."""
        cb = self._create_callback()
        self.assertEqual(cb.order, -9)


# ----- Tests for TrackerCallback -----

class TestTrackerCallback(unittest.TestCase):
    """Test TrackerCallback behavior with mocked learner components."""

    def _create_callback(self, monitor='valid_loss', comp=None, min_delta=0., reset_on_fit=True):
        cb = tracker_module.TrackerCallback(
            monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit
        )
        return cb

    def test_default_comp_loss(self):
        """Monitor with 'loss' in name should use np.less."""
        cb = self._create_callback(monitor='valid_loss')
        self.assertEqual(cb.comp, np.less)

    def test_default_comp_error(self):
        """Monitor with 'error' in name should use np.less."""
        cb = self._create_callback(monitor='error_rate')
        self.assertEqual(cb.comp, np.less)

    def test_default_comp_metric(self):
        """Monitor without 'loss'/'error' should use np.greater."""
        cb = self._create_callback(monitor='accuracy')
        self.assertEqual(cb.comp, np.greater)

    def test_custom_comp(self):
        """Custom comp should override default."""
        cb = self._create_callback(monitor='valid_loss', comp=np.greater)
        self.assertEqual(cb.comp, np.greater)

    def test_min_delta_negative_for_less(self):
        """min_delta should be negated when comp is np.less."""
        cb = self._create_callback(monitor='valid_loss', min_delta=0.01)
        self.assertEqual(cb.min_delta, -0.01)

    def test_min_delta_positive_for_greater(self):
        """min_delta should remain positive when comp is np.greater."""
        cb = self._create_callback(monitor='accuracy', min_delta=0.01)
        self.assertEqual(cb.min_delta, 0.01)

    def test_before_fit_sets_best_inf_for_loss(self):
        """best should be inf when monitoring loss (comp=np.less)."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, float('inf'))

    def test_before_fit_sets_best_neg_inf_for_metric(self):
        """best should be -inf when monitoring metric (comp=np.greater)."""
        cb = self._create_callback(monitor='accuracy')
        cb.recorder = FakeRecorder(['accuracy'])
        cb.before_fit()
        self.assertEqual(cb.best, -float('inf'))

    def test_before_fit_asserts_monitor_exists(self):
        """before_fit should raise AssertionError if monitor not in metrics."""
        cb = self._create_callback(monitor='nonexistent')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        with self.assertRaises(AssertionError):
            cb.before_fit()

    def test_before_fit_run_flag(self):
        """run should be True after before_fit (no lr_finder or gather_preds)."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertTrue(cb.run)

    def test_before_fit_run_false_with_lr_finder(self):
        """run should be False if lr_finder attribute is present."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.lr_finder = True  # Simulate lr_finder mode
        cb.before_fit()
        self.assertFalse(cb.run)

    def test_after_epoch_new_best_on_improvement(self):
        """new_best should be True when metric improves."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Loss improves (decreases)
        cb.recorder.values.append([0.5])
        cb.after_epoch()
        self.assertTrue(cb.new_best)
        self.assertEqual(cb.best, 0.5)

    def test_after_epoch_no_improvement(self):
        """new_best should be False when metric does not improve."""
        cb = self._create_callback(monitor='valid_loss')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # First epoch: improves from inf
        cb.recorder.values.append([0.5])
        cb.after_epoch()
        # Second epoch: worsens
        cb.recorder.values.append([0.6])
        cb.after_epoch()
        self.assertFalse(cb.new_best)
        self.assertEqual(cb.best, 0.5)

    def test_after_epoch_tracks_best_metric(self):
        """best should update only on improvements for a metric (np.greater)."""
        cb = self._create_callback(monitor='accuracy')
        cb.recorder = FakeRecorder(['accuracy'])
        cb.before_fit()
        # Epoch 1: improves from -inf
        cb.recorder.values.append([0.7])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.7)
        # Epoch 2: improves
        cb.recorder.values.append([0.8])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.8)
        # Epoch 3: worsens
        cb.recorder.values.append([0.75])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.8)

    def test_reset_on_fit_true_resets_best(self):
        """With reset_on_fit=True, best should reset on each fit."""
        cb = self._create_callback(monitor='valid_loss', reset_on_fit=True)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        cb.recorder.values.append([0.3])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.3)
        # Simulate second fit
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, float('inf'))

    def test_reset_on_fit_false_preserves_best(self):
        """With reset_on_fit=False, best should persist across fits."""
        cb = self._create_callback(monitor='valid_loss', reset_on_fit=False)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        cb.recorder.values.append([0.3])
        cb.after_epoch()
        self.assertEqual(cb.best, 0.3)
        # Simulate second fit
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.best, 0.3)

    def test_after_fit_resets_run(self):
        """after_fit should set run=True."""
        cb = self._create_callback(monitor='valid_loss')
        cb.run = False
        cb.after_fit()
        self.assertTrue(cb.run)

    def test_min_delta_affects_improvement_threshold(self):
        """min_delta should require more than min_delta improvement."""
        cb = self._create_callback(monitor='valid_loss', min_delta=0.1)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # First epoch: improves from inf
        cb.recorder.values.append([0.5])
        cb.after_epoch()
        self.assertTrue(cb.new_best)
        # Second epoch: improves by 0.05 (less than min_delta=0.1) - not enough
        cb.recorder.values.append([0.45])
        cb.after_epoch()
        self.assertFalse(cb.new_best)
        # Third epoch: improves by 0.15 from best of 0.5 - enough
        cb.recorder.values.append([0.35])
        cb.after_epoch()
        self.assertTrue(cb.new_best)

    def test_order_value(self):
        """TrackerCallback should have order=60."""
        self.assertEqual(tracker_module.TrackerCallback.order, 60)

    def test_idx_correct_for_multiple_metrics(self):
        """idx should correctly index into the recorder values."""
        cb = self._create_callback(monitor='f1_score')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy', 'f1_score'])
        cb.before_fit()
        self.assertEqual(cb.idx, 2)


# ----- Tests for EarlyStoppingCallback -----

class TestEarlyStoppingCallback(unittest.TestCase):
    """Test EarlyStoppingCallback behavior."""

    def _create_callback(self, monitor='valid_loss', patience=2, comp=None, min_delta=0., reset_on_fit=True):
        cb = tracker_module.EarlyStoppingCallback(
            monitor=monitor, comp=comp, min_delta=min_delta,
            patience=patience, reset_on_fit=reset_on_fit
        )
        return cb

    def test_inherits_from_tracker(self):
        """EarlyStoppingCallback should inherit from TrackerCallback."""
        self.assertTrue(issubclass(
            tracker_module.EarlyStoppingCallback,
            tracker_module.TrackerCallback
        ))

    def test_order_greater_than_tracker(self):
        """EarlyStoppingCallback order should be TrackerCallback.order + 3."""
        self.assertEqual(
            tracker_module.EarlyStoppingCallback.order,
            tracker_module.TrackerCallback.order + 3
        )

    def test_before_fit_initializes_wait(self):
        """before_fit should set wait=0."""
        cb = self._create_callback()
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.wait, 0)

    def test_no_stop_when_improving(self):
        """Should not stop when metric keeps improving."""
        cb = self._create_callback(monitor='valid_loss', patience=2)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Continuously improving - no stop
        for val in [0.5, 0.4, 0.3, 0.2]:
            cb.recorder.values.append([val])
            cb.epoch = len(cb.recorder.values)
            cb.after_epoch()

    def test_stop_after_patience_exhausted(self):
        """Should stop after patience epochs without improvement."""
        cb = self._create_callback(monitor='valid_loss', patience=2)
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
        # Epoch 3: worsens (wait=2 >= patience=2) -> stop
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_patience_resets_on_improvement(self):
        """Wait counter should reset when metric improves."""
        cb = self._create_callback(monitor='valid_loss', patience=2)
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
        # Epoch 3: improves - wait resets
        cb.recorder.values.append([0.3])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

    def test_patience_1_stops_immediately(self):
        """With patience=1, should stop after first non-improvement."""
        cb = self._create_callback(monitor='valid_loss', patience=1)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens (wait=1 >= patience=1) -> stop
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_monitors_metric_with_greater(self):
        """Should work correctly when monitoring an increasing metric."""
        cb = self._create_callback(monitor='accuracy', patience=2)
        cb.recorder = FakeRecorder(['accuracy'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.7])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens
        cb.recorder.values.append([0.65])
        cb.epoch = 2
        cb.after_epoch()
        # Epoch 3: worsens -> stop
        cb.recorder.values.append([0.60])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_large_patience(self):
        """Should tolerate many non-improving epochs up to patience."""
        cb = self._create_callback(monitor='valid_loss', patience=5)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epochs 2-5: worsen but within patience
        for i in range(4):
            cb.recorder.values.append([0.6 + i * 0.1])
            cb.epoch = i + 2
            cb.after_epoch()
        # Epoch 6: worsens, patience exhausted
        cb.recorder.values.append([1.0])
        cb.epoch = 6
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()


# ----- Tests for SaveModelCallback -----

class TestSaveModelCallback(unittest.TestCase):
    """Test SaveModelCallback behavior."""

    def _create_callback(self, monitor='valid_loss', fname='model',
                         every_epoch=False, at_end=False, with_opt=False,
                         comp=None, min_delta=0., reset_on_fit=True):
        cb = tracker_module.SaveModelCallback(
            monitor=monitor, comp=comp, min_delta=min_delta,
            fname=fname, every_epoch=every_epoch, at_end=at_end,
            with_opt=with_opt, reset_on_fit=reset_on_fit
        )
        # store_attr is mocked out, so manually set attributes it would have set
        cb.fname = fname
        cb.every_epoch = every_epoch
        cb.at_end = at_end
        cb.with_opt = with_opt
        return cb

    def _setup_learner(self, cb):
        """Attach a mock learner to the callback."""
        cb.learn = MagicMock()
        cb.learn.save = MagicMock(return_value='/path/to/model')
        cb.learn.load = MagicMock()
        return cb

    def test_inherits_from_tracker(self):
        """SaveModelCallback should inherit from TrackerCallback."""
        self.assertTrue(issubclass(
            tracker_module.SaveModelCallback,
            tracker_module.TrackerCallback
        ))

    def test_order_is_tracker_plus_1(self):
        """SaveModelCallback order should be TrackerCallback.order + 1."""
        self.assertEqual(
            tracker_module.SaveModelCallback.order,
            tracker_module.TrackerCallback.order + 1
        )

    def test_every_epoch_and_at_end_cannot_both_be_true(self):
        """Should raise AssertionError if both every_epoch and at_end are True."""
        with self.assertRaises(AssertionError):
            self._create_callback(every_epoch=True, at_end=True)

    def test_saves_on_improvement(self):
        """Should save model when metric improves."""
        cb = self._create_callback(monitor='valid_loss', fname='best_model')
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves from inf
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        cb.learn.save.assert_called_with('best_model', with_opt=False)

    def test_no_save_on_no_improvement(self):
        """Should not save model when metric does not improve."""
        cb = self._create_callback(monitor='valid_loss', fname='best_model')
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        cb.learn.save.reset_mock()
        # Epoch 2: worsens
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        cb.learn.save.assert_not_called()

    def test_every_epoch_saves_each_epoch(self):
        """With every_epoch=True, should save at each epoch with epoch number."""
        cb = self._create_callback(monitor='valid_loss', fname='checkpoint', every_epoch=1)
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 0
        cb.recorder.values.append([0.5])
        cb.epoch = 0
        cb.after_epoch()
        cb.learn.save.assert_called_with('checkpoint_0', with_opt=False)
        # Epoch 1
        cb.recorder.values.append([0.6])
        cb.epoch = 1
        cb.after_epoch()
        cb.learn.save.assert_called_with('checkpoint_1', with_opt=False)

    def test_every_epoch_interval(self):
        """With every_epoch=2, should save every 2nd epoch."""
        cb = self._create_callback(monitor='valid_loss', fname='checkpoint', every_epoch=2)
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 0: saves (0 % 2 == 0)
        cb.epoch = 0
        cb.recorder.values.append([0.5])
        cb.after_epoch()
        cb.learn.save.assert_called_with('checkpoint_0', with_opt=False)
        cb.learn.save.reset_mock()
        # Epoch 1: does not save (1 % 2 != 0)
        cb.epoch = 1
        cb.recorder.values.append([0.6])
        cb.after_epoch()
        cb.learn.save.assert_not_called()
        # Epoch 2: saves (2 % 2 == 0)
        cb.epoch = 2
        cb.recorder.values.append([0.7])
        cb.after_epoch()
        cb.learn.save.assert_called_with('checkpoint_2', with_opt=False)

    def test_at_end_saves_at_fit_end(self):
        """With at_end=True, should save model at end of training."""
        cb = self._create_callback(monitor='valid_loss', fname='final_model', at_end=True)
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        cb.after_fit()
        cb.learn.save.assert_called_with('final_model', with_opt=False)

    def test_not_at_end_loads_best_model(self):
        """Without at_end or every_epoch, should load best model at end."""
        cb = self._create_callback(monitor='valid_loss', fname='model')
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        cb.every_epoch = False
        cb.at_end = False
        cb.after_fit()
        cb.learn.load.assert_called_with('model', with_opt=False)

    def test_with_opt_passes_to_save(self):
        """with_opt=True should pass with_opt=True to learn.save."""
        cb = self._create_callback(monitor='valid_loss', fname='model', with_opt=True)
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        cb.learn.save.assert_called_with('model', with_opt=True)

    def test_last_saved_path_updated(self):
        """last_saved_path should be updated after save."""
        cb = self._create_callback(monitor='valid_loss', fname='model')
        cb = self._setup_learner(cb)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertIsNone(cb.last_saved_path)
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.last_saved_path, '/path/to/model')


# ----- Tests for ReduceLROnPlateau -----

class TestReduceLROnPlateau(unittest.TestCase):
    """Test ReduceLROnPlateau behavior."""

    def _create_callback(self, monitor='valid_loss', patience=2, factor=10.,
                         min_lr=0, comp=None, min_delta=0., reset_on_fit=True):
        cb = tracker_module.ReduceLROnPlateau(
            monitor=monitor, comp=comp, min_delta=min_delta,
            patience=patience, factor=factor, min_lr=min_lr,
            reset_on_fit=reset_on_fit
        )
        return cb

    def _setup_opt(self, cb, lr=0.01, n_groups=1):
        """Attach a mock optimizer to the callback."""
        cb.opt = FakeOptimizer(lr=lr, n_groups=n_groups)
        return cb

    def test_inherits_from_tracker(self):
        """ReduceLROnPlateau should inherit from TrackerCallback."""
        self.assertTrue(issubclass(
            tracker_module.ReduceLROnPlateau,
            tracker_module.TrackerCallback
        ))

    def test_order_is_tracker_plus_2(self):
        """ReduceLROnPlateau order should be TrackerCallback.order + 2."""
        self.assertEqual(
            tracker_module.ReduceLROnPlateau.order,
            tracker_module.TrackerCallback.order + 2
        )

    def test_before_fit_initializes_wait(self):
        """before_fit should set wait=0."""
        cb = self._create_callback()
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.wait, 0)

    def test_no_lr_reduction_on_improvement(self):
        """LR should not change when metric improves."""
        cb = self._create_callback(monitor='valid_loss', patience=2, factor=10.)
        cb = self._setup_opt(cb, lr=0.01)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Improving
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_lr_reduces_after_patience(self):
        """LR should reduce by factor after patience epochs without improvement."""
        cb = self._create_callback(monitor='valid_loss', patience=2, factor=10.)
        cb = self._setup_opt(cb, lr=0.01)
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
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.01)
        # Epoch 3: worsens (wait=2 >= patience=2) -> reduce LR
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.001)

    def test_wait_resets_after_lr_reduction(self):
        """Wait counter should reset to 0 after LR is reduced."""
        cb = self._create_callback(monitor='valid_loss', patience=2, factor=10.)
        cb = self._setup_opt(cb, lr=0.01)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        # Epoch 3: worsens -> reduce
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)

    def test_min_lr_floor(self):
        """LR should not go below min_lr."""
        cb = self._create_callback(monitor='valid_loss', patience=1, factor=10., min_lr=0.005)
        cb = self._setup_opt(cb, lr=0.01)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens -> would reduce to 0.001, but min_lr=0.005
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.005)

    def test_multiple_lr_reductions(self):
        """LR should reduce multiple times when plateau persists."""
        cb = self._create_callback(monitor='valid_loss', patience=1, factor=2.)
        cb = self._setup_opt(cb, lr=0.08)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens -> reduce to 0.04
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.04)
        # Epoch 3: worsens -> reduce to 0.02
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.02)

    def test_multiple_param_groups(self):
        """LR should reduce for all parameter groups."""
        cb = self._create_callback(monitor='valid_loss', patience=1, factor=10.)
        cb = self._setup_opt(cb, lr=0.01, n_groups=3)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens -> reduce all groups
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        for h in cb.opt.hypers:
            self.assertAlmostEqual(h['lr'], 0.001)

    def test_wait_resets_on_improvement(self):
        """Wait counter should reset when metric improves."""
        cb = self._create_callback(monitor='valid_loss', patience=3, factor=10.)
        cb = self._setup_opt(cb, lr=0.01)
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        # Epoch 2: worsens
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.wait, 1)
        # Epoch 3: improves - reset
        cb.recorder.values.append([0.3])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.wait, 0)
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_monitors_metric_with_greater(self):
        """Should correctly monitor metrics that use np.greater."""
        cb = self._create_callback(monitor='accuracy', patience=1, factor=5.)
        cb = self._setup_opt(cb, lr=0.1)
        cb.recorder = FakeRecorder(['accuracy'])
        cb.before_fit()
        # Epoch 1: improves
        cb.recorder.values.append([0.8])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.1)
        # Epoch 2: worsens (lower accuracy) -> reduce
        cb.recorder.values.append([0.7])
        cb.epoch = 2
        cb.after_epoch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.02)


if __name__ == '__main__':
    unittest.main()
