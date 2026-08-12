"""Tests for MultiMetricEarlyStoppingCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""

import sys
import types
import unittest
import numpy as np


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
_make_module('torch', {'isinf': lambda x: False, 'isnan': lambda x: False})
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

# Now we can import the actual tracker module
# First, remove any cached version
if 'fastai.callback.tracker' in sys.modules:
    del sys.modules['fastai.callback.tracker']

# Patch the import mechanism for the tracker module.
# The tracker module does `from ..basics import *` and `from .progress import *`
# and `from .fp16 import MixedPrecision`.
# Since we set up the modules above, we need to make the star imports work.
# The simplest way: manually load and exec the tracker source with our namespace.

import os
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
tracker_module.store_attr = lambda *a, **kw: None
tracker_module.MixedPrecision = fp16_mod.MixedPrecision
tracker_module.__builtins__ = __builtins__

# Execute the tracker source, skipping the import lines
with open(_tracker_path, 'r') as f:
    source = f.read()

# Remove the problematic import lines
lines = source.split('\n')
filtered_lines = []
for line in lines:
    # Skip import lines that pull from fastai internals
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


# ----- Tests -----

class TestMultiMetricEarlyStoppingCallback(unittest.TestCase):
    """Test MultiMetricEarlyStoppingCallback behavior with mocked learner components."""

    def _create_callback(self, monitors, patience=2, logic='all', min_delta=0., comp=None):
        """Create callback instance."""
        cb = tracker_module.MultiMetricEarlyStoppingCallback(
            monitors=monitors, patience=patience, logic=logic,
            min_delta=min_delta, comp=comp
        )
        return cb

    def test_init_default_comp_loss(self):
        """Monitors with 'loss' in name should use np.less."""
        cb = self._create_callback(monitors=['valid_loss', 'train_loss'])
        self.assertEqual(cb.comps[0], np.less)
        self.assertEqual(cb.comps[1], np.less)

    def test_init_default_comp_metric(self):
        """Monitors without 'loss'/'error' should use np.greater."""
        cb = self._create_callback(monitors=['accuracy', 'f1_score'])
        self.assertEqual(cb.comps[0], np.greater)
        self.assertEqual(cb.comps[1], np.greater)

    def test_init_default_comp_error(self):
        """Monitors with 'error' in name should use np.less."""
        cb = self._create_callback(monitors=['error_rate'])
        self.assertEqual(cb.comps[0], np.less)

    def test_init_logic_validation(self):
        """Invalid logic should raise assertion."""
        with self.assertRaises(AssertionError):
            self._create_callback(monitors=['valid_loss'], logic='invalid')

    def test_before_fit_validates_monitors(self):
        """before_fit should assert all monitors exist in recorder metric names."""
        cb = self._create_callback(monitors=['valid_loss', 'nonexistent'])
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        with self.assertRaises(AssertionError):
            cb.before_fit()

    def test_before_fit_sets_bests(self):
        """before_fit should initialize best values correctly."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'])
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()
        self.assertEqual(cb.bests[0], float('inf'))   # loss uses np.less
        self.assertEqual(cb.bests[1], -float('inf'))  # metric uses np.greater

    def test_before_fit_initializes_waits(self):
        """before_fit should zero all wait counters."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'], patience=3)
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()
        self.assertEqual(cb.waits, [0, 0])

    def test_logic_all_no_stop_when_one_improves(self):
        """With logic='all', should NOT stop if at least one metric still improves."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'], patience=2, logic='all')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # Epoch 1: both improve
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: loss worsens, accuracy improves
        cb.recorder.values.append([0.6, 0.85])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: loss worsens again (wait=2 >= patience=2), but accuracy still improves
        cb.recorder.values.append([0.7, 0.9])
        cb.epoch = 3
        # Should NOT stop because logic='all' requires both to stagnate
        cb.after_epoch()  # No exception expected

    def test_logic_all_stop_when_all_stagnate(self):
        """With logic='all', should stop when ALL metrics stagnate beyond patience."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'], patience=2, logic='all')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # Epoch 1: both improve
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: both worsen
        cb.recorder.values.append([0.6, 0.7])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: both worsen again (patience=2 exhausted for both)
        cb.recorder.values.append([0.7, 0.6])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_logic_any_stop_when_one_stagnates(self):
        """With logic='any', should stop when ANY single metric stagnates beyond patience."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'], patience=2, logic='any')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # Epoch 1: both improve
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: loss worsens, accuracy improves
        cb.recorder.values.append([0.6, 0.85])
        cb.epoch = 2
        cb.after_epoch()

        # Epoch 3: loss worsens again (patience=2 for loss), accuracy still improves
        cb.recorder.values.append([0.7, 0.9])
        cb.epoch = 3
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_logic_any_no_stop_when_none_stagnates(self):
        """With logic='any', should NOT stop if no metric has exhausted patience."""
        cb = self._create_callback(monitors=['valid_loss', 'accuracy'], patience=2, logic='any')
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy'])
        cb.before_fit()

        # Epoch 1: both improve
        cb.recorder.values.append([0.5, 0.8])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: loss worsens (wait=1 < patience=2), accuracy improves
        cb.recorder.values.append([0.6, 0.85])
        cb.epoch = 2
        cb.after_epoch()  # No exception - loss only waited 1 epoch

    def test_patience_resets_on_improvement(self):
        """Wait counter should reset when metric improves."""
        cb = self._create_callback(monitors=['valid_loss'], patience=3, logic='any')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.waits[0], 0)

        # Epoch 2: worsens
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        cb.after_epoch()
        self.assertEqual(cb.waits[0], 1)

        # Epoch 3: worsens
        cb.recorder.values.append([0.7])
        cb.epoch = 3
        cb.after_epoch()
        self.assertEqual(cb.waits[0], 2)

        # Epoch 4: improves - wait resets
        cb.recorder.values.append([0.3])
        cb.epoch = 4
        cb.after_epoch()
        self.assertEqual(cb.waits[0], 0)

    def test_single_monitor_equivalent_to_simple_early_stopping(self):
        """Single-monitor usage should behave like regular early stopping."""
        cb = self._create_callback(monitors=['valid_loss'], patience=1, logic='any')
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: improves
        cb.recorder.values.append([0.5])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: worsens (patience=1 exhausted immediately)
        cb.recorder.values.append([0.6])
        cb.epoch = 2
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()

    def test_custom_comp(self):
        """Custom comparators should override the defaults."""
        cb = self._create_callback(
            monitors=['valid_loss'], patience=2, logic='any',
            comp=[np.greater]
        )
        self.assertEqual(cb.comps[0], np.greater)

    def test_reset_on_fit_false_preserves_bests(self):
        """With reset_on_fit=False, bests should persist across fits."""
        cb = tracker_module.MultiMetricEarlyStoppingCallback(
            monitors=['valid_loss'], patience=2, logic='any', reset_on_fit=False
        )
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()

        # Epoch 1: set a good best
        cb.recorder.values.append([0.3])
        cb.epoch = 1
        cb.after_epoch()
        self.assertEqual(cb.bests[0], 0.3)

        # Simulate a second fit call - bests should persist
        cb.recorder = FakeRecorder(['valid_loss'])
        cb.before_fit()
        self.assertEqual(cb.bests[0], 0.3)

    def test_three_monitors_all_logic(self):
        """Three monitors with logic='all' - all must stagnate to stop."""
        cb = self._create_callback(
            monitors=['valid_loss', 'accuracy', 'f1_score'],
            patience=1, logic='all'
        )
        cb.recorder = FakeRecorder(['valid_loss', 'accuracy', 'f1_score'])
        cb.before_fit()

        # Epoch 1: all improve
        cb.recorder.values.append([0.5, 0.8, 0.75])
        cb.epoch = 1
        cb.after_epoch()

        # Epoch 2: all worsen (patience=1 => all stagnate)
        cb.recorder.values.append([0.6, 0.7, 0.65])
        cb.epoch = 2
        with self.assertRaises(_CancelFitException):
            cb.after_epoch()


if __name__ == '__main__':
    unittest.main()
