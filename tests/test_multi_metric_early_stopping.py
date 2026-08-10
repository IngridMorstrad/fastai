"""Tests for MultiMetricEarlyStoppingCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""

import unittest
import numpy as np

from tracker_test_utils import tracker_module, FakeRecorder, _CancelFitException


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
