"""Tests for CheckpointAveragingCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch. Mock infrastructure is shared
via tests/_tracker_mock.py.
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from _tracker_mock import tracker_module, CancelFitException, FakeRecorder


# ----- Helpers -----

class FakeModel:
    """Mock model with state_dict and load_state_dict."""
    def __init__(self, state_dict=None):
        self._state_dict = state_dict or {'layer1.weight': np.array([1.0, 2.0]), 'layer1.bias': np.array([0.5])}

    def state_dict(self):
        return self._state_dict.copy()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class FakeLearner:
    """Mock learner."""
    def __init__(self):
        self.model = FakeModel()
        self.save = MagicMock()


# ----- Tests -----

class TestCheckpointAveragingCallback(unittest.TestCase):
    """Test CheckpointAveragingCallback behavior with mocked learner components."""

    def _create_callback(self, monitor='valid_loss', comp=None, min_delta=0.,
                         k=5, save_averaged=False, fname='averaged_model', reset_on_fit=True):
        """Create a callback instance with given parameters."""
        cb = tracker_module.CheckpointAveragingCallback(
            monitor=monitor, comp=comp, min_delta=min_delta,
            k=k, save_averaged=save_averaged, fname=fname, reset_on_fit=reset_on_fit
        )
        return cb

    def _setup_callback(self, cb, metric_names=None, model_state=None):
        """Attach mocked recorder and model to callback."""
        if metric_names is None:
            metric_names = ['valid_loss']
        cb.recorder = FakeRecorder(metric_names)
        cb.model = FakeModel(model_state)
        cb.learn = FakeLearner()
        cb.learn.model = cb.model
        return cb

    def test_default_parameters(self):
        """Test default parameter values."""
        cb = self._create_callback()
        self.assertEqual(cb.monitor, 'valid_loss')
        self.assertEqual(cb.comp, np.less)
        self.assertEqual(cb.min_delta, 0.)
        self.assertEqual(cb.k, 5)
        self.assertEqual(cb.save_averaged, False)
        self.assertEqual(cb.fname, 'averaged_model')

    def test_comp_inferred_for_metric(self):
        """Monitor without 'loss' or 'error' should use np.greater."""
        cb = self._create_callback(monitor='accuracy')
        self.assertEqual(cb.comp, np.greater)

    def test_comp_inferred_for_loss(self):
        """Monitor with 'loss' should use np.less."""
        cb = self._create_callback(monitor='valid_loss')
        self.assertEqual(cb.comp, np.less)

    def test_comp_inferred_for_error(self):
        """Monitor with 'error' should use np.less."""
        cb = self._create_callback(monitor='error_rate')
        self.assertEqual(cb.comp, np.less)

    def test_custom_comp(self):
        """Custom comp should override auto-detection."""
        cb = self._create_callback(monitor='valid_loss', comp=np.greater)
        self.assertEqual(cb.comp, np.greater)

    def test_before_fit_initializes_top_k(self):
        """before_fit should initialize top_k as empty list."""
        cb = self._create_callback()
        cb = self._setup_callback(cb)
        cb.before_fit()
        self.assertEqual(cb.top_k, [])

    def test_order(self):
        """Order should be TrackerCallback.order + 2."""
        cb = self._create_callback()
        self.assertEqual(cb.order, tracker_module.TrackerCallback.order + 2)

    def test_top_k_tracking_with_k3(self):
        """Should keep only top-3 checkpoints when k=3."""
        cb = self._create_callback(k=3, monitor='valid_loss')
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        losses = [0.5, 0.4, 0.3, 0.6, 0.7]
        for i, loss in enumerate(losses):
            cb.model._state_dict = {'w': np.array([float(i)])}
            cb.recorder.values.append([loss])
            cb.after_epoch()

        self.assertEqual(len(cb.top_k), 3)
        stored_losses = [score for score, _ in cb.top_k]
        self.assertAlmostEqual(stored_losses[0], 0.3)
        self.assertAlmostEqual(stored_losses[1], 0.4)
        self.assertAlmostEqual(stored_losses[2], 0.5)

    def test_averaging_logic_correct_mean(self):
        """Averaging should produce correct mean weights."""
        cb = self._create_callback(k=3, monitor='valid_loss')
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0, 2.0])})
        cb.before_fit()

        weights = [
            np.array([1.0, 4.0]),
            np.array([2.0, 5.0]),
            np.array([3.0, 6.0]),
        ]
        losses = [0.3, 0.2, 0.1]
        for w, loss in zip(weights, losses):
            cb.model._state_dict = {'w': w.copy()}
            cb.recorder.values.append([loss])
            cb.after_epoch()

        cb.after_fit()

        expected = np.array([2.0, 5.0])
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], expected)

    def test_fewer_epochs_than_k(self):
        """Should work correctly when fewer epochs than k."""
        cb = self._create_callback(k=5, monitor='valid_loss')
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        weights = [np.array([2.0]), np.array([4.0])]
        losses = [0.5, 0.3]
        for w, loss in zip(weights, losses):
            cb.model._state_dict = {'w': w.copy()}
            cb.recorder.values.append([loss])
            cb.after_epoch()

        self.assertEqual(len(cb.top_k), 2)
        cb.after_fit()

        expected = np.array([3.0])
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], expected)

    def test_save_averaged_triggers_save(self):
        """save_averaged=True should call learn.save with fname."""
        cb = self._create_callback(k=3, save_averaged=True, fname='my_avg_model')
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        cb.model._state_dict = {'w': np.array([2.0])}
        cb.recorder.values.append([0.5])
        cb.after_epoch()

        cb.after_fit()
        cb.learn.save.assert_called_once_with('my_avg_model')

    def test_save_averaged_false_does_not_save(self):
        """save_averaged=False should not call learn.save."""
        cb = self._create_callback(k=3, save_averaged=False)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        cb.model._state_dict = {'w': np.array([2.0])}
        cb.recorder.values.append([0.5])
        cb.after_epoch()

        cb.after_fit()
        cb.learn.save.assert_not_called()

    def test_only_best_checkpoints_kept(self):
        """When more than k epochs improve, only the top-k best are kept."""
        cb = self._create_callback(k=2, monitor='valid_loss')
        cb = self._setup_callback(cb, model_state={'w': np.array([0.0])})
        cb.before_fit()

        losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        for i, loss in enumerate(losses):
            cb.model._state_dict = {'w': np.array([float(i + 1)])}
            cb.recorder.values.append([loss])
            cb.after_epoch()

        self.assertEqual(len(cb.top_k), 2)
        stored_losses = sorted([score for score, _ in cb.top_k])
        self.assertAlmostEqual(stored_losses[0], 0.1)
        self.assertAlmostEqual(stored_losses[1], 0.2)

        cb.after_fit()
        expected = np.array([4.5])  # (5.0 + 4.0) / 2
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], expected)

    def test_works_with_np_greater_comp(self):
        """Should work with np.greater for metrics like accuracy."""
        cb = self._create_callback(k=3, monitor='accuracy', comp=np.greater)
        cb = self._setup_callback(cb, metric_names=['accuracy'], model_state={'w': np.array([0.0])})
        cb.before_fit()

        accuracies = [0.7, 0.8, 0.75, 0.9, 0.85]
        for i, acc in enumerate(accuracies):
            cb.model._state_dict = {'w': np.array([float(i + 1)])}
            cb.recorder.values.append([acc])
            cb.after_epoch()

        self.assertEqual(len(cb.top_k), 3)
        stored_accs = sorted([score for score, _ in cb.top_k], reverse=True)
        self.assertAlmostEqual(stored_accs[0], 0.9)
        self.assertAlmostEqual(stored_accs[1], 0.85)
        self.assertAlmostEqual(stored_accs[2], 0.8)

        cb.after_fit()
        expected = np.array([11.0 / 3.0])
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], expected)

    def test_empty_top_k_no_error(self):
        """after_fit with empty top_k should not error or modify model."""
        cb = self._create_callback(k=3)
        cb = self._setup_callback(cb, model_state={'w': np.array([42.0])})
        cb.before_fit()
        cb.after_fit()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([42.0]))

    def test_multiple_keys_in_state_dict(self):
        """Should correctly average all keys in state_dict."""
        cb = self._create_callback(k=2, monitor='valid_loss')
        initial_state = {'layer1.weight': np.array([1.0, 2.0]), 'layer1.bias': np.array([0.5])}
        cb = self._setup_callback(cb, model_state=initial_state)
        cb.before_fit()

        cb.model._state_dict = {'layer1.weight': np.array([2.0, 4.0]), 'layer1.bias': np.array([1.0])}
        cb.recorder.values.append([0.3])
        cb.after_epoch()

        cb.model._state_dict = {'layer1.weight': np.array([4.0, 6.0]), 'layer1.bias': np.array([2.0])}
        cb.recorder.values.append([0.2])
        cb.after_epoch()

        cb.after_fit()

        np.testing.assert_array_almost_equal(cb.model._state_dict['layer1.weight'], np.array([3.0, 5.0]))
        np.testing.assert_array_almost_equal(cb.model._state_dict['layer1.bias'], np.array([1.5]))

    def test_in_all_list(self):
        """CheckpointAveragingCallback should be in __all__."""
        self.assertIn('CheckpointAveragingCallback', tracker_module.__all__)

    def test_run_guard_prevents_deep_copies(self):
        """When self.run is False (e.g. during lr_find), after_epoch should not store checkpoints."""
        cb = self._create_callback(k=3, monitor='valid_loss')
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        cb.run = False

        cb.model._state_dict = {'w': np.array([99.0])}
        cb.recorder.values.append([0.1])
        cb.after_epoch()

        self.assertEqual(len(cb.top_k), 0)

    def test_integer_buffer_not_averaged(self):
        """Integer-typed buffers should be copied from best checkpoint, not averaged."""
        cb = self._create_callback(k=2, monitor='valid_loss')
        initial_state = {
            'weight': np.array([1.0, 2.0]),
            'num_batches_tracked': np.array([100], dtype=np.int64)
        }
        cb = self._setup_callback(cb, model_state=initial_state)
        cb.before_fit()

        cb.model._state_dict = {
            'weight': np.array([2.0, 4.0]),
            'num_batches_tracked': np.array([200], dtype=np.int64)
        }
        cb.recorder.values.append([0.2])
        cb.after_epoch()

        cb.model._state_dict = {
            'weight': np.array([4.0, 6.0]),
            'num_batches_tracked': np.array([300], dtype=np.int64)
        }
        cb.recorder.values.append([0.3])
        cb.after_epoch()

        cb.after_fit()

        np.testing.assert_array_almost_equal(cb.model._state_dict['weight'], np.array([3.0, 5.0]))
        np.testing.assert_array_equal(cb.model._state_dict['num_batches_tracked'], np.array([200]))

    def test_memory_warning_printed(self):
        """before_fit should print a memory usage warning."""
        import io
        import contextlib
        cb = self._create_callback(k=7)
        cb = self._setup_callback(cb)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        output = f.getvalue()
        self.assertIn('7', output)
        self.assertIn('checkpoint', output.lower())


if __name__ == '__main__':
    unittest.main()
