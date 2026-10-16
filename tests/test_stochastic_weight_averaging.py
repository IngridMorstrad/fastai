"""Tests for StochasticWeightAveraging callback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
import copy
import numpy as np

from _tracker_test_helpers import tracker_module


# ----- Helpers -----

class FakeModel:
    """Mock model with state_dict and load_state_dict."""
    def __init__(self, state_dict=None):
        self._state_dict = state_dict or {'weight': np.array([1.0, 2.0]), 'bias': np.array([0.5])}

    def state_dict(self):
        return copy.deepcopy(self._state_dict)

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict


class FakeLearner:
    """Mock learner exposing model and training flag."""
    def __init__(self, model=None):
        self.model = model or FakeModel()


# ----- Tests -----

class TestStochasticWeightAveraging(unittest.TestCase):
    """Test StochasticWeightAveraging callback behavior."""

    def _create_callback(self, decay=0.999, start_epoch=0, load_ema_at_end=True):
        """Create a StochasticWeightAveraging instance."""
        return tracker_module.StochasticWeightAveraging(
            decay=decay, start_epoch=start_epoch, load_ema_at_end=load_ema_at_end
        )

    def _setup_callback(self, cb, model_state=None):
        """Attach a mocked model and learner to the callback."""
        model = FakeModel(model_state)
        cb.model = model
        cb.learn = FakeLearner(model)
        cb.training = True
        cb.epoch = 0
        return cb

    def test_default_parameters(self):
        """Test default parameter values."""
        cb = self._create_callback()
        self.assertEqual(cb.decay, 0.999)
        self.assertEqual(cb.start_epoch, 0)
        self.assertTrue(cb.load_ema_at_end)

    def test_custom_parameters(self):
        """Test custom parameter values."""
        cb = self._create_callback(decay=0.99, start_epoch=5, load_ema_at_end=False)
        self.assertEqual(cb.decay, 0.99)
        self.assertEqual(cb.start_epoch, 5)
        self.assertFalse(cb.load_ema_at_end)

    def test_in_all_list(self):
        """StochasticWeightAveraging should be in __all__."""
        self.assertIn('StochasticWeightAveraging', tracker_module.__all__)

    def test_before_fit_initializes_ema(self):
        """before_fit should create ema_state as a copy of model state_dict."""
        cb = self._create_callback()
        cb = self._setup_callback(cb, model_state={'w': np.array([3.0, 4.0])})
        cb.before_fit()
        np.testing.assert_array_equal(cb.ema_state['w'], np.array([3.0, 4.0]))

    def test_before_fit_ema_is_deep_copy(self):
        """ema_state should be independent of model state_dict (deep copy)."""
        cb = self._create_callback()
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()
        # Modify model weights
        cb.model._state_dict['w'] = np.array([99.0])
        # EMA should remain unchanged
        np.testing.assert_array_equal(cb.ema_state['w'], np.array([1.0]))

    def test_after_batch_updates_ema(self):
        """after_batch should update EMA weights using decay formula."""
        decay = 0.9
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        # Simulate training batch: model weights changed to 2.0
        cb.model._state_dict = {'w': np.array([2.0])}
        cb.after_batch()

        # EMA = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected = np.array([1.1])
        np.testing.assert_array_almost_equal(cb.ema_state['w'], expected)

    def test_after_batch_multiple_updates(self):
        """Multiple after_batch calls should accumulate EMA correctly."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([0.0])})
        cb.before_fit()

        # Batch 1: model = 2.0, EMA = 0.5*0.0 + 0.5*2.0 = 1.0
        cb.model._state_dict = {'w': np.array([2.0])}
        cb.after_batch()
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0]))

        # Batch 2: model = 4.0, EMA = 0.5*1.0 + 0.5*4.0 = 2.5
        cb.model._state_dict = {'w': np.array([4.0])}
        cb.after_batch()
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([2.5]))

    def test_after_batch_skipped_during_eval(self):
        """after_batch should not update EMA when not in training mode."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        cb.training = False
        cb.model._state_dict = {'w': np.array([100.0])}
        cb.after_batch()

        # EMA should remain at initial value
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0]))

    def test_after_batch_respects_start_epoch(self):
        """after_batch should skip EMA updates before start_epoch."""
        decay = 0.5
        cb = self._create_callback(decay=decay, start_epoch=2)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        # Epoch 0 - should not update
        cb.epoch = 0
        cb.model._state_dict = {'w': np.array([10.0])}
        cb.after_batch()
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0]))

        # Epoch 1 - should not update
        cb.epoch = 1
        cb.model._state_dict = {'w': np.array([10.0])}
        cb.after_batch()
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.0]))

        # Epoch 2 - should update
        cb.epoch = 2
        cb.model._state_dict = {'w': np.array([10.0])}
        cb.after_batch()
        expected = np.array([0.5 * 1.0 + 0.5 * 10.0])  # 5.5
        np.testing.assert_array_almost_equal(cb.ema_state['w'], expected)

    def test_before_validate_swaps_ema_weights(self):
        """before_validate should load EMA weights into the model."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([2.0])})
        cb.before_fit()

        # After a batch update: EMA = 0.5*2.0 + 0.5*4.0 = 3.0
        cb.model._state_dict = {'w': np.array([4.0])}
        cb.after_batch()

        # before_validate swaps in EMA weights
        cb.before_validate()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([3.0]))

    def test_after_validate_restores_training_weights(self):
        """after_validate should restore the training weights after validation."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([2.0])})
        cb.before_fit()

        # Simulate batch: model -> 4.0, EMA -> 3.0
        cb.model._state_dict = {'w': np.array([4.0])}
        cb.after_batch()

        # Swap to EMA for validation
        cb.before_validate()
        self.assertAlmostEqual(cb.model._state_dict['w'][0], 3.0)

        # Restore training weights
        cb.after_validate()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([4.0]))

    def test_full_training_loop_simulation(self):
        """Simulate a full training loop: batch updates, validate swap, restore."""
        decay = 0.9
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        # Epoch 0, batch 1: model -> 2.0
        cb.epoch = 0
        cb.model._state_dict = {'w': np.array([2.0])}
        cb.after_batch()
        # EMA = 0.9*1.0 + 0.1*2.0 = 1.1
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.1]))

        # Epoch 0, batch 2: model -> 3.0
        cb.model._state_dict = {'w': np.array([3.0])}
        cb.after_batch()
        # EMA = 0.9*1.1 + 0.1*3.0 = 0.99 + 0.3 = 1.29
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([1.29]))

        # Validation: swap EMA in
        cb.before_validate()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([1.29]))

        # After validation: restore training weights
        cb.after_validate()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([3.0]))

    def test_after_fit_loads_ema_when_enabled(self):
        """after_fit should load EMA weights when load_ema_at_end=True."""
        decay = 0.5
        cb = self._create_callback(decay=decay, load_ema_at_end=True)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        # Batch: model -> 3.0, EMA = 0.5*1.0 + 0.5*3.0 = 2.0
        cb.model._state_dict = {'w': np.array([3.0])}
        cb.after_batch()

        cb.after_fit()
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([2.0]))

    def test_after_fit_does_not_load_ema_when_disabled(self):
        """after_fit should NOT load EMA weights when load_ema_at_end=False."""
        decay = 0.5
        cb = self._create_callback(decay=decay, load_ema_at_end=False)
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.before_fit()

        # Batch: model -> 3.0, EMA = 2.0
        cb.model._state_dict = {'w': np.array([3.0])}
        cb.after_batch()

        cb.after_fit()
        # Model should retain training weights (3.0), not EMA (2.0)
        np.testing.assert_array_almost_equal(cb.model._state_dict['w'], np.array([3.0]))

    def test_integer_buffers_not_updated_by_ema(self):
        """Integer buffers should remain unchanged during EMA updates."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        state = {
            'weight': np.array([1.0]),
            'num_batches_tracked': np.array([10], dtype=np.int64)
        }
        cb = self._setup_callback(cb, model_state=state)
        cb.before_fit()

        # Batch: model changes both
        cb.model._state_dict = {
            'weight': np.array([3.0]),
            'num_batches_tracked': np.array([20], dtype=np.int64)
        }
        cb.after_batch()

        # Float weight should be updated: 0.5*1.0 + 0.5*3.0 = 2.0
        np.testing.assert_array_almost_equal(cb.ema_state['weight'], np.array([2.0]))
        # Integer buffer should remain at initial value (10)
        np.testing.assert_array_equal(cb.ema_state['num_batches_tracked'], np.array([10]))

    def test_multiple_keys_in_state_dict(self):
        """EMA should work correctly with multiple keys."""
        decay = 0.5
        cb = self._create_callback(decay=decay)
        state = {'w1': np.array([2.0, 4.0]), 'w2': np.array([1.0])}
        cb = self._setup_callback(cb, model_state=state)
        cb.before_fit()

        cb.model._state_dict = {'w1': np.array([4.0, 8.0]), 'w2': np.array([3.0])}
        cb.after_batch()

        # w1: 0.5*[2,4] + 0.5*[4,8] = [3, 6]
        np.testing.assert_array_almost_equal(cb.ema_state['w1'], np.array([3.0, 6.0]))
        # w2: 0.5*1.0 + 0.5*3.0 = 2.0
        np.testing.assert_array_almost_equal(cb.ema_state['w2'], np.array([2.0]))

    def test_run_guard_lr_finder(self):
        """Callback should not run during lr_find."""
        cb = self._create_callback()
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.lr_finder = True  # Simulate lr_find mode
        cb.before_fit()
        self.assertFalse(cb.run)

    def test_run_guard_gather_preds(self):
        """Callback should not run during gather_preds."""
        cb = self._create_callback()
        cb = self._setup_callback(cb, model_state={'w': np.array([1.0])})
        cb.gather_preds = True  # Simulate gather_preds mode
        cb.before_fit()
        self.assertFalse(cb.run)

    def test_order_value(self):
        """Order should be 65."""
        cb = self._create_callback()
        self.assertEqual(cb.order, 65)

    def test_decay_close_to_one_produces_slow_update(self):
        """High decay (close to 1) should produce very slow updates."""
        decay = 0.999
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([0.0])})
        cb.before_fit()

        # Single batch with large weight change
        cb.model._state_dict = {'w': np.array([100.0])}
        cb.after_batch()

        # EMA = 0.999*0.0 + 0.001*100.0 = 0.1
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([0.1]))

    def test_decay_zero_produces_instant_update(self):
        """Decay of 0.0 means EMA immediately tracks current weights."""
        decay = 0.0
        cb = self._create_callback(decay=decay)
        cb = self._setup_callback(cb, model_state={'w': np.array([5.0])})
        cb.before_fit()

        cb.model._state_dict = {'w': np.array([10.0])}
        cb.after_batch()

        # EMA = 0.0*5.0 + 1.0*10.0 = 10.0
        np.testing.assert_array_almost_equal(cb.ema_state['w'], np.array([10.0]))


if __name__ == '__main__':
    unittest.main()
