"""Tests for ShowGraphCallback.

Regression tests for the empty val_losses case, where validation is skipped
(e.g. on the first epoch) and `rec.values` yields no validation loss. Before
the fix, `after_epoch` computed `max(Tensor(val_losses))` on an empty tensor,
which raises `RuntimeError: operation does not have an identity`.
"""
from unittest.mock import MagicMock

import pytest

from fastai.callback.progress import ShowGraphCallback


class _FakeRecorder:
    """Minimal recorder exposing the attributes ShowGraphCallback reads."""
    def __init__(self, losses, values):
        self.losses = losses
        self.values = values


def _make_callback(losses, values, n_epoch=1):
    """Build a ShowGraphCallback with stubbed learner state for after_epoch."""
    cb = ShowGraphCallback()
    cb.nb_batches = [len(losses) or 1]
    cb.n_epoch = n_epoch
    recorder = _FakeRecorder(losses, values)
    progress = MagicMock()
    learn = MagicMock()
    learn.recorder = recorder
    learn.progress = progress
    cb.learn = learn
    # `self.progress` and `self.n_epoch` resolve via the Callback __getattr__
    # proxy to the learner in real usage; set them directly for the stub.
    cb.progress = progress
    return cb, progress


class TestShowGraphCallbackAfterEpoch:
    def test_empty_val_losses_does_not_raise(self):
        """Validation skipped: val_losses empty, train losses present."""
        cb, progress = _make_callback(losses=[0.9, 0.5, 0.3], values=[])
        cb.after_epoch()  # must not raise RuntimeError
        progress.mbar.update_graph.assert_called_once()
        _, _, y_bounds = progress.mbar.update_graph.call_args.args
        assert y_bounds[0] == 0
        assert float(y_bounds[1]) == pytest.approx(0.9)

    def test_both_losses_present_unchanged_behavior(self):
        """Both lists non-empty: y upper bound is the overall max."""
        cb, progress = _make_callback(losses=[0.4, 0.6], values=[[0.4, 0.8]])
        cb.after_epoch()
        _, _, y_bounds = progress.mbar.update_graph.call_args.args
        assert y_bounds[0] == 0
        assert float(y_bounds[1]) == pytest.approx(0.8)

    def test_val_loss_exceeds_train_loss(self):
        """Validation loss larger than any train loss sets the upper bound."""
        cb, progress = _make_callback(losses=[0.2, 0.1], values=[[0.1, 1.5]])
        cb.after_epoch()
        _, _, y_bounds = progress.mbar.update_graph.call_args.args
        assert float(y_bounds[1]) == pytest.approx(1.5)

    def test_no_batches_returns_early(self):
        """With no recorded batches, after_epoch returns without plotting."""
        cb, progress = _make_callback(losses=[0.5], values=[])
        cb.nb_batches = []
        cb.after_epoch()
        progress.mbar.update_graph.assert_not_called()
