"""Tests for StochasticWeightAveraging callback."""

import copy
import pytest
import torch
import torch.nn as nn

from fastai.callback.swa import StochasticWeightAveraging


class _MockLearn:
    """Minimal mock of a Learner for testing callbacks."""

    def __init__(self, model):
        self.model = model


def _make_model(in_f=4, out_f=2, bias=False):
    """Create a simple Linear model with deterministic weights."""
    m = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        m.weight.fill_(1.0)
    return m


def _make_cb(decay=0.999, swa_start_epoch=1, model=None):
    """Helper: create a callback wired to a mock learner."""
    if model is None:
        model = _make_model()
    cb = StochasticWeightAveraging(decay=decay, swa_start_epoch=swa_start_epoch)
    cb.learn = _MockLearn(model)
    return cb, model


class TestStochasticWeightAveraging:
    """Unit tests for StochasticWeightAveraging."""

    # ------------------------------------------------------------------
    # Construction / defaults
    # ------------------------------------------------------------------

    def test_default_parameters(self):
        cb = StochasticWeightAveraging()
        assert cb.decay == 0.999
        assert cb.swa_start_epoch == 1

    def test_custom_parameters(self):
        cb = StochasticWeightAveraging(decay=0.9, swa_start_epoch=5)
        assert cb.decay == 0.9
        assert cb.swa_start_epoch == 5

    def test_order(self):
        cb = StochasticWeightAveraging()
        assert cb.order == 65

    def test_in_all(self):
        from fastai.callback.swa import __all__ as exports
        assert 'StochasticWeightAveraging' in exports

    def test_repr(self):
        cb = StochasticWeightAveraging()
        assert repr(cb) == 'StochasticWeightAveraging'

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def test_decay_below_zero_raises(self):
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            StochasticWeightAveraging(decay=-0.1)

    def test_decay_above_one_raises(self):
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            StochasticWeightAveraging(decay=1.1)

    def test_negative_swa_start_epoch_raises(self):
        with pytest.raises(ValueError, match="swa_start_epoch must be non-negative"):
            StochasticWeightAveraging(swa_start_epoch=-1)

    # ------------------------------------------------------------------
    # EMA update formula
    # ------------------------------------------------------------------

    def test_ema_update_formula(self):
        """After one batch the EMA should follow decay * init + (1-decay) * current."""
        model = _make_model()  # weights are all 1.0
        cb, _ = _make_cb(decay=0.8, swa_start_epoch=0, model=model)

        # before_fit snapshots the initial weights
        cb.before_fit()
        init_state = copy.deepcopy(cb._ema_state)

        # Shift model weights so they differ from the snapshot
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))  # now all 2.0

        # Now run after_batch
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        # Verify EMA = 0.8 * 1.0 + 0.2 * 2.0 = 1.2
        for key in init_state:
            expected = 0.8 * init_state[key] + 0.2 * model.state_dict()[key]
            assert torch.allclose(cb._ema_state[key], expected, atol=1e-7), \
                f"EMA mismatch for {key}"

    def test_multiple_ema_updates(self):
        """EMA converges toward the model weights over many updates."""
        model = _make_model(2, 1)
        cb, _ = _make_cb(decay=0.5, swa_start_epoch=0, model=model)
        cb.before_fit()
        cb.epoch = 0
        cb.training = True

        # Shift model weights to a constant and update EMA many times
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(10.0)

        for _ in range(50):
            cb.after_batch()

        # After many steps with decay=0.5, EMA should be very close to 10.0
        for key, val in cb._ema_state.items():
            assert torch.allclose(val, torch.full_like(val, 10.0), atol=1e-5)

    # ------------------------------------------------------------------
    # swa_start_epoch gating
    # ------------------------------------------------------------------

    def test_no_update_before_start_epoch(self):
        """EMA should not change before swa_start_epoch is reached."""
        model = _make_model(3, 1)
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=5, model=model)
        cb.before_fit()
        init_state = copy.deepcopy(cb._ema_state)

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 100.0)

        # Simulate batches at epoch 0 through 4
        cb.training = True
        for epoch in range(5):
            cb.epoch = epoch
            cb.after_batch()

        # EMA should still equal the initial snapshot
        for key in init_state:
            assert torch.equal(cb._ema_state[key], init_state[key]), \
                f"EMA changed before swa_start_epoch for {key}"

    def test_update_after_start_epoch(self):
        """EMA should update once swa_start_epoch is reached."""
        model = _make_model(3, 1)
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=2, model=model)
        cb.before_fit()
        init_state = copy.deepcopy(cb._ema_state)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 5.0)

        cb.training = True
        cb.epoch = 2  # at start epoch
        cb.after_batch()

        # EMA should have moved away from initial
        any_changed = any(
            not torch.equal(cb._ema_state[k], init_state[k])
            for k in init_state
        )
        assert any_changed, "EMA did not update at swa_start_epoch"

    # ------------------------------------------------------------------
    # Weight swapping during validation
    # ------------------------------------------------------------------

    def test_before_validate_swaps_to_ema(self):
        """Model should hold EMA weights during validation."""
        model = _make_model()
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=0, model=model)
        cb.before_fit()

        # Mutate model weights so EMA != training weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 3.0)
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        ema_snapshot = {k: v.clone() for k, v in cb._ema_state.items()}

        cb.before_validate()

        for key, val in model.state_dict().items():
            assert torch.allclose(val, ema_snapshot[key], atol=1e-7), \
                f"Model did not hold EMA weights after before_validate for {key}"

    def test_after_validate_restores_training_weights(self):
        """Training weights are restored after validation."""
        model = _make_model()
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=0, model=model)
        cb.before_fit()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        training_snapshot = copy.deepcopy(model.state_dict())

        cb.before_validate()
        cb.after_validate()

        for key, val in model.state_dict().items():
            assert torch.allclose(val, training_snapshot[key], atol=1e-7), \
                f"Training weights not restored after after_validate for {key}"

    # ------------------------------------------------------------------
    # after_fit behaviour
    # ------------------------------------------------------------------

    def test_after_fit_loads_ema_permanently(self):
        """After training, the model should hold the EMA weights."""
        model = _make_model()
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=0, model=model)
        cb.before_fit()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 2.0)
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        ema_snapshot = {k: v.clone() for k, v in cb._ema_state.items()}

        # Restore training weights first (simulates normal after_validate)
        cb.before_validate()
        cb.after_validate()

        cb.after_fit()

        for key, val in model.state_dict().items():
            assert torch.allclose(val, ema_snapshot[key], atol=1e-7), \
                f"after_fit did not load EMA weights permanently for {key}"

    # ------------------------------------------------------------------
    # Edge cases: decay boundaries
    # ------------------------------------------------------------------

    def test_decay_zero_ema_equals_current(self):
        """With decay=0 the EMA should equal the current weights after one update."""
        model = _make_model()
        cb, _ = _make_cb(decay=0.0, swa_start_epoch=0, model=model)
        cb.before_fit()

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(42.0)
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        for key, val in cb._ema_state.items():
            assert torch.allclose(val, model.state_dict()[key], atol=1e-7), \
                "decay=0 should make EMA equal to current weights"

    def test_decay_one_ema_never_updates(self):
        """With decay=1 the EMA should remain at the initial snapshot."""
        model = _make_model()
        cb, _ = _make_cb(decay=1.0, swa_start_epoch=0, model=model)
        cb.before_fit()
        init_state = copy.deepcopy(cb._ema_state)

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        cb.epoch = 0
        cb.training = True
        for _ in range(10):
            cb.after_batch()

        for key in init_state:
            assert torch.equal(cb._ema_state[key], init_state[key]), \
                "decay=1 should leave EMA at the initial snapshot"

    # ------------------------------------------------------------------
    # Non-floating-point buffer handling
    # ------------------------------------------------------------------

    def test_non_float_buffers_copied_not_averaged(self):
        """Integer buffers (e.g. num_batches_tracked) should be copied, not averaged."""
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2))
        # Deterministic init
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[0].bias.fill_(0.0)
            model[1].weight.fill_(0.5)
            model[1].bias.fill_(0.0)
        cb, _ = _make_cb(decay=0.9, swa_start_epoch=0, model=model)
        cb.before_fit()

        # Run a forward pass so BN updates num_batches_tracked
        model.train()
        x = torch.randn(8, 4)
        _ = model(x)

        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        # num_batches_tracked is int64 -- should be copied from current model
        model_nbt = model.state_dict()['0.num_batches_tracked']
        ema_nbt = cb._ema_state['0.num_batches_tracked']
        assert torch.equal(ema_nbt, model_nbt), \
            "Non-float buffer should be copied directly from the model"

    # ------------------------------------------------------------------
    # before_fit reinitializes state
    # ------------------------------------------------------------------

    def test_before_fit_reinitializes_ema(self):
        """Calling before_fit should reset EMA to a fresh snapshot."""
        model = _make_model(2, 1)
        cb, _ = _make_cb(decay=0.5, swa_start_epoch=0, model=model)
        cb.before_fit()

        # Update EMA
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(100.0)
        cb.epoch = 0
        cb.training = True
        cb.after_batch()

        # A second before_fit should reset
        cb.before_fit()

        for key, val in cb._ema_state.items():
            assert torch.allclose(val, model.state_dict()[key], atol=1e-7), \
                "before_fit should reinitialize EMA to current model state"
