"""Tests for LRWarmupCallback."""

import pytest

from fastai.callback.lr_warmup import LRWarmupCallback


class _MockOpt:
    """Minimal mock of a fastai optimizer exposing per-group `hypers`."""

    def __init__(self, lrs):
        self.hypers = [{'lr': lr} for lr in lrs]


class _MockLearn:
    """Minimal mock of a Learner for testing callbacks."""

    def __init__(self, lrs):
        self.opt = _MockOpt(lrs)


def _run_warmup(cb, base_lrs, n_batches):
    """Drive the callback through `before_fit` then `n_batches` of `before_batch`.

    Returns the list of lr values observed for the first param group, one per batch
    (captured after the callback set the lr for that batch).
    """
    learn = _MockLearn(base_lrs)
    cb.learn = learn
    cb.before_fit()
    observed = []
    for _ in range(n_batches):
        cb.before_batch()
        observed.append(learn.opt.hypers[0]['lr'])
    return observed, learn


class TestLRWarmupCallback:
    """Unit tests for LRWarmupCallback."""

    def test_default_parameters(self):
        """Callback initializes with correct default parameters."""
        cb = LRWarmupCallback()
        assert cb.n_warmup == 100
        assert cb.start_lr == 1e-6
        assert cb.mode == 'linear'
        assert cb.order == 50

    def test_custom_parameters(self):
        """Callback accepts custom n_warmup, start_lr and mode."""
        cb = LRWarmupCallback(n_warmup=50, start_lr=1e-4, mode='exponential')
        assert cb.n_warmup == 50
        assert cb.start_lr == 1e-4
        assert cb.mode == 'exponential'

    def test_invalid_n_warmup_zero_raises(self):
        """n_warmup of zero raises ValueError."""
        with pytest.raises(ValueError, match="n_warmup must be a positive integer"):
            LRWarmupCallback(n_warmup=0)

    def test_invalid_n_warmup_negative_raises(self):
        """Negative n_warmup raises ValueError."""
        with pytest.raises(ValueError, match="n_warmup must be a positive integer"):
            LRWarmupCallback(n_warmup=-5)

    def test_invalid_n_warmup_type_raises(self):
        """Non-integer n_warmup raises ValueError."""
        with pytest.raises(ValueError, match="n_warmup must be a positive integer"):
            LRWarmupCallback(n_warmup=10.5)

    def test_negative_start_lr_raises(self):
        """Negative start_lr raises ValueError."""
        with pytest.raises(ValueError, match="start_lr must be non-negative"):
            LRWarmupCallback(start_lr=-1e-3)

    def test_invalid_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            LRWarmupCallback(mode='quadratic')

    def test_exponential_zero_start_lr_raises(self):
        """Exponential mode requires strictly positive start_lr."""
        with pytest.raises(ValueError, match="strictly positive for exponential"):
            LRWarmupCallback(start_lr=0.0, mode='exponential')

    def test_before_fit_records_base_lrs_and_resets_step(self):
        """before_fit captures base lrs from the optimizer and resets the step counter."""
        cb = LRWarmupCallback(n_warmup=10)
        cb._step = 7
        cb.learn = _MockLearn([0.1, 0.2])
        cb.before_fit()
        assert cb._step == 0
        assert cb._base_lrs == [0.1, 0.2]

    def test_linear_ramp_start_and_end(self):
        """Linear warmup starts at start_lr and reaches base lr on the final step."""
        cb = LRWarmupCallback(n_warmup=5, start_lr=0.0, mode='linear')
        observed, _ = _run_warmup(cb, [1.0], n_batches=5)
        assert observed[0] == pytest.approx(0.0)
        assert observed[-1] == pytest.approx(1.0)

    def test_linear_ramp_is_evenly_spaced(self):
        """Linear warmup produces evenly spaced lr increments."""
        cb = LRWarmupCallback(n_warmup=5, start_lr=0.0, mode='linear')
        observed, _ = _run_warmup(cb, [1.0], n_batches=5)
        diffs = [observed[i + 1] - observed[i] for i in range(len(observed) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])
        assert observed == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_exponential_ramp_start_and_end(self):
        """Exponential warmup starts at start_lr and reaches base lr on the final step."""
        cb = LRWarmupCallback(n_warmup=5, start_lr=0.01, mode='exponential')
        observed, _ = _run_warmup(cb, [1.0], n_batches=5)
        assert observed[0] == pytest.approx(0.01)
        assert observed[-1] == pytest.approx(1.0)

    def test_exponential_ramp_constant_ratio(self):
        """Exponential warmup increases by a constant multiplicative factor per step."""
        cb = LRWarmupCallback(n_warmup=5, start_lr=0.01, mode='exponential')
        observed, _ = _run_warmup(cb, [1.0], n_batches=5)
        ratios = [observed[i + 1] / observed[i] for i in range(len(observed) - 1)]
        for r in ratios:
            assert r == pytest.approx(ratios[0])

    def test_warmup_holds_base_lr_after_completion(self):
        """After n_warmup steps the base lr is restored and held constant."""
        cb = LRWarmupCallback(n_warmup=3, start_lr=0.0, mode='linear')
        observed, _ = _run_warmup(cb, [0.5], n_batches=6)
        # Steps 0..2 are the ramp; steps 3..5 hold the base lr exactly.
        assert observed[-1] == pytest.approx(0.5)
        assert observed[-2] == pytest.approx(0.5)
        assert observed[-3] == pytest.approx(0.5)

    def test_step_counter_increments_during_warmup(self):
        """Step counter increments once per batch during the warmup phase."""
        cb = LRWarmupCallback(n_warmup=4, start_lr=0.0)
        learn = _MockLearn([0.1])
        cb.learn = learn
        cb.before_fit()
        assert cb._step == 0
        for i in range(4):
            cb.before_batch()
            assert cb._step == i + 1

    def test_step_counter_stops_at_n_warmup(self):
        """Step counter does not advance past n_warmup once warmup is finished."""
        cb = LRWarmupCallback(n_warmup=2, start_lr=0.0)
        learn = _MockLearn([0.1])
        cb.learn = learn
        cb.before_fit()
        for _ in range(5):
            cb.before_batch()
        assert cb._step == 2
        assert cb.finished is True

    def test_before_fit_resets_step_between_fits(self):
        """A second fit resets the step counter so warmup runs again."""
        cb = LRWarmupCallback(n_warmup=2, start_lr=0.0)
        learn = _MockLearn([0.1])
        cb.learn = learn
        cb.before_fit()
        for _ in range(3):
            cb.before_batch()
        assert cb.finished is True
        cb.before_fit()
        assert cb._step == 0
        assert cb.finished is False

    def test_multiple_param_groups(self):
        """Each param group is warmed relative to its own recorded base lr."""
        cb = LRWarmupCallback(n_warmup=2, start_lr=0.0, mode='linear')
        learn = _MockLearn([1.0, 0.1])
        cb.learn = learn
        cb.before_fit()
        cb.before_batch()  # step 0 -> start_lr
        assert learn.opt.hypers[0]['lr'] == pytest.approx(0.0)
        assert learn.opt.hypers[1]['lr'] == pytest.approx(0.0)
        cb.before_batch()  # step 1 -> base lr of each group
        assert learn.opt.hypers[0]['lr'] == pytest.approx(1.0)
        assert learn.opt.hypers[1]['lr'] == pytest.approx(0.1)

    def test_repr(self):
        """Callback repr reflects its configuration."""
        cb = LRWarmupCallback(n_warmup=20, start_lr=1e-5, mode='exponential')
        r = repr(cb)
        assert "LRWarmupCallback" in r
        assert "n_warmup=20" in r
        assert "exponential" in r
