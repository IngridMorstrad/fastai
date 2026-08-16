"""Tests for LRWarmupCallback.

These tests mock fastai/torch dependencies so the callback logic can be
validated without installing PyTorch or running actual training.
"""
import math
import sys
import os
import types
import unittest


# ---------------------------------------------------------------------------
# Minimal mock setup (same pattern as _tracker_test_helpers)
# ---------------------------------------------------------------------------

def _ensure_mock_modules():
    """Install minimal mocks so fastai.callback.warmup can be exec'd."""
    import numpy as np

    if 'fastai.basics' in sys.modules:
        return  # Already set up

    def _make_module(name, attrs=None):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    _make_module('torch', {
        'isinf': lambda x: False,
        'isnan': lambda x: False,
        'tensor': lambda x: x,
    })
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    class _MockCallback:
        order = 0
        run_valid = True
        run_train = True
        run = True
        learn = None
        _default = 'learn'

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

        def __init__(self, **kwargs):
            pass

    _make_module('fastai.basics', {
        'np': np,
        'Callback': _MockCallback,
        'store_attr': lambda *a, **kw: None,
    })

    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []


def _load_warmup_module():
    """Load the warmup module source with internal imports stripped."""
    _ensure_mock_modules()

    warmup_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'warmup.py')
    warmup_path = os.path.abspath(warmup_path)

    mod = types.ModuleType('fastai.callback.warmup')
    mod.__file__ = warmup_path
    mod.__package__ = 'fastai.callback'

    # Populate namespace with what star-import of basics would provide
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.store_attr = lambda *a, **kw: None
    mod.__builtins__ = __builtins__
    mod.math = math

    with open(warmup_path, 'r') as f:
        source = f.read()

    # Strip internal import lines that would fail without real fastai
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        elif line.strip() == 'from ..basics import *':
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), warmup_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.warmup'] = mod
    return mod


warmup_module = _load_warmup_module()
LRWarmupCallback = warmup_module.LRWarmupCallback


# ---------------------------------------------------------------------------
# Mock optimizer
# ---------------------------------------------------------------------------

class FakeOptimizer:
    """Minimal mock of fastai optimizer with hypers list."""
    def __init__(self, lrs):
        self.hypers = [{'lr': lr} for lr in lrs]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLRWarmupCallbackInit(unittest.TestCase):
    """Test initialization and parameter validation."""

    def test_default_parameters(self):
        cb = LRWarmupCallback()
        self.assertEqual(cb.warmup_steps, 1000)
        self.assertAlmostEqual(cb.start_pct, 0.01)
        self.assertEqual(cb.mode, 'linear')

    def test_custom_parameters(self):
        cb = LRWarmupCallback(warmup_steps=500, start_pct=0.1, mode='exponential')
        self.assertEqual(cb.warmup_steps, 500)
        self.assertAlmostEqual(cb.start_pct, 0.1)
        self.assertEqual(cb.mode, 'exponential')

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            LRWarmupCallback(mode='cosine')
        self.assertIn('cosine', str(ctx.exception))

    def test_start_pct_zero_raises(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(start_pct=0.0)

    def test_start_pct_negative_raises(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(start_pct=-0.1)

    def test_start_pct_above_one_raises(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(start_pct=1.5)

    def test_start_pct_one_is_valid(self):
        cb = LRWarmupCallback(start_pct=1.0)
        self.assertAlmostEqual(cb.start_pct, 1.0)

    def test_negative_warmup_steps_raises(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=-1)

    def test_zero_warmup_steps_allowed(self):
        cb = LRWarmupCallback(warmup_steps=0)
        self.assertEqual(cb.warmup_steps, 0)

    def test_order_attribute(self):
        cb = LRWarmupCallback()
        self.assertEqual(cb.order, -5)

    def test_run_valid_false(self):
        cb = LRWarmupCallback()
        self.assertFalse(cb.run_valid)


class TestLRWarmupLinear(unittest.TestCase):
    """Test linear warmup schedule."""

    def _setup(self, warmup_steps=10, start_pct=0.1, lrs=None):
        if lrs is None:
            lrs = [0.01]
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_pct=start_pct, mode='linear')
        cb.opt = FakeOptimizer(lrs)
        cb.before_fit()
        return cb

    def test_initial_lr_is_start_pct(self):
        cb = self._setup(warmup_steps=10, start_pct=0.1, lrs=[0.01])
        cb.before_batch()
        # At step 0, pct=0, mult = 0.1 + 0.9*0 = 0.1
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.001)

    def test_lr_at_midpoint(self):
        cb = self._setup(warmup_steps=10, start_pct=0.1, lrs=[0.01])
        # Advance 5 steps
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        # At step 5, pct=0.5, mult = 0.1 + 0.9*0.5 = 0.55
        expected = 0.01 * 0.55
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected)

    def test_lr_at_end_of_warmup(self):
        cb = self._setup(warmup_steps=10, start_pct=0.1, lrs=[0.01])
        # Run through all warmup steps
        for _ in range(10):
            cb.before_batch()
            cb.after_batch()
        # After warmup, LR should be exactly target
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_lr_stays_at_target_after_warmup(self):
        cb = self._setup(warmup_steps=5, start_pct=0.1, lrs=[0.01])
        # Complete warmup
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        # Additional batches should not change LR
        for _ in range(10):
            cb.before_batch()
            cb.after_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_multiple_param_groups(self):
        cb = self._setup(warmup_steps=4, start_pct=0.25, lrs=[0.01, 0.001])
        cb.before_batch()
        # At step 0: mult = 0.25
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01 * 0.25)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.001 * 0.25)

    def test_linear_schedule_is_monotonically_increasing(self):
        cb = self._setup(warmup_steps=20, start_pct=0.05, lrs=[0.1])
        prev_lr = 0.0
        for _ in range(20):
            cb.before_batch()
            current_lr = cb.opt.hypers[0]['lr']
            self.assertGreater(current_lr, prev_lr)
            prev_lr = current_lr
            cb.after_batch()


class TestLRWarmupExponential(unittest.TestCase):
    """Test exponential warmup schedule."""

    def _setup(self, warmup_steps=10, start_pct=0.01, lrs=None):
        if lrs is None:
            lrs = [0.01]
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_pct=start_pct, mode='exponential')
        cb.opt = FakeOptimizer(lrs)
        cb.before_fit()
        return cb

    def test_initial_lr_is_start_pct(self):
        cb = self._setup(warmup_steps=10, start_pct=0.01, lrs=[0.1])
        cb.before_batch()
        # At step 0, pct=0, mult = 0.01 * (100)^0 = 0.01
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1 * 0.01)

    def test_lr_at_end_of_warmup(self):
        cb = self._setup(warmup_steps=10, start_pct=0.01, lrs=[0.1])
        for _ in range(10):
            cb.before_batch()
            cb.after_batch()
        # After warmup, LR should be exactly target
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1)

    def test_exponential_midpoint(self):
        cb = self._setup(warmup_steps=10, start_pct=0.01, lrs=[1.0])
        # Advance 5 steps
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        # At step 5, pct=0.5, mult = 0.01 * (1/0.01)^0.5 = 0.01 * 10 = 0.1
        expected = 1.0 * 0.01 * math.pow(1.0 / 0.01, 0.5)
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected)

    def test_exponential_is_monotonically_increasing(self):
        cb = self._setup(warmup_steps=20, start_pct=0.01, lrs=[0.1])
        prev_lr = 0.0
        for _ in range(20):
            cb.before_batch()
            current_lr = cb.opt.hypers[0]['lr']
            self.assertGreater(current_lr, prev_lr)
            prev_lr = current_lr
            cb.after_batch()

    def test_exponential_grows_faster_at_end(self):
        """Exponential schedule should have larger increments near the end."""
        cb = self._setup(warmup_steps=20, start_pct=0.01, lrs=[1.0])
        lrs = []
        for _ in range(20):
            cb.before_batch()
            lrs.append(cb.opt.hypers[0]['lr'])
            cb.after_batch()
        # Check that increments grow
        early_delta = lrs[1] - lrs[0]
        late_delta = lrs[-1] - lrs[-2]
        self.assertGreater(late_delta, early_delta)


class TestLRWarmupEdgeCases(unittest.TestCase):
    """Test edge cases and special configurations."""

    def test_zero_warmup_steps_no_change(self):
        """With warmup_steps=0, LR should never be modified."""
        cb = LRWarmupCallback(warmup_steps=0, start_pct=0.01, mode='linear')
        cb.opt = FakeOptimizer([0.05])
        cb.before_fit()
        self.assertTrue(cb._done)
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)

    def test_start_pct_one_means_no_ramp(self):
        """With start_pct=1.0, LR is always at target."""
        cb = LRWarmupCallback(warmup_steps=10, start_pct=1.0, mode='linear')
        cb.opt = FakeOptimizer([0.01])
        cb.before_fit()
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_warmup_steps_one(self):
        """Single warmup step should ramp from start to target."""
        cb = LRWarmupCallback(warmup_steps=1, start_pct=0.1, mode='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.before_fit()
        # Step 0: pct=0, mult=0.1
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01)
        cb.after_batch()
        # Now done - LR at target
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1)

    def test_in_all_list(self):
        """LRWarmupCallback should be in __all__."""
        self.assertIn('LRWarmupCallback', warmup_module.__all__)

    def test_done_flag_set_after_warmup(self):
        cb = LRWarmupCallback(warmup_steps=3, start_pct=0.1, mode='linear')
        cb.opt = FakeOptimizer([0.01])
        cb.before_fit()
        self.assertFalse(cb._done)
        for _ in range(3):
            cb.before_batch()
            cb.after_batch()
        self.assertTrue(cb._done)

    def test_before_fit_resets_state(self):
        """Calling before_fit again should reset warmup state."""
        cb = LRWarmupCallback(warmup_steps=5, start_pct=0.1, mode='linear')
        cb.opt = FakeOptimizer([0.01])
        cb.before_fit()
        # Complete warmup
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        self.assertTrue(cb._done)
        # Reset with a new LR
        cb.opt = FakeOptimizer([0.02])
        cb.before_fit()
        self.assertFalse(cb._done)
        self.assertEqual(cb._step, 0)
        self.assertAlmostEqual(cb.target_lrs[0], 0.02)


if __name__ == '__main__':
    unittest.main()
