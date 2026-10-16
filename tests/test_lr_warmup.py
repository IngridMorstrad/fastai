"""Tests for LRWarmupCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import sys
import os
import types
import math
import unittest
import numpy as np


# --- Mock module setup ---

def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _store_attr(names=None, **kwargs):
    """Minimal store_attr that sets attributes from the calling frame."""
    import inspect
    frame = inspect.currentframe().f_back
    self = frame.f_locals.get('self')
    if self is None:
        return
    if names and isinstance(names, str):
        names_list = [n.strip() for n in names.split(',')]
        for name in names_list:
            setattr(self, name, frame.f_locals.get(name, kwargs.get(name)))


def _setup_mocks():
    """Install minimal mock modules so LRWarmupCallback can be loaded."""
    _make_module('torch', {
        'isinf': lambda x: False,
        'isnan': lambda x: False,
    })
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    class FakeCallback:
        order = 0
        run_valid = True
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    _make_module('fastai.basics', {
        'np': np,
        'math': math,
        'Callback': FakeCallback,
        'store_attr': _store_attr,
        'docs': lambda cls: cls,  # @docs is a no-op decorator
    })

    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []


def _load_lr_warmup():
    """Load LRWarmupCallback from the schedule module source."""
    schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
    schedule_path = os.path.abspath(schedule_path)

    with open(schedule_path, 'r') as f:
        source = f.read()

    # Extract only the LRWarmupCallback class and its dependencies
    mod = types.ModuleType('fastai.callback.schedule')
    mod.__file__ = schedule_path
    mod.__package__ = 'fastai.callback'
    mod.__builtins__ = __builtins__

    # Provide the necessary symbols
    mod.np = np
    mod.math = math
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.store_attr = _store_attr
    mod.docs = lambda cls: cls

    # Filter lines: skip imports from relative paths and decorators we don't need
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    # Execute only through the LRWarmupCallback definition
    # Find where it ends (next # %% marker after the class)
    code = '\n'.join(filtered_lines)

    # Find start and end of LRWarmupCallback
    start_marker = 'class LRWarmupCallback'
    start_idx = code.find(start_marker)
    if start_idx == -1:
        raise RuntimeError("LRWarmupCallback not found in schedule.py")

    # Find the next cell marker after the class
    end_marker = '\n# %%'
    end_idx = code.find(end_marker, start_idx)
    if end_idx == -1:
        class_code = code[start_idx:]
    else:
        class_code = code[start_idx:end_idx]

    # Execute just the class
    exec(compile(class_code, schedule_path, 'exec'), mod.__dict__)
    return mod.LRWarmupCallback


_setup_mocks()
LRWarmupCallback = _load_lr_warmup()


# --- Test helpers ---

class FakeOptimizer:
    """Mock optimizer with hypers list."""
    def __init__(self, lrs):
        self.hypers = [{'lr': lr} for lr in lrs]


class FakeLearner:
    """Minimal learner-like context for the callback."""
    def __init__(self, lr=0.01, num_param_groups=1):
        self.opt = FakeOptimizer([lr] * num_param_groups)
        self.training = True

    @property
    def model(self):
        return None


# --- Tests ---

class TestLRWarmupCallbackInit(unittest.TestCase):
    """Tests for LRWarmupCallback initialization."""

    def test_default_init(self):
        cb = LRWarmupCallback()
        self.assertEqual(cb.warmup_steps, 100)
        self.assertEqual(cb.start_lr, 1e-7)
        self.assertEqual(cb.schedule, 'linear')

    def test_custom_init(self):
        cb = LRWarmupCallback(warmup_steps=50, start_lr=1e-5, schedule='exponential')
        self.assertEqual(cb.warmup_steps, 50)
        self.assertEqual(cb.start_lr, 1e-5)
        self.assertEqual(cb.schedule, 'exponential')

    def test_invalid_schedule_raises(self):
        with self.assertRaises(ValueError) as ctx:
            LRWarmupCallback(schedule='cosine')
        self.assertIn('cosine', str(ctx.exception))


class TestLRWarmupLinear(unittest.TestCase):
    """Tests for linear warmup schedule."""

    def _make_cb(self, warmup_steps=10, start_lr=0.0, target_lr=0.1):
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_lr=start_lr, schedule='linear')
        cb.opt = FakeOptimizer([target_lr])
        cb.training = True
        cb.before_fit()
        return cb

    def test_before_fit_stores_target(self):
        cb = self._make_cb(target_lr=0.01)
        self.assertEqual(cb.target_lrs, [0.01])
        self.assertEqual(cb._step, 0)

    def test_first_step_uses_start_lr(self):
        cb = self._make_cb(warmup_steps=10, start_lr=0.0, target_lr=0.1)
        # At step 0, pct=0/10=0, lr = start_lr + 0*(target-start) = start_lr
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0)

    def test_midpoint_lr(self):
        cb = self._make_cb(warmup_steps=10, start_lr=0.0, target_lr=0.1)
        # Simulate 5 steps
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        # At step 5, pct=5/10=0.5, lr = 0 + 0.5*0.1 = 0.05
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)

    def test_final_step_restores_target(self):
        cb = self._make_cb(warmup_steps=5, start_lr=0.0, target_lr=0.1)
        # Run all warmup steps
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        # After warmup completes, lr should be target
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1)

    def test_after_warmup_no_change(self):
        cb = self._make_cb(warmup_steps=3, start_lr=0.0, target_lr=0.1)
        for _ in range(3):
            cb.before_batch()
            cb.after_batch()
        # Set lr to something else (simulating another scheduler)
        cb.opt.hypers[0]['lr'] = 0.05
        cb.before_batch()
        # Should not change lr because warmup is done
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)

    def test_linear_ramp_is_monotonic(self):
        cb = self._make_cb(warmup_steps=20, start_lr=0.001, target_lr=0.1)
        lrs = []
        for _ in range(20):
            cb.before_batch()
            lrs.append(cb.opt.hypers[0]['lr'])
            cb.after_batch()
        # Each lr should be >= previous
        for i in range(1, len(lrs)):
            self.assertGreaterEqual(lrs[i], lrs[i-1])


class TestLRWarmupExponential(unittest.TestCase):
    """Tests for exponential warmup schedule."""

    def _make_cb(self, warmup_steps=10, start_lr=1e-4, target_lr=0.1):
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_lr=start_lr, schedule='exponential')
        cb.opt = FakeOptimizer([target_lr])
        cb.training = True
        cb.before_fit()
        return cb

    def test_first_step_uses_start_lr(self):
        cb = self._make_cb(warmup_steps=10, start_lr=1e-4, target_lr=0.1)
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1e-4)

    def test_midpoint_lr_exponential(self):
        cb = self._make_cb(warmup_steps=10, start_lr=1e-4, target_lr=0.1)
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        # At step 5, pct=0.5, lr = 1e-4 * (0.1/1e-4)^0.5 = 1e-4 * 1000^0.5 = 1e-4 * ~31.62 = ~0.003162
        cb.before_batch()
        expected = 1e-4 * (0.1 / 1e-4) ** 0.5
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected, places=8)

    def test_exponential_ramp_is_monotonic(self):
        cb = self._make_cb(warmup_steps=20, start_lr=1e-6, target_lr=0.01)
        lrs = []
        for _ in range(20):
            cb.before_batch()
            lrs.append(cb.opt.hypers[0]['lr'])
            cb.after_batch()
        for i in range(1, len(lrs)):
            self.assertGreaterEqual(lrs[i], lrs[i-1])

    def test_final_step_restores_target(self):
        cb = self._make_cb(warmup_steps=5, start_lr=1e-4, target_lr=0.1)
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1)


class TestLRWarmupMultipleParamGroups(unittest.TestCase):
    """Tests for multiple parameter groups (discriminative LR)."""

    def test_linear_multi_group(self):
        cb = LRWarmupCallback(warmup_steps=4, start_lr=0.0, schedule='linear')
        cb.opt = FakeOptimizer([0.01, 0.1])
        cb.training = True
        cb.before_fit()
        self.assertEqual(cb.target_lrs, [0.01, 0.1])

        # Step 0: pct=0, both start at 0
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.0)
        cb.after_batch()

        # Step 2: pct=2/4=0.5
        cb.before_batch()
        cb.after_batch()
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.005)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.05)


class TestLRWarmupValidationSkip(unittest.TestCase):
    """Tests that warmup doesn't apply during validation."""

    def test_no_change_during_validation(self):
        cb = LRWarmupCallback(warmup_steps=10, start_lr=0.0, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.training = False
        cb.before_fit()

        # before_batch should do nothing when not training
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.1)

    def test_step_not_incremented_during_validation(self):
        cb = LRWarmupCallback(warmup_steps=10, start_lr=0.0, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.training = False
        cb.before_fit()

        cb.before_batch()
        cb.after_batch()
        self.assertEqual(cb._step, 0)


if __name__ == '__main__':
    unittest.main()
