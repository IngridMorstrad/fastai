"""Tests for LRWarmupCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
import sys
import os
import types
import math


def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class FakeCallback:
    """Minimal stand-in for fastai Callback base class."""
    order = 0
    run_valid = True
    def __init_subclass__(cls, **kwargs): pass


class FakeOptimizer:
    """Minimal optimizer mock with hypers list."""
    def __init__(self, lrs):
        self.hypers = [{'lr': lr} for lr in lrs]


def _load_lr_warmup_class():
    """Load LRWarmupCallback from schedule.py with mocked dependencies."""
    import numpy as np

    # Install minimal mock modules
    if 'torch' not in sys.modules:
        _make_module('torch', {
            'isinf': lambda x: False,
            'isnan': lambda x: False,
        })
    if 'torch.multiprocessing' not in sys.modules:
        _make_module('torch.multiprocessing')
    if 'torch.nn' not in sys.modules:
        _make_module('torch.nn')

    schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
    schedule_path = os.path.abspath(schedule_path)

    with open(schedule_path, 'r') as f:
        source = f.read()

    # Extract just the LRWarmupCallback class and its dependencies
    # We'll exec it with a minimal namespace
    namespace = {
        '__builtins__': __builtins__,
        'Callback': FakeCallback,
        'docs': lambda cls: cls,  # no-op decorator
        'store_attr': lambda *a, **kw: None,
        'np': np,
        'math': math,
    }

    # Extract only the LRWarmupCallback class definition
    lines = source.split('\n')
    class_start = None
    class_end = None
    for i, line in enumerate(lines):
        if 'class LRWarmupCallback' in line:
            class_start = i
        elif class_start is not None and (line.startswith('# %%') or (line.startswith('class ') and i > class_start)):
            class_end = i
            break

    if class_start is None:
        raise RuntimeError("Could not find LRWarmupCallback class in schedule.py")
    if class_end is None:
        class_end = len(lines)

    # Include the decorator line if present
    if class_start > 0 and '@docs' in lines[class_start - 1]:
        class_start -= 1

    class_source = '\n'.join(lines[class_start:class_end])
    exec(compile(class_source, schedule_path, 'exec'), namespace)
    return namespace['LRWarmupCallback']


LRWarmupCallback = _load_lr_warmup_class()


class TestLRWarmupCallback(unittest.TestCase):
    """Test LRWarmupCallback behavior with mocked learner components."""

    def _make_cb(self, warmup_steps=100, schedule='linear', start_lr=1e-7):
        cb = LRWarmupCallback(warmup_steps=warmup_steps, schedule=schedule, start_lr=start_lr)
        return cb

    def _setup_cb(self, cb, lrs):
        """Attach a fake optimizer to the callback."""
        cb.opt = FakeOptimizer(lrs)
        cb.training = True
        cb.before_fit()
        return cb

    def test_init_stores_params(self):
        cb = self._make_cb(warmup_steps=50, schedule='exponential', start_lr=1e-5)
        self.assertEqual(cb.warmup_steps, 50)
        self.assertEqual(cb.schedule, 'exponential')
        self.assertEqual(cb.start_lr, 1e-5)

    def test_init_rejects_invalid_schedule(self):
        with self.assertRaises(AssertionError):
            self._make_cb(schedule='invalid')

    def test_init_rejects_zero_steps(self):
        with self.assertRaises(AssertionError):
            self._make_cb(warmup_steps=0)

    def test_init_rejects_negative_steps(self):
        with self.assertRaises(AssertionError):
            self._make_cb(warmup_steps=-5)

    def test_before_fit_stores_target_lrs(self):
        cb = self._make_cb()
        self._setup_cb(cb, [0.01, 0.001])
        self.assertEqual(cb.target_lrs, [0.01, 0.001])
        self.assertEqual(cb.step_count, 0)

    def test_linear_warmup_step_zero(self):
        """At step 0, LR should equal start_lr."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=1e-7)
        self._setup_cb(cb, [0.01])
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1e-7, places=10)

    def test_linear_warmup_midpoint(self):
        """At step 5/10, LR should be halfway between start_lr and target."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01])
        # Advance to step 5
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.005, places=10)

    def test_linear_warmup_last_step(self):
        """At the last warmup step, LR should be near target."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01])
        # Advance to step 9
        for _ in range(9):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.009, places=10)

    def test_linear_warmup_after_completion(self):
        """After warmup completes, LR should not be modified by the callback."""
        cb = self._make_cb(warmup_steps=5, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01])
        # Complete warmup
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        # Manually set LR to something else (simulating another scheduler)
        cb.opt.hypers[0]['lr'] = 0.005
        cb.before_batch()
        # Should not have been changed
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.005)

    def test_exponential_warmup_step_zero(self):
        """At step 0, exponential warmup LR should equal start_lr."""
        cb = self._make_cb(warmup_steps=10, schedule='exponential', start_lr=1e-7)
        self._setup_cb(cb, [0.01])
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1e-7, places=12)

    def test_exponential_warmup_midpoint(self):
        """At midpoint, exponential warmup should give geometric mean of start and target."""
        cb = self._make_cb(warmup_steps=10, schedule='exponential', start_lr=1e-6)
        self._setup_cb(cb, [1e-2])
        # Advance to step 5 (50%)
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        # Geometric mean of 1e-6 and 1e-2 = 1e-4
        expected = 1e-6 * (1e-2 / 1e-6) ** 0.5
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected, places=10)

    def test_multiple_param_groups(self):
        """Warmup should apply independently to each parameter group."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01, 0.001])
        # Advance to step 5 (50%)
        for _ in range(5):
            cb.before_batch()
            cb.after_batch()
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.005, places=10)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.0005, places=10)

    def test_no_update_during_validation(self):
        """Should not modify LR when not in training mode."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01])
        cb.training = False
        cb.opt.hypers[0]['lr'] = 0.01
        cb.before_batch()
        # LR should remain unchanged
        self.assertEqual(cb.opt.hypers[0]['lr'], 0.01)

    def test_step_count_only_increments_during_training(self):
        """step_count should only increment when training=True."""
        cb = self._make_cb(warmup_steps=10, schedule='linear', start_lr=0.0)
        self._setup_cb(cb, [0.01])
        cb.training = True
        cb.after_batch()
        self.assertEqual(cb.step_count, 1)
        cb.training = False
        cb.after_batch()
        # Should still be 1
        self.assertEqual(cb.step_count, 1)

    def test_order_is_after_param_scheduler(self):
        """LRWarmupCallback should run after ParamScheduler (order 60)."""
        cb = self._make_cb()
        self.assertGreater(cb.order, 60)


if __name__ == '__main__':
    unittest.main()
