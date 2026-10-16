"""Tests for LRWarmupCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
import sys
import os
import types
import math


# --- Mock infrastructure (similar to _tracker_test_helpers.py) ---

def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class FakeCallback:
    """Stand-in for fastai.basics.Callback."""
    pass


class FakeOptimizer:
    """Fake optimizer that mimics fastai's optimizer hypers/set_hyper interface."""

    def __init__(self, hypers):
        # hypers is a list of dicts, e.g. [{'lr': 0.01}, {'lr': 0.001}]
        self.hypers = [dict(h) for h in hypers]

    def set_hyper(self, k, v):
        """Set hyper across all param groups. v can be a single value or list."""
        if not isinstance(v, (list, tuple)):
            v = [v] * len(self.hypers)
        for h, val in zip(self.hypers, v):
            h[k] = val


def _setup_mocks():
    """Install minimal mock modules so fastai.callback.schedule can be exec'd."""
    import numpy as np

    _make_module('torch', {
        'isinf': lambda x: False,
        'isnan': lambda x: False,
    })
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    _make_module('fastai.basics', {
        'np': np,
        'Callback': FakeCallback,
        'store_attr': lambda *a, **kw: None,
        'float': float,
    })

    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []

    _make_module('fastai.callback.progress')
    _make_module('fastai.callback.fp16', {
        'MixedPrecision': type('MixedPrecision', (object,), {}),
    })


def _load_schedule_module():
    """Load the schedule module source with import lines stripped."""
    if 'fastai.callback.schedule' in sys.modules:
        del sys.modules['fastai.callback.schedule']

    schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
    schedule_path = os.path.abspath(schedule_path)

    mod = types.ModuleType('fastai.callback.schedule')
    mod.__file__ = schedule_path
    mod.__package__ = 'fastai.callback'

    import numpy as np
    mod.np = np
    mod.Callback = FakeCallback
    mod.store_attr = lambda *a, **kw: None
    mod.__builtins__ = __builtins__

    with open(schedule_path, 'r') as f:
        source = f.read()

    # Strip internal import lines that would fail without real fastai
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    # Remove references to decorators/functions that don't exist in mock
    clean_source = '\n'.join(filtered_lines)
    # Remove @docs, @patch, @delegates decorators
    import re
    clean_source = re.sub(r'^@docs\s*\n', '', clean_source, flags=re.MULTILINE)
    clean_source = re.sub(r'^@patch\s*\n', '', clean_source, flags=re.MULTILINE)
    clean_source = re.sub(r'^@delegates\([^)]*\)\s*\n', '', clean_source, flags=re.MULTILINE)

    # We only need to exec up to LRWarmupCallback; but let's just suppress
    # NameErrors from later code by wrapping in try/except at module level.
    # Actually, easier: just extract the LRWarmupCallback class definition.
    # But let's try the full exec approach with extra builtins.

    # Add missing names to the module namespace
    mod.docs = lambda cls: cls
    mod.patch = lambda f: f
    mod.delegates = lambda *a, **kw: (lambda f: f)
    mod.functools = __import__('functools')
    mod.math = math
    mod.tensor = lambda x: x
    mod.torch = sys.modules['torch']
    mod.L = lambda *a, **kw: list(a[0]) if a else []
    mod.is_listy = lambda x: isinstance(x, (list, tuple))
    mod.mk_class = lambda *a, **kw: None
    mod.collections = __import__('collections')
    mod.tuplify = lambda x: x if isinstance(x, (list, tuple)) else (x,)
    mod.to_np = lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
    mod.even_mults = lambda start, stop, n: [start] * n
    mod.store_attr = lambda *a, **kw: None
    mod.plt = types.ModuleType('matplotlib.pyplot')
    mod.plt.subplots = lambda *a, **kw: (None, None)

    # Fake Learner and Recorder for patches
    mod.Learner = type('Learner', (), {})
    mod.Recorder = type('Recorder', (), {})
    mod.partial = __import__('functools').partial
    mod.tempfile = __import__('tempfile')
    mod.Path = __import__('pathlib').Path
    mod.CancelFitException = Exception
    mod.CancelValidException = Exception

    try:
        exec(compile(clean_source, schedule_path, 'exec'), mod.__dict__)
    except Exception as e:
        # If exec fails on later code, that's OK as long as LRWarmupCallback loaded
        if not hasattr(mod, 'LRWarmupCallback'):
            raise RuntimeError(f"Failed to load LRWarmupCallback: {e}") from e

    sys.modules['fastai.callback.schedule'] = mod
    return mod


# --- Initialize mocks and load module ---
_setup_mocks()
schedule_module = _load_schedule_module()


class TestLRWarmupCallback(unittest.TestCase):
    """Test LRWarmupCallback behavior with mocked learner components."""

    def _create_callback(self, warmup_steps=10, start_lr=0.0, mode='linear'):
        """Create and return a LRWarmupCallback instance."""
        cb = schedule_module.LRWarmupCallback(
            warmup_steps=warmup_steps, start_lr=start_lr, mode=mode
        )
        return cb

    def _attach_opt(self, cb, lrs):
        """Attach a fake optimizer with given learning rates per param group."""
        cb.opt = FakeOptimizer([{'lr': lr} for lr in lrs])

    def test_linear_warmup_ramps_correctly(self):
        """Linear warmup should ramp LR from start_lr to target_lr over warmup_steps."""
        cb = self._create_callback(warmup_steps=5, start_lr=0.0, mode='linear')
        self._attach_opt(cb, [0.01])
        cb.before_fit()

        expected_lrs = []
        for step in range(5):
            pct = step / 5
            expected_lrs.append(0.0 + (0.01 - 0.0) * pct)

        actual_lrs = []
        for step in range(5):
            cb.before_batch()
            actual_lrs.append(cb.opt.hypers[0]['lr'])

        for exp, act in zip(expected_lrs, actual_lrs):
            self.assertAlmostEqual(exp, act, places=10)

    def test_exponential_warmup_ramps_correctly(self):
        """Exponential warmup should ramp LR from start_lr to target_lr."""
        start_lr = 1e-4
        target_lr = 0.01
        warmup_steps = 5
        cb = self._create_callback(warmup_steps=warmup_steps, start_lr=start_lr, mode='exponential')
        self._attach_opt(cb, [target_lr])
        cb.before_fit()

        expected_lrs = []
        for step in range(warmup_steps):
            pct = step / warmup_steps
            expected_lrs.append(start_lr * (target_lr / start_lr) ** pct)

        actual_lrs = []
        for step in range(warmup_steps):
            cb.before_batch()
            actual_lrs.append(cb.opt.hypers[0]['lr'])

        for exp, act in zip(expected_lrs, actual_lrs):
            self.assertAlmostEqual(exp, act, places=10)

    def test_stops_modifying_lr_after_warmup(self):
        """After warmup_steps, the callback should not modify the LR."""
        cb = self._create_callback(warmup_steps=3, start_lr=0.0, mode='linear')
        self._attach_opt(cb, [0.01])
        cb.before_fit()

        # Run through warmup
        for _ in range(3):
            cb.before_batch()

        # Set LR to something else (as if ParamScheduler changed it)
        cb.opt.set_hyper('lr', [0.05])

        # Call before_batch again - should not modify LR
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05, places=10)

    def test_default_parameters(self):
        """Default parameters (start_lr=0.0, mode='linear') should work."""
        cb = self._create_callback(warmup_steps=4)
        self._attach_opt(cb, [0.1])
        cb.before_fit()

        # Step 0: pct=0 -> lr=0.0
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0, places=10)

        # Step 1: pct=0.25 -> lr=0.025
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.025, places=10)

    def test_multiple_param_groups(self):
        """Callback should handle multiple param groups independently."""
        cb = self._create_callback(warmup_steps=4, start_lr=0.0, mode='linear')
        self._attach_opt(cb, [0.01, 0.001])
        cb.before_fit()

        # Step 0: pct=0
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0, places=10)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.0, places=10)

        # Step 1: pct=0.25
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0025, places=10)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.00025, places=10)

        # Step 2: pct=0.5
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.005, places=10)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.0005, places=10)

    def test_exponential_with_zero_start_lr(self):
        """Exponential mode with start_lr=0.0 should use epsilon and still ramp."""
        cb = self._create_callback(warmup_steps=4, start_lr=0.0, mode='exponential')
        self._attach_opt(cb, [0.01])
        cb.before_fit()

        # Step 0: pct=0, lr should be epsilon (1e-8)
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1e-8, places=15)

        # Step 2: pct=0.5, lr = 1e-8 * (0.01 / 1e-8) ** 0.5 = 1e-8 * 1e6**0.5 = 1e-8 * 1000 = 1e-5
        cb.before_batch()  # step 1
        cb.before_batch()  # step 2
        expected = 1e-8 * (0.01 / 1e-8) ** (2 / 4)
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected, places=15)

    def test_order_attribute(self):
        """LRWarmupCallback should have order=61 (after ParamScheduler at 60)."""
        cb = self._create_callback(warmup_steps=10)
        self.assertEqual(cb.order, 61)

    def test_run_valid_attribute(self):
        """LRWarmupCallback should have run_valid=False."""
        cb = self._create_callback(warmup_steps=10)
        self.assertFalse(cb.run_valid)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise AssertionError."""
        with self.assertRaises(AssertionError):
            self._create_callback(warmup_steps=10, mode='cosine')

    def test_linear_final_step_reaches_near_target(self):
        """The last warmup step should set LR very close to (but not exactly at) target."""
        cb = self._create_callback(warmup_steps=10, start_lr=0.0, mode='linear')
        self._attach_opt(cb, [0.1])
        cb.before_fit()

        for _ in range(10):
            cb.before_batch()

        # After step 9 (last warmup step), pct=9/10=0.9, lr=0.09
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.09, places=10)

    def test_exponential_warmup_multiple_groups(self):
        """Exponential warmup with multiple param groups."""
        start_lr = 1e-5
        target_lrs = [0.01, 0.001]
        cb = self._create_callback(warmup_steps=4, start_lr=start_lr, mode='exponential')
        self._attach_opt(cb, target_lrs)
        cb.before_fit()

        # Step 2: pct=2/4=0.5
        cb.before_batch()  # step 0
        cb.before_batch()  # step 1
        cb.before_batch()  # step 2

        expected_0 = start_lr * (target_lrs[0] / start_lr) ** (2 / 4)
        expected_1 = start_lr * (target_lrs[1] / start_lr) ** (2 / 4)
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected_0, places=12)
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], expected_1, places=12)


if __name__ == '__main__':
    unittest.main()
