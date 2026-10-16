"""Tests for LRWarmupCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch or numpy.
"""
import math
import os
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _setup_mock_modules():
    """Install minimal mock modules so fastai.callback.warmup can be exec'd."""
    if 'torch' not in sys.modules:
        _make_module('torch', {'isinf': lambda x: False, 'isnan': lambda x: False})
    if 'torch.multiprocessing' not in sys.modules:
        _make_module('torch.multiprocessing')
    if 'torch.nn' not in sys.modules:
        _make_module('torch.nn')

    if 'fastai' not in sys.modules:
        fastai_pkg = _make_module('fastai')
        fastai_pkg.__path__ = []

    class CallbackBase:
        order = 0
        run_valid = True
        def __init__(self, **kwargs):
            pass
        def __repr__(self):
            return type(self).__name__

    if 'fastai.basics' not in sys.modules:
        _make_module('fastai.basics', {
            'Callback': CallbackBase,
            'store_attr': lambda *a, **kw: None,
            'math': math,
        })

    if 'fastai.callback' not in sys.modules:
        callback_pkg = _make_module('fastai.callback')
        callback_pkg.__path__ = []


def _load_warmup_module():
    """Load and exec the warmup module source with import lines stripped."""
    if 'fastai.callback.warmup' in sys.modules:
        del sys.modules['fastai.callback.warmup']

    warmup_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'warmup.py')
    warmup_path = os.path.abspath(warmup_path)

    mod = types.ModuleType('fastai.callback.warmup')
    mod.__file__ = warmup_path
    mod.__package__ = 'fastai.callback'

    # Populate namespace with what star-imports would provide
    mod.math = math
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.store_attr = lambda *a, **kw: None
    mod.__builtins__ = __builtins__

    with open(warmup_path, 'r') as f:
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

    exec(compile('\n'.join(filtered_lines), warmup_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.warmup'] = mod
    return mod


_setup_mock_modules()
warmup_module = _load_warmup_module()
LRWarmupCallback = warmup_module.LRWarmupCallback


# ---------------------------------------------------------------------------
# Mock learner components
# ---------------------------------------------------------------------------

class FakeHyper(dict):
    """Dict subclass representing one param group's hyperparameters."""
    pass


class FakeOptimizer:
    """Mock optimizer with hypers list."""
    def __init__(self, lrs):
        self.hypers = [FakeHyper(lr=lr) for lr in lrs]


class FakeDataLoader:
    """Mock dataloader that reports a fixed length (number of batches)."""
    def __init__(self, n_batches):
        self._n = n_batches

    def __len__(self):
        return self._n


class FakeDataLoaders:
    """Mock DataLoaders with a train loader."""
    def __init__(self, n_batches):
        self.train = FakeDataLoader(n_batches)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLRWarmupCallbackInit(unittest.TestCase):
    """Test parameter validation in __init__."""

    def test_requires_warmup_steps_or_pct(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback()  # neither specified

    def test_mutual_exclusivity(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, warmup_pct=0.1)

    def test_warmup_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=-5)

    def test_warmup_pct_range(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_pct=0.0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_pct=1.0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_pct=1.5)

    def test_start_lr_must_be_positive(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, start_lr=0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, start_lr=-0.1)

    def test_start_factor_range(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, start_factor=0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, start_factor=1.0)
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, start_factor=1.5)

    def test_schedule_validation(self):
        with self.assertRaises(ValueError):
            LRWarmupCallback(warmup_steps=10, schedule='cosine')

    def test_valid_linear(self):
        cb = LRWarmupCallback(warmup_steps=100, schedule='linear')
        self.assertEqual(cb.warmup_steps, 100)
        self.assertEqual(cb.schedule, 'linear')

    def test_valid_exp(self):
        cb = LRWarmupCallback(warmup_pct=0.1, schedule='exp')
        self.assertAlmostEqual(cb.warmup_pct, 0.1)
        self.assertEqual(cb.schedule, 'exp')


class TestLRWarmupLinearSchedule(unittest.TestCase):
    """Test linear warmup behavior."""

    def _make_cb(self, warmup_steps=10, start_factor=0.01, target_lr=0.1):
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_factor=start_factor, schedule='linear')
        # Attach mock learner state
        cb.opt = FakeOptimizer([target_lr])
        cb.dls = FakeDataLoaders(n_batches=50)
        cb.n_epoch = 5
        return cb

    def test_before_fit_sets_up_state(self):
        cb = self._make_cb(warmup_steps=10, target_lr=0.1, start_factor=0.1)
        cb.before_fit()
        self.assertEqual(cb._warmup_steps, 10)
        self.assertAlmostEqual(cb._target_lrs[0], 0.1)
        self.assertAlmostEqual(cb._start_lrs[0], 0.01)  # 0.1 * 0.1
        self.assertTrue(cb._warmup_active)

    def test_linear_warmup_progression(self):
        cb = self._make_cb(warmup_steps=4, start_factor=0.25, target_lr=1.0)
        cb.before_fit()

        # Step 0: progress=0/4=0.0 -> lr = 0.25 + (1.0-0.25)*0.0 = 0.25
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.25)

        # Step 1: progress=1/4=0.25 -> lr = 0.25 + 0.75*0.25 = 0.4375
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.4375)

        # Step 2: progress=2/4=0.5 -> lr = 0.25 + 0.75*0.5 = 0.625
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.625)

        # Step 3: progress=3/4=0.75 -> lr = 0.25 + 0.75*0.75 = 0.8125
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.8125)

        # Step 4: warmup done, target restored
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1.0)
        self.assertFalse(cb._warmup_active)

    def test_warmup_deactivates_after_completion(self):
        cb = self._make_cb(warmup_steps=2, start_factor=0.5, target_lr=0.1)
        cb.before_fit()
        cb.before_batch()  # step 0
        cb.before_batch()  # step 1
        cb.before_batch()  # step 2: completes warmup
        self.assertFalse(cb._warmup_active)

        # Further calls should not modify LR
        cb.opt.hypers[0]['lr'] = 0.05  # simulate external scheduler
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.05)

    def test_warmup_steps_clamped_to_total(self):
        """If warmup_steps > total training steps, clamp to total."""
        cb = LRWarmupCallback(warmup_steps=1000, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.dls = FakeDataLoaders(n_batches=10)
        cb.n_epoch = 2  # total = 20
        cb.before_fit()
        self.assertEqual(cb._warmup_steps, 20)


class TestLRWarmupExponentialSchedule(unittest.TestCase):
    """Test exponential warmup behavior."""

    def _make_cb(self, warmup_steps=10, start_factor=0.01, target_lr=1.0):
        cb = LRWarmupCallback(warmup_steps=warmup_steps, start_factor=start_factor, schedule='exp')
        cb.opt = FakeOptimizer([target_lr])
        cb.dls = FakeDataLoaders(n_batches=50)
        cb.n_epoch = 5
        return cb

    def test_exp_starts_at_start_lr(self):
        cb = self._make_cb(warmup_steps=10, start_factor=0.01, target_lr=1.0)
        cb.before_fit()
        cb.before_batch()  # step 0, progress=0.0
        # exp: start * exp(log(end/start) * 0.0) = start
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.01, places=5)

    def test_exp_ends_at_target_lr(self):
        cb = self._make_cb(warmup_steps=4, start_factor=0.01, target_lr=1.0)
        cb.before_fit()
        # Run through all warmup steps
        for _ in range(4):
            cb.before_batch()
        # After warmup completes
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 1.0, places=5)

    def test_exp_is_monotonically_increasing(self):
        cb = self._make_cb(warmup_steps=20, start_factor=0.01, target_lr=1.0)
        cb.before_fit()
        lrs = []
        for _ in range(20):
            cb.before_batch()
            lrs.append(cb.opt.hypers[0]['lr'])
        # Verify monotonically increasing
        for i in range(1, len(lrs)):
            self.assertGreater(lrs[i], lrs[i - 1])

    def test_exp_midpoint_value(self):
        """At progress=0.5, exponential schedule gives geometric mean."""
        cb = self._make_cb(warmup_steps=10, start_factor=0.01, target_lr=1.0)
        cb.before_fit()
        # Step 5 has progress=5/10=0.5
        for _ in range(6):  # steps 0-5
            cb.before_batch()
        # At progress=0.5: 0.01 * exp(log(1.0/0.01) * 0.5) = 0.01 * 10 = 0.1
        expected = 0.01 * math.exp(math.log(1.0 / 0.01) * 0.5)
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], expected, places=5)


class TestLRWarmupWithPct(unittest.TestCase):
    """Test warmup_pct-based configuration."""

    def test_pct_computes_correct_steps(self):
        cb = LRWarmupCallback(warmup_pct=0.1, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.dls = FakeDataLoaders(n_batches=100)
        cb.n_epoch = 10  # total = 1000 steps
        cb.before_fit()
        self.assertEqual(cb._warmup_steps, 100)

    def test_pct_rounds_up(self):
        cb = LRWarmupCallback(warmup_pct=0.03, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.dls = FakeDataLoaders(n_batches=10)
        cb.n_epoch = 1  # total = 10 steps; 3% = 0.3 -> ceil = 1
        cb.before_fit()
        self.assertEqual(cb._warmup_steps, 1)


class TestLRWarmupStartLR(unittest.TestCase):
    """Test the start_lr parameter."""

    def test_start_lr_overrides_start_factor(self):
        cb = LRWarmupCallback(warmup_steps=10, start_lr=0.001, start_factor=0.5, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.dls = FakeDataLoaders(n_batches=50)
        cb.n_epoch = 5
        cb.before_fit()
        # start_lr=0.001 should be used, not 0.1*0.5=0.05
        self.assertAlmostEqual(cb._start_lrs[0], 0.001)

    def test_start_lr_clamped_to_target(self):
        """If start_lr > target_lr, clamp to target to avoid overshoot."""
        cb = LRWarmupCallback(warmup_steps=10, start_lr=0.5, schedule='linear')
        cb.opt = FakeOptimizer([0.1])
        cb.dls = FakeDataLoaders(n_batches=50)
        cb.n_epoch = 5
        cb.before_fit()
        # start_lr=0.5 > target=0.1, so clamped to 0.1
        self.assertAlmostEqual(cb._start_lrs[0], 0.1)


class TestLRWarmupMultipleParamGroups(unittest.TestCase):
    """Test with multiple optimizer parameter groups (discriminative LRs)."""

    def test_each_group_warms_independently(self):
        cb = LRWarmupCallback(warmup_steps=4, start_factor=0.25, schedule='linear')
        cb.opt = FakeOptimizer([0.01, 0.1])  # two param groups
        cb.dls = FakeDataLoaders(n_batches=50)
        cb.n_epoch = 5
        cb.before_fit()

        # Step 0: progress=0/4=0.0
        cb.before_batch()
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.0025)   # 0.01 * 0.25
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.025)    # 0.1 * 0.25

        # Step 2: progress=2/4=0.5
        cb.before_batch()  # step 1
        cb.before_batch()  # step 2
        # group 0: 0.0025 + (0.01 - 0.0025) * 0.5 = 0.00625
        self.assertAlmostEqual(cb.opt.hypers[0]['lr'], 0.00625)
        # group 1: 0.025 + (0.1 - 0.025) * 0.5 = 0.0625
        self.assertAlmostEqual(cb.opt.hypers[1]['lr'], 0.0625)


if __name__ == '__main__':
    unittest.main()
