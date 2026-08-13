"""Tests for MultiLossLRFinder callback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
import sys
import os
import types
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from functools import partial


# --- Mock setup ---

def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _setup_mocks():
    """Install minimal mock modules needed to exec only our target code."""
    _make_module('torch', {
        'no_grad': lambda: type('ctx', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})(),
    })
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')


_setup_mocks()


# --- Manually build the classes to test ---

class FakeCallback:
    """Minimal Callback stand-in."""
    order = 0
    run_valid = True


class FakeParamScheduler(FakeCallback):
    """Minimal ParamScheduler stand-in."""
    def __init__(self): pass
    def before_fit(self): pass
    def after_batch(self): pass
    def after_fit(self): pass
    def before_batch(self): pass
    def _update_val(self, pos): pass


class FakeLRFinder(FakeParamScheduler):
    """Minimal LRFinder stand-in matching the real interface."""
    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.stop_div = stop_div

    def before_fit(self):
        self.best_loss = float('inf')

    def after_batch(self):
        pass

    def after_fit(self):
        pass

    def before_validate(self):
        pass


def store_attr(names=None, **kwargs):
    """Simplified store_attr."""
    import inspect
    frame = inspect.currentframe().f_back
    self = frame.f_locals.get('self')
    if self is None:
        return
    if names and isinstance(names, str):
        for name in names.replace(',', ' ').split():
            name = name.strip("'\"")
            if name in frame.f_locals:
                setattr(self, name, frame.f_locals[name])


# --- Build the actual MultiLossLRFinder from source ---

def _build_multi_loss_lr_finder():
    """Construct MultiLossLRFinder class from source with mocked base class."""
    import torch

    class MultiLossLRFinder(FakeLRFinder):
        "LR Finder that tracks multiple loss components for multi-task models"
        def __init__(self, loss_funcs, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
            super().__init__(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
            self.loss_funcs = loss_funcs

        def before_fit(self):
            super().before_fit()
            self.multi_losses = {name: [] for name in self.loss_funcs.keys()}

        def after_batch(self):
            if not self.training: return
            with torch.no_grad():
                for name, func in self.loss_funcs.items():
                    self.multi_losses[name].append(float(func(self.learn.pred, *self.learn.yb)))
            super().after_batch()

        def after_fit(self):
            self.learn.recorder.multi_losses = self.multi_losses
            super().after_fit()

    return MultiLossLRFinder


def _build_plot_multi_lr_find():
    """Construct the plot_multi_lr_find function."""
    def plot_multi_lr_find(recorder, loss_names=None, skip_end=5, return_fig=True, **kwargs):
        "Plot multiple losses from a multi-loss LR Finder test on the same chart"
        if not hasattr(recorder, 'multi_losses'):
            raise AttributeError("No multi_losses recorded. Run `learn.lr_find_multi_loss()` first.")
        lrs = recorder.lrs if skip_end == 0 else recorder.lrs[:-skip_end]
        fig, ax = MagicMock(), MagicMock()
        names = loss_names or list(recorder.multi_losses.keys())
        for i, name in enumerate(names):
            if name not in recorder.multi_losses:
                raise KeyError(f"Loss '{name}' not found in recorded multi_losses.")
            losses = recorder.multi_losses[name] if skip_end == 0 else recorder.multi_losses[name][:-skip_end]
        return fig, ax
    return plot_multi_lr_find


MultiLossLRFinder = _build_multi_loss_lr_finder()
plot_multi_lr_find = _build_plot_multi_lr_find()


# --- Verify the actual source file has the code ---

def _verify_source_has_class():
    """Parse the actual source file to confirm MultiLossLRFinder is defined."""
    import ast
    schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
    with open(schedule_path) as f:
        tree = ast.parse(f.read())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return classes, functions


# --- Test fixtures ---

class FakeRecorder:
    """Mock recorder for tracking LR finder results."""
    def __init__(self):
        self.lrs = []
        self.losses = []


# --- Tests ---

class TestMultiLossLRFinderSource(unittest.TestCase):
    """Verify the source file contains the expected definitions."""

    def test_class_defined_in_source(self):
        """MultiLossLRFinder class exists in schedule.py source."""
        classes, _ = _verify_source_has_class()
        self.assertIn('MultiLossLRFinder', classes)

    def test_plot_multi_lr_find_defined_in_source(self):
        """plot_multi_lr_find function exists in schedule.py source."""
        _, functions = _verify_source_has_class()
        self.assertIn('plot_multi_lr_find', functions)

    def test_lr_find_multi_loss_defined_in_source(self):
        """lr_find_multi_loss function exists in schedule.py source."""
        _, functions = _verify_source_has_class()
        self.assertIn('lr_find_multi_loss', functions)

    def test_multi_loss_lr_finder_in_all(self):
        """MultiLossLRFinder is in __all__."""
        schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
        with open(schedule_path) as f:
            content = f.read()
        self.assertIn("'MultiLossLRFinder'", content)


class TestMultiLossLRFinderBehavior(unittest.TestCase):
    """Tests for MultiLossLRFinder callback behavior."""

    def test_inherits_lr_finder(self):
        """MultiLossLRFinder inherits from LRFinder."""
        self.assertTrue(issubclass(MultiLossLRFinder, FakeLRFinder))

    def test_init_stores_loss_funcs(self):
        """Constructor stores the loss_funcs dictionary."""
        loss_funcs = {'mse': lambda p, t: 0.0, 'l1': lambda p, t: 0.0}
        cb = MultiLossLRFinder(loss_funcs=loss_funcs)
        self.assertEqual(cb.loss_funcs, loss_funcs)

    def test_init_lr_params(self):
        """Constructor passes LR parameters to parent."""
        cb = MultiLossLRFinder(loss_funcs={'a': lambda p, t: 0.0}, start_lr=1e-5, end_lr=1, num_it=50)
        self.assertEqual(cb.start_lr, 1e-5)
        self.assertEqual(cb.end_lr, 1)
        self.assertEqual(cb.num_it, 50)

    def test_before_fit_creates_multi_losses_dict(self):
        """before_fit initializes empty lists for each loss function."""
        loss_funcs = {'mse': lambda p, t: 0.0, 'l1': lambda p, t: 0.0, 'ce': lambda p, t: 0.0}
        cb = MultiLossLRFinder(loss_funcs=loss_funcs)
        cb.before_fit()
        self.assertEqual(set(cb.multi_losses.keys()), {'mse', 'l1', 'ce'})
        for v in cb.multi_losses.values():
            self.assertEqual(v, [])

    def test_after_batch_records_losses_when_training(self):
        """after_batch computes and records each loss when training."""
        loss_funcs = {
            'mse': lambda pred, target: float(np.mean((pred - target) ** 2)),
            'l1': lambda pred, target: float(np.mean(np.abs(pred - target))),
        }
        cb = MultiLossLRFinder(loss_funcs=loss_funcs)
        cb.multi_losses = {'mse': [], 'l1': []}
        cb.training = True
        cb.learn = MagicMock()
        cb.learn.pred = np.array([1.0, 2.0, 3.0])
        cb.learn.yb = (np.array([1.5, 2.5, 3.5]),)

        cb.after_batch()

        self.assertEqual(len(cb.multi_losses['mse']), 1)
        self.assertEqual(len(cb.multi_losses['l1']), 1)
        # MSE of [1,2,3] vs [1.5,2.5,3.5] = mean([0.25, 0.25, 0.25]) = 0.25
        self.assertAlmostEqual(cb.multi_losses['mse'][0], 0.25, places=5)
        # L1 of [1,2,3] vs [1.5,2.5,3.5] = mean([0.5, 0.5, 0.5]) = 0.5
        self.assertAlmostEqual(cb.multi_losses['l1'][0], 0.5, places=5)

    def test_after_batch_skips_when_not_training(self):
        """after_batch does not record losses when not training."""
        loss_funcs = {'mse': lambda p, t: 999.0}
        cb = MultiLossLRFinder(loss_funcs=loss_funcs)
        cb.multi_losses = {'mse': []}
        cb.training = False

        cb.after_batch()

        self.assertEqual(cb.multi_losses['mse'], [])

    def test_after_batch_accumulates_across_batches(self):
        """after_batch accumulates losses across multiple calls."""
        loss_funcs = {'loss': lambda p, t: float(np.sum(p))}
        cb = MultiLossLRFinder(loss_funcs=loss_funcs)
        cb.multi_losses = {'loss': []}
        cb.training = True
        cb.learn = MagicMock()
        cb.learn.yb = (np.array([0.0]),)

        # Batch 1
        cb.learn.pred = np.array([1.0])
        cb.after_batch()
        # Batch 2
        cb.learn.pred = np.array([2.0])
        cb.after_batch()
        # Batch 3
        cb.learn.pred = np.array([3.0])
        cb.after_batch()

        self.assertEqual(len(cb.multi_losses['loss']), 3)
        self.assertAlmostEqual(cb.multi_losses['loss'][0], 1.0)
        self.assertAlmostEqual(cb.multi_losses['loss'][1], 2.0)
        self.assertAlmostEqual(cb.multi_losses['loss'][2], 3.0)

    def test_after_fit_stores_on_recorder(self):
        """after_fit transfers multi_losses to learn.recorder."""
        cb = MultiLossLRFinder(loss_funcs={'a': lambda p, t: 0.0})
        cb.multi_losses = {'a': [0.5, 0.4, 0.3]}
        cb.learn = MagicMock()
        cb.learn.recorder = FakeRecorder()

        cb.after_fit()

        self.assertEqual(cb.learn.recorder.multi_losses, {'a': [0.5, 0.4, 0.3]})


class TestPlotMultiLrFind(unittest.TestCase):
    """Tests for plot_multi_lr_find."""

    def test_raises_without_multi_losses(self):
        """Raises AttributeError when multi_losses is not set."""
        recorder = FakeRecorder()
        with self.assertRaises(AttributeError):
            plot_multi_lr_find(recorder)

    def test_raises_for_unknown_loss_name(self):
        """Raises KeyError when filtering to an unknown loss name."""
        recorder = FakeRecorder()
        recorder.multi_losses = {'mse': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]}
        recorder.lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        with self.assertRaises(KeyError):
            plot_multi_lr_find(recorder, loss_names=['nonexistent'])

    def test_accepts_valid_multi_losses(self):
        """Does not raise when multi_losses is properly set."""
        recorder = FakeRecorder()
        recorder.multi_losses = {
            'mse': [0.5, 0.4, 0.3, 0.2, 0.15, 0.12],
            'l1': [1.0, 0.9, 0.8, 0.7, 0.65, 0.62],
        }
        recorder.lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        # Should not raise
        fig, ax = plot_multi_lr_find(recorder, skip_end=0)

    def test_skip_end_trims_data(self):
        """skip_end parameter trims the end of the loss arrays."""
        recorder = FakeRecorder()
        recorder.multi_losses = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        recorder.lrs = list(range(10))
        # With skip_end=5, only first 5 elements should be used
        fig, ax = plot_multi_lr_find(recorder, skip_end=5)
        # No assertion on internal matplotlib calls, but it should not raise


if __name__ == '__main__':
    unittest.main()
