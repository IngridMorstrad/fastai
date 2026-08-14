"""Tests for GradientAccumulationCallback.

These tests mock the heavy fastai/torch dependencies so the callback logic
can be validated without installing PyTorch.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import sys
import os
import types
import io
import contextlib

import numpy as np


# --- Mock infrastructure ---

class CancelBatchException(Exception):
    """Stand-in for fastai CancelBatchException."""
    pass


class CancelEpochException(Exception):
    """Stand-in for fastai CancelEpochException."""
    pass


class FakeParameter:
    """Mock model parameter with grad attribute."""
    def __init__(self, grad=None):
        self.grad = grad
        self.requires_grad = True


class FakeGradData:
    """Mock .grad.data with a mul_ method that tracks calls."""
    def __init__(self, value=1.0):
        self.value = value
        self.mul_calls = []

    def mul_(self, factor):
        self.value *= factor
        self.mul_calls.append(factor)


class FakeGrad:
    """Mock .grad with a .data attribute."""
    def __init__(self, value=1.0):
        self.data = FakeGradData(value)


class FakeOptimizer:
    """Mock optimizer with step and zero_grad."""
    def __init__(self):
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self):
        self.step_count += 1

    def zero_grad(self):
        self.zero_grad_count += 1


class FakeModel:
    """Mock model with parameters."""
    def __init__(self, params=None):
        self._params = params or [FakeParameter(grad=FakeGrad())]

    def parameters(self):
        return iter(self._params)


class FakeLearner:
    """Mock learner object."""
    def __init__(self, batch_size=8):
        self.model = FakeModel()
        self.opt = FakeOptimizer()
        self.loss_grad = 1.0
        self.yb = (np.zeros(batch_size),)


def _find_bs(yb):
    """Mock find_bs that returns the length of the first element."""
    if isinstance(yb, (list, tuple)):
        return len(yb[0])
    return len(yb)


def _setup_mocks():
    """Install mocks needed to import the gradient_accumulation module."""
    # Create mock modules
    torch_mod = types.ModuleType('torch')
    torch_mod.isinf = lambda x: False
    torch_mod.isnan = lambda x: False
    sys.modules['torch'] = torch_mod

    torch_nn = types.ModuleType('torch.nn')
    torch_nn.utils = MagicMock()
    sys.modules['torch.nn'] = torch_nn
    torch_mod.nn = torch_nn

    torch_mp = types.ModuleType('torch.multiprocessing')
    sys.modules['torch.multiprocessing'] = torch_mp

    # fastai hierarchy
    fastai_pkg = types.ModuleType('fastai')
    fastai_pkg.__path__ = []
    sys.modules['fastai'] = fastai_pkg

    fastai_basics = types.ModuleType('fastai.basics')
    fastai_basics.Callback = type('Callback', (object,), {'order': 0, 'run_valid': True})
    fastai_basics.CancelBatchException = CancelBatchException
    fastai_basics.CancelEpochException = CancelEpochException
    fastai_basics.store_attr = lambda *a, **kw: None
    fastai_basics.find_bs = _find_bs
    fastai_basics.np = np
    sys.modules['fastai.basics'] = fastai_basics

    callback_pkg = types.ModuleType('fastai.callback')
    callback_pkg.__path__ = []
    sys.modules['fastai.callback'] = callback_pkg

    fastai_cb_fp16 = types.ModuleType('fastai.callback.fp16')
    fastai_cb_fp16.MixedPrecision = type('MixedPrecision', (object,), {'order': 0})
    sys.modules['fastai.callback.fp16'] = fastai_cb_fp16

    fastai_cb_progress = types.ModuleType('fastai.callback.progress')
    sys.modules['fastai.callback.progress'] = fastai_cb_progress


def _load_gradient_accumulation_module():
    """Load and exec the gradient_accumulation module with mocked deps."""
    mod_path = os.path.join(
        os.path.dirname(__file__), '..', 'fastai', 'callback', 'gradient_accumulation.py'
    )
    mod_path = os.path.abspath(mod_path)

    mod = types.ModuleType('fastai.callback.gradient_accumulation')
    mod.__file__ = mod_path
    mod.__package__ = 'fastai.callback'

    # Inject namespace
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.CancelBatchException = CancelBatchException
    mod.CancelEpochException = CancelEpochException
    mod.store_attr = lambda *a, **kw: None
    mod.find_bs = _find_bs
    mod.math = __import__('math')
    mod.nn = sys.modules['torch.nn']
    mod.__builtins__ = __builtins__

    with open(mod_path, 'r') as f:
        source = f.read()

    # Strip internal import lines that would fail
    filtered_lines = []
    for line in source.split('\n'):
        stripped = line.lstrip()
        if stripped.startswith('from __future__'):
            filtered_lines.append(line)
        elif stripped.startswith('from ..') or (stripped.startswith('from .') and not stripped.startswith('from fastai')):
            filtered_lines.append(line[:len(line) - len(stripped)] + 'pass  # skipped import')
        elif 'import torch' in line:
            filtered_lines.append(line[:len(line) - len(stripped)] + 'pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), mod_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.gradient_accumulation'] = mod
    return mod


_setup_mocks()
ga_module = _load_gradient_accumulation_module()
GradientAccumulationCallback = ga_module.GradientAccumulationCallback


# --- Tests ---

class TestGradientAccumulationCallback(unittest.TestCase):
    """Test GradientAccumulationCallback with mocked learner."""

    def _create_callback(self, n_acc=4, max_grad_norm=None, norm_type=2.0, drop_remainder=False):
        cb = GradientAccumulationCallback(
            n_acc=n_acc, max_grad_norm=max_grad_norm,
            norm_type=norm_type, drop_remainder=drop_remainder
        )
        return cb

    def _attach_learner(self, cb, batch_size=8):
        cb.learn = FakeLearner(batch_size=batch_size)
        return cb

    def test_default_parameters(self):
        """Test default parameter values."""
        cb = self._create_callback()
        self.assertEqual(cb.n_acc, 4)
        self.assertIsNone(cb.max_grad_norm)
        self.assertEqual(cb.norm_type, 2.0)
        self.assertFalse(cb.drop_remainder)

    def test_custom_parameters(self):
        """Test custom parameter values are stored."""
        cb = self._create_callback(n_acc=8, max_grad_norm=1.0, norm_type=1.0, drop_remainder=True)
        self.assertEqual(cb.n_acc, 8)
        self.assertEqual(cb.max_grad_norm, 1.0)
        self.assertEqual(cb.norm_type, 1.0)
        self.assertTrue(cb.drop_remainder)

    def test_before_fit_initializes_state(self):
        """before_fit should initialize accumulation counters."""
        cb = self._create_callback(n_acc=4)
        cb = self._attach_learner(cb)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        self.assertEqual(cb._acc_count, 0)
        self.assertEqual(cb._total_steps, 0)
        self.assertEqual(cb._total_micro_batches, 0)
        self.assertEqual(cb._samples_since_step, 0)

    def test_before_fit_prints_config(self):
        """before_fit should print accumulation configuration."""
        cb = self._create_callback(n_acc=4)
        cb = self._attach_learner(cb)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        output = f.getvalue()
        self.assertIn('4', output)
        self.assertIn('micro-batch', output)
        self.assertIn('4x', output)

    def test_before_epoch_resets_counters(self):
        """before_epoch should reset per-epoch statistics."""
        cb = self._create_callback(n_acc=2)
        cb = self._attach_learner(cb)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb._total_steps = 5
        cb._total_micro_batches = 10
        cb.before_epoch()
        self.assertEqual(cb._total_steps, 0)
        self.assertEqual(cb._total_micro_batches, 0)

    def test_after_loss_scales_loss(self):
        """after_loss should divide loss_grad by n_acc."""
        cb = self._create_callback(n_acc=4)
        cb = self._attach_learner(cb)
        cb.learn.loss_grad = 2.0
        cb.after_loss()
        self.assertAlmostEqual(cb.learn.loss_grad, 0.5)

    def test_after_loss_scaling_with_different_n_acc(self):
        """Verify loss scaling for various n_acc values."""
        for n_acc in [1, 2, 8, 16]:
            cb = self._create_callback(n_acc=n_acc)
            cb = self._attach_learner(cb)
            cb.learn.loss_grad = 4.0
            cb.after_loss()
            self.assertAlmostEqual(cb.learn.loss_grad, 4.0 / n_acc)

    def test_before_step_skips_until_n_acc(self):
        """before_step should raise CancelBatchException until n_acc micro-batches accumulated."""
        cb = self._create_callback(n_acc=4)
        cb = self._attach_learner(cb, batch_size=8)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()

        # First 3 calls should skip (raise CancelBatchException)
        for i in range(3):
            cb.learn.loss_grad = 0.25  # already scaled by after_loss
            with self.assertRaises(CancelBatchException):
                cb.before_step()

        # 4th call should allow the step
        cb.learn.loss_grad = 0.25
        cb.before_step()  # should NOT raise

        # Verify state
        self.assertEqual(cb._acc_count, 0)  # reset after step
        self.assertEqual(cb._total_steps, 1)
        self.assertEqual(cb._total_micro_batches, 4)

    def test_before_step_restores_loss_on_skip(self):
        """When skipping, loss_grad should be restored for correct logging."""
        cb = self._create_callback(n_acc=4)
        cb = self._attach_learner(cb, batch_size=8)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()

        cb.learn.loss_grad = 0.5  # scaled loss (original was 2.0, /4)
        with self.assertRaises(CancelBatchException):
            cb.before_step()
        # loss_grad should be restored: 0.5 * 4 = 2.0
        self.assertAlmostEqual(cb.learn.loss_grad, 2.0)

    def test_full_accumulation_cycle(self):
        """Simulate a full cycle of n_acc micro-batches."""
        cb = self._create_callback(n_acc=3)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate 6 micro-batches = 2 full optimizer steps
        step_count = 0
        for i in range(6):
            cb.learn.loss_grad = 3.0
            cb.after_loss()  # scales to 1.0
            try:
                cb.before_step()
                step_count += 1
            except CancelBatchException:
                pass

        self.assertEqual(step_count, 2)
        self.assertEqual(cb._total_steps, 2)
        self.assertEqual(cb._total_micro_batches, 6)
        self.assertEqual(cb._acc_count, 0)

    def test_after_train_steps_on_remainder(self):
        """after_train should perform a step when there are leftover accumulated gradients."""
        cb = self._create_callback(n_acc=4, drop_remainder=False)
        cb = self._attach_learner(cb, batch_size=8)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate 2 micro-batches (less than n_acc=4)
        for i in range(2):
            cb.learn.loss_grad = 4.0
            cb.after_loss()  # 1.0
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        self.assertEqual(cb._acc_count, 2)
        self.assertEqual(cb._total_steps, 0)

        # Now call after_train - should force a step
        cb.after_train()

        self.assertEqual(cb.learn.opt.step_count, 1)
        self.assertEqual(cb.learn.opt.zero_grad_count, 1)
        self.assertEqual(cb._total_steps, 1)
        self.assertEqual(cb._acc_count, 0)

    def test_after_train_drops_remainder_when_configured(self):
        """after_train with drop_remainder=True should discard partial accumulation."""
        cb = self._create_callback(n_acc=4, drop_remainder=True)
        cb = self._attach_learner(cb, batch_size=8)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate 2 micro-batches
        for i in range(2):
            cb.learn.loss_grad = 4.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        self.assertEqual(cb._acc_count, 2)

        # after_train should NOT step, just zero_grad
        cb.after_train()

        self.assertEqual(cb.learn.opt.step_count, 0)
        self.assertEqual(cb.learn.opt.zero_grad_count, 1)
        self.assertEqual(cb._acc_count, 0)

    def test_after_train_no_op_when_no_remainder(self):
        """after_train should do nothing when accumulation is at zero."""
        cb = self._create_callback(n_acc=2, drop_remainder=False)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate exactly 2 micro-batches (complete cycle)
        for i in range(2):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        self.assertEqual(cb._acc_count, 0)  # clean

        # after_train should not step
        cb.after_train()
        # No additional step beyond the one from before_step
        self.assertEqual(cb._total_steps, 1)

    def test_after_epoch_prints_statistics(self):
        """after_epoch should print step and micro-batch counts."""
        cb = self._create_callback(n_acc=2)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate 4 micro-batches = 2 optimizer steps
        for i in range(4):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.after_epoch()
        output = f.getvalue()
        self.assertIn('2 optimizer steps', output)
        self.assertIn('4 micro-batches', output)

    def test_gradient_rescaling_on_partial_step(self):
        """Verify gradients are rescaled correctly for partial accumulation at end of epoch."""
        # Create model with known gradients
        grad1 = FakeGrad(1.0)
        grad2 = FakeGrad(2.0)
        param1 = FakeParameter(grad=grad1)
        param2 = FakeParameter(grad=grad2)
        model = FakeModel(params=[param1, param2])

        cb = self._create_callback(n_acc=4, drop_remainder=False)
        cb = self._attach_learner(cb, batch_size=8)
        cb.learn.model = model
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Simulate 2 micro-batches (partial: 2 out of 4)
        for i in range(2):
            cb.learn.loss_grad = 4.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        # Trigger partial step
        cb.after_train()

        # Gradients should be scaled by n_acc / acc_count = 4/2 = 2.0
        self.assertAlmostEqual(grad1.data.value, 2.0)  # 1.0 * 2.0
        self.assertAlmostEqual(grad2.data.value, 4.0)  # 2.0 * 2.0

    def test_gradient_clipping_called_on_step(self):
        """When max_grad_norm is set, clip_grad_norm_ should be called."""
        cb = self._create_callback(n_acc=2, max_grad_norm=1.0, norm_type=2.0)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # Mock nn.utils.clip_grad_norm_ directly on the module's nn reference
        mock_clip = MagicMock()
        if not hasattr(ga_module.nn, 'utils'):
            ga_module.nn.utils = MagicMock()
        ga_module.nn.utils.clip_grad_norm_ = mock_clip

        # Complete accumulation cycle
        for i in range(2):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        # clip_grad_norm_ should have been called
        mock_clip.assert_called_once()
        call_args = mock_clip.call_args
        self.assertEqual(call_args[0][1], 1.0)  # max_grad_norm
        self.assertEqual(call_args[0][2], 2.0)  # norm_type

    def test_no_gradient_clipping_when_not_configured(self):
        """When max_grad_norm is None, clip_grad_norm_ should NOT be called."""
        cb = self._create_callback(n_acc=2, max_grad_norm=None)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        mock_clip = MagicMock()
        if not hasattr(ga_module.nn, 'utils'):
            ga_module.nn.utils = MagicMock()
        ga_module.nn.utils.clip_grad_norm_ = mock_clip

        # Complete accumulation cycle
        for i in range(2):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        mock_clip.assert_not_called()

    def test_effective_batch_size_property(self):
        """effective_batch_size should return n_acc."""
        cb = self._create_callback(n_acc=8)
        self.assertEqual(cb.effective_batch_size, 8)

    def test_total_optimizer_steps_property(self):
        """total_optimizer_steps should track steps taken."""
        cb = self._create_callback(n_acc=2)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        self.assertEqual(cb.total_optimizer_steps, 0)

        # 2 micro-batches = 1 step
        for i in range(2):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        self.assertEqual(cb.total_optimizer_steps, 1)

    def test_n_acc_of_1_no_accumulation(self):
        """With n_acc=1, every micro-batch should trigger a step (no accumulation)."""
        cb = self._create_callback(n_acc=1)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # 5 micro-batches should give 5 steps
        step_count = 0
        for i in range(5):
            cb.learn.loss_grad = 1.0
            cb.after_loss()
            try:
                cb.before_step()
                step_count += 1
            except CancelBatchException:
                pass

        self.assertEqual(step_count, 5)
        self.assertEqual(cb._total_steps, 5)

    def test_run_valid_is_false(self):
        """Callback should not run during validation."""
        cb = self._create_callback()
        self.assertFalse(cb.run_valid)

    def test_order_attribute(self):
        """Order should be -4 to run early in the callback chain."""
        cb = self._create_callback()
        self.assertEqual(cb.order, -4)

    def test_in_all_list(self):
        """GradientAccumulationCallback should be in __all__."""
        self.assertIn('GradientAccumulationCallback', ga_module.__all__)

    def test_samples_tracking(self):
        """_samples_since_step should accumulate batch sizes."""
        cb = self._create_callback(n_acc=3)
        cb = self._attach_learner(cb, batch_size=16)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # First micro-batch
        cb.learn.loss_grad = 3.0
        cb.after_loss()
        try:
            cb.before_step()
        except CancelBatchException:
            pass
        self.assertEqual(cb._samples_since_step, 16)

        # Second micro-batch
        cb.learn.loss_grad = 3.0
        cb.after_loss()
        try:
            cb.before_step()
        except CancelBatchException:
            pass
        self.assertEqual(cb._samples_since_step, 32)

        # Third micro-batch triggers step, resets counter
        cb.learn.loss_grad = 3.0
        cb.after_loss()
        cb.before_step()  # should not raise
        self.assertEqual(cb._samples_since_step, 0)

    def test_multiple_epochs(self):
        """Counters should properly reset across epochs."""
        cb = self._create_callback(n_acc=2)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()

        # Epoch 1: 4 micro-batches
        cb.before_epoch()
        for i in range(4):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass
        self.assertEqual(cb._total_steps, 2)
        self.assertEqual(cb._total_micro_batches, 4)

        # Epoch 2: 6 micro-batches
        cb.before_epoch()
        self.assertEqual(cb._total_steps, 0)
        self.assertEqual(cb._total_micro_batches, 0)
        for i in range(6):
            cb.learn.loss_grad = 2.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass
        self.assertEqual(cb._total_steps, 3)
        self.assertEqual(cb._total_micro_batches, 6)

    def test_gradient_clipping_on_partial_step(self):
        """Gradient clipping should also apply on partial end-of-epoch steps."""
        cb = self._create_callback(n_acc=4, max_grad_norm=0.5, drop_remainder=False)
        cb = self._attach_learner(cb, batch_size=4)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        mock_clip = MagicMock()
        if not hasattr(ga_module.nn, 'utils'):
            ga_module.nn.utils = MagicMock()
        ga_module.nn.utils.clip_grad_norm_ = mock_clip

        # Simulate 1 micro-batch (partial)
        cb.learn.loss_grad = 4.0
        cb.after_loss()
        try:
            cb.before_step()
        except CancelBatchException:
            pass

        # Trigger partial step
        cb.after_train()

        # Clipping should have been called
        mock_clip.assert_called_once()

    def test_params_with_no_grad_skipped_in_rescale(self):
        """Parameters with grad=None should be skipped during rescaling."""
        param_with_grad = FakeParameter(grad=FakeGrad(1.0))
        param_no_grad = FakeParameter(grad=None)
        model = FakeModel(params=[param_with_grad, param_no_grad])

        cb = self._create_callback(n_acc=4, drop_remainder=False)
        cb = self._attach_learner(cb, batch_size=8)
        cb.learn.model = model
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cb.before_fit()
        cb.before_epoch()

        # 3 micro-batches (partial)
        for i in range(3):
            cb.learn.loss_grad = 4.0
            cb.after_loss()
            try:
                cb.before_step()
            except CancelBatchException:
                pass

        # Should not raise even with None grad
        cb.after_train()

        # param with grad should be rescaled by 4/3
        self.assertAlmostEqual(param_with_grad.grad.data.value, 4.0 / 3.0)


if __name__ == '__main__':
    unittest.main()
