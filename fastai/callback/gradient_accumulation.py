"""Gradient accumulation callback that transparently simulates larger batch sizes.

This module provides `GradientAccumulationCallback`, an enhanced gradient accumulation
wrapper that simulates larger effective batch sizes by accumulating gradients across
multiple forward passes. Designed for memory-constrained GPUs where the desired batch
size cannot fit in VRAM.

Key improvements over the basic GradientAccumulation in training.py:
- Proper loss normalization by number of accumulation steps (not batch-size ratio)
- Handles end-of-epoch partial accumulation (forces optimizer step on trailing batches)
- Tracks and reports effective batch size for transparency
- Supports optional gradient clipping integrated with the accumulation cycle
- Provides epoch-level statistics on how many optimizer steps were taken

This file is NOT managed by nbdev - it is a standalone module.
"""

from __future__ import annotations
import math


__all__ = ['GradientAccumulationCallback']


# We import from fastai.basics at runtime. When running under test mocks,
# these names are injected into the module namespace externally.
try:
    from ..basics import Callback, CancelBatchException, CancelEpochException, store_attr, find_bs
    from .fp16 import MixedPrecision
    import torch.nn as nn
except ImportError:
    pass


class GradientAccumulationCallback(Callback):
    """Transparently simulate larger batch sizes by accumulating gradients across multiple forward passes.

    This callback divides the effective batch into `n_acc` micro-batches. Gradients are
    accumulated over `n_acc` micro-batches before a single optimizer step, making the
    training behave as if the batch size were `actual_batch_size * n_acc`.

    The loss is normalized by the number of accumulation steps so that learning rate
    semantics remain consistent regardless of the accumulation factor.

    At the end of each training epoch, if there are leftover accumulated gradients
    (fewer than `n_acc` micro-batches since the last step), a final optimizer step is
    performed with appropriately scaled gradients to avoid discarding work.

    Parameters
    ----------
    n_acc : int
        Number of micro-batches to accumulate before performing an optimizer step.
        The effective batch size is `micro_batch_size * n_acc`.
    max_grad_norm : float or None
        If set, clip gradient norms to this value before each optimizer step.
        Uses L2 norm by default.
    norm_type : float
        The type of norm used for gradient clipping (default: 2.0 for L2 norm).
    drop_remainder : bool
        If True, discard leftover accumulated gradients at the end of an epoch
        instead of performing a partial step. Default is False (always step).
    """

    # Run before mixed precision but after TrainEvalCallback; skip during validation
    order = -4
    run_valid = False

    def __init__(self, n_acc=2, max_grad_norm=None, norm_type=2.0, drop_remainder=False):
        self.n_acc = n_acc
        self.max_grad_norm = max_grad_norm
        self.norm_type = norm_type
        self.drop_remainder = drop_remainder

    def before_fit(self):
        """Initialize accumulation state and report effective batch size configuration."""
        self._acc_count = 0  # number of micro-batches accumulated since last step
        self._total_steps = 0  # optimizer steps taken this epoch
        self._total_micro_batches = 0  # micro-batches processed this epoch
        self._samples_since_step = 0  # samples accumulated since last optimizer step
        print(
            f'GradientAccumulationCallback: accumulating {self.n_acc} micro-batches per optimizer step. '
            f'Effective batch size multiplier: {self.n_acc}x'
        )

    def before_epoch(self):
        """Reset per-epoch counters."""
        self._total_steps = 0
        self._total_micro_batches = 0

    def after_loss(self):
        """Scale loss by the accumulation factor for proper gradient normalization.

        Dividing the loss by n_acc before backward ensures that the accumulated
        gradients have the same magnitude as if we had processed one large batch.
        """
        self.learn.loss_grad = self.learn.loss_grad / self.n_acc

    def before_step(self):
        """Skip the optimizer step until we have accumulated enough micro-batches.

        Also tracks sample counts for reporting purposes.
        """
        bs = find_bs(self.learn.yb)
        self._acc_count += 1
        self._total_micro_batches += 1
        self._samples_since_step += bs

        if self._acc_count < self.n_acc:
            # Undo loss scaling for the logged loss (so metrics display correctly)
            self.learn.loss_grad = self.learn.loss_grad * self.n_acc
            raise CancelBatchException()  # skip step and zero_grad
        else:
            # We have accumulated enough - allow the step to proceed
            self._perform_clip()
            self._acc_count = 0
            self._total_steps += 1
            self._samples_since_step = 0

    def after_train(self):
        """Handle leftover accumulated gradients at the end of the training epoch.

        If there are unstepped gradients remaining and drop_remainder is False,
        perform one final optimizer step. The gradients are already properly scaled
        from the after_loss calls, but we need to account for the fact that fewer
        than n_acc micro-batches contributed.
        """
        if self._acc_count > 0 and not self.drop_remainder:
            # Rescale gradients: they were divided by n_acc in after_loss, but only
            # _acc_count micro-batches contributed. Multiply by n_acc/_acc_count to correct.
            self._rescale_gradients(self.n_acc / self._acc_count)
            self._perform_clip()
            self.learn.opt.step()
            self.learn.opt.zero_grad()
            self._total_steps += 1
            self._acc_count = 0
            self._samples_since_step = 0
        elif self._acc_count > 0 and self.drop_remainder:
            # Discard partial accumulation
            self.learn.opt.zero_grad()
            self._acc_count = 0
            self._samples_since_step = 0

    def after_epoch(self):
        """Report accumulation statistics for the epoch."""
        print(
            f'  gradient_accumulation: {self._total_steps} optimizer steps '
            f'from {self._total_micro_batches} micro-batches (target n_acc={self.n_acc})'
        )

    def _perform_clip(self):
        """Apply gradient clipping if max_grad_norm is configured."""
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.learn.model.parameters(),
                self.max_grad_norm,
                self.norm_type
            )

    def _rescale_gradients(self, scale_factor):
        """Multiply all parameter gradients by a scale factor.

        Used to correct gradient magnitudes when performing a partial step at
        end-of-epoch with fewer than n_acc accumulated micro-batches.
        """
        for p in self.learn.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)

    @property
    def effective_batch_size(self):
        """Return the effective batch size (actual_bs * n_acc). Only valid during training."""
        return self.n_acc

    @property
    def total_optimizer_steps(self):
        """Return the total number of optimizer steps taken in the current epoch."""
        return self._total_steps
