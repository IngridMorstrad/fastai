"""Learning rate warmup callback for stabilizing early training.

This module provides `LRWarmupCallback`, a callback that linearly or
exponentially ramps the learning rate from a low starting value up to the
target learning rate over a configurable number of initial training steps.

This is a standalone callback that can be combined with any other scheduling
callbacks. It only modifies the learning rate during the warmup phase; once
warmup completes, it stops intervening and downstream schedulers take over.

Example usage:
    from fastai.callback.warmup import LRWarmupCallback

    # Linear warmup over 100 steps
    learn.fit(10, lr=3e-4, cbs=LRWarmupCallback(warmup_steps=100))

    # Exponential warmup over 5% of total training steps
    learn.fit(10, lr=3e-4, cbs=LRWarmupCallback(warmup_pct=0.05, schedule='exp'))
"""

from __future__ import annotations

import math
from ..basics import *


__all__ = ['LRWarmupCallback']


class LRWarmupCallback(Callback):
    """Ramp the learning rate during early training to stabilize optimization.

    The callback intercepts learning rate updates during the warmup phase,
    scaling from `start_lr` (or `start_factor * target_lr`) up to the target
    learning rate using either a linear or exponential schedule.

    Parameters
    ----------
    warmup_steps : int or None
        Number of training steps (batches) over which to warm up. Mutually
        exclusive with `warmup_pct`. At least one must be specified.
    warmup_pct : float or None
        Fraction of total training steps to use for warmup (between 0 and 1).
        Mutually exclusive with `warmup_steps`.
    start_lr : float or None
        Absolute starting learning rate. If provided, `start_factor` is
        ignored. Must be positive.
    start_factor : float
        Factor to multiply the target learning rate by to get the starting
        learning rate. Defaults to 0.01 (start at 1% of target LR). Ignored
        when `start_lr` is given.
    schedule : str
        Warmup schedule type. One of 'linear' or 'exp' (exponential).
    """

    # Run early so that the warmup LR is set before other schedulers see it.
    # ParamScheduler has order=60; we want to run after the optimizer is set up
    # but the warmup override happens in before_batch which is called in order.
    order = 55
    run_valid = False

    def __init__(
        self,
        warmup_steps: int | None = None,
        warmup_pct: float | None = None,
        start_lr: float | None = None,
        start_factor: float = 0.01,
        schedule: str = 'linear',
    ):
        if warmup_steps is None and warmup_pct is None:
            raise ValueError("Either 'warmup_steps' or 'warmup_pct' must be specified.")
        if warmup_steps is not None and warmup_pct is not None:
            raise ValueError("'warmup_steps' and 'warmup_pct' are mutually exclusive.")
        if warmup_steps is not None and warmup_steps <= 0:
            raise ValueError("'warmup_steps' must be positive.")
        if warmup_pct is not None and not (0 < warmup_pct < 1):
            raise ValueError("'warmup_pct' must be between 0 and 1 (exclusive).")
        if start_lr is not None and start_lr <= 0:
            raise ValueError("'start_lr' must be positive.")
        if start_factor <= 0 or start_factor >= 1:
            raise ValueError("'start_factor' must be between 0 and 1 (exclusive).")
        if schedule not in ('linear', 'exp'):
            raise ValueError(f"'schedule' must be 'linear' or 'exp', got '{schedule}'.")

        self.warmup_steps = warmup_steps
        self.warmup_pct = warmup_pct
        self.start_lr = start_lr
        self.start_factor = start_factor
        self.schedule = schedule

    def before_fit(self):
        """Compute warmup parameters based on the optimizer's target LR."""
        # Total training steps across all epochs
        n_batches = len(self.dls.train)
        total_steps = self.n_epoch * n_batches

        # Resolve warmup duration
        if self.warmup_steps is not None:
            self._warmup_steps = min(self.warmup_steps, total_steps)
        else:
            self._warmup_steps = int(math.ceil(self.warmup_pct * total_steps))

        # Capture target learning rates from the optimizer (one per param group)
        self._target_lrs = [h['lr'] for h in self.opt.hypers]

        # Determine starting learning rates
        if self.start_lr is not None:
            self._start_lrs = [min(self.start_lr, lr) for lr in self._target_lrs]
        else:
            self._start_lrs = [lr * self.start_factor for lr in self._target_lrs]

        self._step_count = 0
        self._warmup_active = True

    def before_batch(self):
        """Adjust learning rate if still within the warmup phase."""
        if not self._warmup_active:
            return

        if self._step_count >= self._warmup_steps:
            # Warmup complete: restore target LRs and deactivate
            for hp, target_lr in zip(self.opt.hypers, self._target_lrs):
                hp['lr'] = target_lr
            self._warmup_active = False
            return

        # Compute progress through warmup (0.0 to 1.0)
        progress = self._step_count / self._warmup_steps

        for hp, start_lr, target_lr in zip(self.opt.hypers, self._start_lrs, self._target_lrs):
            hp['lr'] = self._interpolate(start_lr, target_lr, progress)

        self._step_count += 1

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Interpolate between start and end LR based on schedule type."""
        if self.schedule == 'linear':
            return start + (end - start) * progress
        else:  # exponential
            # Avoid log(0) by using a small floor for start
            safe_start = max(start, 1e-10)
            log_ratio = math.log(end / safe_start)
            return safe_start * math.exp(log_ratio * progress)
