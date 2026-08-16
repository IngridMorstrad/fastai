"""Learning rate warmup callback for stabilizing early training."""

from __future__ import annotations
import math

from ..basics import *


__all__ = ['LRWarmupCallback']


class LRWarmupCallback(Callback):
    """Gradually ramp learning rate from `start_pct` of target LR over `warmup_steps` batches.

    Supports both linear and exponential warmup schedules. After the warmup
    period completes, the learning rate is set to its full target value and the
    callback becomes a no-op for the remainder of training.

    Args:
        warmup_steps: Number of training batches over which to ramp the LR.
            If <= 0, the callback is effectively disabled.
        start_pct: Starting LR as a fraction of the optimizer's initial LR
            (e.g. 0.01 means start at 1% of target LR). Must be in (0, 1].
        mode: Warmup schedule type -- 'linear' or 'exponential'.
    """

    # Run before most other callbacks but after TrainEvalCallback (-10)
    order = -5
    run_valid = False

    def __init__(self, warmup_steps: int = 1000, start_pct: float = 0.01, mode: str = 'linear'):
        if mode not in ('linear', 'exponential'):
            raise ValueError(f"mode must be 'linear' or 'exponential', got '{mode}'")
        if not (0 < start_pct <= 1):
            raise ValueError(f"start_pct must be in (0, 1], got {start_pct}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        self.start_pct = start_pct
        self.mode = mode

    def before_fit(self):
        """Store the target LRs from the optimizer and initialize step counter."""
        self.target_lrs = [hp['lr'] for hp in self.opt.hypers]
        self._step = 0
        self._done = self.warmup_steps == 0

    def before_batch(self):
        """Adjust LR according to warmup schedule before each training batch."""
        if self._done:
            return
        pct = min(self._step / self.warmup_steps, 1.0) if self.warmup_steps > 0 else 1.0
        mult = self._compute_mult(pct)
        for hp, target_lr in zip(self.opt.hypers, self.target_lrs):
            hp['lr'] = target_lr * mult

    def after_batch(self):
        """Advance warmup step counter after each training batch."""
        if self._done:
            return
        self._step += 1
        if self._step >= self.warmup_steps:
            self._done = True
            # Ensure LR is exactly at target after warmup completes
            for hp, target_lr in zip(self.opt.hypers, self.target_lrs):
                hp['lr'] = target_lr

    def _compute_mult(self, pct: float) -> float:
        """Compute the LR multiplier for the current warmup progress.

        Args:
            pct: Progress through warmup in [0, 1].

        Returns:
            Multiplier to apply to target LR.
        """
        if self.mode == 'linear':
            return self.start_pct + (1.0 - self.start_pct) * pct
        else:  # exponential
            # Exponential ramp: start_pct * (1/start_pct)^pct
            # At pct=0 -> start_pct, at pct=1 -> 1.0
            return self.start_pct * math.pow(1.0 / self.start_pct, pct)
