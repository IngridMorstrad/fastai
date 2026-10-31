"""LRWarmupCallback - Linearly or exponentially ramps the learning rate over the
first few training steps to stabilize early training.

Large learning rates at the very start of training (before the optimizer state and
batch-norm statistics have settled) can destabilize or diverge a model. A short warmup
phase ramps the learning rate from a small starting value up to the base learning rate
configured on the optimizer, after which the base learning rate is held unchanged and
any downstream scheduler takes over.

Two ramp shapes are supported:
    - 'linear'      : lr grows linearly from start_lr to the base lr.
    - 'exponential' : lr grows geometrically (constant multiplicative factor per step)
                      from start_lr to the base lr.

Usage with fastai:
    from fastai.callback.lr_warmup import LRWarmupCallback
    learn.fit(10, cbs=[LRWarmupCallback(n_warmup=100, mode='linear')])
"""

__all__ = ['LRWarmupCallback']


try:
    from fastai.callback.core import Callback as _BaseCallback
except Exception:
    # Provide a minimal base when the full fastai stack is not available
    # (e.g., during isolated unit testing). This base implements the
    # callback protocol so the class remains fully functional with fastai
    # when it IS available.
    class _BaseCallback:
        order = 0
        learn = None
        run = True
        def __init__(self, **kwargs): pass
        def __repr__(self): return type(self).__name__


class LRWarmupCallback(_BaseCallback):
    """Callback that ramps the learning rate during an initial warmup phase.

    At the start of training the base learning rate of every optimizer parameter group
    is recorded. For the first ``n_warmup`` batches the learning rate of each group is
    scaled by a factor that grows from ``start_lr / base_lr`` up to ``1.0``. Once warmup
    completes, the base learning rate is restored exactly so that it is held constant (or
    picked up by any following scheduler).

    Parameters
    ----------
    n_warmup : int, default=100
        Number of initial batches over which to ramp the learning rate. Must be positive.
    start_lr : float, default=1e-6
        Learning rate at the very first step of warmup. Must be non-negative. For the
        'exponential' mode it must be strictly positive (a geometric ramp cannot start
        from zero).
    mode : str, default='linear'
        Ramp shape, either 'linear' or 'exponential'.

    Example
    -------
    >>> from fastai.callback.lr_warmup import LRWarmupCallback
    >>> learn.fit(10, cbs=[LRWarmupCallback(n_warmup=100, start_lr=1e-6, mode='linear')])
    """

    # Run before ParamScheduler (order 60) and most schedulers so the warmup factor is
    # applied on top of a freshly set base lr, but after the base lr has been set.
    order = 50

    _MODES = ('linear', 'exponential')

    def __init__(self, n_warmup=100, start_lr=1e-6, mode='linear'):
        super().__init__()
        if not isinstance(n_warmup, int) or n_warmup <= 0:
            raise ValueError(f"n_warmup must be a positive integer, got {n_warmup!r}")
        if start_lr < 0:
            raise ValueError(f"start_lr must be non-negative, got {start_lr}")
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        if mode == 'exponential' and start_lr <= 0:
            raise ValueError(
                f"start_lr must be strictly positive for exponential mode, got {start_lr}")
        self.n_warmup = n_warmup
        self.start_lr = start_lr
        self.mode = mode
        self._step = 0
        self._base_lrs = None

    def __repr__(self):
        return (f"{type(self).__name__}(n_warmup={self.n_warmup}, "
                f"start_lr={self.start_lr}, mode={self.mode!r})")

    def _hypers(self):
        "Return the list of optimizer hyper-parameter dicts (one per param group)."
        return self.learn.opt.hypers

    def before_fit(self):
        "Reset the step counter and record the base learning rate of each param group."
        self._step = 0
        self._base_lrs = [h['lr'] for h in self._hypers()]

    def before_batch(self):
        "Set the warmed learning rate on every param group during the warmup phase."
        if self._base_lrs is None:
            # before_fit did not run (defensive); nothing to warm up against.
            return
        hypers = self._hypers()
        if self._step >= self.n_warmup:
            # Warmup finished: hold the recorded base lr exactly.
            for h, base in zip(hypers, self._base_lrs):
                h['lr'] = base
            return
        denom = max(self.n_warmup - 1, 1)
        pct = min(self._step, denom) / denom
        for h, base in zip(hypers, self._base_lrs):
            if self.mode == 'linear':
                h['lr'] = self.start_lr + (base - self.start_lr) * pct
            else:  # exponential
                if base <= 0:
                    h['lr'] = 0.0
                else:
                    lo = min(self.start_lr, base)
                    # Geometric interpolation from lo up to base.
                    h['lr'] = lo * (base / lo) ** pct
        self._step += 1

    @property
    def finished(self):
        "Whether the warmup phase has completed."
        return self._step >= self.n_warmup
