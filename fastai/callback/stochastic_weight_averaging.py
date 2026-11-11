"""StochasticWeightAveraging - Maintains an EMA of model weights and evaluates with them.

Keeps an exponential moving average (EMA) of the model's parameters during training.
At evaluation time the callback swaps the live (training) weights for the averaged
weights so that validation metrics reflect the smoother, better-generalizing averaged
model, then restores the live weights afterwards so training continues unaffected.

The EMA update applied after every optimizer step is, for each parameter ``p``:
    ema = decay * ema + (1 - decay) * p

Averaging weights over a trajectory of SGD iterates is a well established technique for
improving generalization (see "Averaging Weights Leads to Wider Optima and Better
Generalization", Izmailov et al., 2018). An EMA gives more weight to recent iterates
while still smoothing the trajectory.

Usage with fastai:
    from fastai.callback.stochastic_weight_averaging import StochasticWeightAveraging
    learn.fit(10, cbs=[StochasticWeightAveraging(decay=0.99, start_epoch=1)])
"""

import torch

__all__ = ['StochasticWeightAveraging']


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


def _is_floating(t):
    "Return True if tensor/array ``t`` holds floating point values."
    # torch.Tensor exposes is_floating_point(); numpy arrays expose dtype.kind.
    is_fp = getattr(t, "is_floating_point", None)
    if callable(is_fp):
        return bool(is_fp())
    dtype = getattr(t, "dtype", None)
    kind = getattr(dtype, "kind", None)
    if kind is not None:
        return kind == "f"
    return isinstance(t, float)


def _detached_clone(t):
    "Return a detached, independent copy of ``t`` (torch tensor or numpy array)."
    detach = getattr(t, "detach", None)
    if callable(detach):
        t = detach()
    clone = getattr(t, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(t, "copy", None)
    if callable(copy):
        return copy()
    return t


class StochasticWeightAveraging(_BaseCallback):
    """Callback that maintains an EMA of model weights and evaluates with them.

    After each optimizer step the EMA is updated for every floating point parameter::

        ema = decay * ema + (1 - decay) * p

    Non-floating buffers (e.g. batchnorm ``num_batches_tracked``) are left untouched.
    At validation the live weights are stashed and the averaged weights are loaded into
    the model; the live weights are restored once validation finishes.

    Parameters
    ----------
    decay : float, default=0.99
        EMA decay in ``[0, 1]``. Higher values smooth over a longer window of iterates.
        Raises ``ValueError`` if outside ``[0, 1]``.
    start_epoch : int, default=0
        Epoch (0-indexed) at which to begin accumulating the EMA. Earlier epochs are
        skipped so warm-up iterates do not pollute the average. Raises ``ValueError`` if
        negative.

    Example
    -------
    >>> from fastai.callback.stochastic_weight_averaging import StochasticWeightAveraging
    >>> learn.fit(10, cbs=[StochasticWeightAveraging(decay=0.99, start_epoch=1)])
    """

    # Run after optimizer-step related callbacks so the EMA tracks post-step weights.
    order = 20

    def __init__(self, decay=0.99, start_epoch=0):
        super().__init__()
        if not (0. <= decay <= 1.):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be non-negative, got {start_epoch}")
        self.decay = decay
        self.start_epoch = start_epoch
        self.ema_state = {}
        self._live_state = None

    def __repr__(self):
        return type(self).__name__

    def before_fit(self):
        "Reset EMA state at the start of training."
        if not self.run:
            return
        self.ema_state = {}
        self._live_state = None

    def after_step(self):
        "Update the EMA from the current model weights after each optimizer step."
        if not self.run:
            return
        if self.epoch < self.start_epoch:
            return
        for name, p in self.learn.model.named_parameters():
            if not _is_floating(p):
                # Non-floating parameters are not averaged.
                continue
            value = _detached_clone(p)
            if name not in self.ema_state:
                # Lazily initialize the EMA buffer from the first observed weights.
                self.ema_state[name] = value
            else:
                self.ema_state[name] = self.decay * self.ema_state[name] + (1. - self.decay) * value

    def before_validate(self):
        "Swap the live weights for the averaged weights before validation."
        if not self.run:
            return
        if not self.ema_state:
            # EMA not yet initialized (no steps taken); nothing to swap.
            return
        model = self.learn.model
        # Stash a copy of the current live weights so they can be restored exactly.
        self._live_state = {k: _detached_clone(v) for k, v in model.state_dict().items()}
        averaged = dict(self._live_state)
        averaged.update(self.ema_state)
        model.load_state_dict(averaged)

    def after_validate(self):
        "Restore the live training weights after validation."
        if not self.run:
            return
        if self._live_state is None:
            return
        self.learn.model.load_state_dict(self._live_state)
        self._live_state = None

    @property
    def averaged_state(self):
        "Return the current averaged (EMA) state dict (useful for logging/debugging)."
        return self.ema_state
