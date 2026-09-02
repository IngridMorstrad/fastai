"""StochasticWeightAveraging - Maintains an exponential moving average of model
weights and swaps to the averaged weights at evaluation time for improved
generalization.

The EMA update rule applied after each training batch (once ``swa_start_epoch``
is reached) is::

    ema_param = decay * ema_param + (1 - decay) * current_param

Non-floating-point buffers (e.g. BatchNorm ``num_batches_tracked``) are copied
directly from the training model rather than averaged.

Usage with fastai::

    from fastai.callback.swa import StochasticWeightAveraging
    learn.fit(20, cbs=[StochasticWeightAveraging(decay=0.999, swa_start_epoch=5)])
"""

import copy
import torch

__all__ = ['StochasticWeightAveraging']


try:
    from fastai.callback.core import Callback as _BaseCallback
except Exception:
    # Minimal base when the full fastai stack is unavailable (e.g. unit tests).
    class _BaseCallback:
        order = 0
        learn = None
        run = True
        run_valid = True
        run_train = True
        def __init__(self, **kwargs): pass
        def __repr__(self): return type(self).__name__


class StochasticWeightAveraging(_BaseCallback):
    """Callback that keeps an exponential moving average (EMA) of model weights.

    During training the model trains normally. Before each validation pass the
    EMA weights are swapped in so that metrics reflect the averaged model; after
    validation the training weights are restored. At the end of training the EMA
    weights are loaded permanently.

    Parameters
    ----------
    decay : float, default=0.999
        EMA decay factor.  Higher values make the average change more slowly.
    swa_start_epoch : int, default=1
        Epoch index (0-based) at which EMA updates begin.  Before this epoch
        the shadow weights are simply a copy of the initial model state.

    Example
    -------
    >>> from fastai.callback.swa import StochasticWeightAveraging
    >>> learn.fit(20, cbs=[StochasticWeightAveraging(decay=0.999, swa_start_epoch=5)])
    """

    # Run after Recorder (50), TrackerCallback (60), SaveModel (61).
    order = 65

    # Prevent inner-loop events (after_batch) from firing during validation.
    # The framework checks this flag and skips the dispatch entirely, giving
    # fail-safe behaviour instead of the fail-open getattr(self, 'training')
    # guard it replaces.
    run_valid = False

    def __init__(self, decay=0.999, swa_start_epoch=1):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        if swa_start_epoch < 0:
            raise ValueError(f"swa_start_epoch must be non-negative, got {swa_start_epoch}")
        self.decay = decay
        self.swa_start_epoch = swa_start_epoch
        self._ema_state = None

    def __repr__(self):
        return type(self).__name__

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def before_fit(self):
        """Deep-copy the model state dict as initial EMA (shadow) weights."""
        self._ema_state = copy.deepcopy(self.learn.model.state_dict())

    def after_batch(self):
        """Update EMA weights after each training batch (if past swa_start_epoch)."""
        if getattr(self, 'epoch', 0) < self.swa_start_epoch:
            return
        d = self.decay
        model_state = self.learn.model.state_dict()
        for key, ema_val in self._ema_state.items():
            cur_val = model_state[key]
            if _is_floating_point(ema_val):
                self._ema_state[key] = d * ema_val + (1.0 - d) * cur_val
            else:
                # Non-float buffers: copy from current model directly
                self._ema_state[key] = cur_val.clone() if hasattr(cur_val, 'clone') else copy.deepcopy(cur_val)

    def before_validate(self):
        """Swap model weights with EMA weights so validation uses the average.

        If ``before_fit`` has not been called yet (e.g. a bare
        ``learn.validate()`` with no prior ``fit()``), skip the swap silently
        since there is no EMA state to swap in.
        """
        if self._ema_state is None:
            return
        self._swap_params()

    def after_validate(self):
        """Restore training weights after validation is done."""
        if self._ema_state is None:
            return
        self._swap_params()

    def after_cancel_validate(self):
        """Restore training weights when validation is cancelled by an exception.

        Without this, an exception between ``before_validate`` and
        ``after_validate`` would leave the model holding EMA weights,
        silently corrupting subsequent training batches.
        """
        self.after_validate()

    def after_fit(self):
        """Load the EMA weights permanently into the model."""
        if self._ema_state is None:
            return
        # Perform one final swap so the model ends up with EMA weights.
        # _swap_params exchanges model <-> _ema_state, so after this call the
        # model holds EMA weights and _ema_state holds the last training weights.
        self._swap_params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _swap_params(self):
        """Swap model parameters/buffers with ``_ema_state`` in-place.

        This is its own inverse: calling it twice restores the original state.
        Uses ``param.data`` assignment for zero-allocation, O(1)-per-parameter
        swapping instead of deep-copying the full state dict.
        """
        named = dict(self.learn.model.named_parameters())
        named.update(dict(self.learn.model.named_buffers()))
        for key, ema_val in self._ema_state.items():
            param = named.get(key)
            if param is not None:
                model_val = param.data.clone()
                param.data = ema_val
                self._ema_state[key] = model_val
            else:
                # Fallback for keys not reachable via named_parameters/buffers
                # (should not happen in practice, but keeps the contract safe).
                model_state = self.learn.model.state_dict()
                old = model_state[key].clone()
                self._ema_state[key] = old


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _is_floating_point(tensor):
    """Check whether *tensor* is a floating-point type."""
    if hasattr(tensor, 'is_floating_point'):
        return tensor.is_floating_point()
    if hasattr(tensor, 'dtype'):
        return tensor.dtype.kind == 'f'
    return True
