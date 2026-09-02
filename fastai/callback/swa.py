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

    def __init__(self, decay=0.999, swa_start_epoch=1):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        if swa_start_epoch < 0:
            raise ValueError(f"swa_start_epoch must be non-negative, got {swa_start_epoch}")
        self.decay = decay
        self.swa_start_epoch = swa_start_epoch
        self._ema_state = None
        self._training_state = None

    def __repr__(self):
        return type(self).__name__

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def before_fit(self):
        """Deep-copy the model state dict as initial EMA (shadow) weights."""
        self._ema_state = copy.deepcopy(self.learn.model.state_dict())
        self._training_state = None

    def after_batch(self):
        """Update EMA weights after each training batch (if past swa_start_epoch)."""
        # Only update during the training phase
        if not getattr(self, 'training', True):
            return
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
        """Swap model weights with EMA weights so validation uses the average."""
        self._training_state = copy.deepcopy(self.learn.model.state_dict())
        self.learn.model.load_state_dict(self._ema_state)

    def after_validate(self):
        """Restore training weights after validation is done."""
        if self._training_state is not None:
            self.learn.model.load_state_dict(self._training_state)
            self._training_state = None

    def after_fit(self):
        """Load the EMA weights permanently into the model."""
        self.learn.model.load_state_dict(self._ema_state)
        self._training_state = None


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
