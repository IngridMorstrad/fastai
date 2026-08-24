"""GradientNoiseCallback - Injects decayed Gaussian noise into gradients during training.

Based on the paper "Adding Gradient Noise Improves Learning for Very Deep Networks"
(Neelakantan et al., 2015). Adds Gaussian noise with variance that decays over training
steps according to: variance = noise_scale / (1 + t)^decay_rate

This helps models escape sharp minima and can improve generalization, particularly
on small datasets.

Usage with fastai:
    from fastai.callback.gradient_noise import GradientNoiseCallback
    learn.fit(10, cbs=[GradientNoiseCallback(noise_scale=0.1, decay_rate=0.55)])
"""

import torch

__all__ = ['GradientNoiseCallback']


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


class GradientNoiseCallback(_BaseCallback):
    """Callback that adds decayed Gaussian noise to gradients after backward pass.

    The noise standard deviation at step t is:
        std = sqrt(noise_scale / (1 + t)^decay_rate)

    Parameters
    ----------
    noise_scale : float, default=0.1
        Initial noise variance scale. Higher values inject more noise early in training.
    decay_rate : float, default=0.55
        Controls how quickly the noise decays. The paper recommends 0.55.
        Higher values decay noise faster.

    Example
    -------
    >>> from fastai.callback.gradient_noise import GradientNoiseCallback
    >>> learn.fit(10, cbs=[GradientNoiseCallback(noise_scale=0.1, decay_rate=0.55)])
    """

    # Run after gradients are computed but before optimizer step.
    # Default Callback order is 0; we use a small positive value to ensure
    # we run after standard gradient computation callbacks but before
    # gradient clipping or optimizer steps.
    order = 10

    def __init__(self, noise_scale=0.1, decay_rate=0.55):
        super().__init__()
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        if decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}")
        self.noise_scale = noise_scale
        self.decay_rate = decay_rate
        self._step = 0

    def __repr__(self):
        return type(self).__name__

    def before_fit(self):
        "Reset step counter at the start of training."
        self._step = 0

    def after_backward(self):
        "Add Gaussian noise to all gradients with trainable parameters."
        variance = self.noise_scale / (1 + self._step) ** self.decay_rate
        std = variance ** 0.5
        for p in self.learn.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * std
                p.grad.add_(noise)
        self._step += 1

    @property
    def current_variance(self):
        "Return the current noise variance (useful for logging/debugging)."
        return self.noise_scale / (1 + self._step) ** self.decay_rate
