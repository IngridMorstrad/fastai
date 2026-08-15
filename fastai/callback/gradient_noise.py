"""Gradient noise injection callback for improved generalization.

Implements the technique from Neelakantan et al. (2015) "Adding Gradient Noise
Improves Learning for Very Deep Networks" which adds decayed Gaussian noise to
gradients during training to help escape sharp minima and improve generalization,
especially on small datasets.
"""

from __future__ import annotations
from ..basics import *


__all__ = ['GradientNoiseCallback']


class GradientNoiseCallback(Callback):
    """Inject decayed Gaussian noise into gradients during training.

    Adds noise sampled from N(0, sigma_t^2) to each gradient after the backward
    pass and before the optimizer step. The noise standard deviation decays as:

        sigma_t = eta / (1 + t)^gamma

    where `t` is the training iteration count, `eta` controls the initial noise
    magnitude, and `gamma` controls the decay rate.

    Args:
        eta: Initial noise scale (default: 0.3).
        gamma: Decay exponent (default: 0.55, as recommended by Neelakantan et al.).
    """
    order = MixedPrecision.order + 1  # Same priority as GradientClip
    run_valid = False

    def __init__(self, eta: float = 0.3, gamma: float = 0.55):
        store_attr()

    def before_fit(self):
        """Reset the iteration counter at the start of training."""
        self._step = 0

    def before_step(self):
        """Add decayed Gaussian noise to all parameter gradients."""
        sigma = self.eta / (1 + self._step) ** self.gamma
        for p in self.parameters():
            if p.grad is not None:
                noise = torch.zeros_like(p.grad).normal_(0, sigma)
                p.grad.add_(noise)
        self._step += 1
