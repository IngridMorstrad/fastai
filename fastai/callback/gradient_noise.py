"""Gradient noise injection callback for improved generalization."""

from __future__ import annotations
from ..basics import *

__all__ = ['GradientNoiseCallback']


class GradientNoiseCallback(Callback):
    """Inject decayed Gaussian noise into gradients during training.

    Implements the schedule from Neelakantan et al. 2015 ("Adding Gradient Noise
    Improves Learning for Very Deep Networks"):

        sigma_t = eta / (1 + t) ^ gamma

    where `t` is the training step, `eta` controls the noise magnitude, and
    `gamma` controls the decay rate.  Adding gradient noise helps escape sharp
    minima and can improve generalization, especially on small datasets.

    Args:
        eta:   Noise magnitude (default 0.3).
        gamma: Decay exponent (default 0.55, as suggested in the paper).
    """
    # Run after gradient clipping but before the optimizer step.
    order = 60
    run_valid = False

    def __init__(self, eta: float = 0.3, gamma: float = 0.55):
        self.eta = eta
        self.gamma = gamma

    def before_fit(self):
        """Reset the step counter at the start of each fit."""
        self.step_count = 0

    def before_step(self):
        """Add Gaussian noise scaled by the decay schedule to every gradient."""
        self.step_count += 1
        sigma = self.eta / (1 + self.step_count) ** self.gamma
        for p in self.parameters():
            if p.grad is not None:
                noise = torch.normal(mean=0.0, std=sigma, size=p.grad.shape,
                                     device=p.grad.device, dtype=p.grad.dtype)
                p.grad.add_(noise)
