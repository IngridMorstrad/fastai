"""Structured weight pruning callback with integrated fine-tuning schedule.

Provides `PruningCallback` for iterative structured pruning during training,
and patches `Learner.prune()` for a convenient one-call pruning + fine-tuning API.

Uses PyTorch's built-in `torch.nn.utils.prune` module.
"""

from __future__ import annotations
from ..basics import *


__all__ = ['PruningCallback', 'prune']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_prunable_modules(model, prune_types=(nn.Conv2d, nn.Linear)):
    """Return list of (module, 'weight') tuples for all prunable layers."""
    return [(m, 'weight') for m in model.modules() if isinstance(m, prune_types)]


def _compute_sparsity(model, prune_types=(nn.Conv2d, nn.Linear)):
    """Compute overall weight sparsity across prunable layers."""
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, prune_types):
            w = getattr(m, 'weight', None)
            if w is not None:
                total += w.nelement()
                zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Pruning Callback
# ---------------------------------------------------------------------------

class PruningCallback(Callback):
    """Apply structured pruning at the end of designated epochs.

    Structured pruning removes entire output channels/neurons (dim=0) based on
    Ln-norm ranking, which produces genuinely smaller models that benefit from
    hardware acceleration (unlike unstructured pruning which only yields sparse
    weight matrices).

    Parameters
    ----------
    amount : float
        Fraction of channels/neurons to prune at each pruning step (0-1).
    prune_epochs : list[int] | None
        Epoch indices (0-based) at which to apply pruning. If None, prunes
        at every epoch.
    method : str
        Pruning method: 'structured' (Ln-norm channel pruning) or
        'unstructured' (magnitude-based weight pruning).
    norm : int
        The Ln norm used to rank channels (only for structured pruning).
    dim : int
        Dimension along which to prune for structured pruning (0 = output channels).
    prune_types : tuple
        Module types eligible for pruning.
    make_permanent : bool
        If True, make pruning permanent (remove re-parametrization) after the
        final pruning step. Set False to keep masks for potential un-pruning.
    """
    order = 60  # Run after most other callbacks

    def __init__(self, amount=0.3, prune_epochs=None, method='structured',
                 norm=2, dim=0, prune_types=(nn.Conv2d, nn.Linear),
                 make_permanent=True):
        store_attr()
        self.pruning_applied = False

    def before_fit(self):
        """Record initial sparsity."""
        self._initial_sparsity = _compute_sparsity(self.model, self.prune_types)
        self._pruned_at = []

    def after_epoch(self):
        """Apply pruning at designated epochs."""
        if not self.training:
            return
        epoch = self.epoch
        should_prune = (self.prune_epochs is None) or (epoch in self.prune_epochs)
        if not should_prune:
            return

        import torch.nn.utils.prune as prune_utils

        modules_to_prune = _get_prunable_modules(self.model, self.prune_types)
        if not modules_to_prune:
            return

        if self.method == 'structured':
            for module, name in modules_to_prune:
                # Only prune if the weight tensor has more than 1 output channel/neuron
                w = getattr(module, name)
                if w.dim() >= 2 and w.shape[self.dim] > 1:
                    prune_utils.ln_structured(
                        module, name=name, amount=self.amount,
                        n=self.norm, dim=self.dim
                    )
        elif self.method == 'unstructured':
            prune_utils.global_unstructured(
                modules_to_prune, pruning_method=prune_utils.L1Unstructured,
                amount=self.amount
            )
        else:
            raise ValueError(f"Unknown pruning method: {self.method!r}. "
                             f"Use 'structured' or 'unstructured'.")

        self._pruned_at.append(epoch)
        self.pruning_applied = True

    def after_fit(self):
        """Make pruning permanent and log final sparsity."""
        if not self.pruning_applied:
            return

        import torch.nn.utils.prune as prune_utils

        if self.make_permanent:
            for module, name in _get_prunable_modules(self.model, self.prune_types):
                if prune_utils.is_pruned(module):
                    prune_utils.remove(module, name)

        final_sparsity = _compute_sparsity(self.model, self.prune_types)
        if hasattr(self, 'recorder') and hasattr(self.recorder, 'log'):
            self.recorder.log = self.recorder.log + (
                f' | Pruning: {self._initial_sparsity:.1%} -> {final_sparsity:.1%}',
            )


# ---------------------------------------------------------------------------
# Learner.prune() convenience method
# ---------------------------------------------------------------------------

@patch
def prune(self: Learner, amount=0.3, n_epochs=3, method='structured', lr=None,
          prune_schedule=None, norm=2, dim=0, prune_types=(nn.Conv2d, nn.Linear),
          make_permanent=True, cbs=None, **kwargs):
    """Apply structured weight pruning with an integrated fine-tuning schedule.

    This method performs iterative pruning: it alternates between pruning a
    fraction of weights and fine-tuning the remaining weights to recover
    accuracy. The pruning is spread across epochs according to `prune_schedule`.

    Parameters
    ----------
    amount : float
        Target fraction of channels/neurons to prune overall (0-1). When using
        iterative pruning (multiple prune epochs), each step prunes a fraction
        such that the cumulative effect approximates `amount`.
    n_epochs : int
        Number of fine-tuning epochs.
    method : str
        'structured' for Ln-norm channel pruning, 'unstructured' for
        magnitude-based weight pruning.
    lr : float | None
        Learning rate for fine-tuning. Defaults to `self.lr / 10`.
    prune_schedule : list[int] | None
        Epoch indices at which to apply pruning. Defaults to pruning at the
        start (epoch 0) and midpoint of training for iterative recovery.
    norm : int
        Ln norm for ranking channels (structured pruning only).
    dim : int
        Dimension along which to prune (0 = output channels).
    prune_types : tuple
        Module types eligible for pruning.
    make_permanent : bool
        If True, remove pruning re-parametrization after training so the model
        is a standard nn.Module again.
    cbs : list | None
        Additional callbacks to include during fine-tuning.
    **kwargs
        Additional arguments passed to `self.fit_one_cycle`.

    Returns
    -------
    Learner
        self, for method chaining.

    Example
    -------
    >>> learn = vision_learner(dls, resnet34, metrics=accuracy)
    >>> learn.fit_one_cycle(10)
    >>> # Prune 30% of channels and fine-tune for 5 epochs
    >>> learn.prune(amount=0.3, n_epochs=5)
    >>> # Check resulting sparsity
    >>> learn.prune_sparsity()
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")
    if not 0 < amount < 1:
        raise ValueError("amount must be between 0 and 1 (exclusive)")

    # Default schedule: prune at epoch 0 and midpoint for iterative recovery
    if prune_schedule is None:
        if n_epochs == 1:
            prune_schedule = [0]
        else:
            prune_schedule = [0, n_epochs // 2]

    # Per-step amount: distribute pruning across multiple steps so cumulative
    # effect approximates the target. Each step prunes `per_step` of remaining.
    # After k steps: remaining = (1-per_step)^k, so per_step = 1 - (1-amount)^(1/k)
    n_steps = len(prune_schedule)
    per_step = 1.0 - (1.0 - amount) ** (1.0 / n_steps)

    pruning_cb = PruningCallback(
        amount=per_step,
        prune_epochs=prune_schedule,
        method=method,
        norm=norm,
        dim=dim,
        prune_types=prune_types,
        make_permanent=make_permanent,
    )

    fine_tune_lr = lr if lr is not None else self.lr / 10
    all_cbs = L(cbs) + L([pruning_cb])
    self.fit_one_cycle(n_epochs, lr_max=fine_tune_lr, cbs=all_cbs, **kwargs)
    return self


@patch
def prune_sparsity(self: Learner, prune_types=(nn.Conv2d, nn.Linear)):
    """Report overall weight sparsity of prunable layers.

    Returns
    -------
    float
        Fraction of zero weights across all prunable layers (0-1).
    """
    return _compute_sparsity(self.model, prune_types)
