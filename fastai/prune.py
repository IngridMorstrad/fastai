"""Structured pruning + fine-tuning for fastai Learners.

Structured pruning removes whole output channels/neurons (rather than individual
weights) by ranking them with an L-n norm along a chosen dimension and zeroing the
lowest-ranked ones. Unlike unstructured pruning, this yields dense, hardware-friendly
weight tensors with entire filters/neurons set to zero, which can translate into real
speedups after a compaction step.

This module provides:
  - ``structured_prune_model(model, amount, n, dim, module_types, exclude)``: a pure
    function that applies ``torch.nn.utils.prune.ln_structured`` to every eligible
    module and then bakes the mask permanently into the weights via ``prune.remove``.
  - a ``@patch``-ed ``Learner.prune(...)`` method that prunes ``self.model`` in place
    and optionally runs a short fine-tuning schedule to recover accuracy.

.. warning::
    Pruning is applied to **every** eligible module by default, *including the model's
    final output/classifier head*. For a classifier this permanently zeroes whole
    output neurons that map directly to classes: after ``amount=0.3`` roughly 30% of
    the class logits are baked to zero and cannot be recovered by fine-tuning (the
    whole neuron is gone). This is usually not what you want for the head. To protect
    it, pass ``exclude`` with the head module(s) or the module type(s) to skip, e.g.
    ``exclude=[learn.model[-1]]`` or ``exclude=(nn.Linear,)`` to prune only convs.

It is a STANDALONE module (no source notebook), mirroring
``fastai/callback/gradient_noise.py``: the fastai imports are wrapped in try/except so
the pure function stays importable and unit-testable without the full fastai stack.

Usage with fastai:
    from fastai.prune import structured_prune_model  # noqa: F401 (registers Learner.prune)
    learn.prune(amount=0.3, fine_tune_epochs=1, base_lr=2e-3)
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as tprune

__all__ = ['structured_prune_model', 'prune']


try:
    from fastai.learner import Learner
    from fastcore.basics import patch
except Exception:
    # Provide fallbacks when the full fastai stack is not available (e.g. during
    # isolated unit testing). ``patch`` becomes a no-op decorator and ``Learner``
    # is None, so ``structured_prune_model`` stays importable and testable.
    Learner = None

    def patch(f):
        "No-op fallback for fastcore.basics.patch when fastai is unavailable."
        return f


def structured_prune_model(model, amount=0.3, n=2, dim=0, module_types=(nn.Linear, nn.Conv2d),
                           exclude=None):
    """Structurally prune eligible modules of ``model`` in place and return it.

    For every submodule that is an instance of ``module_types``, owns a ``weight``
    parameter, and is not excluded, applies ``torch.nn.utils.prune.ln_structured``
    (L-``n`` norm along ``dim``) to zero a fraction ``amount`` of the output
    channels/neurons, then calls ``prune.remove`` to bake the mask permanently into the
    weight tensor (so the module keeps a plain ``weight`` Parameter with no
    ``weight_orig``/``weight_mask`` reparam).

    .. warning::
        By default this prunes **every** eligible module, *including the final
        output/classifier head*, which permanently zeroes output neurons that map to
        classes/targets (fine-tuning cannot recover a removed neuron). Use ``exclude``
        to protect the head, e.g. ``exclude=[model[-1]]`` or ``exclude=(nn.Linear,)``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to prune (modified in place).
    amount : float, default=0.3
        Fraction of channels/neurons to prune, in [0, 1). ``amount == 0`` is a
        validated no-op that returns the model unchanged.
    n : int or float, default=2
        Order of the L-``n`` norm used to rank channels (e.g. 1 or 2). Must be a
        positive number (> 0).
    dim : int, default=0
        Dimension along which to prune. ``dim=0`` prunes output channels/neurons.
        Must be a valid weight dimension for every eligible module.
    module_types : tuple, default=(nn.Linear, nn.Conv2d)
        Module classes eligible for pruning.
    exclude : optional
        Modules to skip even when they match ``module_types``. May be a single module
        instance, an iterable of module instances, a module type, or an iterable of
        module types. Instances are matched by identity; types are matched by
        ``isinstance``. ``None`` (default) excludes nothing, preserving legacy behavior.

    Returns
    -------
    torch.nn.Module
        The same ``model`` instance, pruned in place.

    Raises
    ------
    ValueError
        If ``amount`` is not in [0, 1), ``n`` is not a positive number, or ``dim`` is
        out of range for any eligible module's weight.
    """
    if not (0 <= amount < 1):
        raise ValueError("amount must be in [0, 1)")
    if not (n > 0):
        raise ValueError("n must be a positive number")
    # amount == 0 is a validated no-op: nothing is zeroed, model returned unchanged.
    if amount == 0:
        return model
    # Normalize `exclude` into a set of module instances (matched by identity) and a
    # tuple of module types (matched by isinstance), so callers can protect the head.
    exclude_instances, exclude_types = _normalize_exclude(exclude)
    # Collect eligible modules once so we can validate `dim` against all of them
    # BEFORE mutating any weight (validate-before-mutate: a bad `dim` must not leave a
    # partially pruned model).
    eligible = [
        m for m in model.modules()
        if isinstance(m, module_types) and not isinstance(m, exclude_types)
        and m not in exclude_instances
        and getattr(m, 'weight', None) is not None
    ]
    for m in eligible:
        ndims = m.weight.dim()
        if not (-ndims <= dim < ndims):
            raise ValueError(
                f"dim {dim} is out of range for a weight with {ndims} dimensions "
                f"(expected {-ndims} <= dim < {ndims})")
    for module in eligible:
        tprune.ln_structured(module, name='weight', amount=amount, n=n, dim=dim)
        tprune.remove(module, 'weight')
    return model


def _normalize_exclude(exclude):
    "Split ``exclude`` into (set of module instances, tuple of module types)."
    if exclude is None:
        return set(), tuple()
    # A single module type, or a single module instance, gets wrapped in a list.
    if isinstance(exclude, type) or isinstance(exclude, nn.Module):
        exclude = [exclude]
    instances, types = set(), []
    for item in exclude:
        if isinstance(item, type):
            types.append(item)
        elif isinstance(item, nn.Module):
            instances.add(item)
        else:
            raise ValueError(
                f"exclude entries must be nn.Module instances or module types, got {item!r}")
    return instances, tuple(types)


@patch
def prune(self: Learner, amount=0.3, method='structured', n=2, dim=0, fine_tune_epochs=1,
          base_lr=2e-3, module_types=(nn.Linear, nn.Conv2d), exclude=None, **kwargs):
    """Structurally prune ``self.model`` in place and optionally fine-tune to recover.

    Prunes whole output channels/neurons of eligible modules via
    ``structured_prune_model``, then, when ``fine_tune_epochs > 0``, runs a short
    fine-tuning schedule (``self.fine_tune`` if available, else ``self.fit``) to recover
    accuracy. Returns ``self`` for chaining.

    .. warning::
        By default every eligible module is pruned, *including the model's final
        classification/regression head*, which permanently zeroes output neurons that
        map directly to classes/targets and cannot be recovered by fine-tuning. Pass
        ``exclude`` to protect the head, e.g. ``exclude=[self.model[-1]]`` or
        ``exclude=(nn.Linear,)`` to prune only conv layers.

    Parameters
    ----------
    amount : float, default=0.3
        Fraction of channels/neurons to prune, in [0, 1).
    method : str, default='structured'
        Pruning strategy. Only ``'structured'`` is supported for now.
    n : int or float, default=2
        Order of the L-``n`` norm used to rank channels. Must be a positive number.
    dim : int, default=0
        Dimension along which to prune (0 = output channels/neurons).
    fine_tune_epochs : int, default=1
        Number of epochs to fine-tune after pruning. 0 skips fine-tuning.
    base_lr : float, default=2e-3
        Learning rate for the fine-tuning schedule.
    module_types : tuple, default=(nn.Linear, nn.Conv2d)
        Module classes eligible for pruning.
    exclude : optional
        Modules or module types to skip; forwarded to ``structured_prune_model``. Use
        this to protect the output head from being pruned. ``None`` prunes everything.
    kwargs : dict
        Forwarded to the fine-tuning call (``self.fine_tune``/``self.fit``).

    Returns
    -------
    Learner
        ``self``, so calls can be chained.

    Raises
    ------
    ValueError
        If ``method`` is not ``'structured'`` (or on invalid ``amount``/``n``/``dim``
        from ``structured_prune_model``).
    """
    if method != 'structured':
        raise ValueError(f"Unsupported prune method: {method!r}; only 'structured' is supported")
    structured_prune_model(self.model, amount=amount, n=n, dim=dim, module_types=module_types,
                           exclude=exclude)
    if fine_tune_epochs and fine_tune_epochs > 0:
        # Prefer fine_tune (discriminative LRs + freeze schedule) when present; fall
        # back to fit. Data is required here by design: if dls are missing, fastai
        # raises loudly rather than silently skipping recovery training.
        if hasattr(self, 'fine_tune'):
            self.fine_tune(fine_tune_epochs, base_lr=base_lr, **kwargs)
        else:
            self.fit(fine_tune_epochs, lr=base_lr, **kwargs)
    return self
