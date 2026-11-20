"""FP32ParamGroups - Keep selected layers in full precision (fp32) during AMP training.

Automatic Mixed Precision (AMP, see ``fastai.callback.fp16.MixedPrecision``) runs most
of a model under ``torch.autocast``, casting many ops to a reduced precision (fp16/bf16).
Some layers are numerically sensitive and benefit from running in full precision, batch
norm and layer norm being the canonical examples. While autocast already keeps some norm
ops in fp32 at the backend level, there is no user-facing way to force *specific* module
types or module instances to run completely outside autocast.

This callback fills that gap. You designate module TYPES (e.g. ``nn.BatchNorm2d``,
``nn.LayerNorm``) and/or specific module INSTANCES whose forward passes should run in
fp32 regardless of the surrounding autocast context. It does so by wrapping each target
module's ``forward`` method: the wrapper enters a ``torch.autocast(..., enabled=False)``
region (for both the CPU and CUDA device types, so it works regardless of which device
autocast is running on), casts the module's float inputs to ``torch.float32``, calls the
original forward, and exits the region in a ``finally`` block so autocast re-engages for
the rest of the model even if the forward raises. Entering a disabled-autocast region is
safe even when no outer autocast is active, so the callback is a no-op on CPU / non-AMP
training.

It composes with the existing ``MixedPrecision`` callback: add both to a ``Learner`` and
the designated layers will train in fp32 while everything else trains under AMP.

Usage with fastai:
    from fastai.callback.fp16 import MixedPrecision
    from fastai.callback.fp32_param_groups import FP32ParamGroups
    learn.fit(10, cbs=[MixedPrecision(), FP32ParamGroups()])

    # Keep only layer norm in fp32, plus one specific attention block:
    learn.fit(10, cbs=[MixedPrecision(),
                       FP32ParamGroups(fp32_module_types=nn.LayerNorm,
                                       fp32_modules=[model.attn])])
"""

import torch
import torch.nn as nn

__all__ = ['FP32ParamGroups']


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


# Default module types considered numerically sensitive and worth keeping in fp32.
_DEFAULT_FP32_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)


# Device types for which we disable autocast inside the fp32 region. Disabling both means
# the fp32 region works whether AMP is running on CPU (bf16) or CUDA (fp16/bf16), and is a
# safe no-op when no outer autocast is active.
_AUTOCAST_DEVICE_TYPES = ('cpu', 'cuda')


class _DisableAutocast:
    "Context manager that disables autocast for every device type in `device_types`."
    def __init__(self, device_types=_AUTOCAST_DEVICE_TYPES):
        self._ctxs = [torch.autocast(device_type=d, enabled=False) for d in device_types]

    def __enter__(self):
        for ctx in self._ctxs:
            ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit in reverse order to properly unwind the nested contexts.
        for ctx in reversed(self._ctxs):
            ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


def _cast_floats_to_fp32(obj):
    "Recursively cast floating-point tensors in `obj` (tensor/tuple/list/dict) to fp32."
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.is_floating_point() else obj
    if isinstance(obj, tuple):
        return tuple(_cast_floats_to_fp32(o) for o in obj)
    if isinstance(obj, list):
        return [_cast_floats_to_fp32(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _cast_floats_to_fp32(v) for k, v in obj.items()}
    return obj


class FP32ParamGroups(_BaseCallback):
    """Run designated module types / instances in fp32 during AMP training.

    The callback walks ``learn.model`` at ``before_fit`` and selects every module whose
    type is in ``fp32_module_types`` plus every module explicitly listed in
    ``fp32_modules``. Each selected module's ``forward`` is wrapped so that it disables
    autocast (``torch.autocast(..., enabled=False)`` for the CPU and CUDA device types)
    and casts float inputs to ``torch.float32`` for the duration of its forward pass, so
    the computation runs in genuine full precision. The wrapper exits the fp32 region in a
    ``finally`` block, so autocast is restored even if the forward raises, and nothing is
    left on the per-module re-entrancy stack. Autocast re-engages automatically for the
    rest of the model once the forward returns.

    Parameters
    ----------
    fp32_module_types : type | tuple[type, ...], default=(BatchNorm1d/2d/3d, LayerNorm)
        ``nn.Module`` subclass(es) whose instances should run in fp32. A single class is
        normalized to a one-element tuple. ``None`` is treated as an empty tuple.
    fp32_modules : Iterable[nn.Module] | None, default=None
        Optional specific module instances to also keep in fp32, even if their type is not
        in ``fp32_module_types``. ``None`` is treated as empty.

    Raises
    ------
    TypeError
        If any entry in ``fp32_module_types`` is not an ``nn.Module`` subclass, or any
        entry in ``fp32_modules`` is not an ``nn.Module`` instance.

    Example
    -------
    >>> from fastai.callback.fp16 import MixedPrecision
    >>> from fastai.callback.fp32_param_groups import FP32ParamGroups
    >>> learn.fit(10, cbs=[MixedPrecision(), FP32ParamGroups()])
    """

    # Match MixedPrecision's order so hook registration happens alongside AMP setup.
    order = 10

    def __init__(self, fp32_module_types=_DEFAULT_FP32_TYPES, fp32_modules=None):
        super().__init__()
        self.fp32_module_types = self._validate_types(fp32_module_types)
        self.fp32_modules = self._validate_modules(fp32_modules)
        self._handles = []
        self._resolved = []

    def __repr__(self):
        return type(self).__name__

    @staticmethod
    def _validate_types(fp32_module_types):
        "Normalize and validate `fp32_module_types` into a tuple of nn.Module subclasses."
        if fp32_module_types is None:
            return tuple()
        # Normalize a single class passed instead of a tuple.
        if isinstance(fp32_module_types, type):
            fp32_module_types = (fp32_module_types,)
        else:
            try:
                fp32_module_types = tuple(fp32_module_types)
            except TypeError:
                raise TypeError(
                    f"fp32_module_types must be an nn.Module subclass or an iterable of "
                    f"them, got {fp32_module_types!r} of type "
                    f"{type(fp32_module_types).__name__}"
                )
        for t in fp32_module_types:
            if not (isinstance(t, type) and issubclass(t, nn.Module)):
                raise TypeError(
                    f"fp32_module_types entries must be nn.Module subclasses, "
                    f"got {t!r} of type {type(t).__name__}"
                )
        return fp32_module_types

    @staticmethod
    def _validate_modules(fp32_modules):
        "Normalize and validate `fp32_modules` into a list of nn.Module instances."
        if fp32_modules is None:
            return []
        fp32_modules = list(fp32_modules)
        for m in fp32_modules:
            if not isinstance(m, nn.Module):
                raise TypeError(
                    f"fp32_modules entries must be nn.Module instances, "
                    f"got {m!r} of type {type(m).__name__}"
                )
        return fp32_modules

    def _resolve_modules(self, model):
        "Return the ordered, de-duplicated list of modules to keep in fp32 for `model`."
        explicit = set(id(m) for m in self.fp32_modules)
        resolved, seen = [], set()
        # Explicitly listed instances first, preserving the user's order.
        for m in self.fp32_modules:
            if id(m) not in seen:
                resolved.append(m)
                seen.add(id(m))
        # Then every module matching a target type (skip ones already added).
        if self.fp32_module_types:
            for m in model.modules():
                if isinstance(m, self.fp32_module_types) and id(m) not in seen:
                    resolved.append(m)
                    seen.add(id(m))
        return resolved

    @staticmethod
    def _wrap_forward(module):
        """Replace `module.forward` with an fp32 (autocast-disabled) wrapper.

        Returns a zero-argument ``unwrap`` callable that restores the module to its exact
        prior state. Entry and exit of the fp32 region are paired in a ``try/finally`` so
        the region is always exited (and the per-call re-entrancy stack always popped) even
        if the wrapped forward raises. The per-module stack makes re-entrant / nested calls
        to the same module unwind in LIFO order.
        """
        original_forward = module.forward
        # Snapshot whether `forward` was an instance attribute (vs. resolved on the class)
        # so `unwrap` can restore the module to byte-identical state.
        had_own_forward = 'forward' in module.__dict__
        stack = []

        def _fp32_forward(*args, **kwargs):
            ctx = _DisableAutocast()
            ctx.__enter__()
            stack.append(ctx)
            try:
                args = _cast_floats_to_fp32(args)
                kwargs = _cast_floats_to_fp32(kwargs)
                return original_forward(*args, **kwargs)
            finally:
                # Always pair the enter from this call; LIFO keeps nesting correct.
                stack.pop().__exit__(None, None, None)

        # Expose the per-call re-entrancy stack on the wrapper for introspection/testing.
        _fp32_forward._fp32_stack = stack
        module.forward = _fp32_forward

        def unwrap():
            # Only remove our wrapper; never clobber a forward installed after us.
            if module.__dict__.get('forward') is _fp32_forward:
                if had_own_forward:
                    module.forward = original_forward
                else:
                    del module.__dict__['forward']

        return unwrap

    def register_hooks(self, model):
        "Wrap `forward` with an fp32 autocast-disabling region on every resolved target."
        self.remove_hooks()
        self._resolved = self._resolve_modules(model)
        for m in self._resolved:
            self._handles.append(self._wrap_forward(m))
        return self._resolved

    def remove_hooks(self):
        "Restore every wrapped forward and clear resolved state. Idempotent."
        for unwrap in self._handles:
            unwrap()
        self._handles = []
        self._resolved = []

    @property
    def fp32_modules_resolved(self):
        "List of modules currently selected for fp32 (populated at `before_fit`)."
        return list(self._resolved)

    def before_fit(self):
        "Register fp32 hooks on the model (removing any stale hooks first)."
        self.register_hooks(self.learn.model)

    def after_fit(self):
        "Remove all registered hooks so the callback is reusable across fits."
        self.remove_hooks()
