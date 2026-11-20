"""Tests for FP32ParamGroups callback."""

import pytest
import torch
import torch.nn as nn

from fastai.callback.fp32_param_groups import FP32ParamGroups


class _MockLearn:
    """Minimal mock of a Learner for testing callbacks."""

    def __init__(self, model):
        self.model = model


class _AutocastProbe(nn.Module):
    """Records whether autocast was enabled during its forward pass."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.cpu_autocast_seen = None
        self.cuda_autocast_seen = None

    def forward(self, x):
        self.cpu_autocast_seen = torch.is_autocast_enabled('cpu')
        self.cuda_autocast_seen = torch.is_autocast_enabled('cuda')
        return self.lin(x)


def _model_wrapped_forwards(model):
    """Count submodules whose forward has been wrapped by FP32ParamGroups.

    FP32ParamGroups installs an fp32 region by replacing a target module's ``forward``
    with a closure named ``_fp32_forward`` stored as an instance attribute, so a wrapped
    module has ``'forward'`` in its ``__dict__`` pointing at that closure.
    """
    n = 0
    for m in model.modules():
        f = m.__dict__.get('forward')
        if f is not None and getattr(f, '__name__', '') == '_fp32_forward':
            n += 1
    return n


class TestFP32ParamGroups:
    """Unit tests for FP32ParamGroups."""

    def test_default_module_types(self):
        """Default constructor selects BatchNorm and LayerNorm types."""
        cb = FP32ParamGroups()
        assert nn.BatchNorm1d in cb.fp32_module_types
        assert nn.BatchNorm2d in cb.fp32_module_types
        assert nn.BatchNorm3d in cb.fp32_module_types
        assert nn.LayerNorm in cb.fp32_module_types

    def test_custom_module_types_tuple(self):
        """Custom fp32_module_types tuple is respected."""
        cb = FP32ParamGroups(fp32_module_types=(nn.LayerNorm,))
        assert cb.fp32_module_types == (nn.LayerNorm,)

    def test_single_class_normalized_to_tuple(self):
        """A single class (not a tuple) is normalized to a one-element tuple."""
        cb = FP32ParamGroups(fp32_module_types=nn.LayerNorm)
        assert cb.fp32_module_types == (nn.LayerNorm,)

    def test_none_module_types_is_empty(self):
        """None fp32_module_types is treated as an empty tuple."""
        cb = FP32ParamGroups(fp32_module_types=None)
        assert cb.fp32_module_types == tuple()

    def test_invalid_module_type_raises(self):
        """A non-Module class / value in fp32_module_types raises TypeError."""
        with pytest.raises(TypeError, match="nn.Module subclass"):
            FP32ParamGroups(fp32_module_types=(int,))
        with pytest.raises(TypeError, match="nn.Module subclass"):
            FP32ParamGroups(fp32_module_types=5)

    def test_invalid_fp32_modules_raises(self):
        """A non-Module instance in fp32_modules raises TypeError."""
        with pytest.raises(TypeError, match="nn.Module instances"):
            FP32ParamGroups(fp32_modules=[object()])

    def test_resolved_selects_batchnorm_excludes_linear(self):
        """After before_fit, resolved fp32 set contains BatchNorm and excludes Linear."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        cb = FP32ParamGroups()
        cb.learn = _MockLearn(model)
        cb.before_fit()
        bn = model[1]
        resolved = cb.fp32_modules_resolved
        assert bn in resolved
        assert model[0] not in resolved
        assert model[2] not in resolved
        cb.after_fit()

    def test_explicit_modules_included_regardless_of_type(self):
        """Explicit fp32_modules are included even if their type is not targeted."""
        lin = nn.Linear(4, 4)
        model = nn.Sequential(lin, nn.ReLU())
        cb = FP32ParamGroups(fp32_module_types=None, fp32_modules=[lin])
        cb.learn = _MockLearn(model)
        cb.before_fit()
        assert lin in cb.fp32_modules_resolved
        assert len(cb.fp32_modules_resolved) == 1
        cb.after_fit()

    def test_forward_runs_and_is_finite_on_cpu(self):
        """Forward through the hooked model runs on CPU and produces finite outputs."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        cb = FP32ParamGroups()
        cb.learn = _MockLearn(model)
        cb.before_fit()
        out = model(torch.randn(8, 4))
        assert torch.isfinite(out).all()
        cb.after_fit()

    def test_targeted_runs_without_autocast_untargeted_with(self):
        """Targeted module forward sees autocast DISABLED; untargeted sees it ENABLED."""
        targeted = _AutocastProbe()
        untargeted = _AutocastProbe()
        model = nn.Sequential(targeted, untargeted)
        # Only target the first probe by instance (its type is not a norm layer).
        cb = FP32ParamGroups(fp32_module_types=None, fp32_modules=[targeted])
        cb.learn = _MockLearn(model)
        cb.before_fit()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            model(torch.randn(2, 4))
        assert targeted.cpu_autocast_seen is False
        assert untargeted.cpu_autocast_seen is True
        cb.after_fit()

    def test_fp32_region_computes_in_float32(self):
        """Inside the fp32 region the targeted module computes in float32 under bf16 AMP."""
        class _DtypeProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.randn(4, 4))
                self.out_dtype = None

            def forward(self, x):
                y = x @ self.a
                self.out_dtype = y.dtype
                return y

        probe = _DtypeProbe()
        model = nn.Sequential(probe)
        cb = FP32ParamGroups(fp32_module_types=None, fp32_modules=[probe])
        cb.learn = _MockLearn(model)
        cb.before_fit()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            model(torch.randn(2, 4))
        assert probe.out_dtype == torch.float32
        cb.after_fit()

    def test_after_fit_removes_all_hooks(self):
        """after_fit restores every wrapped forward on the model."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        cb = FP32ParamGroups()
        cb.learn = _MockLearn(model)
        assert _model_wrapped_forwards(model) == 0
        cb.before_fit()
        assert _model_wrapped_forwards(model) > 0
        cb.after_fit()
        assert _model_wrapped_forwards(model) == 0
        assert cb.fp32_modules_resolved == []

    def test_before_fit_idempotent_no_double_register(self):
        """Calling before_fit twice does not double-wrap; callback is reusable."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        cb = FP32ParamGroups()
        cb.learn = _MockLearn(model)
        cb.before_fit()
        count_once = _model_wrapped_forwards(model)
        cb.before_fit()
        count_twice = _model_wrapped_forwards(model)
        assert count_once == count_twice
        # Reusable: a full fit cycle twice leaves no leftover wrapping.
        cb.after_fit()
        cb.before_fit()
        cb.after_fit()
        assert _model_wrapped_forwards(model) == 0

    def test_remove_hooks_method(self):
        """Explicit remove_hooks() restores forwards and clears resolved state."""
        model = nn.Sequential(nn.BatchNorm1d(4))
        cb = FP32ParamGroups()
        cb.learn = _MockLearn(model)
        cb.before_fit()
        assert _model_wrapped_forwards(model) > 0
        cb.remove_hooks()
        assert _model_wrapped_forwards(model) == 0
        assert cb.fp32_modules_resolved == []

    def test_exception_in_forward_restores_autocast_and_state(self):
        """A raising targeted forward must not leave autocast disabled or stale state.

        The fp32 region is entered via the wrapped forward and exited in a ``finally``
        block, so even when the targeted module's forward raises inside an outer autocast
        context the region is unwound: a subsequent forward through an untargeted probe
        still sees autocast ENABLED, the targeted region is not permanently disabled, and
        a later clean forward through the (repaired) targeted module works again.
        """
        class _RaisingProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.should_raise = True
                self.lin = nn.Linear(4, 4)
                self.autocast_seen = None

            def forward(self, x):
                self.autocast_seen = torch.is_autocast_enabled('cpu')
                if self.should_raise:
                    raise ValueError("boom")
                return self.lin(x)

        targeted = _RaisingProbe()
        probe = _AutocastProbe()
        model = nn.Sequential(targeted, probe)
        cb = FP32ParamGroups(fp32_module_types=None, fp32_modules=[targeted])
        cb.learn = _MockLearn(model)
        cb.before_fit()

        # The raising forward happens inside an outer autocast context.
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            with pytest.raises(ValueError, match="boom"):
                model(torch.randn(2, 4))
            # Targeted region did run with autocast disabled before raising.
            assert targeted.autocast_seen is False
            # Autocast is NOT left disabled by the exception: an untargeted probe forward
            # in the SAME outer context still observes autocast ENABLED.
            probe(torch.randn(2, 4))
            assert probe.cpu_autocast_seen is True

        # No stale internal state: the targeted module's per-call re-entrancy stack is
        # empty after the exception unwound the region.
        assert targeted.forward._fp32_stack == []

        # A later clean forward through the targeted module still works and still runs its
        # region with autocast disabled.
        targeted.should_raise = False
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            out = targeted(torch.randn(2, 4))
        assert targeted.autocast_seen is False
        assert torch.isfinite(out).all()
        cb.after_fit()
        # After teardown the targeted forward is fully restored (no wrapper left).
        assert _model_wrapped_forwards(model) == 0

    def test_repr(self):
        """repr returns the class name."""
        cb = FP32ParamGroups()
        assert repr(cb) == "FP32ParamGroups"
