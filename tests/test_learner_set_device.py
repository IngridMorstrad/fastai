"""Tests for Learner._set_device with parameter-free (buffer-only) models.

We test the _set_device logic in isolation to avoid pulling in the full fastai
import chain (which drags in dozens of optional dependencies).  The function
under test is small and self-contained; we extract it and feed it the same
inputs a real Learner would.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace


def _default_device():
    """Stand-in for fastai.torch_core.default_device."""
    return torch.device('cpu')


def _to_device(b, device):
    """Stand-in for fastai.torch_core.to_device -- moves every tensor in *b*."""
    return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in b)


def _set_device(learner, b):
    """Exact copy of the FIXED Learner._set_device (from fastai/learner.py)."""
    dls_device = getattr(learner.dls, 'device', _default_device())
    p = next(learner.model.parameters(), None)
    model_device = p.device if p is not None else dls_device
    if model_device == dls_device:
        return _to_device(b, dls_device)
    else:
        return _to_device(b, model_device)


class BufferOnlyModel(nn.Module):
    """A model with zero parameters -- only registered buffers."""
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


def _make_learner(model, device='cpu'):
    """Build a minimal namespace that quacks like a Learner for _set_device."""
    dls = SimpleNamespace(device=torch.device(device))
    return SimpleNamespace(model=model, dls=dls)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetDeviceBufferOnlyModel:
    """Learner._set_device must not raise StopIteration on buffer-only models."""

    def test_does_not_raise_on_zero_parameter_model(self):
        learner = _make_learner(BufferOnlyModel())
        batch = (torch.randn(2, 3), torch.randn(2, 1))
        # Before the fix this raised StopIteration
        result = _set_device(learner, batch)
        assert result is not None

    def test_returns_batch_on_cpu(self):
        learner = _make_learner(BufferOnlyModel())
        batch = (torch.randn(2, 3), torch.randn(2, 1))
        result = _set_device(learner, batch)
        for t in result:
            assert t.device == torch.device('cpu')

    def test_works_normally_with_parametric_model(self):
        learner = _make_learner(nn.Linear(3, 1))
        batch = (torch.randn(2, 3), torch.randn(2, 1))
        result = _set_device(learner, batch)
        assert result is not None
        for t in result:
            assert t.device == torch.device('cpu')

    def test_completely_empty_model(self):
        """Even a bare nn.Module (no params, no buffers) must not crash."""
        learner = _make_learner(nn.Module())
        batch = (torch.randn(1, 4),)
        result = _set_device(learner, batch)
        assert len(result) == 1
        assert result[0].device == torch.device('cpu')

    def test_original_code_raises_stop_iteration(self):
        """Demonstrate that the OLD code path raises StopIteration."""
        model = BufferOnlyModel()
        # Old code: next(model.parameters()).device  -- no default sentinel
        with_error = False
        try:
            next(model.parameters()).device
        except StopIteration:
            with_error = True
        assert with_error, "Expected StopIteration from next(model.parameters())"
