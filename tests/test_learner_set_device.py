"""Tests for Learner._set_device with buffer-only models (no parameters)."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from fastai.data.core import DataLoaders
from fastai.learner import Learner


class BufferOnlyModel(nn.Module):
    """A model that has buffers but no parameters."""
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(2.0))

    def forward(self, x):
        return x * self.scale


class EmptyModel(nn.Module):
    """A model with neither parameters nor buffers."""
    def forward(self, x):
        return x


class NormalModel(nn.Module):
    """A standard model with parameters."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def _make_dls():
    """Create minimal DataLoaders for testing."""
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    ds = list(zip(x, y))
    dl = torch.utils.data.DataLoader(ds, batch_size=4)
    return DataLoaders(dl, dl)


def test_set_device_buffer_only_model_no_stopiteration():
    """Learner._set_device should not raise StopIteration for buffer-only models."""
    dls = _make_dls()
    model = BufferOnlyModel()
    learn = Learner(dls, model, loss_func=nn.MSELoss())
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    # Should not raise StopIteration
    result = learn._set_device(batch)
    assert len(result) == 2


def test_set_device_empty_model_no_stopiteration():
    """Learner._set_device should not raise StopIteration for empty models."""
    dls = _make_dls()
    model = EmptyModel()
    learn = Learner(dls, model, loss_func=nn.MSELoss())
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    # Should not raise StopIteration
    result = learn._set_device(batch)
    assert len(result) == 2


def test_set_device_normal_model_still_works():
    """Learner._set_device should still work normally for models with parameters."""
    dls = _make_dls()
    model = NormalModel()
    learn = Learner(dls, model, loss_func=nn.MSELoss())
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    result = learn._set_device(batch)
    assert len(result) == 2


def test_set_device_buffer_only_uses_buffer_device():
    """For buffer-only models, _set_device should detect device from buffers."""
    dls = _make_dls()
    model = BufferOnlyModel()
    learn = Learner(dls, model, loss_func=nn.MSELoss())
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    result = learn._set_device(batch)
    # Buffer is on CPU, so result should be on CPU
    assert result[0].device == torch.device('cpu')
