"""Tests for the Learner.to_onnx ONNX export patch.

Builds a tiny real Learner with a CPU nn.Sequential and a small synthetic
DataLoaders, then exercises the real ONNX export path. The whole module skips
gracefully when the `onnx` package (required by torch.onnx.export in torch
2.5.x even to perform the export) is not importable.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# torch.onnx.export requires the `onnx` package importable even to export in
# torch 2.5.x; skip the entire module gracefully if it (or torch.onnx) is absent.
onnx = pytest.importorskip("onnx")
pytest.importorskip("torch.onnx")

from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader


# ============================================================
# Helper: tiny real Learner on a CPU regression model
# ============================================================

def _make_learner():
    X = torch.randn(64, 4)
    Y = torch.randn(64, 1)
    dl = DataLoader(list(zip(X, Y)), bs=16)
    dls = DataLoaders(dl, dl)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    return Learner(dls, model, loss_func=nn.MSELoss())


class TestToOnnx:
    """Tests for Learner.to_onnx."""

    def test_creates_onnx_file(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        out = learn.to_onnx('model.onnx')
        assert out.exists()
        assert out.stat().st_size > 0

    def test_dynamic_batch_axis_on_input_and_output(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        out = learn.to_onnx('model.onnx')
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        assert m.graph.input[0].type.tensor_type.shape.dim[0].dim_param == 'batch'
        assert m.graph.output[0].type.tensor_type.shape.dim[0].dim_param == 'batch'

    def test_custom_names(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        out = learn.to_onnx('model.onnx', input_names=['x'], output_names=['y'])
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        assert m.graph.input[0].name == 'x'
        assert m.graph.output[0].name == 'y'
        assert m.graph.input[0].type.tensor_type.shape.dim[0].dim_param == 'batch'
        assert m.graph.output[0].type.tensor_type.shape.dim[0].dim_param == 'batch'

    def test_restores_training_state_when_training(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        learn.model.train()
        assert learn.model.training is True
        learn.to_onnx('model.onnx')
        assert learn.model.training is True

    def test_restores_training_state_when_eval(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        learn.model.eval()
        assert learn.model.training is False
        learn.to_onnx('model.onnx')
        assert learn.model.training is False

    def test_returns_path(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        out = learn.to_onnx('model')
        from pathlib import Path
        assert isinstance(out, Path)
        assert out.name == 'model.onnx'

    def test_appends_onnx_extension(self, tmp_path):
        learn = _make_learner()
        learn.path = tmp_path
        out = learn.to_onnx('noext')
        assert out.name == 'noext.onnx'
        assert out.exists()
