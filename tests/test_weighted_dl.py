"""Tests for WeightedDL zero-sum weight guard in fastai.callback.data."""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.callback.data import WeightedDL
from fastai.data.core import Datasets
import torch


class TestWeightedDLZeroSumGuard:
    """Tests that WeightedDL raises ValueError when all weights are zero."""

    def test_all_zero_weights_raises_valueerror(self):
        """Passing all-zero weights should raise a clear ValueError instead of ZeroDivisionError."""
        dsets = Datasets(torch.arange(10).float())
        with pytest.raises(ValueError, match="all-zero weights"):
            WeightedDL(dataset=dsets.train, bs=2, wgts=[0.0] * 10)

    def test_valid_weights_do_not_raise(self):
        """Non-zero weights should normalize correctly without errors."""
        dsets = Datasets(torch.arange(10).float())
        dl = WeightedDL(dataset=dsets.train, bs=2, wgts=[1.0] * 10)
        assert np.isclose(dl.wgts.sum(), 1.0)

    def test_single_nonzero_weight_works(self):
        """A single non-zero weight among zeros should still be valid."""
        dsets = Datasets(torch.arange(5).float())
        wgts = [0.0, 0.0, 1.0, 0.0, 0.0]
        dl = WeightedDL(dataset=dsets.train, bs=2, wgts=wgts)
        assert np.isclose(dl.wgts.sum(), 1.0)
        assert np.isclose(dl.wgts[2], 1.0)

    def test_none_weights_default_to_uniform(self):
        """When wgts is None, all items get equal weight."""
        dsets = Datasets(torch.arange(4).float())
        dl = WeightedDL(dataset=dsets.train, bs=2, wgts=None)
        assert np.isclose(dl.wgts.sum(), 1.0)
        assert np.allclose(dl.wgts, 0.25)
