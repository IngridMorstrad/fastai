"""Tests for ClassBalancedDL and the class_balanced_dataloaders patches.

Covers fastai.callback.data.ClassBalancedDL and the `class_balanced_dataloaders`
methods patched onto `Datasets` and `DataBlock`. These implement automatic
class-balanced sampling: per-sample weights are derived from the class labels
using inverse class frequency so that, over an epoch of shuffled sampling, each
class is drawn with (approximately) equal probability. This lets heavily
imbalanced classification datasets be trained without manual oversampling.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importing this module also registers the @patch-ed `class_balanced_dataloaders`
# methods on `Datasets` and `DataBlock`.
from fastai.callback.data import ClassBalancedDL, WeightedDL
from fastai.data.core import Datasets
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import FuncSplitter, ToTensor


def _sampled_idxs(dl, epochs):
    """Concatenate the training indices drawn by `dl.get_idxs` over `epochs`."""
    idxs = []
    for _ in range(epochs):
        idxs += list(dl.get_idxs())
    return np.array(idxs)


# ============================================================
# Tests for ClassBalancedDL construction / weighting
# ============================================================

class TestClassBalancedDLConstruction:
    """Tests for ClassBalancedDL weight computation."""

    def test_is_weighted_dl_subclass(self):
        """ClassBalancedDL should be a WeightedDL so it reuses weighted get_idxs."""
        assert issubclass(ClassBalancedDL, WeightedDL)

    def test_inverse_frequency_weights(self):
        """Each sample's weight should be the inverse of its class frequency."""
        labels = np.array([0, 0, 0, 1])  # class 0 is 3x as frequent as class 1
        dl = ClassBalancedDL(dataset=list(range(4)), bs=2, labels=labels, shuffle=True)
        # Raw weights are 1/count: class 0 -> 1/3, class 1 -> 1/1, then normalized.
        expected = np.array([1 / 3, 1 / 3, 1 / 3, 1.0])
        expected = expected / expected.sum()
        assert np.allclose(dl.wgts, expected)

    def test_weights_are_normalized(self):
        """The stored weights should sum to 1 so they form a probability vector."""
        labels = np.array([0] * 7 + [1] * 2 + [2] * 1)
        dl = ClassBalancedDL(dataset=list(range(10)), bs=2, labels=labels, shuffle=True)
        assert pytest.approx(float(dl.wgts.sum()), abs=1e-6) == 1.0

    def test_equal_total_mass_per_class(self):
        """Summed weight of every class should be equal (that is what 'balanced' means)."""
        labels = np.array([0] * 90 + [1] * 10)
        dl = ClassBalancedDL(dataset=list(range(100)), bs=10, labels=labels, shuffle=True)
        mass0 = dl.wgts[labels == 0].sum()
        mass1 = dl.wgts[labels == 1].sum()
        assert pytest.approx(float(mass0), abs=1e-6) == float(mass1)

    def test_labels_none_falls_back_to_uniform(self):
        """With no labels (as when cloning the validation DL) weights are uniform."""
        dl = ClassBalancedDL(dataset=list(range(5)), bs=2, labels=None, shuffle=True)
        assert np.allclose(dl.wgts, np.full(5, 1 / 5))

    def test_empty_dataset_get_idxs(self):
        """An empty dataset should yield no indices, mirroring WeightedDL."""
        dl = ClassBalancedDL(dataset=[], bs=2, labels=np.array([]), shuffle=True)
        assert dl.get_idxs() == []

    def test_no_shuffle_is_sequential(self):
        """Without shuffle, get_idxs should be sequential (no reweighting applied)."""
        labels = np.array([0, 0, 0, 1, 1])
        dl = ClassBalancedDL(dataset=list(range(5)), bs=2, labels=labels, shuffle=False)
        assert dl.get_idxs() == [0, 1, 2, 3, 4]


# ============================================================
# Tests for Datasets.class_balanced_dataloaders
# ============================================================

class TestDatasetsClassBalanced:
    """Tests for the class_balanced_dataloaders patch on Datasets."""

    def test_builds_class_balanced_training_dl(self):
        """The training dataloader should be a ClassBalancedDL."""
        n = 160
        dsets = Datasets(torch.arange(n).float())
        dls = dsets.class_balanced_dataloaders(labels=np.array([0] * 140 + [1] * 20), bs=16)
        assert isinstance(dls[0], ClassBalancedDL)

    def test_minority_class_is_upsampled(self):
        """The minority class should be sampled far above its raw frequency."""
        n = 160
        labels = np.array([0] * 140 + [1] * 20)  # minority class 1 is 12.5% of data
        dsets = Datasets(torch.arange(n).float())
        dls = dsets.class_balanced_dataloaders(labels=labels, bs=16)
        sampled = _sampled_idxs(dls[0], epochs=100)
        minority_frac = (labels[sampled] == 1).mean()
        raw_frac = (labels == 1).mean()
        assert minority_frac > 3 * raw_frac  # well above the raw 12.5%
        assert minority_frac == pytest.approx(0.5, abs=0.1)  # approximately balanced


# ============================================================
# Tests for DataBlock.class_balanced_dataloaders
# ============================================================

class TestDataBlockClassBalanced:
    """Tests for the class_balanced_dataloaders patch on DataBlock."""

    def test_uses_training_split_labels_only(self):
        """Weights should be derived from the training split's labels and rebalance it."""
        # 100 items: first 90 are training, last 10 are validation.
        labels = np.array([0] * 80 + [1] * 10 + [0] * 8 + [1] * 2)
        is_valid = lambda i: i >= 90
        dblock = DataBlock(blocks=[CategoryBlock],
                           getters=[lambda i: labels[i]],
                           splitter=FuncSplitter(is_valid),
                           item_tfms=[ToTensor()])
        dls = dblock.class_balanced_dataloaders(list(range(100)), labels=labels, bs=1)
        trn_dl = dls[0]
        assert isinstance(trn_dl, ClassBalancedDL)
        assert trn_dl.n == 90  # only the training split

        sampled = _sampled_idxs(trn_dl, epochs=80)
        trn_labels = labels[:90]
        sampled_class1_frac = (trn_labels[sampled] == 1).mean()
        raw_class1_frac = (trn_labels == 1).mean()
        assert sampled_class1_frac > 3 * raw_class1_frac
        assert sampled_class1_frac == pytest.approx(0.5, abs=0.12)
