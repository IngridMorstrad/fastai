"""Tests for ClassBalancedDL - automatic class-balanced sampling DataLoader."""
import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import DataLoader
from fastai.callback.data import ClassBalancedDL


class SimpleDataset:
    """A minimal indexed dataset for testing with imbalanced classes."""
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])


class TestClassBalancedDLConstruction:
    """Tests for ClassBalancedDL initialization."""

    def test_basic_construction(self):
        """ClassBalancedDL can be constructed with an imbalanced dataset."""
        # 90 samples of class 0, 10 samples of class 1
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        assert dl.n == 100
        assert dl.bs == 10

    def test_weights_are_computed(self):
        """Weights should be computed automatically on construction."""
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        assert dl.wgts is not None
        assert len(dl.wgts) == 100

    def test_weights_sum_to_one(self):
        """Computed weights should sum to 1.0."""
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        assert abs(dl.wgts.sum() - 1.0) < 1e-10

    def test_minority_class_gets_higher_weight(self):
        """Minority class samples should have higher individual weights."""
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        # Class 1 (minority) samples should have higher weight than class 0 (majority)
        majority_weight = dl.wgts[0]
        minority_weight = dl.wgts[90]
        assert minority_weight > majority_weight

    def test_equal_class_distribution_equal_weights(self):
        """With balanced classes, all weights should be equal."""
        inputs = list(range(100))
        labels = [0] * 50 + [1] * 50
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        # All weights should be equal (1/100)
        expected_weight = 1.0 / 100
        assert np.allclose(dl.wgts, expected_weight)


class TestClassBalancedDLSampling:
    """Tests for ClassBalancedDL sampling behavior."""

    def test_get_idxs_no_shuffle_sequential(self):
        """Without shuffle, get_idxs should return sequential indices."""
        inputs = list(range(20))
        labels = [0] * 15 + [1] * 5
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=5, shuffle=False, num_workers=0)
        idxs = dl.get_idxs()
        assert idxs == list(range(20))

    def test_get_idxs_with_shuffle_returns_correct_length(self):
        """With shuffle, get_idxs should return n indices."""
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        idxs = dl.get_idxs()
        assert len(idxs) == 100

    def test_get_idxs_empty_dataset(self):
        """Empty dataset should return empty list."""
        ds = SimpleDataset([], [])
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)
        idxs = dl.get_idxs()
        assert idxs == []

    def test_sampling_balances_classes_statistically(self):
        """Over many samples, class representation should be approximately balanced."""
        np.random.seed(42)
        inputs = list(range(100))
        labels = [0] * 90 + [1] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)

        # Sample many times and count class occurrences
        class_counts = {0: 0, 1: 0}
        for _ in range(100):
            idxs = dl.get_idxs()
            for idx in idxs:
                class_counts[labels[idx]] += 1

        total = class_counts[0] + class_counts[1]
        # With balanced sampling, each class should get roughly 50% of samples
        class_0_ratio = class_counts[0] / total
        class_1_ratio = class_counts[1] / total
        # Allow some tolerance since it is stochastic
        assert 0.35 < class_0_ratio < 0.65, f"Class 0 ratio: {class_0_ratio}"
        assert 0.35 < class_1_ratio < 0.65, f"Class 1 ratio: {class_1_ratio}"


class TestClassBalancedDLMultiClass:
    """Tests for ClassBalancedDL with more than 2 classes."""

    def test_three_class_imbalanced(self):
        """Should handle 3+ class imbalanced datasets correctly."""
        inputs = list(range(110))
        labels = [0] * 100 + [1] * 5 + [2] * 5
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)

        # Minority classes should have higher weight
        majority_wgt = dl.wgts[0]  # class 0
        minority_wgt = dl.wgts[100]  # class 1
        assert minority_wgt > majority_wgt

    def test_weights_proportional_to_inverse_frequency(self):
        """Per-class total weight should be equal (each class contributes equally)."""
        inputs = list(range(110))
        labels = [0] * 100 + [1] * 5 + [2] * 5
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=10, shuffle=True, num_workers=0)

        # Each class's total weight should be the same
        class_0_total = dl.wgts[:100].sum()
        class_1_total = dl.wgts[100:105].sum()
        class_2_total = dl.wgts[105:110].sum()
        # All classes should contribute equally
        assert abs(class_0_total - class_1_total) < 1e-10
        assert abs(class_1_total - class_2_total) < 1e-10


class TestClassBalancedDLWithTensors:
    """Tests with tensor labels (common in real usage)."""

    def test_tensor_labels(self):
        """Should handle tensor labels correctly."""
        inputs = [torch.randn(3) for _ in range(50)]
        labels = [torch.tensor(0)] * 40 + [torch.tensor(1)] * 10
        ds = SimpleDataset(inputs, labels)
        dl = ClassBalancedDL(ds, bs=5, shuffle=True, num_workers=0)

        assert len(dl.wgts) == 50
        assert abs(dl.wgts.sum() - 1.0) < 1e-10
        # Minority class should have higher weight
        assert dl.wgts[40] > dl.wgts[0]
