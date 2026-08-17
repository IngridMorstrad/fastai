"""Tests for ClassBalancedDL - class-balanced sampling DataLoader."""
import sys
import os
import pytest
from collections import Counter

# Ensure the repo root is on sys.path so sub-package imports resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from fastai.data.core import ClassBalancedDL, TfmdDL, Datasets


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def imbalanced_dataset():
    """Create an imbalanced dataset with 90% class 0 and 10% class 1."""
    items = list(range(100))
    labels = [0] * 90 + [1] * 10
    dsets = Datasets(items, tfms=[[lambda x: x], [lambda x: labels[x]]])
    return dsets, labels


@pytest.fixture
def multiclass_dataset():
    """Create a 3-class imbalanced dataset: 80% class 0, 15% class 1, 5% class 2."""
    items = list(range(100))
    labels = [0] * 80 + [1] * 15 + [2] * 5
    dsets = Datasets(items, tfms=[[lambda x: x], [lambda x: labels[x]]])
    return dsets, labels


# ============================================================
# Tests for weight computation
# ============================================================

class TestWeightComputation:
    """Tests for _compute_weights method."""

    def test_binary_weights_correct(self, imbalanced_dataset):
        """Weights are inversely proportional to class frequency."""
        dsets, labels = imbalanced_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True, labels=labels)

        # Class 0: 90 samples → weight = 100 / (2 * 90)
        # Class 1: 10 samples → weight = 100 / (2 * 10)
        expected_w0 = 100.0 / (2 * 90)
        expected_w1 = 100.0 / (2 * 10)

        assert abs(dl._sample_weights[0].item() - expected_w0) < 1e-4
        assert abs(dl._sample_weights[90].item() - expected_w1) < 1e-4

    def test_multiclass_weights_correct(self, multiclass_dataset):
        """Weights are correct for a 3-class problem."""
        dsets, labels = multiclass_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True, labels=labels)

        expected_w0 = 100.0 / (3 * 80)
        expected_w1 = 100.0 / (3 * 15)
        expected_w2 = 100.0 / (3 * 5)

        assert abs(dl._sample_weights[0].item() - expected_w0) < 1e-4
        assert abs(dl._sample_weights[80].item() - expected_w1) < 1e-4
        assert abs(dl._sample_weights[95].item() - expected_w2) < 1e-4

    def test_balanced_dataset_equal_weights(self):
        """A perfectly balanced dataset should have equal weights for all samples."""
        items = list(range(20))
        labels = [0] * 10 + [1] * 10
        dsets = Datasets(items, tfms=[[lambda x: x], [lambda x: labels[x]]])
        dl = ClassBalancedDL(dsets, bs=5, shuffle=True, labels=labels)

        # All weights should be equal: 20 / (2 * 10) = 1.0
        for w in dl._sample_weights:
            assert abs(w.item() - 1.0) < 1e-4


# ============================================================
# Tests for get_idxs behavior
# ============================================================

class TestGetIdxs:
    """Tests for get_idxs method."""

    def test_shuffle_produces_balanced_indices(self, imbalanced_dataset):
        """When shuffle=True, sampled indices should balance class representation."""
        dsets, labels = imbalanced_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True, labels=labels)

        # Sample many times to get a stable distribution
        class_counts = Counter()
        for _ in range(50):
            idxs = dl.get_idxs()
            for i in idxs:
                class_counts[0 if i < 90 else 1] += 1

        total = sum(class_counts.values())
        class_1_ratio = class_counts[1] / total
        # Should be approximately 50% (within a wide tolerance for randomness)
        assert 0.35 < class_1_ratio < 0.65, f"Class 1 ratio {class_1_ratio:.2f} not near 0.5"

    def test_no_shuffle_returns_sequential(self, imbalanced_dataset):
        """When shuffle=False, get_idxs returns sequential indices."""
        dsets, labels = imbalanced_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=False, labels=labels)
        idxs = dl.get_idxs()
        assert idxs == list(range(100))

    def test_correct_number_of_indices(self, imbalanced_dataset):
        """get_idxs always returns n indices."""
        dsets, labels = imbalanced_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True, labels=labels)
        idxs = dl.get_idxs()
        assert len(idxs) == 100


# ============================================================
# Tests for label auto-extraction
# ============================================================

class TestLabelExtraction:
    """Tests for automatic label extraction from Datasets."""

    def test_auto_extract_from_datasets(self, imbalanced_dataset):
        """Labels are automatically extracted from Datasets.tls[1]."""
        dsets, labels = imbalanced_dataset
        # Don't pass labels explicitly
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True)

        # Should get the same weights as if labels were passed explicitly
        expected_w0 = 100.0 / (2 * 90)
        expected_w1 = 100.0 / (2 * 10)
        assert abs(dl._sample_weights[0].item() - expected_w0) < 1e-4
        assert abs(dl._sample_weights[90].item() - expected_w1) < 1e-4

    def test_raises_on_incompatible_dataset(self):
        """Raises ValueError when dataset doesn't have extractable labels."""
        # A simple list doesn't have tls
        with pytest.raises(ValueError, match="Could not auto-extract labels"):
            ClassBalancedDL([1, 2, 3], bs=2, shuffle=True)


# ============================================================
# Tests for new() method
# ============================================================

class TestNew:
    """Tests for the new() method creating derived DataLoaders."""

    def test_new_creates_class_balanced_dl(self, imbalanced_dataset):
        """new() creates another ClassBalancedDL instance."""
        dsets, labels = imbalanced_dataset
        dl = ClassBalancedDL(dsets, bs=10, shuffle=True, labels=labels)
        dl_new = dl.new(dsets)
        assert isinstance(dl_new, ClassBalancedDL)
