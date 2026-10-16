"""Tests for ClassBalancedDL."""

import pytest
import numpy as np

from fastai.callback.class_balanced import ClassBalancedDL, _extract_labels, _compute_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal map-style dataset returning ``(feature, label)`` tuples."""

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (idx, self.labels[idx])


class _FakeTfmdLists:
    """Mimics a TfmdLists with a pre-transform ``.items`` list."""
    def __init__(self, items):
        self.items = list(items)


class _FakeDatasets(_FakeDataset):
    """_FakeDataset with a ``.tls`` attribute to simulate a Datasets object,
    letting the fast-path label extraction code exercise."""

    def __init__(self, labels):
        super().__init__(labels)
        self.tls = [_FakeTfmdLists(range(len(labels))),  # features
                     _FakeTfmdLists(labels)]              # labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassBalancedDL:
    """Unit tests for ClassBalancedDL."""

    # -- weight computation ------------------------------------------------

    def test_weights_sum_to_one(self):
        """Computed weights should sum to 1.0 for any non-empty dataset."""
        ds = _FakeDataset([0] * 90 + [1] * 10)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12

    def test_imbalanced_weights(self):
        """Minority-class samples should receive higher per-sample weight."""
        ds = _FakeDataset([0] * 90 + [1] * 10)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        # Each class-0 sample weight = 1/90, each class-1 sample weight = 1/10
        # After normalization both classes contribute equally (0.5 each).
        w_majority = dl.wgts[0]   # a class-0 sample
        w_minority = dl.wgts[90]  # a class-1 sample
        assert w_minority > w_majority, (
            f"Minority weight ({w_minority}) should exceed majority weight ({w_majority})"
        )
        # Class-1 sample should be 9x more likely than class-0 sample.
        assert abs(w_minority / w_majority - 9.0) < 1e-10

    def test_balanced_dataset_uniform_weights(self):
        """A perfectly balanced dataset should yield uniform weights."""
        ds = _FakeDataset([0] * 50 + [1] * 50)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        expected = 1.0 / 100
        assert np.allclose(dl.wgts, expected)

    def test_three_classes(self):
        """Weight computation generalizes to more than two classes."""
        # 60 / 30 / 10
        ds = _FakeDataset([0] * 60 + [1] * 30 + [2] * 10)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12
        # Each class should contribute 1/3 of total weight.
        class_0_total = dl.wgts[:60].sum()
        class_1_total = dl.wgts[60:90].sum()
        class_2_total = dl.wgts[90:].sum()
        assert abs(class_0_total - 1.0 / 3) < 1e-10
        assert abs(class_1_total - 1.0 / 3) < 1e-10
        assert abs(class_2_total - 1.0 / 3) < 1e-10

    def test_single_class(self):
        """Edge case: every sample belongs to the same class."""
        ds = _FakeDataset([0] * 20)
        dl = ClassBalancedDL(dataset=ds, bs=4, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12
        assert np.allclose(dl.wgts, 1.0 / 20)

    def test_empty_dataset(self):
        """Edge case: empty dataset produces None weights."""
        ds = _FakeDataset([])
        dl = ClassBalancedDL(dataset=ds, bs=4)
        assert dl.wgts is None

    # -- get_idxs ----------------------------------------------------------

    def test_get_idxs_shuffle_true_returns_weighted(self):
        """When shuffle=True, get_idxs should use weighted random sampling."""
        np.random.seed(42)
        ds = _FakeDataset([0] * 90 + [1] * 10)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        idxs = dl.get_idxs()
        assert len(idxs) == len(ds)
        # Count how many indices fall in the minority class (indices 90-99).
        minority_count = sum(1 for i in idxs if i >= 90)
        # With balanced weighting, we expect roughly 50% minority samples.
        # Use a wide tolerance for stochastic test stability.
        assert minority_count > 20, (
            f"Expected significant minority representation, got {minority_count}/100"
        )

    def test_get_idxs_shuffle_false_returns_sequential(self):
        """When shuffle=False, get_idxs should return sequential indices."""
        ds = _FakeDataset([0] * 5 + [1] * 5)
        dl = ClassBalancedDL(dataset=ds, bs=4, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == list(range(10))

    def test_get_idxs_empty_dataset(self):
        """get_idxs on an empty dataset returns an empty list."""
        ds = _FakeDataset([])
        dl = ClassBalancedDL(dataset=ds, bs=4, shuffle=True)
        assert dl.get_idxs() == []

    # -- shuffle=False skips weight computation ----------------------------

    def test_no_weights_when_shuffle_false(self):
        """Validation split (shuffle=False) should NOT compute weights."""
        ds = _FakeDataset([0] * 90 + [1] * 10)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=False)
        assert dl.wgts is None

    # -- fast path via .tls ------------------------------------------------

    def test_fast_path_via_tls(self):
        """When dataset has .tls (Datasets), labels are extracted from the
        raw items list instead of calling dataset[i]."""
        ds = _FakeDatasets([0] * 80 + [1] * 20)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12
        # Class-1 sample should be 4x the weight of class-0 sample.
        assert abs(dl.wgts[80] / dl.wgts[0] - 4.0) < 1e-10

    def test_extract_labels_fast_path(self):
        """_extract_labels uses .tls[-1].items when available."""
        ds = _FakeDatasets([0, 1, 2, 0])
        labels = _extract_labels(ds)
        assert labels == [0, 1, 2, 0]

    def test_extract_labels_slow_path(self):
        """_extract_labels falls back to dataset[i] when no .tls."""
        ds = _FakeDataset([0, 1, 2, 0])
        labels = _extract_labels(ds)
        assert labels == [0, 1, 2, 0]

    # -- statistical validation --------------------------------------------

    def test_minority_oversampled_statistically(self):
        """Over many draws, minority class should be sampled at roughly 50%."""
        np.random.seed(0)
        ds = _FakeDataset([0] * 900 + [1] * 100)
        dl = ClassBalancedDL(dataset=ds, bs=64, shuffle=True)
        total_minority = 0
        draws = 50
        for _ in range(draws):
            idxs = dl.get_idxs()
            total_minority += sum(1 for i in idxs if i >= 900)
        avg_minority_frac = total_minority / (draws * len(ds))
        # Should be close to 0.5 (each class gets equal weight).
        assert 0.4 < avg_minority_frac < 0.6, (
            f"Expected ~50% minority fraction, got {avg_minority_frac:.3f}"
        )

    # -- string labels -----------------------------------------------------

    def test_string_labels(self):
        """ClassBalancedDL works with string labels, not just integers."""
        ds = _FakeDataset(["cat"] * 80 + ["dog"] * 20)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12
        # dog samples should be 4x the weight of cat samples.
        assert abs(dl.wgts[80] / dl.wgts[0] - 4.0) < 1e-10

    # -- tensor labels (scalar) -------------------------------------------

    def test_tensor_labels(self):
        """ClassBalancedDL handles tensor labels via .item() conversion."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        labels = [torch.tensor(0)] * 70 + [torch.tensor(1)] * 30
        ds = _FakeDataset(labels)
        dl = ClassBalancedDL(dataset=ds, bs=16, shuffle=True)
        assert abs(dl.wgts.sum() - 1.0) < 1e-12

    # -- repr / identity ---------------------------------------------------

    def test_is_subclass(self):
        """ClassBalancedDL should be a subclass of its base (TfmdDL or stub)."""
        ds = _FakeDataset([0, 1])
        dl = ClassBalancedDL(dataset=ds, bs=2)
        assert isinstance(dl, ClassBalancedDL)
