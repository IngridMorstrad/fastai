"""ClassBalancedDL -- A DataLoader that auto-computes per-sample weights from
class label frequencies so that every class is sampled equally in expectation.

For heavily imbalanced classification datasets, minority classes get
proportionally higher sampling probability, removing the need for manual
oversampling or explicit weight computation.

Usage with fastai::

    from fastai.callback.class_balanced import ClassBalancedDL
    dls = dsets.dataloaders(bs=64, dl_type=ClassBalancedDL)
"""

import numpy as np
from collections import Counter

__all__ = ['ClassBalancedDL']


# ---------------------------------------------------------------------------
# Import the real TfmdDL when the full fastai stack is available; fall back to
# a lightweight stub so the class can be unit-tested in isolation.
# ---------------------------------------------------------------------------
try:
    from ..data.core import TfmdDL as _BaseDL
    from fastcore.meta import delegates as _delegates
except Exception:
    class _BaseDL:
        """Minimal stand-in for TfmdDL used only during isolated unit tests."""
        def __init__(self, dataset=None, bs=64, shuffle=False, **kwargs):
            self.dataset = dataset
            self.bs = bs
            self.shuffle = shuffle
            self.n = len(dataset) if dataset is not None else 0
        def get_idxs(self):
            return list(range(self.n))
    _delegates = None


def _extract_labels(dataset):
    """Extract class labels from *dataset* without triggering the full
    transform pipeline when possible.

    For ``Datasets`` objects the raw label column is available at
    ``dataset.tls[-1].items``, which avoids image decoding / augmentation
    that ``dataset[i]`` would trigger.  Falls back to per-item indexing
    for generic map-style datasets.
    """
    n = len(dataset)
    if n == 0:
        return []

    # Fast path: Datasets stores pre-transform items on each TfmdLists.
    tls = getattr(dataset, 'tls', None)
    if tls is not None:
        try:
            raw = list(tls[-1].items)
            if len(raw) == n:
                labels = []
                for lbl in raw:
                    if hasattr(lbl, 'item'):
                        lbl = lbl.item()
                    labels.append(lbl)
                return labels
        except Exception:
            pass  # fall through to slow path

    # Slow path: iterate the dataset (triggers full transform pipeline).
    labels = []
    for i in range(n):
        item = dataset[i]
        lbl = item[-1]
        if hasattr(lbl, 'item'):
            lbl = lbl.item()
        labels.append(lbl)
    return labels


def _compute_weights(dataset):
    """Return a normalized weight array (sums to 1) where each sample's
    weight is ``1 / count_of_its_class``, or *None* when *dataset* is
    empty."""
    labels = _extract_labels(dataset)
    if not labels:
        return None

    counts = Counter(labels)
    wgts = np.array([1.0 / counts[lbl] for lbl in labels], dtype=np.float64)
    total = wgts.sum()
    if total > 0:
        wgts /= total
    return wgts


# Apply @delegates() when the real fastcore decorator is available so that
# TfmdDL kwargs (after_item, after_batch, num_workers, ...) are accepted
# by ClassBalancedDL the same way WeightedDL and PartialDL accept them.
def _maybe_delegates(cls):
    if _delegates is not None:
        cls = _delegates()(cls)
    return cls


@_maybe_delegates
class ClassBalancedDL(_BaseDL):
    """DataLoader that reweights samples so every class has equal expected
    representation per epoch.

    During training (``shuffle=True``), indices are drawn via
    ``np.random.choice`` with per-sample probability inversely proportional
    to that sample's class frequency.  During validation (``shuffle=False``),
    standard sequential ordering is used and no weight computation is
    performed.

    Parameters
    ----------
    dataset : object
        A map-style dataset.  Each item should be a tuple whose *last*
        element is the label (int, str, or tensor scalar).
    bs : int, default 64
        Batch size.
    **kwargs
        Forwarded to the parent ``TfmdDL`` / ``DataLoader``.
    """

    def __init__(self, dataset=None, bs=None, shuffle=False, **kwargs):
        super().__init__(dataset=dataset, bs=bs, shuffle=shuffle, **kwargs)
        # Only compute weights when they will actually be used (training).
        # Validation splits (shuffle=False) never consult the weights, so
        # skip the potentially expensive label scan.
        if self.shuffle and dataset is not None and len(dataset) > 0:
            self.wgts = _compute_weights(dataset)
        else:
            self.wgts = None

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------
    def get_idxs(self):
        """Weighted random sampling when shuffling; sequential otherwise."""
        if self.n == 0:
            return []
        if not self.shuffle or self.wgts is None:
            return super().get_idxs()
        return list(np.random.choice(self.n, self.n, p=self.wgts))
