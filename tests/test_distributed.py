"""Tests for fastai.distributed.DataLoaders.distributed.

Covers the `DataLoaders.distributed()` method which shards each underlying
`DataLoader` across ranks via `DistributedDL`, giving balanced partition sizes
(padding to a multiple of `world_size`) and deterministic, reproducible ordering
(rank-0's index order is broadcast to every rank).

These tests run on CPU in a single process. There is no `torch.distributed`
setup; instead we replicate the notebook's hidden test harness by monkeypatching
`DistributedDL._broadcast` (save/load the index tensor through a temp file) and
`DistributedDL._to_detach` (fake the all-gather), exactly as the hidden cells in
nbs/20a_distributed.ipynb do. This lets us simulate any (rank, world_size).
"""
import sys
import os
import math
import tempfile

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastcore.basics import patch
from fastcore.foundation import L
from fastcore.test import test_eq as _test_eq  # aliased so pytest does not collect it
from fastai.torch_core import to_detach, apply
from fastai.data.core import DataLoaders, TfmdDL
from fastai.distributed import DistributedDL, _round_to_multiple


# ============================================================
# CPU simulation harness (mirrors nbs/20a_distributed.ipynb hidden cells)
# ============================================================
# We patch _broadcast/_to_detach once at import time so every test exercises the
# real sharding logic without needing a real DDP process group.

_tmp_file = tempfile.NamedTemporaryFile().name


@patch
def _broadcast(self: DistributedDL, t, rank):
    "Mocked broadcast: rank `rank` saves the index tensor, other ranks load it."
    t = torch.LongTensor(t)
    if rank == self.rank: torch.save(t, _tmp_file)
    else:                 t.data = torch.load(_tmp_file)
    return t.tolist()


@patch
def _to_detach(self: DistributedDL, b, cpu=True, gather=True):
    "Mocked gather: returns the right gathered size but -100 for other ranks' tensors."
    b = to_detach(b, cpu, gather)
    if not gather: return b
    def _inner(b, cpu, gather):
        if b.ndim == 0: b = b[None]
        b = torch.cat([b if i == self.rank else torch.full_like(b, -100) for i in range(self.world_size)])
        return b if b.ndim > 0 else b.mean()
    return apply(_inner, b, cpu, gather)


def _base_dls(n=50, bs=12, num_workers=0):
    "A `DataLoaders` with identical train/valid loaders over `range(n)`."
    # num_workers=0 keeps iteration single-process (the fake multiprocessing loader
    # is incompatible with this torch build); sharding/ordering are unaffected.
    dl = TfmdDL(list(range(n)), bs=bs, num_workers=num_workers)
    return DataLoaders(dl, dl)


# ============================================================
# Tests for wrapping behaviour
# ============================================================

class TestDistributedWrapping:
    """`DataLoaders.distributed` wraps each loader in a `DistributedDL`."""

    def test_each_loader_is_distributeddl(self):
        """Every returned loader must be a `DistributedDL` instance."""
        dls = _base_dls()
        res = dls.distributed(rank=0, world_size=4)
        assert len(res.loaders) == 2
        for dl in res.loaders:
            assert isinstance(dl, DistributedDL)

    def test_train_and_valid_both_wrapped(self):
        """Both the train and valid loaders are wrapped (multi-loader case)."""
        dls = _base_dls()
        res = dls.distributed(rank=1, world_size=4)
        assert isinstance(res.train, DistributedDL)
        assert isinstance(res.valid, DistributedDL)

    def test_already_distributed_not_double_wrapped(self):
        """An already-`DistributedDL` loader is returned as-is, not re-wrapped."""
        dl = TfmdDL(list(range(50)), bs=12, num_workers=0)
        wrapped = DistributedDL(dl, 0, 4)
        dls = DataLoaders(wrapped)
        res = dls.distributed(rank=0, world_size=4)
        # same object, and not a DistributedDL-of-a-DistributedDL
        assert res.loaders[0] is wrapped
        assert not isinstance(res.loaders[0].dl, DistributedDL)

    def test_original_dls_untouched(self):
        """`distributed()` returns a new `DataLoaders`; the original is unchanged."""
        dls = _base_dls()
        orig_loaders = list(dls.loaders)
        res = dls.distributed(rank=0, world_size=4)
        assert res is not dls
        assert list(dls.loaders) == orig_loaders
        assert not isinstance(dls.loaders[0], DistributedDL)

    def test_path_preserved(self):
        """`path` carries over to the returned `DataLoaders`."""
        dl = TfmdDL(list(range(50)), bs=12, num_workers=0)
        dls = DataLoaders(dl, dl, path='some/path')
        res = dls.distributed(rank=0, world_size=4)
        assert str(res.path) == str(dls.path)


# ============================================================
# Tests for balanced partition sizes
# ============================================================

class TestBalancedPartitions:
    """Partitions are balanced by padding to a multiple of `world_size`."""

    def test_equal_len_across_ranks_uneven_split(self):
        """50 items / world_size=4 does not divide evenly; every rank's len is equal."""
        world_size = 4
        lens = []
        for r in range(world_size):
            res = _base_dls(n=50, bs=12).distributed(rank=r, world_size=world_size)
            lens.append(len(res.train))
        assert len(set(lens)) == 1, f"ranks have unequal lengths: {lens}"

    def test_len_equals_padded_formula(self):
        """Each rank's len == _round_to_multiple(len(base), world_size)//world_size."""
        base_len = 50
        bs = 12
        world_size = 4
        base = TfmdDL(list(range(base_len)), bs=bs, num_workers=0)
        expected = _round_to_multiple(len(base), world_size) // world_size
        for r in range(world_size):
            res = DataLoaders(base, base).distributed(rank=r, world_size=world_size)
            assert len(res.train) == expected
            assert len(res.valid) == expected

    def test_balanced_for_various_world_sizes(self):
        """Balanced lengths hold for several non-dividing world sizes."""
        for world_size in (2, 3, 4, 5, 7):
            base = TfmdDL(list(range(50)), bs=12, num_workers=0)
            expected = _round_to_multiple(len(base), world_size) // world_size
            lens = [len(DataLoaders(base).distributed(rank=r, world_size=world_size).train)
                    for r in range(world_size)]
            assert all(l == expected for l in lens), f"world_size={world_size}: {lens}"


# ============================================================
# Tests for deterministic / reproducible ordering
# ============================================================

class TestDeterministicOrdering:
    """Ordering is deterministic and matches the known arange pattern."""

    def test_matches_arange_pattern(self):
        """50-item, bs=12, world_size=4: each rank matches the notebook's arange pattern."""
        world_size = 4
        for i in range(world_size):
            res = _base_dls(n=50, bs=12).distributed(rank=i, world_size=world_size)
            for dl in res.loaders:
                _test_eq(list(dl), (torch.arange(i * 13, i * 13 + 12) % 50,
                                   torch.tensor([i * 13 + 12]) % 50))

    def test_repeated_calls_identical_per_rank(self):
        """Calling distributed() twice with same rank/world_size gives identical order."""
        world_size = 4
        for i in range(world_size):
            first = list(_base_dls(n=50, bs=12).distributed(rank=i, world_size=world_size).train)
            second = list(_base_dls(n=50, bs=12).distributed(rank=i, world_size=world_size).train)
            assert len(first) == len(second)
            for a, b in zip(first, second):
                assert torch.equal(a, b)

    def test_matches_direct_distributeddl(self):
        """DataLoaders.distributed matches wrapping each loader in DistributedDL directly."""
        world_size = 4
        for i in range(world_size):
            base = TfmdDL(list(range(50)), bs=12, num_workers=0)
            via_method = list(DataLoaders(base).distributed(rank=i, world_size=world_size).train)
            direct = list(DistributedDL(base, i, world_size))
            assert len(via_method) == len(direct)
            for a, b in zip(via_method, direct):
                assert torch.equal(a, b)

    def test_ranks_cover_whole_dataset(self):
        """Concatenating all ranks' (deduped) indices covers the whole dataset."""
        world_size = 4
        n = 50
        seen = set()
        for i in range(world_size):
            res = _base_dls(n=n, bs=12).distributed(rank=i, world_size=world_size)
            for batch in res.train:
                seen.update(int(x) % n for x in batch.tolist())
        assert seen == set(range(n))
