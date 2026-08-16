"""Tests for fastai/data/load.py module.

Covers fa_collate, fa_convert, SkipItemException, collate_error,
DataLoader class initialization, iteration, batching, shuffling,
device placement, and edge cases.
"""
import sys
import os
import pytest

# Ensure the repo root is on sys.path so sub-package imports resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Patch missing functions from fastcore/fasttransform that are needed at runtime.
# The installed fastcore version moved retain_types/cast to fasttransform;
# we patch them into the relevant module namespaces so DataLoader iteration works.
import fasttransform
import fastcore.basics

# Patch retain_types and cast into the fastcore.dispatch module namespace
# since fastai imports them via `from fastcore.dispatch import *`
class _FakeDispatch:
    """Shim module providing retain_types and cast that fastai expects."""
    pass

# If fastcore.dispatch doesn't properly export these, patch them in
try:
    from fastcore.dispatch import retain_types, cast
except (ImportError, AttributeError):
    from fasttransform import retain_types, cast
    # Patch into the modules that need them
    import fastai.torch_core as _tc
    if not hasattr(_tc, 'retain_types'):
        _tc.retain_types = retain_types
    if not hasattr(_tc, 'cast'):
        _tc.cast = cast

    # Also ensure they exist in the data.load module's global scope
    import fastai.data.load as _dl_mod
    if not hasattr(_dl_mod, 'retain_types'):
        _dl_mod.retain_types = retain_types

import torch
import numpy as np
from torch import Tensor

from fastai.data.load import (
    fa_collate, fa_convert, SkipItemException, collate_error, DataLoader,
    _FakeLoader,
)


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors produces a stacked tensor."""
        items = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = fa_collate(items)
        assert isinstance(result, Tensor)
        assert result.shape == (2, 3)
        assert result[0].tolist() == [1, 2, 3]
        assert result[1].tolist() == [4, 5, 6]

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays produces a tensor."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        assert isinstance(result, Tensor)
        assert result.shape == (2, 2)

    def test_collate_strings(self):
        """Collating strings produces a list of strings."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_tuples_preserves_type(self):
        """Collating tuples preserves the tuple type and collates each element."""
        items = [(torch.tensor([1]), torch.tensor([2])),
                 (torch.tensor([3]), torch.tensor([4]))]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].tolist() == [[1], [3]]
        assert result[1].tolist() == [[2], [4]]

    def test_collate_lists_preserves_type(self):
        """Collating lists preserves the list type and collates each element."""
        items = [[torch.tensor([1]), torch.tensor([2])],
                 [torch.tensor([3]), torch.tensor([4])]]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].tolist() == [[1], [3]]
        assert result[1].tolist() == [[2], [4]]

    def test_collate_scalars(self):
        """Collating scalar tensors produces a 1D tensor."""
        items = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        result = fa_collate(items)
        assert result.tolist() == [1, 2, 3]

    def test_collate_dicts(self):
        """Collating dicts produces a dict with collated values."""
        items = [{'a': torch.tensor([1]), 'b': torch.tensor([2])},
                 {'a': torch.tensor([3]), 'b': torch.tensor([4])}]
        result = fa_collate(items)
        assert isinstance(result, dict)
        assert result['a'].tolist() == [[1], [3]]
        assert result['b'].tolist() == [[2], [4]]


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor returns the same tensor."""
        t = torch.tensor([1, 2, 3])
        result = fa_convert(t)
        assert isinstance(result, Tensor)
        assert torch.equal(result, t)

    def test_convert_numpy(self):
        """Converting a numpy array returns a tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = fa_convert(arr)
        assert isinstance(result, Tensor)

    def test_convert_string(self):
        """Converting a string returns the string unchanged."""
        result = fa_convert("hello")
        assert result == "hello"

    def test_convert_list_of_tensors(self):
        """Converting a list of tensors returns a list of converted items."""
        items = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_convert_tuple_of_tensors(self):
        """Converting a tuple preserves tuple type."""
        items = (torch.tensor([1, 2]), torch.tensor([3, 4]))
        result = fa_convert(items)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for SkipItemException."""

    def test_is_exception(self):
        """SkipItemException is an Exception subclass."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException can be raised and caught."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()


# ============================================================
# Tests for DataLoader initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_basic_init_with_list(self):
        """DataLoader can be initialized with a simple list dataset."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=2)
        assert dl.n == 10
        assert dl.bs == 2
        assert dl.shuffle is False
        assert dl.drop_last is False
        assert dl.indexed is True

    def test_batch_size_alias(self):
        """batch_size parameter is an alias for bs."""
        dataset = list(range(10))
        dl = DataLoader(dataset, batch_size=4)
        assert dl.bs == 4

    def test_default_bs_is_none_allowed(self):
        """bs=None means prebatched mode (no chunking)."""
        dataset = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(dataset, bs=None)
        assert dl.prebatched is True

    def test_drop_last_requires_bs(self):
        """drop_last=True without bs raises AssertionError."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(10)), bs=None, drop_last=True)

    def test_shuffle_requires_indexed(self):
        """shuffle=True with non-indexed dataset raises ValueError."""
        class IterDS:
            def __iter__(self):
                return iter(range(10))
            def __len__(self):
                return 10
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(IterDS(), shuffle=True)

    def test_n_inferred_from_dataset_len(self):
        """n is inferred from len(dataset) if not provided."""
        dataset = list(range(20))
        dl = DataLoader(dataset, bs=5)
        assert dl.n == 20

    def test_n_explicit(self):
        """Explicit n overrides len(dataset)."""
        dataset = list(range(20))
        dl = DataLoader(dataset, bs=5, n=10)
        assert dl.n == 10

    def test_n_none_for_unsized_dataset(self):
        """n is None for datasets without __len__."""
        class InfDS:
            def __getitem__(self, idx):
                return idx
        dl = DataLoader(InfDS(), bs=2, n=None)
        assert dl.n is None


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader __len__."""

    def test_len_basic(self):
        """__len__ returns correct number of batches."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3)
        # 10 / 3 = 3 full batches + 1 partial = 4
        assert len(dl) == 4

    def test_len_exact_division(self):
        """__len__ with exact division gives exact count."""
        dataset = list(range(12))
        dl = DataLoader(dataset, bs=4)
        assert len(dl) == 3

    def test_len_drop_last(self):
        """__len__ with drop_last drops incomplete final batch."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3, drop_last=True)
        # 10 / 3 = 3 full batches, drop partial
        assert len(dl) == 3

    def test_len_no_bs(self):
        """__len__ with bs=None returns n (prebatched mode)."""
        dataset = [[1, 2], [3, 4], [5, 6]]
        dl = DataLoader(dataset, bs=None)
        assert len(dl) == 3

    def test_len_raises_if_n_none(self):
        """__len__ raises TypeError when n is None."""
        class InfDS:
            def __getitem__(self, idx):
                return idx
        dl = DataLoader(InfDS(), bs=2, n=None)
        with pytest.raises(TypeError):
            len(dl)


# ============================================================
# Tests for DataLoader iteration
# ============================================================

class TestDataLoaderIteration:
    """Tests for DataLoader iteration and batching."""

    def test_basic_iteration(self):
        """DataLoader produces correct batches from a list dataset."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        batches = list(dl)
        assert len(batches) == 2
        # Each batch should be a tensor of size 5
        assert batches[0].shape == (5,)
        assert batches[1].shape == (5,)

    def test_iteration_drop_last(self):
        """DataLoader with drop_last omits the last incomplete batch."""
        dataset = list(range(7))
        dl = DataLoader(dataset, bs=3, num_workers=0, drop_last=True)
        batches = list(dl)
        assert len(batches) == 2
        for b in batches:
            assert b.shape == (3,)

    def test_iteration_with_tuples(self):
        """DataLoader handles tuple items correctly."""
        dataset = [(torch.tensor([i]), torch.tensor([i * 2])) for i in range(6)]
        dl = DataLoader(dataset, bs=3, num_workers=0)
        batches = list(dl)
        assert len(batches) == 2
        # Each batch is a tuple of two tensors
        for b in batches:
            assert isinstance(b, tuple)
            assert b[0].shape == (3, 1)
            assert b[1].shape == (3, 1)

    def test_iteration_preserves_all_data(self):
        """All dataset items appear in the output when not shuffled."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=4, num_workers=0, shuffle=False)
        batches = list(dl)
        all_items = torch.cat(batches).tolist()
        assert sorted(all_items) == list(range(10))

    def test_shuffle_changes_order(self):
        """Shuffle produces a different order (with high probability)."""
        dataset = list(range(100))
        dl = DataLoader(dataset, bs=100, num_workers=0, shuffle=True)
        batch = list(dl)[0]
        # Very unlikely that a shuffled 100-item list is in order
        assert batch.tolist() != list(range(100))

    def test_shuffle_preserves_all_items(self):
        """Shuffle does not lose or duplicate items."""
        dataset = list(range(50))
        dl = DataLoader(dataset, bs=50, num_workers=0, shuffle=True)
        batch = list(dl)[0]
        assert sorted(batch.tolist()) == list(range(50))

    def test_prebatched_mode(self):
        """With bs=None, each dataset item is treated as a pre-formed batch."""
        dataset = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        dl = DataLoader(dataset, bs=None, num_workers=0)
        batches = list(dl)
        assert len(batches) == 2
        assert batches[0].tolist() == [1, 2, 3]
        assert batches[1].tolist() == [4, 5, 6]


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_returns_first_batch(self):
        """one_batch returns the first batch of data."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        batch = dl.one_batch()
        assert isinstance(batch, Tensor)
        assert batch.shape == (5,)

    def test_one_batch_empty_raises(self):
        """one_batch raises ValueError for empty DataLoader."""
        dl = DataLoader([], bs=1, num_workers=0)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_creates_copy_with_same_params(self):
        """new() creates a DataLoader with the same configuration."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0, shuffle=True, drop_last=True)
        dl2 = dl.new()
        assert dl2.bs == dl.bs
        assert dl2.shuffle == dl.shuffle
        assert dl2.drop_last == dl.drop_last
        assert dl2.n == dl.n

    def test_new_with_different_dataset(self):
        """new(dataset=...) uses a different dataset."""
        dataset1 = list(range(10))
        dataset2 = list(range(20))
        dl = DataLoader(dataset1, bs=5, num_workers=0)
        dl2 = dl.new(dataset=dataset2)
        assert dl2.n == 20
        assert len(dl2) == 4

    def test_new_with_different_bs(self):
        """new(bs=...) uses a different batch size."""
        dataset = list(range(12))
        dl = DataLoader(dataset, bs=4, num_workers=0)
        dl2 = dl.new(bs=6)
        assert dl2.bs == 6
        assert len(dl2) == 2


# ============================================================
# Tests for DataLoader.to and device property
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device management."""

    def test_to_sets_device(self):
        """to() method sets the device property."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_device_none_by_default(self):
        """device is None by default."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        assert dl.device is None

    def test_device_init(self):
        """device can be set during initialization."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0, device='cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_no_shuffle(self):
        """get_idxs returns sequential indices when shuffle=False."""
        dataset = list(range(5))
        dl = DataLoader(dataset, bs=2, num_workers=0, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_shuffle(self):
        """get_idxs returns shuffled indices when shuffle=True."""
        dataset = list(range(50))
        dl = DataLoader(dataset, bs=10, num_workers=0, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(50))
        # Likely not in order
        assert idxs != list(range(50))


# ============================================================
# Tests for DataLoader.shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn method."""

    def test_shuffle_fn_preserves_elements(self):
        """shuffle_fn returns a permutation of input."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=2, num_workers=0)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(10))
        assert len(shuffled) == 10


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item with item skipping."""

    def test_skip_item_exception_skips(self):
        """Items raising SkipItemException are skipped (return None)."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)

        # Override after_item to skip even numbers
        def skip_evens(x):
            if x % 2 == 0:
                raise SkipItemException()
            return x
        dl.after_item = skip_evens

        result = dl.do_item(0)
        assert result is None

        result = dl.do_item(1)
        assert result == 1


# ============================================================
# Tests for DataLoader.randomize
# ============================================================

class TestDataLoaderRandomize:
    """Tests for DataLoader.randomize method."""

    def test_randomize_changes_rng_state(self):
        """randomize() changes the internal rng state."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=2, num_workers=0, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_with_bs(self):
        """chunkify splits items into chunks of size bs."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3, num_workers=0)
        items = iter(range(9))
        chunks = list(dl.chunkify(items))
        assert len(chunks) == 3
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]

    def test_chunkify_prebatched(self):
        """chunkify in prebatched mode returns items as-is."""
        dataset = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(dataset, bs=None, num_workers=0)
        items = iter([[1, 2, 3], [4, 5, 6]])
        chunks = list(dl.chunkify(items))
        assert chunks == [[1, 2, 3], [4, 5, 6]]


# ============================================================
# Tests for _FakeLoader
# ============================================================

class TestFakeLoader:
    """Tests for the _FakeLoader helper class."""

    def test_fake_loader_init(self):
        """_FakeLoader stores its configuration."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        fl = dl.fake_l
        assert fl.num_workers == 0
        assert fl.pin_memory is False
        assert fl.d is dl

    def test_fake_loader_no_multiproc_context(self):
        """no_multiproc context manager temporarily sets num_workers=0."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=5, num_workers=0)
        fl = dl.fake_l
        with fl.no_multiproc() as d:
            assert fl.num_workers == 0
            assert d is dl


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for collate_error function."""

    def test_collate_error_gives_informative_message(self):
        """collate_error modifies the exception message to be informative."""
        batch = [
            (torch.zeros(3, 4), torch.zeros(2)),
            (torch.zeros(3, 5), torch.zeros(2)),  # Mismatch: (3,4) vs (3,5)
        ]
        e = RuntimeError("original error")
        # collate_error uses bare `raise` so it must be called from an except block
        with pytest.raises(RuntimeError) as exc_info:
            try:
                raise e
            except RuntimeError:
                collate_error(e, batch)
        assert "Mismatch found" in str(exc_info.value)

    def test_collate_error_no_mismatch_no_raise(self):
        """collate_error with matching shapes does not raise."""
        batch = [
            (torch.zeros(3, 4), torch.zeros(2)),
            (torch.zeros(3, 4), torch.zeros(2)),
        ]
        e = RuntimeError("original error")
        # Should not raise since there's no mismatch
        collate_error(e, batch)


# ============================================================
# Tests for DataLoader with callbacks
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback/hook methods."""

    def test_before_batch_hook(self):
        """before_batch is called on each batch before collation."""
        dataset = list(range(6))
        dl = DataLoader(dataset, bs=3, num_workers=0)
        calls = []

        def track_before_batch(b):
            calls.append(1)
            return b
        dl.before_batch = track_before_batch

        batches = list(dl)
        assert len(calls) == 2

    def test_after_batch_hook(self):
        """after_batch is called on each batch after collation."""
        dataset = list(range(6))
        dl = DataLoader(dataset, bs=3, num_workers=0)
        calls = []

        def track_after_batch(b):
            calls.append(b)
            return b
        dl.after_batch = track_after_batch

        batches = list(dl)
        assert len(calls) == 2

    def test_after_item_transforms_items(self):
        """after_item can transform individual items."""
        dataset = list(range(6))
        dl = DataLoader(dataset, bs=3, num_workers=0)

        def double(x):
            return x * 2
        dl.after_item = double

        batches = list(dl)
        all_items = torch.cat(batches).tolist()
        assert sorted(all_items) == [0, 2, 4, 6, 8, 10]


# ============================================================
# Integration tests
# ============================================================

class TestDataLoaderIntegration:
    """End-to-end integration tests for DataLoader."""

    def test_multiple_epochs_produce_different_order_when_shuffled(self):
        """Multiple iterations with shuffle produce different orderings."""
        dataset = list(range(100))
        dl = DataLoader(dataset, bs=100, num_workers=0, shuffle=True)
        epoch1 = list(dl)[0].tolist()
        epoch2 = list(dl)[0].tolist()
        # Different epochs should produce different orderings
        # (astronomically unlikely to be the same for 100 items)
        assert epoch1 != epoch2

    def test_multiple_epochs_same_order_when_not_shuffled(self):
        """Multiple iterations without shuffle produce the same ordering."""
        dataset = list(range(20))
        dl = DataLoader(dataset, bs=10, num_workers=0, shuffle=False)
        epoch1 = [b.tolist() for b in dl]
        epoch2 = [b.tolist() for b in dl]
        assert epoch1 == epoch2

    def test_tensor_dataset(self):
        """DataLoader works with a TensorDataset-like indexed dataset."""
        class TensorDS:
            def __init__(self, x, y):
                self.x, self.y = x, y
            def __getitem__(self, idx):
                return (self.x[idx], self.y[idx])
            def __len__(self):
                return len(self.x)

        x = torch.arange(12).float()
        y = torch.arange(12).float() * 2
        ds = TensorDS(x, y)
        dl = DataLoader(ds, bs=4, num_workers=0)
        batches = list(dl)
        assert len(batches) == 3
        for b in batches:
            assert isinstance(b, tuple)
            assert b[0].shape == (4,)
            assert b[1].shape == (4,)

    def test_single_item_dataset(self):
        """DataLoader handles a single-item dataset."""
        dataset = [42]
        dl = DataLoader(dataset, bs=1, num_workers=0)
        batches = list(dl)
        assert len(batches) == 1
        assert batches[0].item() == 42

    def test_large_bs_smaller_dataset(self):
        """When bs > n, one batch contains all items."""
        dataset = list(range(5))
        dl = DataLoader(dataset, bs=100, num_workers=0)
        batches = list(dl)
        assert len(batches) == 1
        assert batches[0].shape == (5,)


# ============================================================
# Tests for DataLoader prebatched mode (bs=None)
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for DataLoader with prebatched data (bs=None)."""

    def test_prebatched_property(self):
        """prebatched is True when bs is None."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        assert dl.prebatched is True

    def test_not_prebatched(self):
        """prebatched is False when bs is set."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.prebatched is False

    def test_prebatched_iteration(self):
        """Prebatched DataLoader applies fa_convert instead of fa_collate."""
        ds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        dl = DataLoader(ds, bs=None)
        batches = list(dl)
        assert len(batches) == 2
        # Each batch is a converted numpy array (now a tensor)
        assert isinstance(batches[0], torch.Tensor)
        assert torch.equal(batches[0], torch.tensor([1.0, 2.0]))

    def test_prebatched_len(self):
        """len of prebatched DataLoader equals n."""
        ds = [[1, 2], [3, 4], [5, 6], [7, 8]]
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 4


# ============================================================
# Tests for SkipItemException handling in do_item
# ============================================================

class TestSkipItemHandling:
    """Tests for SkipItemException handling during iteration."""

    def test_do_item_returns_none_on_skip(self):
        """do_item returns None when SkipItemException is raised."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2)

        def after_item_that_skips(x):
            if x == 2:
                raise SkipItemException()
            return x

        dl.after_item = after_item_that_skips
        result = dl.do_item(2)
        assert result is None

    def test_do_item_normal_return(self):
        """do_item returns the item when no exception is raised."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2)
        result = dl.do_item(0)
        assert result == 0

    def test_skip_item_during_iteration(self):
        """Items that raise SkipItemException are filtered out during iteration."""
        ds = list(range(6))
        dl = DataLoader(ds, bs=2, shuffle=False)

        def skip_odds(x):
            if x % 2 != 0:
                raise SkipItemException()
            return x

        dl.after_item = skip_odds
        batches = list(dl)
        # Only even items remain: 0, 2, 4 -> one batch of 2 and one partial of 1
        all_items = torch.cat(batches).tolist()
        assert all_items == [0, 2, 4]


# ============================================================
# Tests for DataLoader construction parameters
# ============================================================

class TestDataLoaderConstruction:
    """Tests for DataLoader initialization with various parameters."""

    def test_construction_with_shuffle(self):
        """DataLoader can be constructed with shuffle=True for indexed datasets."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        assert dl.shuffle is True

    def test_construction_with_drop_last(self):
        """DataLoader can be constructed with drop_last=True."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        assert dl.drop_last is True

    def test_construction_indexed_explicit(self):
        """DataLoader accepts explicit indexed parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, indexed=False)
        assert dl.indexed is False

    def test_construction_with_device(self):
        """DataLoader can be constructed with a device parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_construction_bs_none_is_prebatched(self):
        """When bs is None, the DataLoader is in prebatched mode."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        assert dl.prebatched is True


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_create_item_indexed(self):
        """With indexed dataset, create_item should return dataset[s]."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_create_item_non_indexed(self):
        """With non-indexed dataset, create_item(None) should use the iterator."""
        ds = iter([10, 20, 30])
        dl = DataLoader(ds, bs=None, indexed=False)
        dl.it = iter([10, 20, 30])
        assert dl.create_item(None) == 10
        assert dl.create_item(None) == 20
        assert dl.create_item(None) == 30

    def test_create_item_non_indexed_raises_on_numeric_index(self):
        """Non-indexed dataset should raise IndexError when given a numeric index."""
        ds = iter([10, 20, 30])
        dl = DataLoader(ds, bs=None, indexed=False)
        dl.it = iter([10, 20, 30])
        with pytest.raises(IndexError, match="Cannot index an iterable dataset"):
            dl.create_item(0)
