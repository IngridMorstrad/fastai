"""Tests for fastai/data/load.py module.

Covers fa_collate, fa_convert, SkipItemException, collate_error,
DataLoader class initialization, iteration, batching, shuffling,
device placement, callbacks, and edge cases.
"""
import sys
import os
import pytest

# Ensure the repo root is on sys.path so sub-package imports resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Patch missing functions from fastcore/fasttransform that are needed at runtime.
import fasttransform
import fastcore.basics

try:
    from fastcore.dispatch import retain_types, cast
except (ImportError, AttributeError):
    from fasttransform import retain_types, cast
    import fastai.torch_core as _tc
    if not hasattr(_tc, 'retain_types'):
        _tc.retain_types = retain_types
    if not hasattr(_tc, 'cast'):
        _tc.cast = cast
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

    def test_collate_integers(self):
        """Collating plain integers via default_collate."""
        items = [1, 2, 3, 4]
        result = fa_collate(items)
        expected = torch.tensor([1, 2, 3, 4])
        assert torch.equal(result, expected)

    def test_collate_floats(self):
        """Collating plain floats via default_collate."""
        items = [1.5, 2.5, 3.5]
        result = fa_collate(items)
        expected = torch.tensor([1.5, 2.5, 3.5])
        assert torch.equal(result, expected)


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor returns the same tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = fa_convert(t)
        assert isinstance(result, Tensor)
        assert torch.equal(result, t)

    def test_convert_numpy(self):
        """Converting a numpy array returns a tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = fa_convert(arr)
        assert isinstance(result, Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_convert_string(self):
        """Converting a string returns it unchanged."""
        result = fa_convert("hello")
        assert result == "hello"

    def test_convert_tuple(self):
        """Converting a tuple preserves type and converts each element."""
        items = (np.array([1.0]), np.array([2.0]))
        result = fa_convert(items)
        assert isinstance(result, tuple)
        assert isinstance(result[0], Tensor)
        assert isinstance(result[1], Tensor)

    def test_convert_list(self):
        """Converting a list preserves type and converts each element."""
        items = [np.array([1.0]), np.array([2.0])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert isinstance(result[0], Tensor)
        assert isinstance(result[1], Tensor)

    def test_convert_mapping(self):
        """Converting a mapping converts each value."""
        d = {'a': np.array([1.0]), 'b': np.array([2.0])}
        result = fa_convert(d)
        assert isinstance(result, dict)
        assert isinstance(result['a'], Tensor)
        assert isinstance(result['b'], Tensor)

    def test_convert_integer(self):
        """Plain integers (non-collate, non-Sequence) pass through unchanged."""
        result = fa_convert(42)
        assert result == 42


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

    def test_with_message(self):
        """SkipItemException can carry a message."""
        with pytest.raises(SkipItemException, match="skip this"):
            raise SkipItemException("skip this")


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for collate_error function."""

    def test_raises_on_shape_mismatch(self):
        """collate_error raises with informative message on shape mismatch."""
        batch = [
            (torch.zeros(3, 4), torch.zeros(2)),
            (torch.zeros(3, 5), torch.zeros(2)),
        ]
        e = RuntimeError("original error")
        with pytest.raises(RuntimeError) as exc_info:
            try:
                raise e
            except RuntimeError:
                collate_error(e, batch)
        error_msg = str(exc_info.value)
        assert "Mismatch found" in error_msg
        assert "axis 0" in error_msg

    def test_error_message_contains_shapes(self):
        """The error message includes both shapes that differ."""
        batch = [
            (torch.zeros(3, 4),),
            (torch.zeros(3, 5),),
        ]
        e = RuntimeError("collate failed")
        with pytest.raises(RuntimeError) as exc_info:
            try:
                raise e
            except RuntimeError:
                collate_error(e, batch)
        error_msg = str(exc_info.value)
        assert 'torch.Size([3, 4])' in error_msg
        assert 'torch.Size([3, 5])' in error_msg

    def test_no_raise_when_shapes_match(self):
        """Does not raise when all items have the same shape."""
        batch = [
            (torch.zeros(3, 4), torch.zeros(2)),
            (torch.zeros(3, 4), torch.zeros(2)),
        ]
        e = RuntimeError("original error")
        collate_error(e, batch)


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

    def test_prebatched_mode(self):
        """bs=None means prebatched mode (no chunking)."""
        dataset = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(dataset, bs=None)
        assert dl.prebatched is True

    def test_not_prebatched(self):
        """prebatched is False when bs is set."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.prebatched is False

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

    def test_device_none_default(self):
        """Device defaults to None."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.device is None

    def test_device_init_param(self):
        """Device can be set via init parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, device='cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader __len__."""

    def test_len_exact_division(self):
        """n evenly divisible by bs gives exact count."""
        dataset = list(range(12))
        dl = DataLoader(dataset, bs=4)
        assert len(dl) == 3

    def test_len_with_remainder(self):
        """Includes partial batch when drop_last=False."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3)
        assert len(dl) == 4  # 3+3+3+1

    def test_len_with_drop_last(self):
        """Excludes partial batch when drop_last=True."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3, drop_last=True)
        assert len(dl) == 3

    def test_len_prebatched(self):
        """With bs=None, length equals n."""
        dataset = [[1, 2], [3, 4], [5, 6]]
        dl = DataLoader(dataset, bs=None)
        assert len(dl) == 3

    def test_len_raises_if_n_none(self):
        """Raises TypeError when n is None."""
        class InfDS:
            def __getitem__(self, idx):
                return idx
        dl = DataLoader(InfDS(), bs=2, n=None)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_item(self):
        """Dataset with 1 item, bs=1."""
        ds = [42]
        dl = DataLoader(ds, bs=1)
        assert len(dl) == 1

    def test_len_bs_larger_than_n(self):
        """When bs > n, there is still 1 batch."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=10)
        assert len(dl) == 1

    def test_len_bs_larger_than_n_drop_last(self):
        """When bs > n and drop_last, there are 0 batches."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=10, drop_last=True)
        assert len(dl) == 0


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_no_shuffle(self):
        """Without shuffle, indices are sequential."""
        dataset = list(range(5))
        dl = DataLoader(dataset, bs=2, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_with_shuffle(self):
        """With shuffle, indices are a permutation of the same set."""
        dataset = list(range(50))
        dl = DataLoader(dataset, bs=10, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(50))
        # Very unlikely to be in order
        assert idxs != list(range(50))

    def test_get_idxs_length_matches_n(self):
        """get_idxs returns exactly n indices."""
        ds = list(range(15))
        dl = DataLoader(ds, bs=4, shuffle=True)
        idxs = dl.get_idxs()
        assert len(idxs) == 15

    def test_get_idxs_with_explicit_n(self):
        """get_idxs respects explicit n parameter."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=5, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10


# ============================================================
# Tests for DataLoader.shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn method."""

    def test_shuffle_fn_preserves_elements(self):
        """shuffle_fn returns a permutation of input."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(10))
        assert len(shuffled) == 10

    def test_shuffle_fn_produces_different_orders(self):
        """shuffle_fn produces different orderings on repeated calls."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(100))
        result1 = dl.shuffle_fn(idxs)
        dl.randomize()
        result2 = dl.shuffle_fn(idxs)
        assert result1 != result2


# ============================================================
# Tests for DataLoader.randomize
# ============================================================

class TestDataLoaderRandomize:
    """Tests for DataLoader.randomize method."""

    def test_randomize_changes_rng_state(self):
        """randomize() changes the internal rng state."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


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

    def test_multiple_epochs(self):
        """DataLoader can be iterated multiple times."""
        dataset = list(range(4))
        dl = DataLoader(dataset, bs=2, num_workers=0)
        epoch1 = list(dl)
        epoch2 = list(dl)
        assert len(epoch1) == 2
        assert len(epoch2) == 2

    def test_shuffle_different_across_epochs(self):
        """Shuffled DataLoader produces different orders across epochs."""
        dataset = list(range(100))
        dl = DataLoader(dataset, bs=100, num_workers=0, shuffle=True)
        epoch1 = list(dl)[0].tolist()
        epoch2 = list(dl)[0].tolist()
        assert sorted(epoch1) == sorted(epoch2) == list(range(100))
        assert epoch1 != epoch2

    def test_no_shuffle_same_across_epochs(self):
        """Non-shuffled DataLoader produces the same order across epochs."""
        dataset = list(range(20))
        dl = DataLoader(dataset, bs=10, num_workers=0, shuffle=False)
        epoch1 = [b.tolist() for b in dl]
        epoch2 = [b.tolist() for b in dl]
        assert epoch1 == epoch2


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_returns_correct_size(self):
        """one_batch returns a batch of size bs."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=4, num_workers=0)
        batch = dl.one_batch()
        assert isinstance(batch, Tensor)
        assert batch.shape == (4,)

    def test_one_batch_tuple_dataset(self):
        """one_batch works with tuple datasets."""
        dataset = [(torch.tensor([float(i)]), torch.tensor([float(i * 2)])) for i in range(6)]
        dl = DataLoader(dataset, bs=3, num_workers=0)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (3, 1)
        assert batch[1].shape == (3, 1)

    def test_one_batch_empty_raises(self):
        """one_batch on empty DataLoader raises ValueError."""
        dl = DataLoader([], bs=1, num_workers=0)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_creates_copy(self):
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

    def test_new_is_independent(self):
        """new() creates an independent DataLoader."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, shuffle=True)
        dl2 = dl.new(shuffle=False)
        assert dl.shuffle is True
        assert dl2.shuffle is False


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item with item skipping."""

    def test_do_item_returns_item(self):
        """do_item returns the dataset item for the given index."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)
        assert dl.do_item(0) == 10
        assert dl.do_item(2) == 30

    def test_do_item_skip_exception_returns_none(self):
        """do_item returns None when SkipItemException is raised."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)

        def skip_after_item(x):
            if x == 20:
                raise SkipItemException()
            return x

        dl.after_item = skip_after_item
        assert dl.do_item(0) == 10
        assert dl.do_item(1) is None
        assert dl.do_item(2) == 30

    def test_skip_item_during_iteration(self):
        """Items raising SkipItemException are filtered out during iteration."""
        ds = list(range(6))
        dl = DataLoader(ds, bs=6, num_workers=0, shuffle=False)

        def skip_odds(x):
            if x % 2 != 0:
                raise SkipItemException()
            return x

        dl.after_item = skip_odds
        batches = list(dl)
        all_items = torch.cat(batches).tolist()
        assert all_items == [0, 2, 4]


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_create_item_indexed(self):
        """With indexed dataset, create_item returns dataset[s]."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_create_item_non_indexed(self):
        """With non-indexed dataset, create_item(None) uses the iterator."""
        ds = iter([10, 20, 30])
        dl = DataLoader(ds, bs=None, indexed=False)
        dl.it = iter([10, 20, 30])
        assert dl.create_item(None) == 10
        assert dl.create_item(None) == 20
        assert dl.create_item(None) == 30

    def test_create_item_non_indexed_raises_on_numeric_index(self):
        """Non-indexed dataset raises IndexError when given a numeric index."""
        ds = iter([10, 20, 30])
        dl = DataLoader(ds, bs=None, indexed=False)
        dl.it = iter([10, 20, 30])
        with pytest.raises(IndexError, match="Cannot index an iterable dataset"):
            dl.create_item(0)


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_with_bs(self):
        """chunkify splits items into chunks of size bs."""
        dataset = list(range(10))
        dl = DataLoader(dataset, bs=3, num_workers=0)
        chunks = list(dl.chunkify(iter(range(9))))
        assert len(chunks) == 3
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]

    def test_chunkify_prebatched(self):
        """In prebatched mode, chunkify returns items as-is."""
        dataset = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(dataset, bs=None, num_workers=0)
        items = iter([[1, 2, 3], [4, 5, 6]])
        chunks = list(dl.chunkify(items))
        assert chunks == [[1, 2, 3], [4, 5, 6]]

    def test_chunkify_with_drop_last(self):
        """With drop_last=True, chunkify drops incomplete final chunk."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        chunks = list(dl.chunkify(iter(range(10))))
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(list(chunk)) == 3


# ============================================================
# Tests for DataLoader.device property
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device management."""

    def test_device_setter_string(self):
        """Setting device with string works."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_setter_torch_device(self):
        """Setting device with a torch.device works."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.device = torch.device('cpu')
        assert dl.device == torch.device('cpu')

    def test_to_sets_device(self):
        """to() method sets the device property."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader callbacks
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback/hook methods."""

    def test_before_batch_hook(self):
        """before_batch is called on each batch."""
        dataset = list(range(6))
        dl = DataLoader(dataset, bs=3, num_workers=0)
        calls = []

        def track_before_batch(b):
            calls.append(1)
            return b
        dl.before_batch = track_before_batch

        list(dl)
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

        list(dl)
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
# Integration tests
# ============================================================

class TestDataLoaderIntegration:
    """End-to-end integration tests for DataLoader."""

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
