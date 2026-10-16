"""Tests for fastai.data.load module.

Covers the DataLoader class and related utility functions: fa_collate, fa_convert,
SkipItemException, and collate_error.
"""
import sys
import os
import pytest

# Ensure the repo root is on sys.path so sub-package imports resolve correctly.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
import numpy as np
from torch.utils.data import IterableDataset

from fastai.data.load import DataLoader, fa_collate, fa_convert, SkipItemException, collate_error


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors produces a stacked tensor."""
        items = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_scalars(self):
        """Collating a list of scalar tensors produces a 1-D tensor."""
        items = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        result = fa_collate(items)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays produces a tensor."""
        items = [np.array([1, 2]), np.array([3, 4])]
        result = fa_collate(items)
        expected = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(result, expected)

    def test_collate_tuples_maintains_type(self):
        """Collating tuples of tensors maintains the tuple structure."""
        items = [
            (torch.tensor([1.0]), torch.tensor(0)),
            (torch.tensor([2.0]), torch.tensor(1)),
        ]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(result[1], torch.tensor([0, 1]))

    def test_collate_lists_maintains_type(self):
        """Collating lists of tensors maintains the list structure."""
        items = [
            [torch.tensor([1.0]), torch.tensor(0)],
            [torch.tensor([2.0]), torch.tensor(1)],
        ]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_collate_strings(self):
        """Collating strings produces a list of strings."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_single_item(self):
        """Collating a single-element list works correctly."""
        items = [torch.tensor([5.0, 6.0])]
        result = fa_collate(items)
        expected = torch.tensor([[5.0, 6.0]])
        assert torch.equal(result, expected)


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_numpy_array(self):
        """Converting a numpy array produces a tensor."""
        arr = np.array([1, 2, 3])
        result = fa_convert(arr)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(result, expected)

    def test_convert_tensor_passthrough(self):
        """Converting a tensor returns the same tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_tuple_maintains_type(self):
        """Converting a tuple maintains the tuple type."""
        tup = (torch.tensor([1, 2]), torch.tensor([3, 4]))
        result = fa_convert(tup)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))

    def test_convert_list_maintains_type(self):
        """Converting a list maintains the list type."""
        lst = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = fa_convert(lst)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_convert_string(self):
        """Converting a string returns it unchanged."""
        s = "hello"
        result = fa_convert(s)
        assert result == "hello"

    def test_convert_nested_tuple(self):
        """Converting a nested tuple with numpy arrays converts inner arrays."""
        tup = (np.array([1, 2]), np.array([3, 4]))
        result = fa_convert(tup)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException is a proper Exception subclass."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException can be raised and caught normally."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()

    def test_dataloader_skips_item(self):
        """DataLoader skips items when SkipItemException is raised in after_item."""
        dl = DataLoader(list(range(10)), bs=5)

        def skip_evens(x):
            if x % 2 == 0:
                raise SkipItemException
            return x

        dl.after_item = skip_evens
        batch = dl.one_batch()
        # Only odd numbers should be in the batch
        for item in batch.tolist():
            assert item % 2 == 1


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for the collate_error function."""

    def test_raises_with_mismatch_info(self):
        """collate_error re-raises with helpful info about shape mismatches."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([1.0])),
            (torch.tensor([1, 2]), torch.tensor([1.0])),  # different shape at idx 0
        ]
        # collate_error uses bare `raise` so must be called inside an except block
        with pytest.raises(RuntimeError, match="Mismatch found"):
            try:
                raise RuntimeError("collate failed")
            except RuntimeError as e:
                collate_error(e, batch)

    def test_no_raise_when_shapes_match(self):
        """collate_error does not raise when shapes all match."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([1.0])),
            (torch.tensor([4, 5, 6]), torch.tensor([2.0])),
        ]
        # Should not raise since all shapes match; bare `raise` is not triggered
        try:
            raise RuntimeError("collate failed")
        except RuntimeError as e:
            collate_error(e, batch)
            # If we get here, it means collate_error did not re-raise


# ============================================================
# Tests for DataLoader initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization and configuration."""

    def test_basic_creation(self):
        """DataLoader can be created with a simple list dataset."""
        dl = DataLoader(list(range(10)), bs=4)
        assert dl.bs == 4
        assert dl.n == 10
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """The batch_size kwarg is an alias for bs (PyTorch compat)."""
        dl = DataLoader(list(range(10)), batch_size=5)
        assert dl.bs == 5

    def test_indexed_auto_detection(self):
        """Indexed is auto-detected for list datasets."""
        dl = DataLoader(list(range(10)), bs=4)
        assert dl.indexed is True

    def test_iterable_dataset_not_indexed(self):
        """IterableDataset is correctly detected as non-indexed."""
        class SimpleIter(IterableDataset):
            def __iter__(self):
                return iter(range(10))
        dl = DataLoader(SimpleIter(), bs=4)
        assert dl.indexed is False

    def test_shuffle_iterable_raises(self):
        """Shuffling an iterable dataset raises ValueError."""
        class SimpleIter(IterableDataset):
            def __iter__(self):
                return iter(range(10))
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(SimpleIter(), bs=4, shuffle=True)

    def test_custom_n_overrides_len(self):
        """Custom n parameter overrides the dataset length."""
        dl = DataLoader(list(range(100)), bs=10, n=30)
        assert dl.n == 30

    def test_device_setting(self):
        """Device can be set during initialization."""
        dl = DataLoader(list(range(10)), bs=4, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_device_none_default(self):
        """Device defaults to None when not specified."""
        dl = DataLoader(list(range(10)), bs=4)
        assert dl.device is None

    def test_drop_last_requires_bs(self):
        """drop_last=True requires bs to be set (assertion)."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(10)), bs=None, drop_last=True)


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length calculation."""

    def test_len_exact_division(self):
        """Length is exact when n divides evenly by bs."""
        dl = DataLoader(list(range(12)), bs=4)
        assert len(dl) == 3

    def test_len_with_remainder(self):
        """Length includes partial batch when drop_last=False."""
        dl = DataLoader(list(range(10)), bs=4)
        assert len(dl) == 3  # 4 + 4 + 2

    def test_len_drop_last(self):
        """Length drops partial batch when drop_last=True."""
        dl = DataLoader(list(range(10)), bs=4, drop_last=True)
        assert len(dl) == 2  # 4 + 4 only

    def test_len_bs_none(self):
        """Length equals n when bs is None (prebatched mode)."""
        dl = DataLoader([torch.tensor([1, 2]), torch.tensor([3, 4])], bs=None)
        assert len(dl) == 2

    def test_len_n_none_raises(self):
        """Length raises TypeError when n is None."""
        class NoLen:
            def __getitem__(self, i):
                return i
        dl = DataLoader(NoLen(), bs=4, n=None, indexed=True)
        with pytest.raises(TypeError):
            len(dl)


# ============================================================
# Tests for DataLoader iteration
# ============================================================

class TestDataLoaderIteration:
    """Tests for DataLoader iteration behavior."""

    def test_iterate_all_items(self):
        """Iterating produces all items in the dataset."""
        dl = DataLoader(list(range(12)), bs=4, shuffle=False)
        all_items = []
        for batch in dl:
            all_items.extend(batch.tolist())
        assert sorted(all_items) == list(range(12))

    def test_iterate_preserves_order(self):
        """Without shuffle, iteration preserves dataset order."""
        dl = DataLoader(list(range(12)), bs=4, shuffle=False)
        all_items = []
        for batch in dl:
            all_items.extend(batch.tolist())
        assert all_items == list(range(12))

    def test_iterate_with_shuffle(self):
        """With shuffle, iteration produces all items but in different order."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        all_items = []
        for batch in dl:
            all_items.extend(batch.tolist())
        assert sorted(all_items) == list(range(20))
        # Very unlikely to be in original order with 20 items
        assert all_items != list(range(20))

    def test_iterate_drop_last(self):
        """drop_last=True discards the final incomplete batch."""
        dl = DataLoader(list(range(10)), bs=4, drop_last=True)
        all_items = []
        for batch in dl:
            all_items.extend(batch.tolist())
            assert len(batch) == 4  # all batches should be full
        assert len(all_items) == 8

    def test_iterate_tuple_dataset(self):
        """DataLoader correctly batches tuple-element datasets."""
        ds = [(torch.tensor([float(i)]), torch.tensor(i)) for i in range(8)]
        dl = DataLoader(ds, bs=4)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (4, 1)
        assert batch[1].shape == (4,)

    def test_iterate_iterable_dataset(self):
        """DataLoader works with IterableDataset."""
        class CountDataset(IterableDataset):
            def __iter__(self):
                return iter(range(8))
        dl = DataLoader(CountDataset(), bs=4)
        batch = dl.one_batch()
        assert batch.shape == (4,)

    def test_multiple_iterations(self):
        """DataLoader can be iterated multiple times."""
        dl = DataLoader(list(range(8)), bs=4, shuffle=False)
        first_pass = [b.tolist() for b in dl]
        second_pass = [b.tolist() for b in dl]
        assert first_pass == second_pass

    def test_shuffle_different_across_epochs(self):
        """Shuffled DataLoader produces different orderings across epochs."""
        dl = DataLoader(list(range(40)), bs=40, shuffle=True)
        batch1 = next(iter(dl)).tolist()
        batch2 = next(iter(dl)).tolist()
        # Same items but very likely different order
        assert sorted(batch1) == sorted(batch2)
        # With 40 items, it's nearly impossible to get the same shuffle twice
        assert batch1 != batch2


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for the one_batch method."""

    def test_one_batch_returns_first_batch(self):
        """one_batch returns the first batch of data."""
        dl = DataLoader(list(range(12)), bs=4, shuffle=False)
        batch = dl.one_batch()
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(batch, expected)

    def test_one_batch_empty_raises(self):
        """one_batch raises ValueError when DataLoader has no batches."""
        dl = DataLoader(list(range(3)), bs=4, drop_last=True, n=3)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()

    def test_one_batch_correct_size(self):
        """one_batch returns a batch of the correct size."""
        dl = DataLoader(list(range(20)), bs=8)
        batch = dl.one_batch()
        assert len(batch) == 8


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for the get_idxs method."""

    def test_get_idxs_sequential(self):
        """get_idxs returns sequential indices when shuffle=False."""
        dl = DataLoader(list(range(10)), bs=4, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == list(range(10))

    def test_get_idxs_shuffled(self):
        """get_idxs returns shuffled indices when shuffle=True."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(20))
        assert idxs != list(range(20))  # very likely shuffled

    def test_get_idxs_respects_n(self):
        """get_idxs respects the n parameter."""
        dl = DataLoader(list(range(100)), bs=4, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for the new() method."""

    def test_new_preserves_settings(self):
        """new() preserves original settings when no overrides given."""
        dl = DataLoader(list(range(10)), bs=4, shuffle=True, drop_last=True)
        dl2 = dl.new()
        assert dl2.bs == 4
        assert dl2.shuffle is True
        assert dl2.drop_last is True

    def test_new_overrides_bs(self):
        """new() can override the batch size."""
        dl = DataLoader(list(range(10)), bs=4)
        dl2 = dl.new(bs=2)
        assert dl2.bs == 2

    def test_new_overrides_dataset(self):
        """new() can override the dataset."""
        dl = DataLoader(list(range(10)), bs=4)
        new_ds = list(range(20))
        dl2 = dl.new(dataset=new_ds)
        assert dl2.n == 20

    def test_new_preserves_device(self):
        """new() preserves device setting."""
        dl = DataLoader(list(range(10)), bs=4, device='cpu')
        dl2 = dl.new()
        assert dl2.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader.to
# ============================================================

class TestDataLoaderTo:
    """Tests for the to() method (device placement)."""

    def test_to_cpu(self):
        """to('cpu') sets the device to CPU."""
        dl = DataLoader(list(range(10)), bs=4)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_to_changes_device(self):
        """to() updates the device property."""
        dl = DataLoader(list(range(10)), bs=4, device=None)
        assert dl.device is None
        dl.to('cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader prebatched mode
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for prebatched mode (bs=None)."""

    def test_prebatched_flag(self):
        """prebatched property is True when bs is None."""
        dl = DataLoader([torch.tensor([1, 2, 3])], bs=None)
        assert dl.prebatched is True

    def test_not_prebatched(self):
        """prebatched property is False when bs is set."""
        dl = DataLoader(list(range(10)), bs=4)
        assert dl.prebatched is False

    def test_prebatched_iteration(self):
        """Prebatched mode yields items directly from the dataset."""
        batches = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        dl = DataLoader(batches, bs=None)
        result = dl.one_batch()
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_prebatched_len(self):
        """Prebatched mode length equals number of items."""
        batches = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
        dl = DataLoader(batches, bs=None)
        assert len(dl) == 3


# ============================================================
# Tests for DataLoader callbacks
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback hooks."""

    def test_before_iter_called(self):
        """before_iter is called at the start of iteration."""
        called = []
        dl = DataLoader(list(range(8)), bs=4)
        original_before_iter = dl.before_iter
        def my_before_iter(x=None):
            called.append('before_iter')
            return original_before_iter(x)
        dl.before_iter = my_before_iter
        list(dl)
        assert 'before_iter' in called

    def test_after_iter_called(self):
        """after_iter is called after iteration completes."""
        called = []
        dl = DataLoader(list(range(8)), bs=4)
        original_after_iter = dl.after_iter
        def my_after_iter(x=None):
            called.append('after_iter')
            return original_after_iter(x)
        dl.after_iter = my_after_iter
        list(dl)
        assert 'after_iter' in called

    def test_after_item_transforms_items(self):
        """after_item is applied to each item."""
        dl = DataLoader(list(range(8)), bs=4)
        dl.after_item = lambda x: x * 10
        batch = dl.one_batch()
        expected = torch.tensor([0, 10, 20, 30])
        assert torch.equal(batch, expected)

    def test_before_batch_transforms_batch(self):
        """before_batch is applied to the batch list before collation."""
        dl = DataLoader(list(range(8)), bs=4)
        dl.before_batch = lambda b: [x * 2 for x in b]
        batch = dl.one_batch()
        expected = torch.tensor([0, 2, 4, 6])
        assert torch.equal(batch, expected)

    def test_after_batch_transforms_batch(self):
        """after_batch is applied to the collated batch."""
        dl = DataLoader(list(range(8)), bs=4)
        dl.after_batch = lambda b: b + 100
        batch = dl.one_batch()
        expected = torch.tensor([100, 101, 102, 103])
        assert torch.equal(batch, expected)


# ============================================================
# Tests for DataLoader with various dataset types
# ============================================================

class TestDataLoaderDatasets:
    """Tests for DataLoader with different dataset types."""

    def test_list_of_ints(self):
        """DataLoader works with a plain list of integers."""
        dl = DataLoader(list(range(8)), bs=4)
        batch = dl.one_batch()
        assert batch.shape == (4,)
        assert batch.dtype == torch.int64

    def test_list_of_floats(self):
        """DataLoader works with a plain list of floats."""
        dl = DataLoader([float(i) for i in range(8)], bs=4)
        batch = dl.one_batch()
        assert batch.shape == (4,)
        assert batch.dtype == torch.float64

    def test_list_of_tensors(self):
        """DataLoader works with a list of tensors."""
        ds = [torch.randn(3) for _ in range(8)]
        dl = DataLoader(ds, bs=4)
        batch = dl.one_batch()
        assert batch.shape == (4, 3)

    def test_list_of_tuples(self):
        """DataLoader works with a list of tuples (x, y pairs)."""
        ds = [(torch.randn(5), torch.tensor(i % 3)) for i in range(12)]
        dl = DataLoader(ds, bs=4)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (4, 5)
        assert batch[1].shape == (4,)

    def test_list_of_numpy_arrays(self):
        """DataLoader works with a list of numpy arrays."""
        ds = [np.array([i, i + 1, i + 2]) for i in range(8)]
        dl = DataLoader(ds, bs=4)
        batch = dl.one_batch()
        assert batch.shape == (4, 3)


# ============================================================
# Tests for DataLoader shuffle_fn and randomize
# ============================================================

class TestDataLoaderShuffle:
    """Tests for shuffle functionality."""

    def test_shuffle_fn_permutes(self):
        """shuffle_fn returns a permutation of the input indices."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        idxs = list(range(20))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == idxs
        assert shuffled != idxs  # very likely different

    def test_randomize_changes_rng(self):
        """randomize() changes the internal random state."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for the create_item method."""

    def test_create_item_indexed(self):
        """create_item returns the indexed dataset element."""
        dl = DataLoader([10, 20, 30, 40], bs=2)
        item = dl.create_item(2)
        assert item == 30

    def test_create_item_iterable_raises_on_index(self):
        """create_item raises IndexError for numeric index on iterable dataset."""
        class SimpleIter(IterableDataset):
            def __iter__(self):
                return iter(range(5))
        dl = DataLoader(SimpleIter(), bs=2)
        with pytest.raises(IndexError, match="Cannot index an iterable dataset"):
            dl.create_item(0)


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for the do_item method."""

    def test_do_item_returns_item(self):
        """do_item returns the item when no exception."""
        dl = DataLoader([10, 20, 30], bs=2)
        result = dl.do_item(1)
        assert result == 20

    def test_do_item_skips_on_exception(self):
        """do_item returns None when SkipItemException is raised."""
        dl = DataLoader([10, 20, 30], bs=2)
        dl.after_item = lambda x: (_ for _ in ()).throw(SkipItemException) if x == 20 else x
        # Simpler approach: direct test
        def skip_20(x):
            if x == 20:
                raise SkipItemException
            return x
        dl.after_item = skip_20
        result = dl.do_item(1)  # index 1 -> item 20 -> skipped
        assert result is None


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for the chunkify method."""

    def test_chunkify_splits_into_batches(self):
        """chunkify splits items into chunks of size bs."""
        dl = DataLoader(list(range(10)), bs=3)
        items = iter(range(9))
        chunks = list(dl.chunkify(items))
        assert len(chunks) == 3
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]

    def test_chunkify_prebatched_passthrough(self):
        """chunkify passes items through unchanged in prebatched mode."""
        dl = DataLoader([torch.tensor([1, 2])], bs=None)
        items = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = list(dl.chunkify(iter(items)))
        # In prebatched mode, items pass through as-is
        assert len(result) == 2


# ============================================================
# Tests for DataLoader device placement
# ============================================================

class TestDataLoaderDevice:
    """Tests for device placement during iteration."""

    def test_batches_on_cpu_device(self):
        """When device='cpu', batches are on CPU."""
        dl = DataLoader(list(range(8)), bs=4, device='cpu')
        batch = dl.one_batch()
        assert batch.device == torch.device('cpu')

    def test_batches_no_device(self):
        """When device=None, batches are on default device (CPU)."""
        dl = DataLoader(list(range(8)), bs=4, device=None)
        batch = dl.one_batch()
        assert batch.device == torch.device('cpu')
