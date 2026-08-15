"""Tests for fastai/data/load.py - DataLoader, fa_collate, fa_convert, SkipItemException.

Covers:
- fa_collate with various types (tensors, numpy arrays, sequences)
- fa_convert with various types
- SkipItemException behavior
- DataLoader __init__ with various parameters
- DataLoader __len__ with different bs and drop_last combinations
- DataLoader get_idxs with shuffle on/off
- DataLoader iteration (simple list dataset)
- DataLoader one_batch
- DataLoader.new() method
- DataLoader with prebatched (bs=None)
- DataLoader device property setting
- SkipItemException being handled (do_item returning None)
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import fa_collate, fa_convert, SkipItemException, collate_error, DataLoader


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for fa_collate function."""

    def test_collate_tensors(self):
        """fa_collate should stack a list of tensors into a batch."""
        items = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """fa_collate should handle numpy arrays."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_strings(self):
        """fa_collate should handle strings (which are a _collate_type)."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_tuples(self):
        """fa_collate should handle tuples (Sequence type) by collating each position."""
        items = [
            (torch.tensor([1.0]), torch.tensor([10.0])),
            (torch.tensor([2.0]), torch.tensor([20.0])),
        ]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(result[1], torch.tensor([[10.0], [20.0]]))

    def test_collate_lists(self):
        """fa_collate should handle lists of lists (Sequence type)."""
        items = [
            [torch.tensor([1.0]), torch.tensor([10.0])],
            [torch.tensor([2.0]), torch.tensor([20.0])],
        ]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(result[1], torch.tensor([[10.0], [20.0]]))

    def test_collate_integers(self):
        """fa_collate should handle plain integers via default_collate."""
        items = [1, 2, 3, 4]
        result = fa_collate(items)
        expected = torch.tensor([1, 2, 3, 4])
        assert torch.equal(result, expected)

    def test_collate_floats(self):
        """fa_collate should handle plain floats via default_collate."""
        items = [1.5, 2.5, 3.5]
        result = fa_collate(items)
        expected = torch.tensor([1.5, 2.5, 3.5])
        assert torch.equal(result, expected)

    def test_collate_dicts(self):
        """fa_collate should handle dicts (Mapping type) via default_collate."""
        items = [{"a": torch.tensor(1.0)}, {"a": torch.tensor(2.0)}]
        result = fa_collate(items)
        assert "a" in result
        assert torch.equal(result["a"], torch.tensor([1.0, 2.0]))


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for fa_convert function."""

    def test_convert_tensor(self):
        """fa_convert should pass tensors through default_convert."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_numpy_array(self):
        """fa_convert should convert numpy arrays to tensors."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_convert_tuple(self):
        """fa_convert should handle tuples (Sequence) by converting each element."""
        t = (np.array([1.0]), np.array([2.0]))
        result = fa_convert(t)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert torch.equal(result[1], torch.tensor([2.0]))

    def test_convert_list(self):
        """fa_convert should handle lists (Sequence) by converting each element."""
        t = [np.array([1.0]), np.array([2.0])]
        result = fa_convert(t)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert torch.equal(result[1], torch.tensor([2.0]))

    def test_convert_string(self):
        """fa_convert should pass strings through (they are a _collate_type)."""
        result = fa_convert("hello")
        assert result == "hello"

    def test_convert_integer(self):
        """fa_convert passes plain integers through default_convert (returns as-is for non-collate types)."""
        result = fa_convert(42)
        # Plain integers are not a _collate_type or Sequence, so default_convert returns them unchanged
        assert result == 42

    def test_convert_dict(self):
        """fa_convert should handle dicts (Mapping type) via default_convert."""
        d = {"x": np.array([1.0, 2.0])}
        result = fa_convert(d)
        assert "x" in result
        assert torch.equal(result["x"], torch.tensor([1.0, 2.0]))


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for SkipItemException."""

    def test_is_exception(self):
        """SkipItemException should be an Exception subclass."""
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
# Tests for DataLoader __init__
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_basic_init_with_list(self):
        """DataLoader can be initialized with a simple list dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.n == 10
        assert dl.bs == 2
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_init_infers_length(self):
        """DataLoader infers n from len(dataset)."""
        ds = list(range(25))
        dl = DataLoader(ds, bs=5)
        assert dl.n == 25

    def test_init_explicit_n(self):
        """DataLoader respects explicitly provided n."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, n=50)
        assert dl.n == 50

    def test_init_batch_size_alias(self):
        """batch_size parameter is an alias for bs (PyTorch compat)."""
        ds = list(range(10))
        dl = DataLoader(ds, batch_size=4)
        assert dl.bs == 4

    def test_init_indexed_auto_detection(self):
        """DataLoader auto-detects indexed datasets via __getitem__."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.indexed is True

    def test_init_iterable_dataset_not_indexed(self):
        """Iterable datasets should not be indexed."""
        from torch.utils.data import IterableDataset

        class MyIterableDs(IterableDataset):
            def __iter__(self):
                return iter(range(10))

        ds = MyIterableDs()
        dl = DataLoader(ds, bs=2, n=10)
        assert dl.indexed is False

    def test_init_shuffle_requires_indexed(self):
        """Shuffling a non-indexed dataset should raise ValueError."""
        from torch.utils.data import IterableDataset

        class MyIterableDs(IterableDataset):
            def __iter__(self):
                return iter(range(10))

        ds = MyIterableDs()
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(ds, bs=2, n=10, shuffle=True)

    def test_init_drop_last_requires_bs(self):
        """drop_last=True with bs=None should raise AssertionError."""
        ds = list(range(10))
        with pytest.raises(AssertionError):
            DataLoader(ds, bs=None, drop_last=True)

    def test_init_device_none_default(self):
        """Device defaults to None."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.device is None


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader __len__ calculation."""

    def test_len_exact_division(self):
        """When n is exactly divisible by bs, len = n // bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 2

    def test_len_with_remainder(self):
        """When n is not divisible by bs, len includes partial batch."""
        ds = list(range(11))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 3  # 5 + 5 + 1

    def test_len_with_drop_last(self):
        """drop_last=True drops the partial batch."""
        ds = list(range(11))
        dl = DataLoader(ds, bs=5, drop_last=True)
        assert len(dl) == 2  # only full batches

    def test_len_drop_last_exact(self):
        """drop_last with exact division gives same result."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, drop_last=True)
        assert len(dl) == 2

    def test_len_prebatched(self):
        """With bs=None (prebatched), len equals n."""
        ds = [[1, 2], [3, 4], [5, 6]]
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 3

    def test_len_none_n_raises(self):
        """If n is None, len() should raise TypeError."""
        dl = DataLoader(None, bs=2, n=None)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_item(self):
        """Dataset with 1 item, bs=1."""
        ds = [42]
        dl = DataLoader(ds, bs=1)
        assert len(dl) == 1

    def test_len_bs_larger_than_n(self):
        """When bs > n, there is still 1 batch (not dropped)."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=10)
        assert len(dl) == 1

    def test_len_bs_larger_than_n_drop_last(self):
        """When bs > n and drop_last, there are 0 batches."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=10, drop_last=True)
        assert len(dl) == 0


# ============================================================
# Tests for DataLoader get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader get_idxs method."""

    def test_get_idxs_no_shuffle(self):
        """Without shuffle, get_idxs returns sequential indices."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_with_shuffle(self):
        """With shuffle, get_idxs returns a permutation of indices."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=4, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(20))
        # Very unlikely to be in order with 20 items
        assert idxs != list(range(20))

    def test_get_idxs_length_matches_n(self):
        """get_idxs returns exactly n indices."""
        ds = list(range(7))
        dl = DataLoader(ds, bs=3)
        idxs = dl.get_idxs()
        assert len(idxs) == 7

    def test_get_idxs_with_explicit_n(self):
        """get_idxs respects explicit n parameter."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=5, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10


# ============================================================
# Tests for DataLoader iteration
# ============================================================

def _make_dl_iterable(dl):
    """Override retain to identity since retain_types is not available in this stripped-down repo."""
    dl.retain = lambda res, b: res
    return dl


class TestDataLoaderIteration:
    """Tests for DataLoader iteration."""

    def test_iterate_simple_list(self):
        """Iterating a DataLoader over a simple list dataset yields batches."""
        ds = list(range(6))
        dl = _make_dl_iterable(DataLoader(ds, bs=3))
        batches = list(dl)
        assert len(batches) == 2
        # Each batch should be a tensor of size 3
        assert batches[0].shape == (3,)
        assert batches[1].shape == (3,)

    def test_iterate_tuple_dataset(self):
        """DataLoader handles datasets of tuples."""
        ds = [(torch.tensor([float(i)]), torch.tensor([float(i * 10)])) for i in range(4)]
        dl = _make_dl_iterable(DataLoader(ds, bs=2))
        batches = list(dl)
        assert len(batches) == 2
        # Each batch is a tuple of (inputs, targets)
        x_batch, y_batch = batches[0]
        assert x_batch.shape == (2, 1)
        assert y_batch.shape == (2, 1)

    def test_iterate_all_items_present(self):
        """All items from the dataset appear in the iteration (no shuffle)."""
        ds = list(range(5))
        dl = _make_dl_iterable(DataLoader(ds, bs=2, shuffle=False))
        batches = list(dl)
        all_items = torch.cat(batches).tolist()
        assert all_items == [0, 1, 2, 3, 4]

    def test_iterate_with_drop_last(self):
        """drop_last=True omits partial last batch."""
        ds = list(range(7))
        dl = _make_dl_iterable(DataLoader(ds, bs=3, drop_last=True))
        batches = list(dl)
        assert len(batches) == 2
        total_items = sum(b.shape[0] for b in batches)
        assert total_items == 6

    def test_iterate_multiple_epochs(self):
        """DataLoader can be iterated multiple times."""
        ds = list(range(4))
        dl = _make_dl_iterable(DataLoader(ds, bs=2))
        epoch1 = list(dl)
        epoch2 = list(dl)
        assert len(epoch1) == 2
        assert len(epoch2) == 2

    def test_iterate_shuffle_different_order(self):
        """Shuffled DataLoader produces different orders across epochs."""
        ds = list(range(50))
        dl = _make_dl_iterable(DataLoader(ds, bs=50, shuffle=True))
        epoch1 = list(dl)[0].tolist()
        epoch2 = list(dl)[0].tolist()
        # Both contain same elements
        assert sorted(epoch1) == sorted(epoch2) == list(range(50))
        # But very unlikely to be same order
        assert epoch1 != epoch2


# ============================================================
# Tests for DataLoader one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader one_batch method."""

    def test_one_batch_returns_first_batch(self):
        """one_batch returns a single batch."""
        ds = list(range(10))
        dl = _make_dl_iterable(DataLoader(ds, bs=4, shuffle=False))
        batch = dl.one_batch()
        assert batch.shape == (4,)
        assert torch.equal(batch, torch.tensor([0, 1, 2, 3]))

    def test_one_batch_tuple_dataset(self):
        """one_batch works with tuple datasets."""
        ds = [(torch.tensor([float(i)]), torch.tensor([float(i * 2)])) for i in range(6)]
        dl = _make_dl_iterable(DataLoader(ds, bs=3))
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (3, 1)
        assert batch[1].shape == (3, 1)

    def test_one_batch_empty_raises(self):
        """one_batch on empty DataLoader raises ValueError."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=10, drop_last=True)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader.new()
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new() method."""

    def test_new_preserves_bs(self):
        """new() preserves batch size by default."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl2 = dl.new()
        assert dl2.bs == 3

    def test_new_preserves_shuffle(self):
        """new() preserves shuffle setting."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, shuffle=True)
        dl2 = dl.new()
        assert dl2.shuffle is True

    def test_new_override_bs(self):
        """new() allows overriding bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl2 = dl.new(bs=5)
        assert dl2.bs == 5

    def test_new_override_dataset(self):
        """new() allows providing a new dataset."""
        ds1 = list(range(10))
        ds2 = list(range(20))
        dl = DataLoader(ds1, bs=3)
        dl2 = dl.new(dataset=ds2)
        assert dl2.n == 20

    def test_new_is_independent(self):
        """new() creates an independent DataLoader."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, shuffle=True)
        dl2 = dl.new(shuffle=False)
        assert dl.shuffle is True
        assert dl2.shuffle is False


# ============================================================
# Tests for DataLoader prebatched (bs=None)
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
        dl = _make_dl_iterable(DataLoader(ds, bs=None))
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
# Tests for DataLoader device property
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device property."""

    def test_device_default_none(self):
        """Device is None by default."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.device is None

    def test_device_set_cpu(self):
        """Device can be set to cpu."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_via_to(self):
        """to() method sets device."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_device_init_param(self):
        """Device can be set via init parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, device='cpu')
        assert dl.device == torch.device('cpu')


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
        dl.retain = lambda res, b: res
        batches = list(dl)
        # Only even items remain: 0, 2, 4 -> one batch of 2 and one partial of 1
        all_items = torch.cat(batches).tolist()
        assert all_items == [0, 2, 4]


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for collate_error function."""

    def test_collate_error_raises_with_mismatch(self):
        """collate_error should re-raise with informative message on shape mismatch."""
        # Each item in batch is a tuple/list of tensors, and we check by position
        batch = [
            [torch.randn(3, 4), torch.randn(2)],
            [torch.randn(3, 5), torch.randn(2)],  # shape mismatch at index 0
        ]
        e = RuntimeError("original error")
        # collate_error uses bare `raise` so it must be called within an exception context
        with pytest.raises(RuntimeError, match="Mismatch found on axis 0"):
            try:
                raise e
            except RuntimeError:
                collate_error(e, batch)

    def test_collate_error_no_mismatch(self):
        """collate_error should not raise if all shapes match."""
        batch = [
            [torch.randn(3, 4), torch.randn(2)],
            [torch.randn(3, 4), torch.randn(2)],
        ]
        e = RuntimeError("original error")
        # Should not raise since shapes match
        collate_error(e, batch)


# ============================================================
# Tests for DataLoader callbacks (before_iter, after_batch, etc.)
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback hooks."""

    def test_before_iter_called(self):
        """before_iter is called at start of iteration."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.retain = lambda res, b: res
        called = []

        def track_before_iter(x=None):
            called.append("before_iter")
            return x

        dl.before_iter = track_before_iter
        list(dl)
        assert "before_iter" in called

    def test_after_iter_called(self):
        """after_iter is called at end of iteration."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.retain = lambda res, b: res
        called = []

        def track_after_iter(x=None):
            called.append("after_iter")
            return x

        dl.after_iter = track_after_iter
        list(dl)
        assert "after_iter" in called

    def test_after_batch_transforms_batch(self):
        """after_batch can transform each batch."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.retain = lambda res, b: res

        def double_batch(b):
            return b * 2

        dl.after_batch = double_batch
        batches = list(dl)
        # First batch should be [0,1]*2 = [0,2]
        assert torch.equal(batches[0], torch.tensor([0, 2]))


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


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify behavior."""

    def test_chunkify_normal_mode(self):
        """In normal mode (bs is set), chunkify should chunk items into batches."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        chunks = list(dl.chunkify(iter(range(10))))
        assert len(chunks) == 4  # [0,1,2], [3,4,5], [6,7,8], [9]
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[-1]) == [9]

    def test_chunkify_prebatched_mode(self):
        """In prebatched mode (bs=None), chunkify should pass through items unchanged."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        items = [100, 200, 300]
        result = list(dl.chunkify(iter(items)))
        assert result == [100, 200, 300]

    def test_chunkify_with_drop_last(self):
        """With drop_last=True, chunkify should drop incomplete final chunk."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        chunks = list(dl.chunkify(iter(range(10))))
        assert len(chunks) == 3  # [0,1,2], [3,4,5], [6,7,8] - last [9] dropped
        for chunk in chunks:
            assert len(list(chunk)) == 3


# ============================================================
# Tests for DataLoader.shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn method."""

    def test_shuffle_fn_same_length(self):
        """shuffle_fn should return a permutation of the same length."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert len(shuffled) == len(idxs)

    def test_shuffle_fn_same_elements(self):
        """shuffle_fn should contain all the same elements."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == sorted(idxs)

    def test_shuffle_fn_produces_different_orders(self):
        """shuffle_fn should produce different orderings on repeated calls."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(100))
        result1 = dl.shuffle_fn(idxs)
        dl.randomize()
        result2 = dl.shuffle_fn(idxs)
        # With 100 elements, probability of same order is essentially 0
        assert result1 != result2
