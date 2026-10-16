"""Tests for fastai.data.load module.

Covers the DataLoader class and helper functions: fa_collate, fa_convert,
SkipItemException, collate_error, and DataLoader with various configurations.
"""
import sys
import os
import pytest
import torch
import numpy as np
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import fa_collate, fa_convert, SkipItemException, DataLoader


@pytest.fixture
def patched_dl():
    """Create a DataLoader with retain patched to a no-op identity.

    The retain method calls retain_types which has been moved to the
    fasttransform package and is unavailable in this test environment.
    Patching it to identity allows testing iteration logic.
    """
    def _make_dl(*args, **kwargs):
        dl = DataLoader(*args, **kwargs)
        dl.retain = lambda res, b: res
        return dl
    return _make_dl


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should produce a stacked tensor."""
        items = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays should produce a tensor."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_collate_tuples(self):
        """Collating tuples of tensors should return a tuple of stacked tensors."""
        items = [
            (torch.tensor([1.0]), torch.tensor([10.0])),
            (torch.tensor([2.0]), torch.tensor([20.0])),
        ]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1.0], [2.0]]))
        assert torch.equal(result[1], torch.tensor([[10.0], [20.0]]))

    def test_collate_lists(self):
        """Collating lists of tensors should return a list of stacked tensors."""
        items = [
            [torch.tensor([1.0]), torch.tensor([10.0])],
            [torch.tensor([2.0]), torch.tensor([20.0])],
        ]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_collate_scalars(self):
        """Collating scalar tensors should produce a 1D tensor."""
        items = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(items)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_collate_strings(self):
        """Collating strings should use default_collate behavior."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor should return the same tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_numpy(self):
        """Converting a numpy array should return a tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_convert_tuple(self):
        """Converting a tuple of arrays should return a tuple of tensors."""
        items = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        result = fa_convert(items)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_convert_list(self):
        """Converting a list of arrays should return a list of tensors."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)

    def test_convert_string(self):
        """Converting a string should return the string unchanged (it's a Mapping-like type)."""
        s = "hello"
        result = fa_convert(s)
        # strings are in _collate_types, so default_convert handles them
        assert result == "hello"


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException should be a subclass of Exception."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException should be raiseable and catchable."""
        with pytest.raises(SkipItemException):
            raise SkipItemException("skip this item")


# ============================================================
# Tests for DataLoader.__init__
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_basic_init_with_list(self):
        """DataLoader should accept a simple list dataset."""
        data = list(range(10))
        dl = DataLoader(data, bs=2)
        assert dl.bs == 2
        assert dl.n == 10
        assert dl.shuffle is False
        assert dl.drop_last is False
        assert dl.indexed is True

    def test_init_with_batch_size_alias(self):
        """DataLoader should accept batch_size as alias for bs."""
        data = list(range(10))
        dl = DataLoader(data, batch_size=4)
        assert dl.bs == 4

    def test_init_indexed_auto_detection(self):
        """DataLoader should auto-detect indexed datasets."""
        data = list(range(10))
        dl = DataLoader(data, bs=2)
        assert dl.indexed is True

    def test_init_indexed_override(self):
        """DataLoader should respect explicit indexed=False."""
        data = list(range(10))
        dl = DataLoader(data, bs=2, indexed=False)
        assert dl.indexed is False

    def test_init_shuffle(self):
        """DataLoader should accept shuffle parameter."""
        data = list(range(10))
        dl = DataLoader(data, bs=2, shuffle=True)
        assert dl.shuffle is True

    def test_init_shuffle_iterable_raises(self):
        """DataLoader should raise ValueError for shuffle with non-indexed dataset."""
        data = iter(range(10))
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(data, bs=2, shuffle=True, indexed=False)

    def test_init_drop_last_requires_bs(self):
        """DataLoader should assert if drop_last is True but bs is None."""
        data = list(range(10))
        with pytest.raises(AssertionError):
            DataLoader(data, bs=None, drop_last=True)

    def test_init_n_auto_from_dataset_len(self):
        """DataLoader should auto-detect n from len(dataset)."""
        data = list(range(15))
        dl = DataLoader(data, bs=3)
        assert dl.n == 15

    def test_init_explicit_n(self):
        """DataLoader should use explicit n when provided."""
        data = list(range(100))
        dl = DataLoader(data, bs=5, n=20)
        assert dl.n == 20

    def test_init_device_none(self):
        """DataLoader device should default to None."""
        data = list(range(10))
        dl = DataLoader(data, bs=2)
        assert dl.device is None


# ============================================================
# Tests for DataLoader.__len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length computation."""

    def test_len_exact_division(self):
        """Length should be n//bs when n is divisible by bs."""
        data = list(range(10))
        dl = DataLoader(data, bs=5)
        assert len(dl) == 2

    def test_len_with_remainder(self):
        """Length should include partial batch when drop_last is False."""
        data = list(range(11))
        dl = DataLoader(data, bs=5)
        assert len(dl) == 3  # 5 + 5 + 1

    def test_len_with_drop_last(self):
        """Length should exclude partial batch when drop_last is True."""
        data = list(range(11))
        dl = DataLoader(data, bs=5, drop_last=True)
        assert len(dl) == 2  # only full batches

    def test_len_bs_none(self):
        """Length should be n when bs is None (prebatched mode)."""
        data = list(range(7))
        dl = DataLoader(data, bs=None)
        assert len(dl) == 7

    def test_len_n_none_raises(self):
        """Length should raise TypeError when n is None."""
        dl = DataLoader(iter(range(10)), bs=2, indexed=False)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_item(self):
        """Length of single-item dataset with bs=1."""
        data = [42]
        dl = DataLoader(data, bs=1)
        assert len(dl) == 1

    def test_len_bs_larger_than_n(self):
        """When bs > n and drop_last=False, length should be 1."""
        data = list(range(3))
        dl = DataLoader(data, bs=10)
        assert len(dl) == 1

    def test_len_bs_larger_than_n_drop_last(self):
        """When bs > n and drop_last=True, length should be 0."""
        data = list(range(3))
        dl = DataLoader(data, bs=10, drop_last=True)
        assert len(dl) == 0


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_no_shuffle(self):
        """Without shuffle, get_idxs should return sequential indices."""
        data = list(range(5))
        dl = DataLoader(data, bs=2, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_with_shuffle(self):
        """With shuffle, get_idxs should return permuted indices."""
        data = list(range(20))
        dl = DataLoader(data, bs=4, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(20))
        # Very unlikely to be in order for 20 items
        assert idxs != list(range(20))

    def test_get_idxs_respects_n(self):
        """get_idxs should return exactly n indices."""
        data = list(range(100))
        dl = DataLoader(data, bs=5, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10

    def test_get_idxs_not_indexed(self):
        """For non-indexed datasets, get_idxs should return Nones."""
        data = list(range(5))
        dl = DataLoader(data, bs=2, indexed=False)
        idxs = dl.get_idxs()
        assert all(x is None for x in idxs)
        assert len(idxs) == 5


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_create_item_indexed(self):
        """create_item should return dataset[s] for indexed datasets."""
        data = [10, 20, 30, 40, 50]
        dl = DataLoader(data, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_create_item_iterable(self):
        """create_item should call next(it) for iterable datasets."""
        data = list(range(5))
        dl = DataLoader(data, bs=2, indexed=False)
        dl.it = iter(data)
        assert dl.create_item(None) == 0
        assert dl.create_item(None) == 1

    def test_create_item_iterable_with_index_raises(self):
        """create_item should raise IndexError if given a non-None index on iterable."""
        data = list(range(5))
        dl = DataLoader(data, bs=2, indexed=False)
        dl.it = iter(data)
        with pytest.raises(IndexError):
            dl.create_item(3)


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_basic(self):
        """chunkify should split items into chunks of size bs."""
        data = list(range(10))
        dl = DataLoader(data, bs=3)
        chunks = list(dl.chunkify(iter(data)))
        assert len(chunks) == 4  # 3+3+3+1
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]
        assert list(chunks[3]) == [9]

    def test_chunkify_drop_last(self):
        """chunkify with drop_last should exclude incomplete last chunk."""
        data = list(range(10))
        dl = DataLoader(data, bs=3, drop_last=True)
        chunks = list(dl.chunkify(iter(data)))
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(list(chunk)) == 3

    def test_chunkify_prebatched(self):
        """chunkify with bs=None should pass through without chunking."""
        data = list(range(5))
        dl = DataLoader(data, bs=None)
        result = list(dl.chunkify(iter(data)))
        assert result == [0, 1, 2, 3, 4]


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_basic(self, patched_dl):
        """one_batch should return a single batch from the DataLoader."""
        data = list(range(10))
        dl = patched_dl(data, bs=4)
        batch = dl.one_batch()
        assert len(batch) == 4

    def test_one_batch_tensor_dataset(self, patched_dl):
        """one_batch with tensor items should collate into a tensor."""
        data = [torch.tensor([float(i)]) for i in range(10)]
        dl = patched_dl(data, bs=3)
        batch = dl.one_batch()
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (3, 1)

    def test_one_batch_tuple_dataset(self, patched_dl):
        """one_batch with tuple items should return a tuple of tensors."""
        data = [(torch.tensor([float(i)]), torch.tensor([float(i * 10)])) for i in range(8)]
        dl = patched_dl(data, bs=4)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert len(batch) == 2
        assert batch[0].shape == (4, 1)
        assert batch[1].shape == (4, 1)

    def test_one_batch_empty_raises(self):
        """one_batch on an empty DataLoader should raise ValueError."""
        data = list(range(3))
        dl = DataLoader(data, bs=10, drop_last=True)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader iteration
# ============================================================

class TestDataLoaderIteration:
    """Tests for iterating over the DataLoader."""

    def test_iterate_all_batches(self, patched_dl):
        """Iterating should yield all batches."""
        data = [torch.tensor([float(i)]) for i in range(10)]
        dl = patched_dl(data, bs=3)
        batches = list(dl)
        assert len(batches) == 4  # 3+3+3+1

    def test_iterate_all_items_covered(self, patched_dl):
        """All items should appear in the iteration output."""
        data = [torch.tensor([float(i)]) for i in range(6)]
        dl = patched_dl(data, bs=2, shuffle=False)
        batches = list(dl)
        all_items = torch.cat(batches, dim=0)
        expected = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        assert torch.equal(all_items, expected)

    def test_iterate_drop_last(self, patched_dl):
        """With drop_last, only full batches should be yielded."""
        data = [torch.tensor([float(i)]) for i in range(7)]
        dl = patched_dl(data, bs=3, drop_last=True)
        batches = list(dl)
        assert len(batches) == 2
        for b in batches:
            assert b.shape == (3, 1)

    def test_iterate_shuffled_covers_all(self, patched_dl):
        """Shuffled iteration should still cover all items."""
        data = [torch.tensor([float(i)]) for i in range(10)]
        dl = patched_dl(data, bs=5, shuffle=True)
        batches = list(dl)
        all_items = torch.cat(batches, dim=0).flatten().sort()[0]
        expected = torch.arange(10, dtype=torch.float)
        assert torch.equal(all_items, expected)

    def test_iterate_multiple_epochs(self, patched_dl):
        """DataLoader should be re-iterable for multiple epochs."""
        data = [torch.tensor([float(i)]) for i in range(6)]
        dl = patched_dl(data, bs=3)
        batches_epoch1 = list(dl)
        batches_epoch2 = list(dl)
        assert len(batches_epoch1) == 2
        assert len(batches_epoch2) == 2

    def test_iterate_prebatched(self, patched_dl):
        """With bs=None (prebatched), each item is yielded as-is after conversion."""
        data = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        dl = patched_dl(data, bs=None)
        batches = list(dl)
        assert len(batches) == 2
        assert torch.equal(batches[0], torch.tensor([1.0, 2.0]))
        assert torch.equal(batches[1], torch.tensor([3.0, 4.0]))


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_preserves_params(self):
        """new() without args should create a DataLoader with same params."""
        data = list(range(10))
        dl = DataLoader(data, bs=4, shuffle=True, drop_last=True)
        dl2 = dl.new()
        assert dl2.bs == 4
        assert dl2.shuffle is True
        assert dl2.drop_last is True
        assert dl2.n == 10

    def test_new_overrides_bs(self):
        """new(bs=...) should override the batch size."""
        data = list(range(10))
        dl = DataLoader(data, bs=4)
        dl2 = dl.new(bs=2)
        assert dl2.bs == 2
        assert dl2.n == 10

    def test_new_overrides_dataset(self):
        """new(dataset=...) should use the new dataset."""
        data1 = list(range(10))
        data2 = list(range(20))
        dl = DataLoader(data1, bs=4)
        dl2 = dl.new(dataset=data2)
        assert dl2.n == 20

    def test_new_overrides_shuffle(self):
        """new(shuffle=...) should override the shuffle setting."""
        data = list(range(10))
        dl = DataLoader(data, bs=4, shuffle=False)
        dl2 = dl.new(shuffle=True)
        assert dl2.shuffle is True


# ============================================================
# Tests for DataLoader.prebatched property
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for the prebatched property."""

    def test_prebatched_true_when_bs_none(self):
        """prebatched should be True when bs is None."""
        data = list(range(10))
        dl = DataLoader(data, bs=None)
        assert dl.prebatched is True

    def test_prebatched_false_when_bs_set(self):
        """prebatched should be False when bs is set."""
        data = list(range(10))
        dl = DataLoader(data, bs=4)
        assert dl.prebatched is False


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item handling SkipItemException."""

    def test_do_item_normal(self):
        """do_item should return the item normally."""
        data = [10, 20, 30]
        dl = DataLoader(data, bs=2)
        result = dl.do_item(0)
        assert result == 10

    def test_do_item_skip_exception(self):
        """do_item should return None when SkipItemException is raised."""
        data = [10, 20, 30]
        dl = DataLoader(data, bs=2)

        def after_item_skip(x):
            if x == 20:
                raise SkipItemException()
            return x

        dl.after_item = after_item_skip
        assert dl.do_item(0) == 10
        assert dl.do_item(1) is None
        assert dl.do_item(2) == 30


# ============================================================
# Tests for DataLoader.to (device placement)
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device handling."""

    def test_to_cpu(self):
        """DataLoader.to('cpu') should set device to CPU."""
        data = list(range(10))
        dl = DataLoader(data, bs=4)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_device_setter(self):
        """Setting device directly should work."""
        data = list(range(10))
        dl = DataLoader(data, bs=4)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_none(self):
        """Device can be set to None."""
        data = list(range(10))
        dl = DataLoader(data, bs=4, device='cpu')
        dl.device = None
        assert dl.device is None


# ============================================================
# Tests for DataLoader.shuffle_fn and randomize
# ============================================================

class TestDataLoaderShuffle:
    """Tests for shuffle functionality."""

    def test_shuffle_fn_returns_permutation(self):
        """shuffle_fn should return a permutation of the input."""
        data = list(range(20))
        dl = DataLoader(data, bs=4, shuffle=True)
        idxs = list(range(20))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(20))

    def test_randomize_changes_rng(self):
        """randomize should change the internal RNG state."""
        data = list(range(20))
        dl = DataLoader(data, bs=4, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2

    def test_different_epochs_different_order(self, patched_dl):
        """Different iterations should produce different orderings when shuffled."""
        data = [torch.tensor([float(i)]) for i in range(20)]
        dl = patched_dl(data, bs=20, shuffle=True)
        batch1 = list(dl)[0].flatten().tolist()
        batch2 = list(dl)[0].flatten().tolist()
        # Very unlikely to be identical with 20 items
        assert batch1 != batch2
