"""Tests for fastai.data.load module.

Covers: DataLoader (creation, length, indexing, shuffling, batching, device
management, prebatched mode, drop_last, new), fa_collate, fa_convert,
SkipItemException, and collate_error.
"""
import sys
import os
import pytest
import torch
import numpy as np
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import (
    DataLoader,
    fa_collate,
    fa_convert,
    SkipItemException,
    collate_error,
)


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should stack them into a batch."""
        batch = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(batch)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays should convert them to a tensor batch."""
        batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(batch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_collate_strings(self):
        """Collating strings should return a list of strings."""
        batch = ['hello', 'world', 'test']
        result = fa_collate(batch)
        assert result == ['hello', 'world', 'test']

    def test_collate_nested_tuples(self):
        """Collating tuples of tensors should collate element-wise."""
        batch = [
            (torch.tensor([1]), torch.tensor([10])),
            (torch.tensor([2]), torch.tensor([20])),
        ]
        result = fa_collate(batch)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1], [2]]))
        assert torch.equal(result[1], torch.tensor([[10], [20]]))

    def test_collate_nested_lists(self):
        """Collating lists of tensors should collate element-wise."""
        batch = [
            [torch.tensor([1.0]), torch.tensor([2.0])],
            [torch.tensor([3.0]), torch.tensor([4.0])],
        ]
        result = fa_collate(batch)
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1.0], [3.0]]))
        assert torch.equal(result[1], torch.tensor([[2.0], [4.0]]))

    def test_collate_scalar_tensors(self):
        """Collating scalar tensors should produce a 1-D tensor."""
        batch = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        result = fa_collate(batch)
        assert result.tolist() == [1, 2, 3]

    def test_collate_dicts(self):
        """Collating dicts (Mapping types) should collate by key."""
        batch = [{'a': torch.tensor(1), 'b': torch.tensor(2)},
                 {'a': torch.tensor(3), 'b': torch.tensor(4)}]
        result = fa_collate(batch)
        assert isinstance(result, dict)
        assert result['a'].tolist() == [1, 3]
        assert result['b'].tolist() == [2, 4]

    def test_collate_single_element(self):
        """Collating a batch with a single element should still work."""
        batch = [torch.tensor([1.0, 2.0, 3.0])]
        result = fa_collate(batch)
        assert result.shape == (1, 3)

    def test_collate_2d_tensors(self):
        """Collating 2D tensors should produce a 3D batch."""
        batch = [torch.randn(3, 4), torch.randn(3, 4)]
        result = fa_collate(batch)
        assert result.shape == (2, 3, 4)


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor should return it unchanged."""
        t = torch.tensor([1, 2, 3])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_numpy_array(self):
        """Converting a numpy array should produce a tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_convert_string(self):
        """Converting a string should return it unchanged."""
        result = fa_convert('hello')
        assert result == 'hello'

    def test_convert_list_of_tensors(self):
        """Converting a list of tensors should convert each element."""
        data = [torch.tensor([1]), torch.tensor([2])]
        result = fa_convert(data)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_convert_numpy_int_array(self):
        """Converting a numpy integer array should produce an integer tensor."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = fa_convert(arr)
        assert result.dtype == torch.int64

    def test_convert_numpy_float32(self):
        """Converting a numpy float32 array should produce a float32 tensor."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = fa_convert(arr)
        assert result.dtype == torch.float32


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException should be an Exception subclass."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException should be raisable and catchable."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()

    def test_message(self):
        """SkipItemException should support custom messages."""
        exc = SkipItemException("skip this")
        assert str(exc) == "skip this"


# ============================================================
# Tests for DataLoader initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader construction and parameter handling."""

    def test_basic_creation(self):
        """DataLoader should be creatable with just a dataset and bs."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5)
        assert dl.bs == 5
        assert dl.n == 20
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """batch_size parameter should be an alias for bs."""
        ds = list(range(10))
        dl = DataLoader(ds, batch_size=4)
        assert dl.bs == 4

    def test_default_no_shuffle(self):
        """By default, DataLoader should not shuffle."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.shuffle is False

    def test_shuffle_flag(self):
        """DataLoader should accept shuffle=True for indexed datasets."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, shuffle=True)
        assert dl.shuffle is True

    def test_shuffle_raises_for_iterable(self):
        """Shuffling a non-indexed (iterable) dataset should raise ValueError."""
        class IterDS:
            def __iter__(self):
                return iter(range(10))

        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(IterDS(), bs=5, shuffle=True)

    def test_n_override(self):
        """Providing n should override the length detection."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, n=50)
        assert dl.n == 50

    def test_n_none_for_no_len(self):
        """n should be None if dataset has no __len__."""
        class NoLenDS:
            def __getitem__(self, idx):
                return idx

        dl = DataLoader(NoLenDS(), bs=5)
        assert dl.n is None

    def test_indexed_auto_detection(self):
        """indexed should be True when dataset has __getitem__."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.indexed is True

    def test_indexed_false_for_iterable(self):
        """indexed should be False for iterable datasets without __getitem__."""
        class IterDS:
            def __iter__(self):
                return iter(range(10))

        dl = DataLoader(IterDS(), bs=5)
        assert dl.indexed is False

    def test_device_setting(self):
        """DataLoader device should be settable at init."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_device_none_by_default(self):
        """DataLoader device should be None by default."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.device is None

    def test_drop_last_requires_bs(self):
        """drop_last without bs should raise AssertionError."""
        ds = list(range(10))
        with pytest.raises(AssertionError):
            DataLoader(ds, bs=None, drop_last=True)

    def test_pin_memory(self):
        """pin_memory parameter should be stored."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, pin_memory=True)
        assert dl.pin_memory is True

    def test_num_workers_stored(self):
        """num_workers should be stored in fake_l."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, num_workers=0)
        assert dl.fake_l.num_workers == 0


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length computation."""

    def test_exact_division(self):
        """When n is exactly divisible by bs, len should be n/bs."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 4

    def test_remainder_no_drop(self):
        """When n is not divisible by bs and drop_last=False, add 1."""
        ds = list(range(22))
        dl = DataLoader(ds, bs=5, drop_last=False)
        assert len(dl) == 5  # 22//5 + 1

    def test_remainder_with_drop(self):
        """When n is not divisible by bs and drop_last=True, no extra batch."""
        ds = list(range(22))
        dl = DataLoader(ds, bs=5, drop_last=True)
        assert len(dl) == 4  # 22//5

    def test_len_single_batch(self):
        """A dataset smaller than bs should produce one batch."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=5, drop_last=False)
        assert len(dl) == 1

    def test_len_empty_with_drop_last(self):
        """A dataset smaller than bs with drop_last=True should produce 0 batches."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=5, drop_last=True)
        assert len(dl) == 0

    def test_len_prebatched(self):
        """With bs=None (prebatched), len should equal n."""
        ds = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 2

    def test_len_raises_without_n(self):
        """When n is None, len should raise TypeError."""
        class NoLenDS:
            def __getitem__(self, idx):
                return idx

        dl = DataLoader(NoLenDS(), bs=5)
        with pytest.raises(TypeError):
            len(dl)


# ============================================================
# Tests for DataLoader get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_sequential_idxs(self):
        """Without shuffling, indices should be sequential."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == list(range(10))

    def test_shuffled_idxs_permutation(self):
        """Shuffled indices should be a permutation of the original."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(100))

    def test_shuffled_idxs_different_order(self):
        """Shuffled indices should differ from sequential (with high probability)."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, shuffle=True)
        idxs = dl.get_idxs()
        # Very unlikely that a random permutation of 100 items equals the identity
        assert idxs != list(range(100))

    def test_idxs_respect_n(self):
        """get_idxs should only return n indices."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, n=50)
        idxs = dl.get_idxs()
        assert len(idxs) == 50

    def test_idxs_with_n_override_values(self):
        """With n override, indices should be 0..n-1 for sequential."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, n=30, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == list(range(30))


# ============================================================
# Tests for DataLoader shuffle
# ============================================================

class TestDataLoaderShuffle:
    """Tests for DataLoader shuffle behavior."""

    def test_randomize_changes_order(self):
        """Calling randomize should change the next shuffle order."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, shuffle=True)
        idxs1 = dl.get_idxs()
        dl.randomize()
        idxs2 = dl.get_idxs()
        # Different random seeds should produce different permutations
        assert idxs1 != idxs2

    def test_shuffle_fn_returns_permutation(self):
        """shuffle_fn should return a permutation of the input."""
        ds = list(range(50))
        dl = DataLoader(ds, bs=10, shuffle=True)
        original = list(range(50))
        shuffled = dl.shuffle_fn(original)
        assert sorted(shuffled) == original
        assert len(shuffled) == 50


# ============================================================
# Tests for DataLoader create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_indexed_dataset(self):
        """create_item should index into dataset for indexed datasets."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_indexed_with_none_returns_first(self):
        """create_item(None) on indexed dataset should return dataset[0]."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(None) == 10

    def test_non_indexed_raises_with_index(self):
        """create_item with index on non-indexed dataset should raise IndexError."""
        class IterDS:
            def __iter__(self):
                return iter(range(5))

        dl = DataLoader(IterDS(), bs=2)
        with pytest.raises(IndexError):
            dl.create_item(3)


# ============================================================
# Tests for DataLoader prebatched mode
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for DataLoader in prebatched mode (bs=None)."""

    def test_prebatched_flag(self):
        """prebatched should be True when bs is None."""
        ds = [torch.tensor([1, 2, 3])]
        dl = DataLoader(ds, bs=None)
        assert dl.prebatched is True

    def test_not_prebatched(self):
        """prebatched should be False when bs is set."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.prebatched is False


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_preserves_dataset(self):
        """new() without dataset arg should preserve the original dataset."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, shuffle=True)
        dl2 = dl.new()
        assert dl2.dataset is dl.dataset

    def test_new_overrides_bs(self):
        """new(bs=X) should change the batch size."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5)
        dl2 = dl.new(bs=10)
        assert dl2.bs == 10
        assert len(dl2) == 2

    def test_new_overrides_shuffle(self):
        """new(shuffle=X) should change the shuffle setting."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, shuffle=True)
        dl2 = dl.new(shuffle=False)
        assert dl2.shuffle is False

    def test_new_with_different_dataset(self):
        """new(dataset=X) should use the new dataset."""
        ds1 = list(range(20))
        ds2 = list(range(50))
        dl = DataLoader(ds1, bs=5)
        dl2 = dl.new(dataset=ds2)
        assert dl2.dataset is ds2
        assert dl2.n == 50

    def test_new_preserves_drop_last(self):
        """new() should preserve drop_last unless overridden."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, drop_last=True)
        dl2 = dl.new()
        assert dl2.drop_last is True


# ============================================================
# Tests for DataLoader device management
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device property."""

    def test_device_setter(self):
        """Setting device property should update the device."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_none(self):
        """Device should be None initially."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.device is None

    def test_to_method(self):
        """to() method should set the device."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_empty_dataloader_raises(self):
        """one_batch on empty dataloader should raise ValueError."""
        ds = list(range(3))
        dl = DataLoader(ds, bs=5, drop_last=True)
        # len(dl) == 0 because 3 < 5 with drop_last
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for the collate_error helper function."""

    def test_raises_with_message(self):
        """collate_error should enhance the error message and re-raise."""
        batch = [
            (torch.tensor([1, 2]), torch.tensor([10, 20])),
            (torch.tensor([3, 4, 5]), torch.tensor([30, 40, 50])),  # different shape
        ]
        exc = RuntimeError("original error")
        # collate_error uses bare `raise` so must be called inside an except block
        with pytest.raises(RuntimeError, match="Error when trying to collate"):
            try:
                raise exc
            except RuntimeError:
                collate_error(exc, batch)

    def test_identifies_mismatched_axis(self):
        """collate_error should identify which axis has the mismatch."""
        batch = [
            (torch.tensor([1, 2]),),
            (torch.tensor([3, 4, 5]),),  # shape mismatch
        ]
        exc = RuntimeError("collation failed")
        with pytest.raises(RuntimeError, match="Mismatch found on axis 0"):
            try:
                raise exc
            except RuntimeError:
                collate_error(exc, batch)


# ============================================================
# Tests for DataLoader chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_with_bs(self):
        """chunkify should split items into chunks of size bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        items = iter(range(10))
        chunks = list(dl.chunkify(items))
        assert len(chunks) == 4  # 3+3+3+1
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]
        assert list(chunks[3]) == [9]

    def test_chunkify_with_drop_last(self):
        """chunkify with drop_last should drop incomplete last chunk."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        items = iter(range(10))
        chunks = list(dl.chunkify(items))
        assert len(chunks) == 3  # 3+3+3, drops 1

    def test_chunkify_prebatched(self):
        """chunkify in prebatched mode should pass through unchanged."""
        ds = [torch.tensor([1, 2, 3])]
        dl = DataLoader(ds, bs=None)
        items = iter(['a', 'b', 'c'])
        result = dl.chunkify(items)
        # prebatched just returns the iterator as-is
        assert list(result) == ['a', 'b', 'c']


# ============================================================
# Integration-style tests
# ============================================================

class TestDataLoaderIntegration:
    """Integration tests combining multiple DataLoader features."""

    def test_len_consistency_with_n_and_drop_last(self):
        """Length calculation should be consistent with n and drop_last."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=7, n=50, drop_last=True)
        expected = 50 // 7  # 7
        assert len(dl) == expected

    def test_len_consistency_without_drop_last(self):
        """Length calculation should be consistent without drop_last."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=7, n=50, drop_last=False)
        expected = 50 // 7 + 1  # 8 (because 50 % 7 != 0)
        assert len(dl) == expected

    def test_new_changes_effective_length(self):
        """Creating a new DataLoader with different bs should change length."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10)
        assert len(dl) == 10
        dl2 = dl.new(bs=20)
        assert len(dl2) == 5
        dl3 = dl.new(bs=3)
        assert len(dl3) == 34  # 100//3 + 1

    def test_shuffle_does_not_change_length(self):
        """Enabling shuffle should not affect length."""
        ds = list(range(47))
        dl_no_shuf = DataLoader(ds, bs=10, shuffle=False)
        dl_shuf = DataLoader(ds, bs=10, shuffle=True)
        assert len(dl_no_shuf) == len(dl_shuf)

    def test_multiple_randomize_produces_different_orders(self):
        """Multiple randomize calls should produce different orders each time."""
        ds = list(range(200))
        dl = DataLoader(ds, bs=10, shuffle=True)
        orders = []
        for _ in range(5):
            dl.randomize()
            orders.append(dl.get_idxs())
        # At least some should differ (probability of collision is negligible)
        unique_orders = set(tuple(o) for o in orders)
        assert len(unique_orders) > 1
