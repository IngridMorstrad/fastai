"""Tests for fastai.data.load module.

Covers the core DataLoader class and helper functions: fa_collate, fa_convert,
SkipItemException, collate_error, and the DataLoader methods including __len__,
get_idxs, create_batches, create_item, create_batch, do_item, chunkify,
shuffle_fn, one_batch, new, iteration, device handling, and prebatched mode.
"""
import sys
import os
import pytest
import torch
import numpy as np
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import DataLoader, fa_collate, fa_convert, SkipItemException, collate_error


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """fa_collate should stack tensors into a batch."""
        items = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """fa_collate should handle numpy arrays via default_collate."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        assert torch.equal(result, expected)

    def test_collate_tuples(self):
        """fa_collate should collate tuples element-wise, preserving tuple type."""
        items = [(torch.tensor(1), torch.tensor(2)),
                 (torch.tensor(3), torch.tensor(4))]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1, 3]))
        assert torch.equal(result[1], torch.tensor([2, 4]))

    def test_collate_lists(self):
        """fa_collate should collate lists element-wise, preserving list type."""
        items = [[torch.tensor(1), torch.tensor(2)],
                 [torch.tensor(3), torch.tensor(4)]]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([1, 3]))
        assert torch.equal(result[1], torch.tensor([2, 4]))

    def test_collate_strings(self):
        """fa_collate should handle strings (a collate type) properly."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_scalars(self):
        """fa_collate should collate scalar tensors into a 1D tensor."""
        items = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(items)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_numpy_array(self):
        """fa_convert should convert numpy array to tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_convert_tensor(self):
        """fa_convert should pass through tensors unchanged."""
        t = torch.tensor([1.0, 2.0])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_tuple(self):
        """fa_convert should convert each element in a tuple."""
        data = (np.array([1.0]), np.array([2.0]))
        result = fa_convert(data)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1.0], dtype=torch.float64))
        assert torch.equal(result[1], torch.tensor([2.0], dtype=torch.float64))

    def test_convert_list(self):
        """fa_convert should convert each element in a list."""
        data = [np.array([1.0]), np.array([2.0])]
        result = fa_convert(data)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([1.0], dtype=torch.float64))

    def test_convert_string(self):
        """fa_convert should handle strings (a collate type) properly."""
        result = fa_convert("hello")
        assert result == "hello"


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException should be an Exception subclass."""
        assert issubclass(SkipItemException, Exception)

    def test_can_raise(self):
        """SkipItemException can be raised and caught."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for the collate_error function."""

    def test_error_message_on_shape_mismatch(self):
        """collate_error should raise with informative message when shapes differ."""
        batch = [
            (torch.randn(3, 4), torch.tensor(0)),
            (torch.randn(3, 5), torch.tensor(1)),  # different shape
        ]
        with pytest.raises(Exception) as exc_info:
            try:
                fa_collate(batch)
            except Exception as e:
                collate_error(e, batch)
        assert "Mismatch found" in str(exc_info.value)


# ============================================================
# Tests for DataLoader.__len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length computation."""

    def test_len_exact_division(self):
        """Length should be n//bs when n is divisible by bs."""
        dl = DataLoader(list(range(20)), bs=4)
        assert len(dl) == 5

    def test_len_with_remainder(self):
        """Length should include partial batch when drop_last=False."""
        dl = DataLoader(list(range(10)), bs=3)
        assert len(dl) == 4  # 3 + 3 + 3 + 1

    def test_len_with_drop_last(self):
        """Length should exclude partial batch when drop_last=True."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        assert len(dl) == 3

    def test_len_prebatched(self):
        """Length should be n when bs=None (prebatched mode)."""
        ds = [torch.randn(4, 3) for _ in range(7)]
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 7

    def test_len_raises_when_n_is_none(self):
        """Length should raise TypeError when n cannot be determined."""
        class NoLenDS(IterableDataset):
            def __iter__(self):
                return iter(range(10))

        dl = DataLoader(NoLenDS(), bs=3)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_element_batches(self):
        """bs=1 should give n batches."""
        dl = DataLoader(list(range(5)), bs=1)
        assert len(dl) == 5


# ============================================================
# Tests for DataLoader.__init__
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_basic_init(self):
        """DataLoader should initialize with a list dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.bs == 2
        assert dl.n == 10
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """batch_size param should be mapped to bs for PyTorch compatibility."""
        dl = DataLoader(list(range(10)), batch_size=5)
        assert dl.bs == 5

    def test_indexed_auto_detection(self):
        """indexed should be True for list datasets (they have __getitem__)."""
        dl = DataLoader(list(range(10)), bs=2)
        assert dl.indexed is True

    def test_iterable_dataset_not_indexed(self):
        """IterableDataset should have indexed=False."""
        class MyIterDS(IterableDataset):
            def __iter__(self):
                return iter(range(10))
            def __len__(self):
                return 10

        dl = DataLoader(MyIterDS(), bs=2)
        assert dl.indexed is False

    def test_shuffle_requires_indexed(self):
        """shuffle=True with iterable dataset should raise ValueError."""
        class MyIterDS(IterableDataset):
            def __iter__(self):
                return iter(range(10))
            def __len__(self):
                return 10

        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(MyIterDS(), bs=2, shuffle=True)

    def test_drop_last_requires_bs(self):
        """drop_last=True with bs=None should raise AssertionError."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(10)), bs=None, drop_last=True)

    def test_n_from_dataset_len(self):
        """n should be inferred from dataset __len__."""
        dl = DataLoader(list(range(15)), bs=4)
        assert dl.n == 15

    def test_explicit_n(self):
        """Explicitly provided n should override dataset length."""
        dl = DataLoader(list(range(100)), bs=4, n=10)
        assert dl.n == 10
        assert len(dl) == 3  # 10 // 4 + 1


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for the get_idxs method."""

    def test_sequential_indices(self):
        """Without shuffle, get_idxs should return sequential indices."""
        dl = DataLoader(list(range(5)), bs=2, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_shuffled_indices(self):
        """With shuffle, get_idxs should return permuted indices."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        idxs = dl.get_idxs()
        # Should contain same elements but (very likely) different order
        assert sorted(idxs) == list(range(20))
        # With 20 items, extremely unlikely to be in order
        assert idxs != list(range(20))

    def test_indices_respect_n(self):
        """Indices should be limited to n items."""
        dl = DataLoader(list(range(100)), bs=4, n=7)
        idxs = dl.get_idxs()
        assert len(idxs) == 7


# ============================================================
# Tests for DataLoader.shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for the shuffle_fn method."""

    def test_shuffle_fn_permutes(self):
        """shuffle_fn should return a permutation of the input."""
        dl = DataLoader(list(range(10)), bs=2)
        idxs = list(range(20))
        result = dl.shuffle_fn(idxs)
        assert sorted(result) == list(range(20))
        assert len(result) == 20

    def test_shuffle_fn_produces_different_orders(self):
        """Multiple calls to shuffle_fn should produce different results (randomized)."""
        dl = DataLoader(list(range(10)), bs=2)
        idxs = list(range(50))
        result1 = dl.shuffle_fn(idxs)
        dl.randomize()
        result2 = dl.shuffle_fn(idxs)
        # Extremely unlikely both are identical for 50 items
        assert result1 != result2


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for the create_item method."""

    def test_indexed_create_item(self):
        """create_item should index into dataset for indexed datasets."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_iterable_create_item(self):
        """create_item should use next(iterator) for iterable datasets."""
        class MyIterDS(IterableDataset):
            def __iter__(self):
                return iter([100, 200, 300])
            def __len__(self):
                return 3

        dl = DataLoader(MyIterDS(), bs=2)
        dl.it = iter(dl.dataset)
        assert dl.create_item(None) == 100
        assert dl.create_item(None) == 200

    def test_indexed_with_none_uses_zero(self):
        """For indexed datasets, create_item(None) should use index 0."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(None) == 10


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for the do_item method and SkipItemException handling."""

    def test_do_item_returns_item(self):
        """do_item should return the item from the dataset."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)
        assert dl.do_item(0) == 10
        assert dl.do_item(1) == 20

    def test_do_item_skip_returns_none(self):
        """do_item should return None when SkipItemException is raised."""
        class SkipOdds:
            def __len__(self): return 6
            def __getitem__(self, i):
                if i % 2 == 1:
                    raise SkipItemException()
                return i

        dl = DataLoader(SkipOdds(), bs=3)
        assert dl.do_item(0) == 0
        assert dl.do_item(1) is None
        assert dl.do_item(2) == 2
        assert dl.do_item(3) is None

    def test_skip_item_filters_from_batch(self):
        """Items that raise SkipItemException should be excluded from batches."""
        class SkipEvens:
            def __len__(self): return 10
            def __getitem__(self, i):
                if i % 2 == 0:
                    raise SkipItemException()
                return i

        dl = DataLoader(SkipEvens(), bs=3)
        batch = dl.one_batch()
        # Only odd items should be in the batch
        for item in batch.tolist():
            assert item % 2 == 1


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for the chunkify method."""

    def test_chunkify_with_bs(self):
        """chunkify should split items into chunks of size bs."""
        dl = DataLoader(list(range(10)), bs=3)
        items = iter(range(9))
        chunks = list(dl.chunkify(items))
        assert len(chunks) == 3
        assert list(chunks[0]) == [0, 1, 2]
        assert list(chunks[1]) == [3, 4, 5]
        assert list(chunks[2]) == [6, 7, 8]

    def test_chunkify_prebatched(self):
        """In prebatched mode (bs=None), chunkify should pass through unchanged."""
        ds = [torch.randn(4, 3) for _ in range(5)]
        dl = DataLoader(ds, bs=None)
        items = iter(range(5))
        result = list(dl.chunkify(items))
        assert result == [0, 1, 2, 3, 4]


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for the one_batch method."""

    def test_one_batch_returns_first_batch(self):
        """one_batch should return the first batch of data."""
        ds = list(range(12))
        dl = DataLoader(ds, bs=4, shuffle=False)
        batch = dl.one_batch()
        expected = torch.tensor([0, 1, 2, 3])
        assert torch.equal(batch, expected)

    def test_one_batch_with_tuples(self):
        """one_batch should work with tuple datasets."""
        ds = [(torch.tensor([float(i)]), torch.tensor(i % 2)) for i in range(10)]
        dl = DataLoader(ds, bs=3, shuffle=False)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (3, 1)
        assert batch[1].shape == (3,)

    def test_one_batch_empty_raises(self):
        """one_batch should raise ValueError for empty DataLoader."""
        dl = DataLoader(list(range(2)), bs=4, drop_last=True)
        # 2 items with bs=4 and drop_last=True means 0 batches
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()

    def test_one_batch_prebatched(self):
        """one_batch should work in prebatched mode."""
        ds = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
        dl = DataLoader(ds, bs=None)
        batch = dl.one_batch()
        assert torch.equal(batch, torch.tensor([1.0, 2.0, 3.0]))


# ============================================================
# Tests for DataLoader iteration
# ============================================================

class TestDataLoaderIteration:
    """Tests for iterating over the DataLoader."""

    def test_full_iteration(self):
        """Iterating should yield all batches."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, shuffle=False)
        batches = list(dl)
        assert len(batches) == 4
        assert torch.equal(batches[0], torch.tensor([0, 1, 2]))
        assert torch.equal(batches[1], torch.tensor([3, 4, 5]))
        assert torch.equal(batches[2], torch.tensor([6, 7, 8]))
        assert torch.equal(batches[3], torch.tensor([9]))

    def test_iteration_with_drop_last(self):
        """drop_last should exclude the final partial batch."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        batches = list(dl)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 3

    def test_iteration_shuffled_covers_all_items(self):
        """Shuffled iteration should cover all items exactly once."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, shuffle=True)
        batches = list(dl)
        all_items = sorted(torch.cat(batches).tolist())
        assert all_items == list(range(20))

    def test_multiple_iterations_different_order(self):
        """Multiple iterations with shuffle should produce different orders."""
        ds = list(range(50))
        dl = DataLoader(ds, bs=50, shuffle=True)
        batch1 = list(dl)[0].tolist()
        batch2 = list(dl)[0].tolist()
        # Same elements, very likely different order
        assert sorted(batch1) == sorted(batch2)
        assert batch1 != batch2

    def test_iteration_with_tensor_dataset(self):
        """DataLoader should handle datasets of tensors properly."""
        ds = [torch.randn(3) for _ in range(8)]
        dl = DataLoader(ds, bs=4, shuffle=False)
        batches = list(dl)
        assert len(batches) == 2
        assert batches[0].shape == (4, 3)
        assert batches[1].shape == (4, 3)

    def test_iteration_prebatched(self):
        """Prebatched mode should yield items as-is (converted)."""
        ds = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        dl = DataLoader(ds, bs=None)
        batches = list(dl)
        assert len(batches) == 3
        assert torch.equal(batches[0], torch.tensor([1.0, 2.0]))
        assert torch.equal(batches[1], torch.tensor([3.0, 4.0]))


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for creating a new DataLoader with modified parameters."""

    def test_new_preserves_dataset(self):
        """new() without dataset arg should keep the same dataset."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=4)
        dl2 = dl.new(bs=2)
        assert dl2.n == 20
        assert dl2.bs == 2

    def test_new_with_different_bs(self):
        """new() should allow changing batch size."""
        dl = DataLoader(list(range(20)), bs=4)
        dl2 = dl.new(bs=10)
        assert len(dl2) == 2

    def test_new_with_different_shuffle(self):
        """new() should allow changing shuffle."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        dl2 = dl.new(shuffle=False)
        assert dl2.shuffle is False

    def test_new_with_new_dataset(self):
        """new() should allow replacing the dataset."""
        ds1 = list(range(20))
        ds2 = list(range(50))
        dl = DataLoader(ds1, bs=4)
        dl2 = dl.new(dataset=ds2)
        assert dl2.n == 50


# ============================================================
# Tests for DataLoader.device
# ============================================================

class TestDataLoaderDevice:
    """Tests for device setting and to() method."""

    def test_default_device_is_none(self):
        """Default device should be None."""
        dl = DataLoader(list(range(10)), bs=2)
        assert dl.device is None

    def test_to_cpu(self):
        """to('cpu') should set device to cpu."""
        dl = DataLoader(list(range(10)), bs=2)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_device_setter(self):
        """Setting device property directly should work."""
        dl = DataLoader(list(range(10)), bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_iteration_with_device(self):
        """Batches should be on the correct device after setting device."""
        dl = DataLoader(list(range(10)), bs=3, device='cpu')
        batch = dl.one_batch()
        assert batch.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader.prebatched property
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for the prebatched property."""

    def test_prebatched_true_when_bs_none(self):
        """prebatched should be True when bs is None."""
        dl = DataLoader([torch.randn(3) for _ in range(5)], bs=None)
        assert dl.prebatched is True

    def test_prebatched_false_when_bs_set(self):
        """prebatched should be False when bs is set."""
        dl = DataLoader(list(range(10)), bs=4)
        assert dl.prebatched is False


# ============================================================
# Tests for DataLoader with IterableDataset
# ============================================================

class TestDataLoaderIterableDataset:
    """Tests for DataLoader with IterableDataset."""

    def test_iterable_dataset_basic(self):
        """DataLoader should work with IterableDataset."""
        class SimpleIterDS(IterableDataset):
            def __iter__(self):
                return iter(range(10))
            def __len__(self):
                return 10

        dl = DataLoader(SimpleIterDS(), bs=3)
        batch = dl.one_batch()
        assert torch.equal(batch, torch.tensor([0, 1, 2]))

    def test_iterable_dataset_full_iteration(self):
        """Full iteration over IterableDataset should work."""
        class SimpleIterDS(IterableDataset):
            def __iter__(self):
                return iter(range(9))
            def __len__(self):
                return 9

        dl = DataLoader(SimpleIterDS(), bs=3)
        batches = list(dl)
        assert len(batches) == 3
        assert torch.equal(batches[0], torch.tensor([0, 1, 2]))
        assert torch.equal(batches[1], torch.tensor([3, 4, 5]))
        assert torch.equal(batches[2], torch.tensor([6, 7, 8]))


# ============================================================
# Tests for DataLoader.randomize
# ============================================================

class TestDataLoaderRandomize:
    """Tests for the randomize method."""

    def test_randomize_changes_rng_state(self):
        """randomize should change the internal RNG state."""
        dl = DataLoader(list(range(20)), bs=4, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


# ============================================================
# Tests for DataLoader callbacks (noop methods)
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback/hook methods."""

    def test_before_iter_is_noop(self):
        """Default before_iter should be a no-op."""
        dl = DataLoader(list(range(10)), bs=2)
        result = dl.before_iter()
        assert result is None

    def test_after_item_is_noop(self):
        """Default after_item should pass through."""
        dl = DataLoader(list(range(10)), bs=2)
        result = dl.after_item(42)
        assert result == 42

    def test_before_batch_is_noop(self):
        """Default before_batch should pass through."""
        dl = DataLoader(list(range(10)), bs=2)
        items = [1, 2, 3]
        result = dl.before_batch(items)
        assert result == items

    def test_after_batch_is_noop(self):
        """Default after_batch should pass through."""
        dl = DataLoader(list(range(10)), bs=2)
        batch = torch.tensor([1, 2, 3])
        result = dl.after_batch(batch)
        assert torch.equal(result, batch)

    def test_custom_after_item(self):
        """Custom after_item should be applied to each item."""
        dl = DataLoader(list(range(10)), bs=4, shuffle=False, after_item=lambda x: x * 10)
        batch = dl.one_batch()
        expected = torch.tensor([0, 10, 20, 30])
        assert torch.equal(batch, expected)

    def test_custom_before_batch(self):
        """Custom before_batch should transform items before collation."""
        def double_items(items):
            return [x * 2 for x in items]

        dl = DataLoader(list(range(10)), bs=4, shuffle=False, before_batch=double_items)
        batch = dl.one_batch()
        expected = torch.tensor([0, 2, 4, 6])
        assert torch.equal(batch, expected)


# ============================================================
# Tests for DataLoader.create_batch
# ============================================================

class TestDataLoaderCreateBatch:
    """Tests for the create_batch method."""

    def test_create_batch_collates_items(self):
        """create_batch should collate a list of items into a tensor batch."""
        dl = DataLoader(list(range(10)), bs=4)
        batch = dl.create_batch([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 3, 4])
        assert torch.equal(batch, expected)

    def test_create_batch_with_tuples(self):
        """create_batch should collate tuple items element-wise."""
        dl = DataLoader([(1, 2)], bs=2)
        batch = dl.create_batch([(torch.tensor(1), torch.tensor(10)),
                                  (torch.tensor(2), torch.tensor(20))])
        assert isinstance(batch, tuple)
        assert torch.equal(batch[0], torch.tensor([1, 2]))
        assert torch.equal(batch[1], torch.tensor([10, 20]))

    def test_create_batch_prebatched_uses_convert(self):
        """In prebatched mode, create_batch should use fa_convert."""
        ds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        dl = DataLoader(ds, bs=None)
        result = dl.create_batch(np.array([5.0, 6.0]))
        assert isinstance(result, torch.Tensor)


# ============================================================
# Tests for DataLoader with complex data types
# ============================================================

class TestDataLoaderComplexData:
    """Tests for DataLoader with various data types."""

    def test_dataset_of_tuples(self):
        """DataLoader should handle (input, target) tuple datasets."""
        ds = [(torch.randn(5), torch.tensor(i % 3)) for i in range(12)]
        dl = DataLoader(ds, bs=4, shuffle=False)
        batch = dl.one_batch()
        assert batch[0].shape == (4, 5)
        assert batch[1].shape == (4,)

    def test_dataset_of_3_element_tuples(self):
        """DataLoader should handle tuples with 3 elements."""
        ds = [(torch.randn(3), torch.randn(2), torch.tensor(i)) for i in range(8)]
        dl = DataLoader(ds, bs=4, shuffle=False)
        batch = dl.one_batch()
        assert len(batch) == 3
        assert batch[0].shape == (4, 3)
        assert batch[1].shape == (4, 2)
        assert batch[2].shape == (4,)

    def test_dataset_of_2d_tensors(self):
        """DataLoader should handle 2D tensor datasets (like images)."""
        ds = [torch.randn(3, 8, 8) for _ in range(10)]
        dl = DataLoader(ds, bs=4, shuffle=False)
        batch = dl.one_batch()
        assert batch.shape == (4, 3, 8, 8)
