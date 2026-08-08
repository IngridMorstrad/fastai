"""Tests for fastai.data.load module.

Covers: fa_collate, fa_convert, SkipItemException, collate_error,
and the DataLoader class (construction, iteration, batching, shuffling, etc.).
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import DataLoader, fa_collate, fa_convert, SkipItemException, collate_error

# Patch retain_types into the load module's namespace since it may be missing
# due to fastcore/fasttransform version split. This allows iteration tests to work.
import fastai.data.load as _load_module
if not hasattr(_load_module, 'retain_types') or _load_module.__dict__.get('retain_types') is None:
    try:
        from fasttransform import retain_types as _rt
        _load_module.retain_types = _rt
    except ImportError:
        _load_module.retain_types = lambda res, *args, **kwargs: res


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should stack them."""
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = fa_collate(batch)
        expected = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(result, expected)

    def test_collate_ndarrays(self):
        """Collating a list of numpy arrays should produce a tensor."""
        batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(batch)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_sequences_tuples(self):
        """Collating tuples of tensors should produce a tuple of stacked tensors."""
        batch = [
            (torch.tensor([1]), torch.tensor([10])),
            (torch.tensor([2]), torch.tensor([20])),
        ]
        result = fa_collate(batch)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([[1], [2]]))
        assert torch.equal(result[1], torch.tensor([[10], [20]]))

    def test_collate_sequences_lists(self):
        """Collating lists of tensors should produce a list of stacked tensors."""
        batch = [
            [torch.tensor([1, 2]), torch.tensor([10, 20])],
            [torch.tensor([3, 4]), torch.tensor([30, 40])],
        ]
        result = fa_collate(batch)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([[1, 2], [3, 4]]))
        assert torch.equal(result[1], torch.tensor([[10, 20], [30, 40]]))

    def test_collate_strings(self):
        """Collating strings (a _collate_type) should use default_collate behavior."""
        batch = ['hello', 'world']
        result = fa_collate(batch)
        assert result == ['hello', 'world']

    def test_collate_scalar_tensors(self):
        """Collating scalar tensors should stack into a 1D tensor."""
        batch = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(batch)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_ndarray(self):
        """Converting a numpy array should produce a tensor."""
        arr = np.array([1, 2, 3])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_convert_tensor_passthrough(self):
        """Converting an existing tensor should return it (or equivalent)."""
        t = torch.tensor([4, 5, 6])
        result = fa_convert(t)
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, t)

    def test_convert_sequence(self):
        """Converting a list of numpy arrays should convert each element."""
        data = [np.array([1, 2]), np.array([3, 4])]
        result = fa_convert(data)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))

    def test_convert_tuple_sequence(self):
        """Converting a tuple of arrays should maintain tuple type."""
        data = (np.array([1.0]), np.array([2.0]))
        result = fa_convert(data)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert torch.equal(result[1], torch.tensor([2.0]))

    def test_convert_string(self):
        """Converting a string (a _collate_type) should use default_convert."""
        result = fa_convert('hello')
        assert result == 'hello'


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for SkipItemException."""

    def test_is_exception(self):
        """SkipItemException should be a subclass of Exception."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException should be raisable and catchable."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()

    def test_with_message(self):
        """SkipItemException should support a message."""
        exc = SkipItemException("skip this item")
        assert str(exc) == "skip this item"


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for the collate_error function."""

    def test_raises_with_mismatch_info(self):
        """collate_error should re-raise with a meaningful error about shape mismatch."""
        batch = [
            (torch.tensor([1, 2]), torch.tensor([10])),
            (torch.tensor([3, 4]), torch.tensor([20, 30])),
        ]
        # First trigger the actual collation error
        with pytest.raises(RuntimeError) as exc_info:
            try:
                fa_collate(batch)
            except Exception as e:
                collate_error(e, batch)
                raise
        error_msg = str(exc_info.value)
        assert 'Mismatch' in error_msg
        assert 'shape' in error_msg

    def test_no_raise_when_no_mismatch(self):
        """collate_error should not modify the exception if shapes match (no mismatch found)."""
        batch = [
            (torch.tensor([1, 2]), torch.tensor([10, 20])),
            (torch.tensor([3, 4]), torch.tensor([30, 40])),
        ]
        # If there's no mismatch, collate_error just returns (does nothing)
        e = RuntimeError("some other error")
        # This should not raise since there's no actual shape mismatch
        collate_error(e, batch)


# ============================================================
# Tests for DataLoader.__init__
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader construction."""

    def test_basic_construction(self):
        """DataLoader should be constructable with a simple list dataset."""
        dl = DataLoader(list(range(10)), bs=3)
        assert dl.bs == 3
        assert dl.shuffle is False
        assert dl.drop_last is False
        assert dl.n == 10

    def test_construction_with_shuffle(self):
        """DataLoader should accept shuffle=True for indexed datasets."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        assert dl.shuffle is True

    def test_construction_with_drop_last(self):
        """DataLoader should accept drop_last=True."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        assert dl.drop_last is True

    def test_indexed_auto_detected_for_list(self):
        """A list dataset should be auto-detected as indexed."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.indexed is True

    def test_indexed_explicit_false(self):
        """indexed can be explicitly set to False."""
        dl = DataLoader(list(range(5)), bs=2, indexed=False)
        assert dl.indexed is False

    def test_device_none_by_default(self):
        """Device should be None by default."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.device is None

    def test_device_can_be_set(self):
        """Device can be set during construction."""
        dl = DataLoader(list(range(5)), bs=2, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_batch_size_alias(self):
        """batch_size parameter should be aliased to bs."""
        dl = DataLoader(list(range(10)), batch_size=4)
        assert dl.bs == 4

    def test_n_auto_detected_from_len(self):
        """n should be automatically set from len(dataset)."""
        dl = DataLoader(list(range(7)), bs=2)
        assert dl.n == 7

    def test_n_explicit(self):
        """n can be explicitly overridden."""
        dl = DataLoader(list(range(10)), bs=2, n=5)
        assert dl.n == 5

    def test_shuffle_iterable_raises(self):
        """Cannot shuffle a non-indexed (iterable) dataset."""
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(iter(range(10)), bs=2, shuffle=True, indexed=False)

    def test_drop_last_requires_bs(self):
        """drop_last=True with bs=None should raise AssertionError."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(10)), bs=None, drop_last=True)


# ============================================================
# Tests for DataLoader.__len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length computation."""

    def test_len_exact_division(self):
        """Length should be n // bs when n is evenly divisible."""
        dl = DataLoader(list(range(9)), bs=3)
        assert len(dl) == 3

    def test_len_with_remainder(self):
        """Length should include a partial batch when drop_last=False."""
        dl = DataLoader(list(range(10)), bs=3)
        assert len(dl) == 4  # 3 full + 1 partial

    def test_len_with_drop_last(self):
        """Length should exclude partial batch when drop_last=True."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        assert len(dl) == 3

    def test_len_bs_none(self):
        """Length with bs=None should equal n (prebatched mode)."""
        dl = DataLoader(list(range(5)), bs=None)
        assert len(dl) == 5

    def test_len_single_item(self):
        """Length with single item dataset."""
        dl = DataLoader([42], bs=1)
        assert len(dl) == 1

    def test_len_bs_larger_than_n(self):
        """When bs > n and drop_last=False, should have 1 batch."""
        dl = DataLoader(list(range(3)), bs=10)
        assert len(dl) == 1

    def test_len_bs_larger_than_n_drop_last(self):
        """When bs > n and drop_last=True, should have 0 batches."""
        dl = DataLoader(list(range(3)), bs=10, drop_last=True)
        assert len(dl) == 0

    def test_len_n_none_raises(self):
        """When n is None (iterable with no length), __len__ should raise TypeError."""
        dl = DataLoader(iter(range(10)), bs=2, indexed=False)
        dl.n = None
        with pytest.raises(TypeError):
            len(dl)


# ============================================================
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs."""

    def test_returns_sequential_indices(self):
        """get_idxs without shuffle should return sequential indices."""
        dl = DataLoader(list(range(5)), bs=2, shuffle=False)
        assert dl.get_idxs() == [0, 1, 2, 3, 4]

    def test_shuffled_indices_contain_all(self):
        """get_idxs with shuffle should return all indices in different order."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(10))

    def test_shuffled_indices_differ_from_sequential(self):
        """Shuffled indices should (very likely) differ from sequential order."""
        dl = DataLoader(list(range(100)), bs=2, shuffle=True)
        idxs = dl.get_idxs()
        # With 100 items, probability of getting sequential order is ~1/100!
        assert idxs != list(range(100))

    def test_indices_length_matches_n(self):
        """get_idxs should return exactly n indices."""
        dl = DataLoader(list(range(7)), bs=3, shuffle=False)
        assert len(dl.get_idxs()) == 7

    def test_indices_with_custom_n(self):
        """get_idxs should respect custom n value."""
        dl = DataLoader(list(range(100)), bs=5, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10


# ============================================================
# Tests for DataLoader.__iter__
# ============================================================

class TestDataLoaderIter:
    """Tests for DataLoader iteration."""

    def test_iterates_all_items(self):
        """Iteration should cover all items in the dataset."""
        data = [torch.tensor([i]) for i in range(6)]
        dl = DataLoader(data, bs=2)
        batches = list(dl)
        assert len(batches) == 3
        # Concatenate all batches and verify we got all items
        all_items = torch.cat(batches).flatten().tolist()
        assert sorted(all_items) == list(range(6))

    def test_batch_sizes_correct(self):
        """Each batch should have the expected size."""
        data = [torch.tensor([i]) for i in range(7)]
        dl = DataLoader(data, bs=3)
        batches = list(dl)
        assert batches[0].shape[0] == 3
        assert batches[1].shape[0] == 3
        assert batches[2].shape[0] == 1  # remainder

    def test_drop_last_drops_remainder(self):
        """With drop_last=True, the last incomplete batch is dropped."""
        data = [torch.tensor([i]) for i in range(7)]
        dl = DataLoader(data, bs=3, drop_last=True)
        batches = list(dl)
        assert len(batches) == 2
        for b in batches:
            assert b.shape[0] == 3

    def test_iteration_with_tuples(self):
        """Iteration should handle tuple datasets (multi-element items)."""
        data = [(torch.tensor([i]), torch.tensor([i * 10])) for i in range(4)]
        dl = DataLoader(data, bs=2)
        batches = list(dl)
        assert len(batches) == 2
        # Each batch should be a tuple of two tensors
        assert isinstance(batches[0], tuple)
        assert batches[0][0].shape == (2, 1)
        assert batches[0][1].shape == (2, 1)

    def test_iteration_with_shuffle(self):
        """Iteration with shuffle should produce batches with all items but different order."""
        data = [torch.tensor([i]) for i in range(10)]
        dl = DataLoader(data, bs=5, shuffle=True)
        batches = list(dl)
        all_items = torch.cat(batches).flatten().tolist()
        assert sorted(all_items) == list(range(10))

    def test_multiple_iterations(self):
        """Multiple iterations over the same DataLoader should work."""
        data = [torch.tensor([i]) for i in range(4)]
        dl = DataLoader(data, bs=2)
        batches1 = list(dl)
        batches2 = list(dl)
        assert len(batches1) == 2
        assert len(batches2) == 2


# ============================================================
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch."""

    def test_returns_single_batch(self):
        """one_batch should return a single batch."""
        data = [torch.tensor([i]) for i in range(10)]
        dl = DataLoader(data, bs=3)
        batch = dl.one_batch()
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] == 3

    def test_one_batch_with_tuple_data(self):
        """one_batch with tuple items should return a tuple of tensors."""
        data = [(torch.tensor([i]), torch.tensor([i * 2])) for i in range(6)]
        dl = DataLoader(data, bs=2)
        batch = dl.one_batch()
        assert isinstance(batch, tuple)
        assert batch[0].shape == (2, 1)

    def test_one_batch_empty_raises(self):
        """one_batch on an empty DataLoader should raise ValueError."""
        dl = DataLoader(list(range(3)), bs=10, drop_last=True)
        # len(dl) == 0
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new."""

    def test_new_preserves_dataset(self):
        """new() without dataset arg should preserve original dataset."""
        data = list(range(10))
        dl = DataLoader(data, bs=3)
        dl2 = dl.new()
        assert dl2.dataset == data

    def test_new_overrides_bs(self):
        """new(bs=...) should create a DataLoader with new batch size."""
        dl = DataLoader(list(range(10)), bs=3)
        dl2 = dl.new(bs=5)
        assert dl2.bs == 5
        assert dl2.n == 10

    def test_new_with_different_dataset(self):
        """new(dataset=...) should use the new dataset."""
        dl = DataLoader(list(range(10)), bs=2)
        new_data = list(range(20))
        dl2 = dl.new(dataset=new_data)
        assert dl2.dataset == new_data
        assert dl2.n == 20

    def test_new_preserves_shuffle(self):
        """new() should preserve shuffle setting."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        dl2 = dl.new()
        assert dl2.shuffle is True

    def test_new_preserves_drop_last(self):
        """new() should preserve drop_last setting."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        dl2 = dl.new()
        assert dl2.drop_last is True

    def test_new_overrides_drop_last(self):
        """new(drop_last=...) should override original setting."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        dl2 = dl.new(drop_last=False)
        assert dl2.drop_last is False


# ============================================================
# Tests for DataLoader.create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item."""

    def test_indexed_dataset_returns_item(self):
        """create_item on an indexed dataset returns dataset[s]."""
        data = ['a', 'b', 'c', 'd']
        dl = DataLoader(data, bs=2)
        assert dl.create_item(0) == 'a'
        assert dl.create_item(2) == 'c'

    def test_indexed_dataset_none_returns_first(self):
        """create_item(None) on indexed dataset returns dataset[0]."""
        data = ['a', 'b', 'c']
        dl = DataLoader(data, bs=2)
        assert dl.create_item(None) == 'a'

    def test_iterable_dataset_uses_iterator(self):
        """create_item(None) on iterable dataset returns next from iterator."""
        dl = DataLoader(iter(range(5)), bs=2, indexed=False)
        dl.it = iter(range(5))
        assert dl.create_item(None) == 0
        assert dl.create_item(None) == 1
        assert dl.create_item(None) == 2

    def test_iterable_dataset_with_index_raises(self):
        """create_item with numeric index on iterable dataset raises IndexError."""
        dl = DataLoader(iter(range(5)), bs=2, indexed=False)
        dl.it = iter(range(5))
        with pytest.raises(IndexError, match="Cannot index an iterable dataset"):
            dl.create_item(3)


# ============================================================
# Tests for DataLoader.do_item
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item."""

    def test_returns_item_normally(self):
        """do_item should return the item from the dataset."""
        data = [10, 20, 30, 40]
        dl = DataLoader(data, bs=2)
        assert dl.do_item(0) == 10
        assert dl.do_item(2) == 30

    def test_skip_item_returns_none(self):
        """do_item should return None when SkipItemException is raised."""
        data = list(range(10))
        dl = DataLoader(data, bs=2)

        # Override after_item to skip even numbers
        original_after_item = dl.after_item
        def skip_evens(x):
            if x % 2 == 0:
                raise SkipItemException()
            return x
        dl.after_item = skip_evens

        assert dl.do_item(0) is None  # 0 is even, should be skipped
        assert dl.do_item(1) == 1     # 1 is odd, should pass through
        assert dl.do_item(2) is None  # 2 is even, should be skipped
        assert dl.do_item(3) == 3     # 3 is odd, should pass through


# ============================================================
# Tests for DataLoader.chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify."""

    def test_prebatched_passthrough(self):
        """With bs=None (prebatched), chunkify should pass through the iterable."""
        dl = DataLoader([[1, 2], [3, 4]], bs=None)
        data = iter(['batch1', 'batch2', 'batch3'])
        result = list(dl.chunkify(data))
        assert result == ['batch1', 'batch2', 'batch3']

    def test_chunks_with_bs(self):
        """With bs set, chunkify should group items into chunks of size bs."""
        dl = DataLoader(list(range(10)), bs=3)
        result = list(dl.chunkify(iter(range(10))))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_chunks_with_drop_last(self):
        """With drop_last=True, incomplete chunks should be dropped."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        result = list(dl.chunkify(iter(range(10))))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_chunks_exact_division(self):
        """When n is evenly divisible by bs, all chunks should be full."""
        dl = DataLoader(list(range(9)), bs=3)
        result = list(dl.chunkify(iter(range(9))))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


# ============================================================
# Tests for DataLoader.shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn."""

    def test_returns_all_indices(self):
        """shuffle_fn should return a permutation containing all original indices."""
        dl = DataLoader(list(range(20)), bs=5, shuffle=True)
        idxs = list(range(20))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(20))

    def test_returns_different_order(self):
        """shuffle_fn should (very likely) produce a different ordering."""
        dl = DataLoader(list(range(100)), bs=5, shuffle=True)
        idxs = list(range(100))
        shuffled = dl.shuffle_fn(idxs)
        # Extremely unlikely (1/100!) that shuffled matches original
        assert shuffled != idxs

    def test_preserves_length(self):
        """shuffle_fn output should have the same length as input."""
        dl = DataLoader(list(range(15)), bs=3, shuffle=True)
        idxs = list(range(15))
        shuffled = dl.shuffle_fn(idxs)
        assert len(shuffled) == 15


# ============================================================
# Tests for DataLoader.to
# ============================================================

class TestDataLoaderTo:
    """Tests for DataLoader.to (device setter)."""

    def test_set_device_cpu(self):
        """to('cpu') should set device to cpu."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.device is None
        dl.to('cpu')
        assert dl.device == torch.device('cpu')

    def test_set_device_none(self):
        """to(None) should set device back to None (requires workaround)."""
        dl = DataLoader(list(range(5)), bs=2, device='cpu')
        assert dl.device == torch.device('cpu')
        # Setting via property setter directly
        dl.device = None
        assert dl.device is None

    def test_to_returns_none(self):
        """The to() method does not explicitly return self."""
        dl = DataLoader(list(range(5)), bs=2)
        result = dl.to('cpu')
        # to() doesn't return anything (returns None)
        assert result is None
