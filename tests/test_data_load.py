"""Tests for fastai.data.load module.

Covers the DataLoader class and helper functions: fa_collate, fa_convert,
SkipItemException, collate_error, and DataLoader initialization, length,
indexing, shuffling, chunking, device handling, and batch creation.
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import (
    fa_collate,
    fa_convert,
    SkipItemException,
    collate_error,
    DataLoader,
    _FakeLoader,
)


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should stack them into a batch."""
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = fa_collate(batch)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays should convert and stack them."""
        batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(batch)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(result, expected)

    def test_collate_strings(self):
        """Collating strings should return a list of strings."""
        batch = ['hello', 'world']
        result = fa_collate(batch)
        assert result == ['hello', 'world']

    def test_collate_tuples_preserves_type(self):
        """Collating tuples of tensors should preserve tuple type."""
        batch = [
            (torch.tensor([1]), torch.tensor([2])),
            (torch.tensor([3]), torch.tensor([4])),
        ]
        result = fa_collate(batch)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([[1], [3]]))
        assert torch.equal(result[1], torch.tensor([[2], [4]]))

    def test_collate_lists_preserves_type(self):
        """Collating lists of tensors should preserve list type."""
        batch = [
            [torch.tensor([1]), torch.tensor([2])],
            [torch.tensor([3]), torch.tensor([4])],
        ]
        result = fa_collate(batch)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([[1], [3]]))
        assert torch.equal(result[1], torch.tensor([[2], [4]]))

    def test_collate_scalar_tensors(self):
        """Collating scalar tensors should produce a 1D tensor."""
        batch = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(batch)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_collate_2d_tensors(self):
        """Collating 2D tensors should stack along a new batch dimension."""
        batch = [torch.ones(2, 3), torch.zeros(2, 3)]
        result = fa_collate(batch)
        assert result.shape == (2, 2, 3)

    def test_collate_dicts(self):
        """Collating dicts (Mapping type) should merge values."""
        batch = [{'a': torch.tensor(1), 'b': torch.tensor(2)},
                 {'a': torch.tensor(3), 'b': torch.tensor(4)}]
        result = fa_collate(batch)
        assert torch.equal(result['a'], torch.tensor([1, 3]))
        assert torch.equal(result['b'], torch.tensor([2, 4]))


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
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_convert_string(self):
        """Converting a string should return it unchanged."""
        result = fa_convert('hello')
        assert result == 'hello'

    def test_convert_list_of_tensors(self):
        """Converting a list of tensors should preserve list type."""
        data = [torch.tensor(1), torch.tensor(2)]
        result = fa_convert(data)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor(1))
        assert torch.equal(result[1], torch.tensor(2))

    def test_convert_tuple_of_arrays(self):
        """Converting a tuple of numpy arrays should preserve tuple type."""
        data = (np.array([1, 2]), np.array([3, 4]))
        result = fa_convert(data)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))

    def test_convert_integer_passthrough(self):
        """Converting a plain integer (non-collate type) falls through to default_convert."""
        result = fa_convert(42)
        # default_convert returns scalars as tensors
        assert isinstance(result, (int, torch.Tensor))

    def test_convert_float_passthrough(self):
        """Converting a plain float (non-collate type) falls through to default_convert."""
        result = fa_convert(3.14)
        # default_convert returns scalars as tensors
        assert isinstance(result, (float, torch.Tensor))


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException should be an Exception subclass."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException should be raiseable and catchable."""
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
    """Tests for the collate_error function."""

    def test_mismatched_shapes_raises(self):
        """collate_error should raise when batch items have mismatched shapes."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
        ]
        with pytest.raises(Exception):
            # collate_error uses bare 'raise' so we need to give it a context
            try:
                raise RuntimeError("collation failed")
            except RuntimeError as e:
                collate_error(e, batch)

    def test_matching_shapes_no_raise(self):
        """collate_error should not raise when all shapes match."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
            (torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])),
        ]
        # Should not raise - shapes all match
        try:
            raise RuntimeError("collation failed")
        except RuntimeError as e:
            # When shapes match, collate_error just returns without re-raising
            collate_error(e, batch)

    def test_error_message_contains_mismatch_info(self):
        """Error message should contain shape mismatch details."""
        batch = [
            (torch.zeros(3, 4), torch.zeros(2, 2)),
            (torch.zeros(3, 4), torch.zeros(3, 3)),
        ]
        try:
            raise RuntimeError("collation failed")
        except RuntimeError as e:
            try:
                collate_error(e, batch)
            except RuntimeError as raised:
                assert 'Mismatch' in str(raised.args[0])
                assert 'shape' in str(raised.args[0])


# ============================================================
# Tests for DataLoader initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization and configuration."""

    def test_basic_init(self):
        """DataLoader should initialize with a dataset and batch size."""
        dl = DataLoader(list(range(10)), bs=2)
        assert dl.bs == 2
        assert dl.n == 10
        assert dl.indexed is True
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """batch_size parameter should map to bs for PyTorch compatibility."""
        dl = DataLoader(list(range(10)), batch_size=4)
        assert dl.bs == 4

    def test_shuffle_flag(self):
        """DataLoader should accept shuffle flag for indexed datasets."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        assert dl.shuffle is True

    def test_shuffle_non_indexed_raises(self):
        """Shuffling a non-indexed (iterable) dataset should raise ValueError."""
        class IterDS:
            def __iter__(self):
                yield from range(5)

        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(IterDS(), bs=2, shuffle=True)

    def test_drop_last_without_bs_raises(self):
        """drop_last requires bs to be set."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(10)), bs=None, drop_last=True)

    def test_indexed_auto_detection(self):
        """indexed should be auto-detected based on __getitem__."""
        # List has __getitem__
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.indexed is True

        # Iterator does not have __getitem__
        class IterDS:
            def __iter__(self):
                yield from range(5)

        dl2 = DataLoader(IterDS(), bs=2)
        assert dl2.indexed is False

    def test_device_init(self):
        """DataLoader should accept and store device."""
        dl = DataLoader(list(range(5)), bs=2, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_device_none(self):
        """DataLoader with no device should have device=None."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.device is None

    def test_n_auto_from_len(self):
        """n should be automatically set from len(dataset)."""
        dl = DataLoader(list(range(7)), bs=2)
        assert dl.n == 7

    def test_n_explicit_override(self):
        """Explicit n should override len(dataset)."""
        dl = DataLoader(list(range(10)), bs=2, n=5)
        assert dl.n == 5

    def test_pin_memory(self):
        """pin_memory flag should be stored."""
        dl = DataLoader(list(range(5)), bs=2, pin_memory=True)
        assert dl.pin_memory is True


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length computation."""

    def test_len_exact_division(self):
        """Length with exact division should be n // bs."""
        dl = DataLoader(list(range(10)), bs=5)
        assert len(dl) == 2

    def test_len_with_remainder(self):
        """Length with remainder should round up (no drop_last)."""
        dl = DataLoader(list(range(11)), bs=5)
        assert len(dl) == 3

    def test_len_drop_last(self):
        """Length with drop_last should be n // bs."""
        dl = DataLoader(list(range(11)), bs=5, drop_last=True)
        assert len(dl) == 2

    def test_len_prebatched(self):
        """Length when bs=None should be n."""
        dl = DataLoader(list(range(5)), bs=None)
        assert len(dl) == 5

    def test_len_raises_when_n_is_none(self):
        """Length should raise TypeError when n is None."""
        class NoLen:
            def __getitem__(self, idx):
                return idx

        dl = DataLoader(NoLen(), bs=2, n=None)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_item(self):
        """DataLoader with 1 item and bs=1."""
        dl = DataLoader([42], bs=1)
        assert len(dl) == 1

    def test_len_bs_larger_than_n(self):
        """When bs > n, should have exactly 1 batch (no drop_last)."""
        dl = DataLoader(list(range(3)), bs=10)
        assert len(dl) == 1

    def test_len_bs_larger_than_n_drop_last(self):
        """When bs > n and drop_last, should have 0 batches."""
        dl = DataLoader(list(range(3)), bs=10, drop_last=True)
        assert len(dl) == 0


# ============================================================
# Tests for DataLoader get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_sequential_indices(self):
        """get_idxs without shuffle should return sequential indices."""
        dl = DataLoader(list(range(5)), bs=2)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_shuffled_indices(self):
        """get_idxs with shuffle should return permuted indices."""
        dl = DataLoader(list(range(20)), bs=2, shuffle=True)
        idxs = dl.get_idxs()
        # Should contain same elements
        assert sorted(idxs) == list(range(20))
        # Very unlikely to be in order for 20 items
        assert idxs != list(range(20))

    def test_shuffled_indices_different_each_call(self):
        """Repeated calls to get_idxs should produce different orderings."""
        dl = DataLoader(list(range(50)), bs=2, shuffle=True)
        dl.randomize()
        idxs1 = dl.get_idxs()
        dl.randomize()
        idxs2 = dl.get_idxs()
        # With 50 items, extremely unlikely to get same permutation twice
        assert idxs1 != idxs2


# ============================================================
# Tests for DataLoader shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn method."""

    def test_shuffle_fn_permutes(self):
        """shuffle_fn should return a permutation of the input."""
        dl = DataLoader(list(range(10)), bs=2)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(10))

    def test_shuffle_fn_preserves_length(self):
        """shuffle_fn output should have same length as input."""
        dl = DataLoader(list(range(10)), bs=2)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert len(shuffled) == len(idxs)


# ============================================================
# Tests for DataLoader chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_basic(self):
        """chunkify should split items into batches of size bs."""
        dl = DataLoader(list(range(6)), bs=2)
        chunks = list(dl.chunkify(iter([1, 2, 3, 4, 5, 6])))
        assert chunks == [[1, 2], [3, 4], [5, 6]]

    def test_chunkify_with_remainder(self):
        """chunkify should include a partial final batch."""
        dl = DataLoader(list(range(5)), bs=2)
        chunks = list(dl.chunkify(iter([1, 2, 3, 4, 5])))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_chunkify_drop_last(self):
        """chunkify with drop_last should exclude the partial final batch."""
        dl = DataLoader(list(range(5)), bs=2, drop_last=True)
        chunks = list(dl.chunkify(iter([1, 2, 3, 4, 5])))
        assert chunks == [[1, 2], [3, 4]]

    def test_chunkify_prebatched(self):
        """chunkify with prebatched (bs=None) should pass through unchanged."""
        dl = DataLoader(list(range(5)), bs=None)
        items = [10, 20, 30]
        chunks = list(dl.chunkify(iter(items)))
        assert chunks == [10, 20, 30]


# ============================================================
# Tests for DataLoader create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_indexed_dataset(self):
        """create_item should index into the dataset for indexed datasets."""
        dl = DataLoader(['a', 'b', 'c', 'd'], bs=2)
        assert dl.create_item(0) == 'a'
        assert dl.create_item(2) == 'c'

    def test_non_indexed_raises_with_numeric_index(self):
        """create_item should raise IndexError for non-indexed datasets."""
        class IterDS:
            def __iter__(self):
                yield from range(5)

        dl = DataLoader(IterDS(), bs=2)
        with pytest.raises(IndexError, match="Cannot index an iterable"):
            dl.create_item(0)

    def test_non_indexed_with_none_uses_iterator(self):
        """create_item(None) on non-indexed datasets should use next(iterator)."""
        class IterDS:
            def __iter__(self):
                yield 100
                yield 200

        dl = DataLoader(IterDS(), bs=2)
        dl.it = iter(dl.dataset)
        assert dl.create_item(None) == 100
        assert dl.create_item(None) == 200


# ============================================================
# Tests for DataLoader do_item
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item method."""

    def test_do_item_returns_item(self):
        """do_item should return the item from the dataset."""
        dl = DataLoader([10, 20, 30], bs=2)
        assert dl.do_item(0) == 10
        assert dl.do_item(1) == 20

    def test_do_item_skip_returns_none(self):
        """do_item should return None when SkipItemException is raised."""
        dl = DataLoader([10, 20, 30], bs=2)
        dl.after_item = lambda x: (_ for _ in ()).throw(SkipItemException()) if x == 20 else x
        # Item at index 1 is 20, should be skipped
        assert dl.do_item(1) is None
        # Item at index 0 is 10, should be returned
        assert dl.do_item(0) == 10


# ============================================================
# Tests for DataLoader device property
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device management."""

    def test_device_setter(self):
        """Setting device should update the stored device."""
        dl = DataLoader(list(range(5)), bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_none(self):
        """Device can be set to None."""
        dl = DataLoader(list(range(5)), bs=2, device='cpu')
        dl.device = None
        assert dl.device is None

    def test_to_method(self):
        """The .to() method should set the device."""
        dl = DataLoader(list(range(5)), bs=2)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader prebatched property
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for DataLoader.prebatched property."""

    def test_prebatched_true(self):
        """prebatched should be True when bs is None."""
        dl = DataLoader(list(range(5)), bs=None)
        assert dl.prebatched is True

    def test_prebatched_false(self):
        """prebatched should be False when bs is set."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.prebatched is False


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_preserves_dataset(self):
        """new() without args should preserve the dataset."""
        data = list(range(10))
        dl = DataLoader(data, bs=2)
        dl2 = dl.new()
        assert dl2.dataset is data

    def test_new_with_different_bs(self):
        """new() should allow overriding batch size."""
        dl = DataLoader(list(range(10)), bs=2)
        dl2 = dl.new(bs=5)
        assert dl2.bs == 5
        assert len(dl2) == 2

    def test_new_with_different_dataset(self):
        """new() should allow providing a different dataset."""
        dl = DataLoader(list(range(10)), bs=2)
        new_data = list(range(20))
        dl2 = dl.new(dataset=new_data)
        assert dl2.dataset is new_data
        assert dl2.n == 20

    def test_new_preserves_shuffle(self):
        """new() should preserve shuffle setting."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        dl2 = dl.new()
        assert dl2.shuffle is True

    def test_new_preserves_drop_last(self):
        """new() should preserve drop_last setting."""
        dl = DataLoader(list(range(10)), bs=2, drop_last=True)
        dl2 = dl.new()
        assert dl2.drop_last is True


# ============================================================
# Tests for DataLoader.one_batch (error cases)
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch error handling."""

    def test_one_batch_empty_raises(self):
        """one_batch on empty DataLoader should raise ValueError."""
        dl = DataLoader([], bs=2, n=0)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()

    def test_one_batch_drop_last_empty_raises(self):
        """one_batch when drop_last leaves 0 batches should raise ValueError."""
        dl = DataLoader(list(range(3)), bs=10, drop_last=True)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for DataLoader randomize
# ============================================================

class TestDataLoaderRandomize:
    """Tests for DataLoader.randomize method."""

    def test_randomize_changes_rng_state(self):
        """randomize should change the internal RNG state."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


# ============================================================
# Tests for _FakeLoader
# ============================================================

class TestFakeLoader:
    """Tests for the _FakeLoader helper class."""

    def test_no_multiproc_context_manager(self):
        """no_multiproc should temporarily set num_workers to 0."""
        dl = DataLoader(list(range(6)), bs=2, num_workers=0)
        # On macOS/Windows in notebook, num_workers gets forced to 0 anyway
        original = dl.fake_l.num_workers
        with dl.fake_l.no_multiproc() as d:
            assert dl.fake_l.num_workers == 0
            assert d is dl
        assert dl.fake_l.num_workers == original

    def test_fn_noops(self):
        """_fn_noops should return its first argument."""
        fl = _FakeLoader.__new__(_FakeLoader)
        assert fl._fn_noops(42) == 42
        assert fl._fn_noops('hello') == 'hello'
        assert fl._fn_noops(None) is None

    def test_fake_loader_attributes(self):
        """_FakeLoader should have expected class-level attributes."""
        assert _FakeLoader._IterableDataset_len_called is None
        assert _FakeLoader._auto_collation is False
        assert _FakeLoader.drop_last is False


# ============================================================
# Tests for DataLoader create_batch
# ============================================================

class TestDataLoaderCreateBatch:
    """Tests for DataLoader.create_batch method."""

    def test_create_batch_collates_tensors(self):
        """create_batch should collate a list of tensors into a batch."""
        dl = DataLoader(list(range(6)), bs=2)
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = dl.create_batch(batch)
        expected = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(result, expected)

    def test_create_batch_prebatched_converts(self):
        """create_batch in prebatched mode uses fa_convert."""
        dl = DataLoader([np.array([1, 2, 3])], bs=None)
        batch = np.array([1, 2, 3])
        result = dl.create_batch(batch)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(result, expected)

    def test_create_batch_collate_error_on_mismatch(self):
        """create_batch should raise on shape mismatch with tuple items."""
        dl = DataLoader(list(range(6)), bs=2)
        # collate_error expects batch items to be tuples/sequences of tensors
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
        ]
        with pytest.raises(Exception):
            dl.create_batch(batch)
