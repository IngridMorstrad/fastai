"""Tests for fastai.data.load module.

Covers: fa_collate, fa_convert, SkipItemException, collate_error,
DataLoader class (init, __len__, get_idxs, create_item, do_item,
create_batch, chunkify, shuffle_fn, new, device, one_batch, etc.)
and the _FakeLoader helper.
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
    _collate_types,
)


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """fa_collate should stack a list of tensors into a batch."""
        items = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        result = fa_collate(items)
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """fa_collate should collate numpy arrays into a tensor."""
        items = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        result = fa_collate(items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_collate_strings(self):
        """fa_collate should collate strings using default_collate behavior."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_tuples_of_tensors(self):
        """fa_collate should collate tuples element-wise, preserving tuple type."""
        items = [
            (torch.tensor([1, 2]), torch.tensor([10])),
            (torch.tensor([3, 4]), torch.tensor([20])),
        ]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1, 2], [3, 4]]))
        assert torch.equal(result[1], torch.tensor([[10], [20]]))

    def test_collate_lists_of_tensors(self):
        """fa_collate should collate lists element-wise, preserving list type."""
        items = [
            [torch.tensor([1]), torch.tensor([2])],
            [torch.tensor([3]), torch.tensor([4])],
        ]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([[1], [3]]))
        assert torch.equal(result[1], torch.tensor([[2], [4]]))

    def test_collate_scalars(self):
        """fa_collate should collate scalar tensors."""
        items = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(items)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_collate_dicts(self):
        """fa_collate should handle dicts (Mapping type)."""
        items = [{"a": torch.tensor([1])}, {"a": torch.tensor([2])}]
        result = fa_collate(items)
        assert isinstance(result, dict)
        assert torch.equal(result["a"], torch.tensor([[1], [2]]))


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """fa_convert should pass tensors through unchanged."""
        t = torch.tensor([1, 2, 3])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_numpy(self):
        """fa_convert should convert numpy arrays to tensors."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = fa_convert(a)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_convert_list_of_tensors(self):
        """fa_convert should convert each element in a list, preserving list type."""
        items = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))

    def test_convert_tuple_of_tensors(self):
        """fa_convert should convert each element in a tuple, preserving tuple type."""
        items = (torch.tensor([1]), torch.tensor([2]))
        result = fa_convert(items)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_convert_string(self):
        """fa_convert should handle strings (they are a _collate_type)."""
        result = fa_convert("hello")
        assert result == "hello"

    def test_convert_nested_list(self):
        """fa_convert should handle nested sequences."""
        items = [[torch.tensor(1), torch.tensor(2)], [torch.tensor(3), torch.tensor(4)]]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert len(result) == 2
        # Each inner element should also be converted
        assert isinstance(result[0], list)


# ============================================================
# Tests for SkipItemException
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception(self):
        """SkipItemException should be a subclass of Exception."""
        assert issubclass(SkipItemException, Exception)

    def test_can_raise_and_catch(self):
        """SkipItemException should be raisable and catchable."""
        with pytest.raises(SkipItemException):
            raise SkipItemException()

    def test_message(self):
        """SkipItemException should support custom messages."""
        exc = SkipItemException("skip this item")
        assert str(exc) == "skip this item"


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for the collate_error function."""

    def test_raises_on_shape_mismatch(self):
        """collate_error should re-raise with informative message on shape mismatch."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
            (torch.tensor([1, 2, 3]), torch.tensor([6, 7, 8])),
        ]
        e = Exception("original error")
        # Simulate being in an except block by using raise from within try/except
        with pytest.raises(Exception) as exc_info:
            try:
                raise e
            except Exception as caught:
                collate_error(caught, batch)

        assert "Mismatch found" in str(exc_info.value)
        assert "shape" in str(exc_info.value)

    def test_no_raise_when_shapes_match(self):
        """collate_error should not raise if all shapes match."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])),
            (torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])),
        ]
        e = Exception("original error")
        # Should not raise since shapes all match
        try:
            raise e
        except Exception as caught:
            collate_error(caught, batch)
        # If we get here, no exception was re-raised


# ============================================================
# Tests for DataLoader initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization and configuration."""

    def test_basic_init(self):
        """DataLoader should initialize with a simple list dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert dl.bs == 5
        assert dl.n == 10
        assert dl.indexed is True
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """batch_size parameter should set bs for PyTorch compatibility."""
        ds = list(range(10))
        dl = DataLoader(ds, batch_size=4)
        assert dl.bs == 4

    def test_custom_n_parameter(self):
        """n parameter should override the dataset length."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=5, n=20)
        assert dl.n == 20

    def test_indexed_detection_with_list(self):
        """Lists (with __getitem__) should be detected as indexed."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.indexed is True

    def test_indexed_detection_with_iterable(self):
        """Iterables without __getitem__ should not be detected as indexed."""
        class IterOnly:
            def __iter__(self):
                return iter(range(5))
            def __len__(self):
                return 5
        dl = DataLoader(IterOnly(), bs=2)
        assert dl.indexed is False

    def test_explicit_indexed_false(self):
        """Should allow explicitly setting indexed=False."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2, indexed=False)
        assert dl.indexed is False

    def test_shuffle_raises_for_iterable(self):
        """Shuffle should raise ValueError for non-indexed datasets."""
        class IterOnly:
            def __iter__(self):
                return iter(range(5))
            def __len__(self):
                return 5
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(IterOnly(), bs=2, shuffle=True)

    def test_drop_last_requires_bs(self):
        """drop_last=True with bs=None should raise AssertionError."""
        with pytest.raises(AssertionError):
            DataLoader(list(range(5)), bs=None, drop_last=True)

    def test_none_dataset(self):
        """DataLoader should handle None dataset."""
        dl = DataLoader(dataset=None, bs=2, n=0)
        assert dl.dataset is None

    def test_device_initially_none(self):
        """Device should be None by default."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.device is None


# ============================================================
# Tests for DataLoader __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader.__len__ method."""

    def test_len_exact_division(self):
        """Length should be n/bs when evenly divisible."""
        dl = DataLoader(list(range(20)), bs=5)
        assert len(dl) == 4

    def test_len_with_remainder(self):
        """Length should round up when not evenly divisible."""
        dl = DataLoader(list(range(22)), bs=5)
        assert len(dl) == 5  # ceil(22/5) = 5

    def test_len_with_drop_last(self):
        """Length should floor when drop_last=True."""
        dl = DataLoader(list(range(22)), bs=5, drop_last=True)
        assert len(dl) == 4  # floor(22/5) = 4

    def test_len_prebatched(self):
        """Length should be n when bs is None (prebatched)."""
        dl = DataLoader(list(range(10)), bs=None)
        assert len(dl) == 10

    def test_len_no_n_raises_type_error(self):
        """Length should raise TypeError when n is None."""
        class NoLen:
            def __iter__(self):
                return iter(range(5))
        dl = DataLoader(NoLen(), bs=2)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_with_custom_n(self):
        """Length should use custom n parameter."""
        dl = DataLoader(list(range(100)), bs=10, n=35)
        assert len(dl) == 4  # ceil(35/10) = 4


# ============================================================
# Tests for DataLoader get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_sequential(self):
        """get_idxs should return sequential indices for indexed datasets."""
        dl = DataLoader(list(range(5)), bs=2)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_shuffled(self):
        """get_idxs should return shuffled indices when shuffle=True."""
        dl = DataLoader(list(range(20)), bs=5, shuffle=True)
        idxs = dl.get_idxs()
        assert len(idxs) == 20
        assert set(idxs) == set(range(20))
        # Highly unlikely to remain sorted after shuffle
        assert idxs != list(range(20))

    def test_get_idxs_respects_n(self):
        """get_idxs should only return n indices."""
        dl = DataLoader(list(range(100)), bs=5, n=10)
        idxs = dl.get_idxs()
        assert len(idxs) == 10


# ============================================================
# Tests for DataLoader create_item
# ============================================================

class TestDataLoaderCreateItem:
    """Tests for DataLoader.create_item method."""

    def test_create_item_indexed(self):
        """create_item should index into the dataset for indexed loaders."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=2)
        assert dl.create_item(0) == 10
        assert dl.create_item(2) == 30
        assert dl.create_item(4) == 50

    def test_create_item_non_indexed_next(self):
        """create_item should call next(self.it) for non-indexed datasets with None."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2, indexed=False)
        dl.it = iter(ds)
        assert dl.create_item(None) == 0
        assert dl.create_item(None) == 1

    def test_create_item_non_indexed_raises_for_int(self):
        """create_item should raise IndexError for non-indexed with int index."""
        class IterOnly:
            def __iter__(self):
                return iter(range(5))
            def __len__(self):
                return 5
        dl = DataLoader(IterOnly(), bs=2)
        with pytest.raises(IndexError, match="Cannot index an iterable dataset numerically"):
            dl.create_item(0)

    def test_create_item_with_tensor_dataset(self):
        """create_item should work with tensor datasets."""
        ds = [torch.tensor([i, i + 1]) for i in range(5)]
        dl = DataLoader(ds, bs=2)
        result = dl.create_item(3)
        assert torch.equal(result, torch.tensor([3, 4]))


# ============================================================
# Tests for DataLoader do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item method."""

    def test_do_item_returns_item(self):
        """do_item should return the dataset item at the given index."""
        ds = [100, 200, 300]
        dl = DataLoader(ds, bs=2)
        assert dl.do_item(0) == 100
        assert dl.do_item(1) == 200

    def test_do_item_skip_exception_returns_none(self):
        """do_item should return None when SkipItemException is raised in after_item."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)

        def skip_evens(x):
            if x % 2 == 0:
                raise SkipItemException()
            return x

        dl.after_item = skip_evens
        assert dl.do_item(0) is None  # 0 is even, skipped
        assert dl.do_item(1) == 1     # 1 is odd, kept
        assert dl.do_item(2) is None  # 2 is even, skipped
        assert dl.do_item(3) == 3     # 3 is odd, kept

    def test_do_item_applies_after_item(self):
        """do_item should apply after_item transformation."""
        ds = [1, 2, 3, 4]
        dl = DataLoader(ds, bs=2)
        dl.after_item = lambda x: x * 10
        assert dl.do_item(0) == 10
        assert dl.do_item(2) == 30


# ============================================================
# Tests for DataLoader create_batch
# ============================================================

class TestDataLoaderCreateBatch:
    """Tests for DataLoader.create_batch method."""

    def test_create_batch_collates_tensors(self):
        """create_batch should collate a list of tensors into a batch."""
        dl = DataLoader(list(range(10)), bs=3)
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])]
        result = dl.create_batch(batch)
        expected = torch.tensor([[1, 2], [3, 4], [5, 6]])
        assert torch.equal(result, expected)

    def test_create_batch_prebatched_converts(self):
        """create_batch should use fa_convert when prebatched (bs=None)."""
        dl = DataLoader(list(range(10)), bs=None)
        item = torch.tensor([1, 2, 3])
        result = dl.create_batch(item)
        assert torch.equal(result, item)

    def test_create_batch_calls_collate_error_on_mismatch(self):
        """create_batch should raise when tensor shapes mismatch."""
        dl = DataLoader(list(range(10)), bs=3)
        # collate_error expects each item in the batch to be a tuple of tensors.
        # Use properly structured batch where inner elements have mismatched shapes.
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([10, 20])),
            (torch.tensor([4, 5, 6]), torch.tensor([30, 40, 50])),
        ]
        with pytest.raises((RuntimeError, Exception)):
            dl.create_batch(batch)


# ============================================================
# Tests for DataLoader chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_with_bs(self):
        """chunkify should chunk items into groups of bs."""
        dl = DataLoader(list(range(10)), bs=3)
        items = iter(range(7))
        chunks = list(dl.chunkify(items))
        assert chunks == [[0, 1, 2], [3, 4, 5], [6]]

    def test_chunkify_with_drop_last(self):
        """chunkify should drop the last incomplete batch when drop_last=True."""
        dl = DataLoader(list(range(10)), bs=3, drop_last=True)
        items = iter(range(7))
        chunks = list(dl.chunkify(items))
        assert chunks == [[0, 1, 2], [3, 4, 5]]

    def test_chunkify_prebatched(self):
        """chunkify should pass through items unchanged when prebatched."""
        dl = DataLoader(list(range(5)), bs=None)
        items = [10, 20, 30]
        result = list(dl.chunkify(items))
        assert result == [10, 20, 30]


# ============================================================
# Tests for DataLoader shuffle_fn
# ============================================================

class TestDataLoaderShuffleFn:
    """Tests for DataLoader.shuffle_fn method."""

    def test_shuffle_fn_permutes(self):
        """shuffle_fn should return a permutation of the input."""
        dl = DataLoader(list(range(20)), bs=5, shuffle=True)
        idxs = list(range(20))
        shuffled = dl.shuffle_fn(idxs)
        assert len(shuffled) == 20
        assert set(shuffled) == set(idxs)

    def test_shuffle_fn_preserves_elements(self):
        """shuffle_fn should not add or remove elements."""
        dl = DataLoader(list(range(10)), bs=2, shuffle=True)
        idxs = [5, 10, 15, 20, 25]
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == sorted(idxs)


# ============================================================
# Tests for DataLoader randomize
# ============================================================

class TestDataLoaderRandomize:
    """Tests for DataLoader.randomize method."""

    def test_randomize_changes_rng_state(self):
        """randomize should change the internal RNG state."""
        dl = DataLoader(list(range(20)), bs=5, shuffle=True)
        # Get first shuffle result
        idxs1 = dl.shuffle_fn(list(range(20)))
        dl.randomize()
        # After randomize, shuffle should produce different results
        idxs2 = dl.shuffle_fn(list(range(20)))
        # With 20 items, extremely unlikely to get same order
        assert idxs1 != idxs2


# ============================================================
# Tests for DataLoader properties
# ============================================================

class TestDataLoaderProperties:
    """Tests for DataLoader property accessors."""

    def test_prebatched_true(self):
        """prebatched should be True when bs is None."""
        dl = DataLoader(list(range(5)), bs=None)
        assert dl.prebatched is True

    def test_prebatched_false(self):
        """prebatched should be False when bs is set."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.prebatched is False

    def test_device_setter_getter(self):
        """device property should be settable and gettable."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.device is None
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_to_method(self):
        """to method should set the device."""
        dl = DataLoader(list(range(5)), bs=2)
        dl.to('cpu')
        assert dl.device == torch.device('cpu')


# ============================================================
# Tests for DataLoader new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_preserves_defaults(self):
        """new should preserve original settings by default."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, shuffle=True, drop_last=True)
        dl2 = dl.new()
        assert dl2.bs == 5
        assert dl2.shuffle is True
        assert dl2.drop_last is True
        assert dl2.dataset is ds

    def test_new_overrides_bs(self):
        """new should allow overriding bs."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5)
        dl2 = dl.new(bs=10)
        assert dl2.bs == 10
        assert dl2.dataset is ds

    def test_new_with_different_dataset(self):
        """new should allow specifying a different dataset."""
        ds1 = list(range(10))
        ds2 = list(range(20))
        dl = DataLoader(ds1, bs=5)
        dl2 = dl.new(dataset=ds2)
        assert dl2.dataset is ds2
        assert dl2.n == 20

    def test_new_returns_same_type(self):
        """new should return an instance of the same type."""
        dl = DataLoader(list(range(10)), bs=5)
        dl2 = dl.new()
        assert type(dl2) is DataLoader


# ============================================================
# Tests for DataLoader one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_empty_raises_value_error(self):
        """one_batch should raise ValueError when DataLoader has no batches."""
        dl = DataLoader([], bs=5, n=0)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


# ============================================================
# Tests for _FakeLoader
# ============================================================

class TestFakeLoader:
    """Tests for the _FakeLoader helper class."""

    def test_no_multiproc_context_manager(self):
        """no_multiproc should temporarily set num_workers to 0."""
        dl = DataLoader(list(range(10)), bs=3)
        # On macOS/Windows in notebook, num_workers is forced to 0
        # but we can test the context manager logic
        original_workers = dl.fake_l.num_workers
        with dl.fake_l.no_multiproc() as d:
            assert dl.fake_l.num_workers == 0
            assert d is dl
        assert dl.fake_l.num_workers == original_workers

    def test_fake_loader_attributes(self):
        """_FakeLoader should have required DataLoader protocol attributes."""
        dl = DataLoader(list(range(10)), bs=3)
        fl = dl.fake_l
        assert fl._auto_collation is False
        assert fl.drop_last is False
        assert fl._IterableDataset_len_called is None
        assert fl.prefetch_factor == 2

    def test_fn_noops(self):
        """_FakeLoader._fn_noops should return its input unchanged."""
        dl = DataLoader(list(range(10)), bs=3)
        fl = dl.fake_l
        assert fl._fn_noops(42) == 42
        assert fl._fn_noops("test") == "test"
        assert fl._fn_noops(None) is None

    def test_fake_loader_dataset_is_self(self):
        """_FakeLoader.dataset should be self (for PyTorch protocol)."""
        dl = DataLoader(list(range(10)), bs=3)
        fl = dl.fake_l
        assert fl.dataset is fl


# ============================================================
# Tests for _collate_types
# ============================================================

class TestCollateTypes:
    """Tests for the _collate_types tuple used in type checks."""

    def test_includes_ndarray(self):
        """_collate_types should include numpy ndarray."""
        assert np.ndarray in _collate_types

    def test_includes_tensor(self):
        """_collate_types should include torch Tensor."""
        assert torch.Tensor in _collate_types

    def test_includes_str(self):
        """_collate_types should include str."""
        assert str in _collate_types


# ============================================================
# Tests for DataLoader noop methods
# ============================================================

class TestDataLoaderNoopMethods:
    """Tests for DataLoader noop callback methods."""

    def test_wif_noop(self):
        """wif should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.wif(42) == 42

    def test_before_iter_noop(self):
        """before_iter should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.before_iter("test") == "test"

    def test_after_item_noop(self):
        """after_item should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        x = torch.tensor([1, 2, 3])
        assert torch.equal(dl.after_item(x), x)

    def test_before_batch_noop(self):
        """before_batch should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        batch = [1, 2, 3]
        assert dl.before_batch(batch) == batch

    def test_after_batch_noop(self):
        """after_batch should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        batch = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(dl.after_batch(batch), batch)

    def test_after_iter_noop(self):
        """after_iter should be a no-op that returns its input."""
        dl = DataLoader(list(range(5)), bs=2)
        assert dl.after_iter(None) is None


# ============================================================
# Tests for DataLoader edge cases
# ============================================================

class TestDataLoaderEdgeCases:
    """Tests for DataLoader edge case handling."""

    def test_single_item_dataset(self):
        """DataLoader should handle a single-item dataset."""
        dl = DataLoader([42], bs=1)
        assert len(dl) == 1
        assert dl.create_item(0) == 42

    def test_bs_larger_than_dataset(self):
        """DataLoader should handle bs larger than dataset."""
        dl = DataLoader(list(range(3)), bs=10)
        assert len(dl) == 1  # ceil(3/10) = 1

    def test_bs_equals_n(self):
        """DataLoader should handle bs equal to n."""
        dl = DataLoader(list(range(10)), bs=10)
        assert len(dl) == 1

    def test_drop_last_exact_division(self):
        """drop_last should not affect length when n is divisible by bs."""
        dl = DataLoader(list(range(10)), bs=5, drop_last=True)
        assert len(dl) == 2

    def test_large_dataset_len(self):
        """DataLoader should handle large dataset sizes correctly."""
        dl = DataLoader(list(range(1000)), bs=64)
        assert len(dl) == 16  # ceil(1000/64) = 16

    def test_device_property_with_none(self):
        """Setting device to None should work."""
        dl = DataLoader(list(range(5)), bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')
        dl.device = None
        assert dl.device is None
