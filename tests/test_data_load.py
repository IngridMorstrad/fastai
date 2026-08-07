"""Tests for fastai.data.load module.

Covers: fa_collate, fa_convert, SkipItemException, collate_error, DataLoader.
Tests batching, shuffling, indexing, skip item, length calculation, device
placement, and other DataLoader behaviors.
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Patch retain_types/retain_type before importing DataLoader - these functions
# were moved from fastcore.dispatch to the fasttransform package, but the fastai
# code still expects them in the global namespace via star-imports.
from fasttransform import retain_types, retain_type
import fastai.data.load as _load_mod
import fastai.torch_core as _torch_core_mod
_load_mod.retain_types = retain_types
_torch_core_mod.retain_type = retain_type

from fastai.data.load import (
    fa_collate,
    fa_convert,
    SkipItemException,
    collate_error,
    DataLoader,
)


# ============================================================
# Tests for fa_collate
# ============================================================

class TestFaCollate:
    """Tests for fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should stack them."""
        items = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = fa_collate(items)
        expected = torch.tensor([[1, 2], [3, 4]])
        assert torch.equal(result, expected)

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays should convert them into a tensor."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_collate_strings(self):
        """Collating strings should return a list of strings (no stacking)."""
        items = ["hello", "world"]
        result = fa_collate(items)
        assert result == ["hello", "world"]

    def test_collate_tuples(self):
        """Collating a list of tuples should maintain the tuple type."""
        items = [
            (torch.tensor([1, 2]), torch.tensor([10])),
            (torch.tensor([3, 4]), torch.tensor([20])),
        ]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([[1, 2], [3, 4]]))
        assert torch.equal(result[1], torch.tensor([[10], [20]]))

    def test_collate_lists(self):
        """Collating a list of lists should maintain the list type."""
        items = [
            [torch.tensor([1]), torch.tensor([10])],
            [torch.tensor([2]), torch.tensor([20])],
        ]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert torch.equal(result[0], torch.tensor([[1], [2]]))
        assert torch.equal(result[1], torch.tensor([[10], [20]]))

    def test_collate_scalar_tensors(self):
        """Collating scalar tensors should produce a 1D tensor."""
        items = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        result = fa_collate(items)
        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_collate_dicts(self):
        """Collating dicts (Mapping type) should use default_collate behavior."""
        items = [{"a": torch.tensor(1)}, {"a": torch.tensor(2)}]
        result = fa_collate(items)
        assert isinstance(result, dict)
        assert torch.equal(result["a"], torch.tensor([1, 2]))


# ============================================================
# Tests for fa_convert
# ============================================================

class TestFaConvert:
    """Tests for fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor should return the same tensor."""
        t = torch.tensor([1, 2, 3])
        result = fa_convert(t)
        assert torch.equal(result, t)

    def test_convert_numpy_array(self):
        """Converting a numpy array should return a tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        result = fa_convert(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_convert_string(self):
        """Converting a string should return it as-is."""
        result = fa_convert("hello")
        assert result == "hello"

    def test_convert_tuple_of_tensors(self):
        """Converting a tuple of tensors should maintain tuple type."""
        t = (torch.tensor([1, 2]), torch.tensor([3, 4]))
        result = fa_convert(t)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.tensor([1, 2]))
        assert torch.equal(result[1], torch.tensor([3, 4]))

    def test_convert_list_of_numpy(self):
        """Converting a list of numpy arrays should maintain list type."""
        items = [np.array([1.0]), np.array([2.0])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_convert_dict(self):
        """Converting a dict should use default_convert behavior."""
        d = {"x": np.array([1, 2, 3])}
        result = fa_convert(d)
        assert isinstance(result, dict)
        assert isinstance(result["x"], torch.Tensor)


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
            raise SkipItemException("skip this item")


# ============================================================
# Tests for collate_error
# ============================================================

class TestCollateError:
    """Tests for collate_error function."""

    def test_raises_with_shape_mismatch_info(self):
        """collate_error should re-raise with informative shape mismatch message."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([10])),
            (torch.tensor([4, 5]), torch.tensor([20])),  # shape mismatch on axis 0
        ]
        # collate_error uses bare 'raise' so it must be called inside an except block
        with pytest.raises(RuntimeError) as exc_info:
            try:
                raise RuntimeError("original collation error")
            except RuntimeError as e:
                collate_error(e, batch)
        assert "Mismatch found on axis 0" in str(exc_info.value)
        assert "shape" in str(exc_info.value).lower()

    def test_no_error_when_shapes_match(self):
        """collate_error should not raise if all shapes match."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([10])),
            (torch.tensor([4, 5, 6]), torch.tensor([20])),
        ]
        # When shapes match, collate_error just returns without raising
        try:
            raise RuntimeError("original")
        except RuntimeError as e:
            # Should not raise since shapes match
            collate_error(e, batch)


# ============================================================
# Tests for DataLoader - Initialization
# ============================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_basic_creation(self):
        """DataLoader should be created with a simple list dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.bs == 2
        assert dl.n == 10
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_batch_size_alias(self):
        """batch_size kwarg should be accepted as alias for bs."""
        ds = list(range(10))
        dl = DataLoader(ds, batch_size=4)
        assert dl.bs == 4

    def test_indexed_auto_detection_list(self):
        """A list dataset should be detected as indexed."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2)
        assert dl.indexed is True

    def test_indexed_auto_detection_iterable(self):
        """An iterable dataset (without __getitem__) should not be indexed."""
        class IterDS:
            def __iter__(self):
                return iter(range(5))
        dl = DataLoader(IterDS(), bs=2, n=5)
        assert dl.indexed is False

    def test_shuffle_requires_indexed(self):
        """Shuffling a non-indexed dataset should raise ValueError."""
        class IterDS:
            def __iter__(self):
                return iter(range(5))
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(IterDS(), bs=2, n=5, shuffle=True)

    def test_drop_last_requires_bs(self):
        """drop_last=True without bs should raise AssertionError."""
        ds = list(range(10))
        with pytest.raises(AssertionError):
            DataLoader(ds, bs=None, drop_last=True)

    def test_n_from_dataset_len(self):
        """n should be inferred from len(dataset) if not given."""
        ds = list(range(7))
        dl = DataLoader(ds, bs=3)
        assert dl.n == 7

    def test_explicit_n_overrides_dataset_len(self):
        """Explicitly passing n should override len(dataset)."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, n=10)
        assert dl.n == 10


# ============================================================
# Tests for DataLoader - __len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader __len__."""

    def test_len_exact_division(self):
        """Length should be n // bs when evenly divisible."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 2

    def test_len_with_remainder(self):
        """Length should include partial batch when drop_last=False."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        assert len(dl) == 4  # 10 // 3 = 3, plus 1 remainder

    def test_len_drop_last(self):
        """Length should exclude partial batch when drop_last=True."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        assert len(dl) == 3  # 10 // 3 = 3, remainder dropped

    def test_len_no_bs(self):
        """With bs=None, length should be n (prebatched mode)."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 2

    def test_len_raises_when_n_is_none(self):
        """Length should raise TypeError when n is None."""
        class InfDS:
            def __getitem__(self, i):
                return i
        dl = DataLoader(InfDS(), bs=2, n=None)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_single_batch(self):
        """A dataset fitting exactly one batch should have length 1."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 1

    def test_len_one_item(self):
        """A single-item dataset with bs=1 should have length 1."""
        ds = [42]
        dl = DataLoader(ds, bs=1)
        assert len(dl) == 1


# ============================================================
# Tests for DataLoader - Iteration and Batching
# ============================================================

class TestDataLoaderIteration:
    """Tests for DataLoader iteration and batching."""

    def test_basic_iteration(self):
        """Iterating should produce batches of the correct size."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        batches = list(dl)
        assert len(batches) == 4
        # First batch should have 3 items
        assert len(batches[0]) == 3
        # Last batch has 1 item (remainder)
        assert len(batches[-1]) == 1

    def test_iteration_drop_last(self):
        """With drop_last=True, partial batches should be dropped."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        batches = list(dl)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 3

    def test_all_items_present_no_shuffle(self):
        """Without shuffle, all items should appear in order."""
        ds = list(range(6))
        dl = DataLoader(ds, bs=2)
        all_items = []
        for b in dl:
            all_items.extend(b.tolist())
        assert all_items == [0, 1, 2, 3, 4, 5]

    def test_shuffle_changes_order(self):
        """With shuffle=True, item order should differ from the original."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, shuffle=True)
        all_items = []
        for b in dl:
            all_items.extend(b.tolist())
        # All items present
        assert sorted(all_items) == list(range(100))
        # Very unlikely that shuffled order equals original
        assert all_items != list(range(100))

    def test_shuffle_produces_different_epochs(self):
        """Shuffling should produce different orderings across epochs."""
        ds = list(range(50))
        dl = DataLoader(ds, bs=50, shuffle=True)
        epoch1 = list(dl)[0].tolist()
        epoch2 = list(dl)[0].tolist()
        # Different epochs should produce different orderings
        assert epoch1 != epoch2

    def test_tensor_dataset(self):
        """DataLoader should work with a list of tensors."""
        ds = [torch.tensor([i, i * 2]) for i in range(6)]
        dl = DataLoader(ds, bs=3)
        batches = list(dl)
        assert len(batches) == 2
        assert batches[0].shape == (3, 2)
        assert batches[1].shape == (3, 2)

    def test_tuple_dataset(self):
        """DataLoader should work with a list of tuples (x, y)."""
        ds = [(torch.tensor([i]), torch.tensor([i * 10])) for i in range(4)]
        dl = DataLoader(ds, bs=2)
        batches = list(dl)
        assert len(batches) == 2
        x, y = batches[0]
        assert x.shape == (2, 1)
        assert y.shape == (2, 1)

    def test_multiple_iterations(self):
        """DataLoader should be re-iterable."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        batches1 = list(dl)
        batches2 = list(dl)
        assert len(batches1) == len(batches2) == 2


# ============================================================
# Tests for DataLoader - SkipItemException
# ============================================================

class TestDataLoaderSkipItem:
    """Tests for SkipItemException handling in DataLoader."""

    def test_skip_item_filters_items(self):
        """Items that raise SkipItemException in after_item should be skipped."""
        ds = list(range(10))

        def skip_evens(x):
            if x % 2 == 0:
                raise SkipItemException()
            return x

        dl = DataLoader(ds, bs=5, after_item=skip_evens)
        all_items = []
        for b in dl:
            all_items.extend(b.tolist())
        # Only odd numbers should remain
        assert all_items == [1, 3, 5, 7, 9]

    def test_skip_all_items_empty_result(self):
        """If all items are skipped, no batches should be produced."""
        ds = list(range(5))

        def skip_all(x):
            raise SkipItemException()

        dl = DataLoader(ds, bs=2, after_item=skip_all)
        batches = list(dl)
        assert batches == []


# ============================================================
# Tests for DataLoader - get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_returns_sequential(self):
        """Without shuffle, get_idxs should return sequential indices."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_shuffled(self):
        """With shuffle, get_idxs should return permuted indices."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=5, shuffle=True)
        dl.randomize()
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(20))
        # Should be shuffled
        assert idxs != list(range(20))

    def test_get_idxs_length_matches_n(self):
        """get_idxs should return exactly n indices."""
        ds = list(range(15))
        dl = DataLoader(ds, bs=4)
        idxs = dl.get_idxs()
        assert len(idxs) == 15


# ============================================================
# Tests for DataLoader - Prebatched mode (bs=None)
# ============================================================

class TestDataLoaderPrebatched:
    """Tests for prebatched mode (bs=None)."""

    def test_prebatched_property(self):
        """prebatched should be True when bs is None."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        assert dl.prebatched is True

    def test_prebatched_false(self):
        """prebatched should be False when bs is set."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.prebatched is False

    def test_prebatched_iteration(self):
        """In prebatched mode, each item is yielded as its own batch."""
        ds = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        dl = DataLoader(ds, bs=None)
        batches = list(dl)
        assert len(batches) == 2
        assert torch.equal(batches[0], torch.tensor([1, 2, 3]))
        assert torch.equal(batches[1], torch.tensor([4, 5, 6]))


# ============================================================
# Tests for DataLoader - one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_returns_first_batch(self):
        """one_batch should return the first batch."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        batch = dl.one_batch()
        assert len(batch) == 3
        assert torch.equal(batch, torch.tensor([0, 1, 2]))

    def test_one_batch_empty_raises(self):
        """one_batch should raise ValueError when DataLoader is empty."""
        ds = []
        dl = DataLoader(ds, bs=2)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()

    def test_one_batch_with_tensors(self):
        """one_batch should work with tensor datasets."""
        ds = [torch.tensor([i, i + 1]) for i in range(6)]
        dl = DataLoader(ds, bs=3)
        batch = dl.one_batch()
        assert batch.shape == (3, 2)


# ============================================================
# Tests for DataLoader - Device placement
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader device placement."""

    def test_to_device(self, cpu_device):
        """DataLoader.to() should set the device."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.to(cpu_device)
        assert dl.device == cpu_device

    def test_device_setter(self, cpu_device):
        """Setting device property should move data to that device."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2, device=cpu_device)
        batch = dl.one_batch()
        assert batch.device == cpu_device

    def test_device_none_by_default(self):
        """Device should be None when not specified."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        assert dl.device is None


# ============================================================
# Tests for DataLoader - new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_creates_copy(self):
        """new() should create a new DataLoader with the same settings."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, shuffle=True, drop_last=True)
        dl2 = dl.new()
        assert dl2.bs == 3
        assert dl2.shuffle is True
        assert dl2.drop_last is True
        assert dl2.n == 10

    def test_new_with_different_bs(self):
        """new() should allow overriding settings."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl2 = dl.new(bs=5)
        assert dl2.bs == 5

    def test_new_with_different_dataset(self):
        """new() should allow passing a different dataset."""
        ds1 = list(range(10))
        ds2 = list(range(20))
        dl = DataLoader(ds1, bs=3)
        dl2 = dl.new(dataset=ds2)
        assert dl2.n == 20

    def test_new_preserves_indexed(self):
        """new() should preserve the indexed setting."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl2 = dl.new()
        assert dl2.indexed == dl.indexed


# ============================================================
# Tests for DataLoader - Callbacks
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback methods."""

    def test_before_batch_called(self):
        """before_batch should be applied to each batch."""
        ds = list(range(6))
        call_count = [0]

        def count_calls(b):
            call_count[0] += 1
            return b

        dl = DataLoader(ds, bs=3, before_batch=count_calls)
        list(dl)
        assert call_count[0] == 2  # 6 items / bs=3 = 2 batches

    def test_after_batch_called(self):
        """after_batch should be applied to each collated batch."""
        ds = list(range(6))

        def double_batch(b):
            return b * 2

        dl = DataLoader(ds, bs=3, after_batch=double_batch)
        batches = list(dl)
        # Each batch should be doubled
        assert torch.equal(batches[0], torch.tensor([0, 1, 2]) * 2)

    def test_after_item_called(self):
        """after_item should be applied to each item."""
        ds = list(range(5))

        def add_ten(x):
            return x + 10

        dl = DataLoader(ds, bs=5, after_item=add_ten)
        batch = dl.one_batch()
        assert torch.equal(batch, torch.tensor([10, 11, 12, 13, 14]))

    def test_create_item_indexed(self):
        """create_item should use indexing for indexed datasets."""
        ds = [10, 20, 30, 40, 50]
        dl = DataLoader(ds, bs=5)
        assert dl.create_item(0) == 10
        assert dl.create_item(4) == 50

    def test_create_item_iterable(self):
        """create_item should use next(iter) for non-indexed datasets."""
        class IterDS:
            def __iter__(self):
                return iter([100, 200, 300])
        dl = DataLoader(IterDS(), bs=3, n=3)
        dl.it = iter(dl.dataset)
        assert dl.create_item(None) == 100
        assert dl.create_item(None) == 200


# ============================================================
# Tests for DataLoader - shuffle_fn and randomize
# ============================================================

class TestDataLoaderShuffle:
    """Tests for shuffle behavior."""

    def test_shuffle_fn_permutes(self):
        """shuffle_fn should return a permutation of all indices."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, shuffle=True)
        idxs = list(range(10))
        shuffled = dl.shuffle_fn(idxs)
        assert sorted(shuffled) == list(range(10))

    def test_randomize_changes_rng_state(self):
        """randomize should change the RNG state."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, shuffle=True)
        state1 = dl.rng.getstate()
        dl.randomize()
        state2 = dl.rng.getstate()
        assert state1 != state2


# ============================================================
# Tests for DataLoader - chunkify
# ============================================================

class TestDataLoaderChunkify:
    """Tests for DataLoader.chunkify method."""

    def test_chunkify_batched(self):
        """chunkify should split items into chunks of size bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        chunks = list(dl.chunkify(iter(range(7))))
        assert len(chunks) == 3  # 3, 3, 1
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6]

    def test_chunkify_drop_last(self):
        """chunkify with drop_last should drop partial chunks."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        chunks = list(dl.chunkify(iter(range(7))))
        assert len(chunks) == 2  # 3, 3 (drop the last 1)
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]

    def test_chunkify_prebatched_passthrough(self):
        """In prebatched mode, chunkify should pass through items as-is."""
        ds = [[1, 2], [3, 4]]
        dl = DataLoader(ds, bs=None)
        items = [10, 20, 30]
        result = list(dl.chunkify(iter(items)))
        assert result == [10, 20, 30]
