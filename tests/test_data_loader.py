"""Tests for fastai.data.load module.

Covers fa_collate, fa_convert, SkipItemException, collate_error,
and the DataLoader class with various configurations and behaviors.
"""
import sys
import os
import pytest
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.load import fa_collate, fa_convert, SkipItemException, collate_error, DataLoader


def _identity_retain(self, res, b):
    """A simple retain replacement that returns res unchanged.

    The real retain calls retain_types (from fasttransform) which may not be
    available in all environments. This stand-in lets us test iteration logic
    without that dependency.
    """
    return res


# ============================================================
# Tests for `fa_collate`
# ============================================================

class TestFaCollate:
    """Tests for the fa_collate function."""

    def test_collate_tensors(self):
        """Collating a list of tensors should stack them into a batch."""
        items = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = fa_collate(items)
        assert isinstance(result, Tensor)
        assert result.shape == (2, 2)
        assert result[0].tolist() == [1.0, 2.0]
        assert result[1].tolist() == [3.0, 4.0]

    def test_collate_numpy_arrays(self):
        """Collating numpy arrays should produce a tensor."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = fa_collate(items)
        assert isinstance(result, Tensor)
        assert result.shape == (2, 2)

    def test_collate_strings(self):
        """Collating strings should return a list of strings (default_collate behavior)."""
        items = ['hello', 'world']
        result = fa_collate(items)
        assert result == ['hello', 'world']

    def test_collate_mappings(self):
        """Collating mappings (dicts) should collate each key separately."""
        items = [{'a': torch.tensor(1), 'b': torch.tensor(2)},
                 {'a': torch.tensor(3), 'b': torch.tensor(4)}]
        result = fa_collate(items)
        assert isinstance(result, dict)
        assert result['a'].tolist() == [1, 3]
        assert result['b'].tolist() == [2, 4]

    def test_collate_sequences(self):
        """Collating sequences (tuples) should collate element-wise while preserving type."""
        items = [(torch.tensor(1), torch.tensor(2)),
                 (torch.tensor(3), torch.tensor(4))]
        result = fa_collate(items)
        assert isinstance(result, tuple)
        assert result[0].tolist() == [1, 3]
        assert result[1].tolist() == [2, 4]

    def test_collate_sequences_as_lists(self):
        """Collating list sequences should collate element-wise while preserving list type."""
        items = [[torch.tensor(1), torch.tensor(2)],
                 [torch.tensor(3), torch.tensor(4)]]
        result = fa_collate(items)
        assert isinstance(result, list)
        assert result[0].tolist() == [1, 3]
        assert result[1].tolist() == [2, 4]

    def test_collate_scalar_tensors(self):
        """Collating scalar tensors should stack them."""
        items = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = fa_collate(items)
        assert result.shape == (3,)
        assert result.tolist() == [1.0, 2.0, 3.0]

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


# ============================================================
# Tests for `fa_convert`
# ============================================================

class TestFaConvert:
    """Tests for the fa_convert function."""

    def test_convert_tensor(self):
        """Converting a tensor should return the same tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = fa_convert(t)
        assert isinstance(result, Tensor)
        assert torch.equal(result, t)

    def test_convert_numpy_array(self):
        """Converting a numpy array should produce a tensor."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = fa_convert(arr)
        assert isinstance(result, Tensor)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_convert_string(self):
        """Converting a string should return it unchanged."""
        s = 'hello'
        result = fa_convert(s)
        assert result == 'hello'

    def test_convert_mapping(self):
        """Converting a mapping should convert each value."""
        d = {'a': np.array([1.0]), 'b': np.array([2.0])}
        result = fa_convert(d)
        assert isinstance(result, dict)
        assert isinstance(result['a'], Tensor)
        assert isinstance(result['b'], Tensor)

    def test_convert_sequence(self):
        """Converting a sequence should convert each element while preserving type."""
        items = (np.array([1.0]), np.array([2.0]))
        result = fa_convert(items)
        assert isinstance(result, tuple)
        assert isinstance(result[0], Tensor)
        assert isinstance(result[1], Tensor)

    def test_convert_list_sequence(self):
        """Converting a list sequence should convert each element and keep list type."""
        items = [np.array([1.0]), np.array([2.0])]
        result = fa_convert(items)
        assert isinstance(result, list)
        assert isinstance(result[0], Tensor)
        assert isinstance(result[1], Tensor)

    def test_convert_integer(self):
        """fa_convert passes plain integers through default_convert."""
        result = fa_convert(42)
        assert result == 42

    def test_convert_dict(self):
        """fa_convert should handle dicts (Mapping type) via default_convert."""
        d = {"x": np.array([1.0, 2.0])}
        result = fa_convert(d)
        assert "x" in result
        assert torch.equal(result["x"], torch.tensor([1.0, 2.0]))


# ============================================================
# Tests for `SkipItemException`
# ============================================================

class TestSkipItemException:
    """Tests for the SkipItemException class."""

    def test_is_exception_subclass(self):
        """SkipItemException should be a subclass of Exception."""
        assert issubclass(SkipItemException, Exception)

    def test_can_be_raised_and_caught(self):
        """SkipItemException should be raiseable and catchable."""
        with pytest.raises(SkipItemException):
            raise SkipItemException("skip this item")

    def test_instance_check(self):
        """An instance of SkipItemException should be an Exception."""
        exc = SkipItemException()
        assert isinstance(exc, Exception)


# ============================================================
# Tests for `collate_error`
# ============================================================

class TestCollateError:
    """Tests for the collate_error function."""

    def test_raises_on_shape_mismatch(self):
        """Should raise with a descriptive error message when shapes mismatch."""
        # Create a batch with mismatched shapes
        batch = [
            (torch.tensor([1, 2, 3]),),    # shape (3,)
            (torch.tensor([1, 2, 3, 4]),), # shape (4,)
        ]
        e = RuntimeError("original error")
        with pytest.raises(RuntimeError) as exc_info:
            try:
                raise e
            except RuntimeError:
                collate_error(e, batch)
        error_msg = str(exc_info.value)
        assert 'Mismatch found' in error_msg
        assert 'axis 0' in error_msg

    def test_error_message_contains_shapes(self):
        """The error message should include both shapes that differ."""
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
        """Should not raise when all items have the same shape."""
        batch = [
            (torch.zeros(3, 4),),
            (torch.zeros(3, 4),),
        ]
        e = RuntimeError("original error")
        # No exception should propagate since shapes match
        try:
            raise e
        except RuntimeError:
            collate_error(e, batch)


# ============================================================
# Tests for DataLoader construction
# ============================================================

class TestDataLoaderConstruction:
    """Tests for DataLoader initialization with various parameters."""

    def test_basic_construction(self):
        """DataLoader can be constructed with a simple list dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.bs == 2
        assert dl.n == 10
        assert dl.shuffle is False
        assert dl.drop_last is False

    def test_construction_with_shuffle(self):
        """DataLoader can be constructed with shuffle=True for indexed datasets."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=True)
        assert dl.shuffle is True

    def test_construction_with_drop_last(self):
        """DataLoader can be constructed with drop_last=True."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        assert dl.drop_last is True

    def test_construction_indexed_auto_detect(self):
        """DataLoader auto-detects indexed datasets via __getitem__."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.indexed is True

    def test_construction_indexed_explicit(self):
        """DataLoader accepts explicit indexed parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, indexed=False)
        assert dl.indexed is False

    def test_construction_batch_size_alias(self):
        """batch_size parameter should alias to bs for PyTorch compatibility."""
        ds = list(range(10))
        dl = DataLoader(ds, batch_size=4)
        assert dl.bs == 4

    def test_construction_with_device(self):
        """DataLoader can be constructed with a device parameter."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, device='cpu')
        assert dl.device == torch.device('cpu')

    def test_construction_bs_none_is_prebatched(self):
        """When bs is None, the DataLoader is in prebatched mode."""
        ds = [[1, 2, 3], [4, 5, 6]]
        dl = DataLoader(ds, bs=None)
        assert dl.prebatched is True

    def test_construction_explicit_n(self):
        """DataLoader respects explicitly provided n."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=10, n=50)
        assert dl.n == 50

    def test_construction_iterable_not_indexed(self):
        """Iterable datasets should not be indexed."""
        from torch.utils.data import IterableDataset

        class MyIterableDs(IterableDataset):
            def __iter__(self):
                return iter(range(10))

        ds = MyIterableDs()
        dl = DataLoader(ds, bs=2, n=10)
        assert dl.indexed is False

    def test_construction_shuffle_requires_indexed(self):
        """Shuffling a non-indexed dataset should raise ValueError."""
        from torch.utils.data import IterableDataset

        class MyIterableDs(IterableDataset):
            def __iter__(self):
                return iter(range(10))

        ds = MyIterableDs()
        with pytest.raises(ValueError, match="Can only shuffle an indexed dataset"):
            DataLoader(ds, bs=2, n=10, shuffle=True)

    def test_construction_drop_last_requires_bs(self):
        """drop_last=True with bs=None should raise AssertionError."""
        ds = list(range(10))
        with pytest.raises(AssertionError):
            DataLoader(ds, bs=None, drop_last=True)


# ============================================================
# Tests for DataLoader.__len__
# ============================================================

class TestDataLoaderLen:
    """Tests for DataLoader length calculation."""

    def test_len_exact_division(self):
        """Length should be n//bs when n is evenly divisible by bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5)
        assert len(dl) == 2

    def test_len_with_remainder(self):
        """Length should include partial batch when drop_last=False."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        assert len(dl) == 4  # 3+3+3+1

    def test_len_with_drop_last(self):
        """Length should exclude partial batch when drop_last=True."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        assert len(dl) == 3  # 3+3+3, last 1 dropped

    def test_len_bs_none_returns_n(self):
        """When bs is None, length equals number of items."""
        ds = list(range(7))
        dl = DataLoader(ds, bs=None)
        assert len(dl) == 7

    def test_len_raises_when_n_is_none(self):
        """Length should raise TypeError when n cannot be determined."""
        dl = DataLoader(iter(range(10)), bs=2, indexed=False)
        with pytest.raises(TypeError):
            len(dl)

    def test_len_drop_last_exact(self):
        """drop_last with exact division gives same result."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=5, drop_last=True)
        assert len(dl) == 2

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
# Tests for DataLoader.get_idxs
# ============================================================

class TestDataLoaderGetIdxs:
    """Tests for DataLoader.get_idxs method."""

    def test_get_idxs_no_shuffle(self):
        """Without shuffle, indices should be sequential."""
        ds = list(range(5))
        dl = DataLoader(ds, bs=2, shuffle=False)
        idxs = dl.get_idxs()
        assert idxs == [0, 1, 2, 3, 4]

    def test_get_idxs_with_shuffle(self):
        """With shuffle, indices should be a permutation of the same set."""
        ds = list(range(20))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = dl.get_idxs()
        assert sorted(idxs) == list(range(20))
        # Very unlikely (1/20! chance) to be identical to sequential
        # But we just check it's a valid permutation

    def test_get_idxs_length_matches_n(self):
        """get_idxs should return exactly n indices."""
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
# Tests for DataLoader.one_batch
# ============================================================

class TestDataLoaderOneBatch:
    """Tests for DataLoader.one_batch method."""

    def test_one_batch_returns_correct_size(self):
        """one_batch should return a batch of size bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=4)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batch = dl.one_batch()
        assert len(batch) == 4

    def test_one_batch_tensor_output(self):
        """one_batch on numeric data should return a tensor."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batch = dl.one_batch()
        assert isinstance(batch, Tensor)

    def test_one_batch_empty_raises(self):
        """one_batch on empty DataLoader should raise ValueError."""
        ds = []
        dl = DataLoader(ds, bs=2)
        with pytest.raises(ValueError, match="does not contain any batches"):
            dl.one_batch()


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
        """shuffle_fn should produce different orderings on repeated calls (with randomization)."""
        ds = list(range(100))
        dl = DataLoader(ds, bs=2, shuffle=True)
        idxs = list(range(100))
        result1 = dl.shuffle_fn(idxs)
        dl.randomize()
        result2 = dl.shuffle_fn(idxs)
        # With 100 elements, probability of same order is essentially 0
        assert result1 != result2


# ============================================================
# Tests for DataLoader.new
# ============================================================

class TestDataLoaderNew:
    """Tests for DataLoader.new method."""

    def test_new_creates_copy(self):
        """new() should create a new DataLoader with the same dataset."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=False)
        dl2 = dl.new()
        assert dl2.dataset is ds
        assert dl2.bs == 2
        assert dl2.shuffle is False

    def test_new_with_overridden_params(self):
        """new() should allow overriding parameters."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2, shuffle=False)
        dl2 = dl.new(bs=4, shuffle=True)
        assert dl2.bs == 4
        assert dl2.shuffle is True
        assert dl2.dataset is ds

    def test_new_with_different_dataset(self):
        """new() can specify a different dataset."""
        ds1 = list(range(10))
        ds2 = list(range(20))
        dl = DataLoader(ds1, bs=2)
        dl2 = dl.new(dataset=ds2)
        assert dl2.dataset is ds2
        assert dl2.n == 20


# ============================================================
# Tests for DataLoader.do_item with SkipItemException
# ============================================================

class TestDataLoaderDoItem:
    """Tests for DataLoader.do_item with SkipItemException handling."""

    def test_do_item_returns_item(self):
        """do_item should return the dataset item for the given index."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)
        result = dl.do_item(0)
        assert result == 10

    def test_do_item_skip_exception_returns_none(self):
        """do_item should return None when SkipItemException is raised."""
        ds = [10, 20, 30]
        dl = DataLoader(ds, bs=2)

        # Override after_item to raise SkipItemException for item 20
        def skip_after_item(x):
            if x == 20:
                raise SkipItemException()
            return x

        dl.after_item = skip_after_item
        assert dl.do_item(0) == 10
        assert dl.do_item(1) is None
        assert dl.do_item(2) == 30


# ============================================================
# Tests for DataLoader.device property
# ============================================================

class TestDataLoaderDevice:
    """Tests for DataLoader.device property setter."""

    def test_device_setter_string(self):
        """Setting device with string should work."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.device = 'cpu'
        assert dl.device == torch.device('cpu')

    def test_device_setter_torch_device(self):
        """Setting device with a torch.device should work."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        dl.device = torch.device('cpu')
        assert dl.device == torch.device('cpu')

    def test_device_initial_none(self):
        """Device should be None by default."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=2)
        assert dl.device is None

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
# Tests for DataLoader iteration
# ============================================================

class TestDataLoaderIteration:
    """Tests for DataLoader iteration behavior."""

    def test_iteration_produces_correct_num_batches(self):
        """Iterating should produce the correct number of batches."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batches = list(dl)
        assert len(batches) == 4  # 3+3+3+1

    def test_iteration_batch_sizes(self):
        """All batches except possibly the last should have size bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batches = list(dl)
        for b in batches[:-1]:
            assert len(b) == 3
        assert len(batches[-1]) <= 3

    def test_iteration_drop_last(self):
        """With drop_last=True, all batches should have size bs."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=3, drop_last=True)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batches = list(dl)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 3

    def test_iteration_all_items_covered(self):
        """Without shuffle, all items should be present across batches."""
        ds = list(range(10))
        dl = DataLoader(ds, bs=4)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batches = list(dl)
        all_items = []
        for b in batches:
            all_items.extend(b.tolist())
        assert sorted(all_items) == list(range(10))

    def test_iteration_with_tuples(self):
        """DataLoader should handle tuple datasets properly."""
        ds = [(i, i * 2) for i in range(8)]
        dl = DataLoader(ds, bs=4)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batch = dl.one_batch()
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 2  # two elements per tuple
        assert len(batch[0]) == 4  # batch size

    def test_drop_last_drops_incomplete_batch(self):
        """drop_last=True should drop the incomplete final batch."""
        ds = list(range(7))
        dl = DataLoader(ds, bs=3, drop_last=True)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        batches = list(dl)
        # 7 items, bs=3, drop_last -> only 2 full batches (6 items), last 1 dropped
        assert len(batches) == 2
        total_items = sum(len(b) for b in batches)
        assert total_items == 6

    def test_iteration_multiple_epochs(self):
        """DataLoader can be iterated multiple times."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        epoch1 = list(dl)
        epoch2 = list(dl)
        assert len(epoch1) == 2
        assert len(epoch2) == 2

    def test_iteration_shuffle_different_order(self):
        """Shuffled DataLoader produces different orders across epochs."""
        ds = list(range(50))
        dl = DataLoader(ds, bs=50, shuffle=True)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
        epoch1 = list(dl)[0].tolist()
        epoch2 = list(dl)[0].tolist()
        # Both contain same elements
        assert sorted(epoch1) == sorted(epoch2) == list(range(50))
        # But very unlikely to be same order
        assert epoch1 != epoch2


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
        dl = DataLoader(ds, bs=None)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
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
# Tests for DataLoader callbacks (before_iter, after_batch)
# ============================================================

class TestDataLoaderCallbacks:
    """Tests for DataLoader callback hooks."""

    def test_before_iter_called(self):
        """before_iter is called at start of iteration."""
        ds = list(range(4))
        dl = DataLoader(ds, bs=2)
        dl.retain = _identity_retain.__get__(dl, DataLoader)
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
        dl.retain = _identity_retain.__get__(dl, DataLoader)
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
        dl.retain = _identity_retain.__get__(dl, DataLoader)

        def double_batch(b):
            return b * 2

        dl.after_batch = double_batch
        batches = list(dl)
        # First batch should be [0,1]*2 = [0,2]
        assert torch.equal(batches[0], torch.tensor([0, 2]))
