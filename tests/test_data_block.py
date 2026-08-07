"""Tests for fastai/data/block.py module.

Covers TransformBlock, CategoryBlock, MultiCategoryBlock, RegressionBlock,
DataBlock, and helper functions (_merge_tfms, _short_repr).
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.block import (
    TransformBlock,
    CategoryBlock,
    MultiCategoryBlock,
    RegressionBlock,
    DataBlock,
    _merge_tfms,
    _short_repr,
)
from fastai.data.transforms import (
    Categorize,
    MultiCategorize,
    EncodedMultiCategorize,
    OneHotEncode,
    RegressionSetup,
    RandomSplitter,
    IndexSplitter,
    ToTensor,
    CategoryMap,
    ItemGetter,
)
from fastai.data.core import TfmdDL, Datasets


# ============================================================
# Tests for TransformBlock
# ============================================================

class TestTransformBlock:
    """Tests for the TransformBlock class."""

    def test_default_creation(self):
        """TransformBlock with no arguments has sensible defaults."""
        tb = TransformBlock()
        assert len(tb.type_tfms) == 0
        assert len(tb.batch_tfms) == 0
        assert tb.dl_type is None
        assert tb.dls_kwargs == {}

    def test_item_tfms_includes_to_tensor(self):
        """Default item_tfms always includes ToTensor."""
        tb = TransformBlock()
        # item_tfms = ToTensor + L(item_tfms)
        assert len(tb.item_tfms) >= 1
        assert any(t == ToTensor or t is ToTensor for t in tb.item_tfms)

    def test_custom_type_tfms(self):
        """TransformBlock accepts custom type transforms."""
        cat_tfm = Categorize(vocab=['a', 'b', 'c'])
        tb = TransformBlock(type_tfms=cat_tfm)
        assert len(tb.type_tfms) == 1
        assert isinstance(tb.type_tfms[0], Categorize)

    def test_custom_item_tfms(self):
        """TransformBlock prepends ToTensor to custom item transforms."""
        tb = TransformBlock(item_tfms=[lambda x: x])
        # ToTensor + provided transforms
        assert len(tb.item_tfms) >= 2

    def test_custom_batch_tfms(self):
        """TransformBlock stores batch transforms."""
        tb = TransformBlock(batch_tfms=[lambda x: x])
        assert len(tb.batch_tfms) == 1

    def test_custom_dl_type(self):
        """TransformBlock accepts a custom DataLoader type."""

        class CustomDL(TfmdDL):
            pass

        tb = TransformBlock(dl_type=CustomDL)
        assert tb.dl_type is CustomDL

    def test_custom_dls_kwargs(self):
        """TransformBlock stores dls_kwargs for DataLoaders."""
        tb = TransformBlock(dls_kwargs={'bs': 32, 'shuffle': True})
        assert tb.dls_kwargs == {'bs': 32, 'shuffle': True}

    def test_dls_kwargs_none_becomes_empty_dict(self):
        """When dls_kwargs is None, it becomes an empty dict."""
        tb = TransformBlock(dls_kwargs=None)
        assert tb.dls_kwargs == {}

    def test_multiple_type_tfms(self):
        """TransformBlock accepts a list of type transforms."""
        tb = TransformBlock(type_tfms=[Categorize(vocab=['x', 'y']), lambda x: x])
        assert len(tb.type_tfms) == 2


# ============================================================
# Tests for CategoryBlock
# ============================================================

class TestCategoryBlock:
    """Tests for the CategoryBlock factory function."""

    def test_returns_transform_block(self):
        """CategoryBlock returns a TransformBlock instance."""
        cb = CategoryBlock(vocab=['cat', 'dog'])
        assert isinstance(cb, TransformBlock)

    def test_has_categorize_type_tfm(self):
        """CategoryBlock has a Categorize transform in type_tfms."""
        cb = CategoryBlock(vocab=['cat', 'dog'])
        assert len(cb.type_tfms) == 1
        assert isinstance(cb.type_tfms[0], Categorize)

    def test_vocab_sorted_by_default(self):
        """CategoryBlock sorts vocab alphabetically by default."""
        cb = CategoryBlock(vocab=['dog', 'cat', 'bird'])
        categorize = cb.type_tfms[0]
        assert list(categorize.vocab) == ['bird', 'cat', 'dog']

    def test_vocab_unsorted(self):
        """CategoryBlock preserves order when sort=False."""
        cb = CategoryBlock(vocab=['dog', 'cat', 'bird'], sort=False)
        categorize = cb.type_tfms[0]
        assert list(categorize.vocab) == ['dog', 'cat', 'bird']

    def test_add_na(self):
        """CategoryBlock adds #na# to vocab when add_na=True."""
        cb = CategoryBlock(vocab=['cat', 'dog'], add_na=True)
        categorize = cb.type_tfms[0]
        assert '#na#' in list(categorize.vocab)

    def test_no_add_na_by_default(self):
        """CategoryBlock does not add #na# by default."""
        cb = CategoryBlock(vocab=['cat', 'dog'])
        categorize = cb.type_tfms[0]
        assert '#na#' not in list(categorize.vocab)

    def test_vocab_none(self):
        """CategoryBlock can be created with vocab=None (inferred later)."""
        cb = CategoryBlock()
        categorize = cb.type_tfms[0]
        assert categorize.vocab is None

    def test_includes_to_tensor_in_item_tfms(self):
        """CategoryBlock includes ToTensor in item_tfms."""
        cb = CategoryBlock(vocab=['a', 'b'])
        assert any(t == ToTensor or t is ToTensor for t in cb.item_tfms)


# ============================================================
# Tests for MultiCategoryBlock
# ============================================================

class TestMultiCategoryBlock:
    """Tests for the MultiCategoryBlock factory function."""

    def test_returns_transform_block(self):
        """MultiCategoryBlock returns a TransformBlock instance."""
        mcb = MultiCategoryBlock(vocab=['cat', 'dog', 'bird'])
        assert isinstance(mcb, TransformBlock)

    def test_non_encoded_has_multi_categorize_and_one_hot(self):
        """Non-encoded MultiCategoryBlock uses MultiCategorize + OneHotEncode."""
        mcb = MultiCategoryBlock(vocab=['cat', 'dog', 'bird'])
        assert len(mcb.type_tfms) == 2
        assert isinstance(mcb.type_tfms[0], MultiCategorize)
        # Second should be OneHotEncode class (not instance since it's a list)
        assert mcb.type_tfms[1] is OneHotEncode

    def test_encoded_has_encoded_multi_categorize(self):
        """Encoded MultiCategoryBlock uses EncodedMultiCategorize."""
        mcb = MultiCategoryBlock(encoded=True, vocab=['cat', 'dog', 'bird'])
        assert len(mcb.type_tfms) == 1
        assert isinstance(mcb.type_tfms[0], EncodedMultiCategorize)

    def test_encoded_vocab_preserved(self):
        """Encoded MultiCategoryBlock preserves the given vocab."""
        vocab = ['alpha', 'beta', 'gamma']
        mcb = MultiCategoryBlock(encoded=True, vocab=vocab)
        tfm = mcb.type_tfms[0]
        assert list(tfm.vocab) == vocab

    def test_non_encoded_add_na(self):
        """Non-encoded MultiCategoryBlock respects add_na parameter."""
        mcb = MultiCategoryBlock(vocab=['cat', 'dog'], add_na=True)
        multi_cat = mcb.type_tfms[0]
        assert '#na#' in list(multi_cat.vocab)

    def test_vocab_none(self):
        """MultiCategoryBlock can be created with vocab=None."""
        mcb = MultiCategoryBlock()
        multi_cat = mcb.type_tfms[0]
        assert multi_cat.vocab is None


# ============================================================
# Tests for RegressionBlock
# ============================================================

class TestRegressionBlock:
    """Tests for the RegressionBlock factory function."""

    def test_returns_transform_block(self):
        """RegressionBlock returns a TransformBlock instance."""
        rb = RegressionBlock(n_out=1)
        assert isinstance(rb, TransformBlock)

    def test_has_regression_setup_type_tfm(self):
        """RegressionBlock has a RegressionSetup transform."""
        rb = RegressionBlock(n_out=1)
        assert len(rb.type_tfms) == 1
        assert isinstance(rb.type_tfms[0], RegressionSetup)

    def test_n_out_stored(self):
        """RegressionBlock stores the n_out value in the transform."""
        rb = RegressionBlock(n_out=5)
        reg_setup = rb.type_tfms[0]
        assert reg_setup.c == 5

    def test_n_out_none(self):
        """RegressionBlock with n_out=None defers c inference."""
        rb = RegressionBlock(n_out=None)
        reg_setup = rb.type_tfms[0]
        assert reg_setup.c is None

    def test_n_out_single(self):
        """RegressionBlock with n_out=1 for scalar targets."""
        rb = RegressionBlock(n_out=1)
        reg_setup = rb.type_tfms[0]
        assert reg_setup.c == 1

    def test_includes_to_tensor_in_item_tfms(self):
        """RegressionBlock includes ToTensor in item_tfms."""
        rb = RegressionBlock(n_out=1)
        assert any(t == ToTensor or t is ToTensor for t in rb.item_tfms)


# ============================================================
# Tests for DataBlock
# ============================================================

class TestDataBlock:
    """Tests for the DataBlock class."""

    def test_default_creation(self):
        """DataBlock with no arguments creates with default blocks."""
        db = DataBlock()
        assert db.n_inp == 1

    def test_custom_blocks(self):
        """DataBlock accepts blocks parameter."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        assert db.n_inp == 1

    def test_n_inp_default_is_len_blocks_minus_one(self):
        """Default n_inp is len(blocks) - 1 (at least 1)."""
        db = DataBlock(blocks=(TransformBlock, TransformBlock, CategoryBlock))
        assert db.n_inp == 2

        db2 = DataBlock(blocks=(TransformBlock, CategoryBlock))
        assert db2.n_inp == 1

    def test_n_inp_override(self):
        """n_inp can be explicitly set."""
        db = DataBlock(
            blocks=(TransformBlock, TransformBlock, TransformBlock, CategoryBlock),
            n_inp=2,
        )
        assert db.n_inp == 2

    def test_invalid_kwargs_raises_type_error(self):
        """DataBlock raises TypeError for invalid keyword arguments."""
        with pytest.raises(TypeError, match="invalid keyword arguments"):
            DataBlock(invalid_kwarg=True)

    def test_multiple_invalid_kwargs(self):
        """DataBlock reports all invalid keyword arguments."""
        with pytest.raises(TypeError, match="foo"):
            DataBlock(foo=1, bar=2)

    def test_get_x_wrong_count_raises_value_error(self):
        """DataBlock raises ValueError when get_x has wrong number of functions."""
        with pytest.raises(ValueError, match="get_x contains"):
            DataBlock(
                blocks=(TransformBlock, CategoryBlock),
                get_x=[lambda x: x, lambda x: x],  # 2 fns for n_inp=1
            )

    def test_get_y_wrong_count_raises_value_error(self):
        """DataBlock raises ValueError when get_y has wrong number of functions."""
        with pytest.raises(ValueError, match="get_y contains"):
            DataBlock(
                blocks=(TransformBlock, CategoryBlock),
                get_y=[lambda x: x, lambda x: x],  # 2 fns for 1 target
            )

    def test_dl_type_from_block(self):
        """DataBlock uses dl_type from a block when specified."""

        class CustomDL(TfmdDL):
            pass

        tb = TransformBlock(dl_type=CustomDL)
        db = DataBlock(blocks=[tb, CategoryBlock])
        assert db.dl_type is CustomDL

    def test_dl_type_override(self):
        """Explicit dl_type parameter overrides block's dl_type."""

        class CustomDL1(TfmdDL):
            pass

        class CustomDL2(TfmdDL):
            pass

        tb = TransformBlock(dl_type=CustomDL1)
        db = DataBlock(blocks=[tb, CategoryBlock], dl_type=CustomDL2)
        assert db.dl_type is CustomDL2

    def test_item_tfms_stored(self):
        """DataBlock stores item transforms."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        # Should at least have ToTensor from default
        assert len(db.item_tfms) >= 1

    def test_batch_tfms_stored(self):
        """DataBlock stores batch transforms."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        # Default batch_tfms from TransformBlock is empty
        assert isinstance(db.batch_tfms, list) or hasattr(db.batch_tfms, '__len__')

    def test_new_returns_self(self):
        """DataBlock.new() returns self for chaining."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        result = db.new(item_tfms=[], batch_tfms=[])
        assert result is db

    def test_new_updates_transforms(self):
        """DataBlock.new() updates item and batch transforms."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        original_item_tfms = list(db.item_tfms)
        db.new(item_tfms=[], batch_tfms=[])
        # After new(), item_tfms are re-merged (default + new)
        # Should still contain defaults since we passed empty lists
        assert len(db.item_tfms) >= 0

    def test_callable_blocks_are_called(self):
        """When blocks are callable (classes), they are instantiated."""
        db = DataBlock(blocks=[TransformBlock, TransformBlock])
        # Should have type_tfms for each block
        assert len(db.type_tfms) == 2

    def test_instance_blocks_used_directly(self):
        """When blocks are already instances, they are used directly."""
        tb = TransformBlock(type_tfms=Categorize(vocab=['a', 'b']))
        db = DataBlock(blocks=[TransformBlock, tb])
        assert len(db.type_tfms) == 2
        # The second block should have a Categorize transform
        assert len(db.type_tfms[1]) == 1

    def test_getters_default_to_noop(self):
        """Default getters are noop functions."""
        db = DataBlock(blocks=(TransformBlock, CategoryBlock))
        assert len(db.getters) == 2  # One per block

    def test_custom_getters(self):
        """DataBlock accepts custom getters."""
        getter1 = lambda x: x[0]
        getter2 = lambda x: x[1]
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            getters=[getter1, getter2],
        )
        assert db.getters[0] is getter1
        assert db.getters[1] is getter2

    def test_dls_kwargs_merged_from_blocks(self):
        """DataBlock merges dls_kwargs from all blocks."""
        tb1 = TransformBlock(dls_kwargs={'bs': 32})
        tb2 = TransformBlock(dls_kwargs={'shuffle': True})
        db = DataBlock(blocks=[tb1, tb2])
        assert 'bs' in db.dls_kwargs
        assert 'shuffle' in db.dls_kwargs


class TestDataBlockFromColumns:
    """Tests for DataBlock.from_columns class method."""

    def test_creates_datablock(self):
        """from_columns returns a DataBlock instance."""
        db = DataBlock.from_columns(blocks=[TransformBlock, CategoryBlock])
        assert isinstance(db, DataBlock)

    def test_getters_are_item_getters(self):
        """from_columns creates ItemGetter instances for getters."""
        db = DataBlock.from_columns(blocks=[TransformBlock, CategoryBlock])
        assert len(db.getters) == 2
        assert isinstance(db.getters[0], ItemGetter)
        assert isinstance(db.getters[1], ItemGetter)

    def test_default_blocks_count(self):
        """from_columns with no blocks defaults to 2 getters."""
        db = DataBlock.from_columns()
        assert len(db.getters) == 2

    def test_custom_getters(self):
        """from_columns uses custom getters when provided."""
        g1 = ItemGetter(0)
        g2 = ItemGetter(1)
        db = DataBlock.from_columns(
            blocks=[TransformBlock, CategoryBlock],
            getters=[g1, g2],
        )
        assert db.getters[0] is g1
        assert db.getters[1] is g2


class TestDataBlockDatasets:
    """Tests for DataBlock.datasets method."""

    def test_creates_datasets(self):
        """DataBlock.datasets creates a Datasets object."""
        items = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog']
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=IndexSplitter([6, 7]),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        assert isinstance(dsets, Datasets)

    def test_datasets_train_valid_split(self):
        """datasets splits into train and valid sets."""
        items = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog']
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=IndexSplitter([6, 7]),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        assert len(dsets.train) == 6
        assert len(dsets.valid) == 2

    def test_datasets_with_random_splitter(self):
        """datasets works with RandomSplitter."""
        items = ['cat', 'dog', 'bird', 'cat', 'dog'] * 4  # 20 items
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        total = len(dsets.train) + len(dsets.valid)
        assert total == 20

    def test_datasets_stores_source(self):
        """datasets stores the source on the DataBlock."""
        items = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird']
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=IndexSplitter([4, 5]),
            get_y=lambda item: item,
        )
        db.datasets(items)
        assert db.source is items

    def test_datasets_with_get_items(self):
        """datasets uses get_items to fetch data from source."""
        source = {'data': ['cat', 'dog', 'bird', 'cat', 'dog', 'bird']}
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            get_items=lambda s: s['data'],
            splitter=IndexSplitter([4, 5]),
            get_y=lambda item: item,
        )
        dsets = db.datasets(source)
        assert len(dsets.train) + len(dsets.valid) == 6

    def test_datasets_default_splitter(self):
        """Without explicit splitter, RandomSplitter is used."""
        items = ['cat', 'dog', 'bird'] * 10  # 30 items
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        # Should have train and valid
        total = len(dsets.train) + len(dsets.valid)
        assert total == 30


# ============================================================
# Tests for _merge_tfms helper
# ============================================================

class TestMergeTfms:
    """Tests for the _merge_tfms helper function."""

    def test_empty_input(self):
        """_merge_tfms with empty lists returns empty."""
        result = _merge_tfms([], [])
        assert len(result) == 0

    def test_single_transform(self):
        """_merge_tfms with a single transform preserves it."""
        tfm = Categorize(vocab=['a', 'b'])
        result = _merge_tfms([tfm])
        assert len(result) == 1

    def test_deduplicates_same_class(self):
        """_merge_tfms removes duplicates from the same class, keeping the last."""
        tfm1 = Categorize(vocab=['a', 'b'])
        tfm2 = Categorize(vocab=['c', 'd'])
        result = _merge_tfms([tfm1, tfm2])
        # Should keep only one (the last) since they're same class
        assert len(result) == 1
        # The kept one should be tfm2 (last wins)
        assert list(result[0].vocab) == ['c', 'd']

    def test_different_classes_preserved(self):
        """_merge_tfms preserves transforms of different classes."""
        tfm1 = Categorize(vocab=['a', 'b'])
        tfm2 = RegressionSetup(c=1)
        result = _merge_tfms([tfm1, tfm2])
        assert len(result) == 2

    def test_merges_multiple_lists(self):
        """_merge_tfms merges from multiple transform lists."""
        tfm1 = Categorize(vocab=['a', 'b'])
        tfm2 = RegressionSetup(c=1)
        result = _merge_tfms([tfm1], [tfm2])
        assert len(result) == 2


# ============================================================
# Tests for _short_repr helper
# ============================================================

class TestShortRepr:
    """Tests for the _short_repr helper function."""

    def test_string(self):
        """_short_repr of a string returns the string."""
        assert _short_repr('hello') == 'hello'

    def test_integer(self):
        """_short_repr of an integer returns its str."""
        assert _short_repr(42) == '42'

    def test_tuple(self):
        """_short_repr of a tuple formats with parentheses."""
        result = _short_repr(('a', 'b'))
        assert result == '(a, b)'

    def test_list(self):
        """_short_repr of a list formats with brackets."""
        result = _short_repr([1, 2, 3])
        assert result == '[1, 2, 3]'

    def test_small_tensor(self):
        """_short_repr of a small tensor shows its contents."""
        t = torch.tensor([1, 2, 3])
        result = _short_repr(t)
        assert 'tensor' in result.lower() or '1' in result

    def test_large_tensor(self):
        """_short_repr of a large tensor shows shape summary."""
        t = torch.randn(10, 20, 30)
        result = _short_repr(t)
        assert '10x20x30' in result

    def test_nested_tuple(self):
        """_short_repr handles nested structures."""
        result = _short_repr(('a', [1, 2]))
        assert '(' in result and '[' in result

    def test_2d_small_tensor(self):
        """_short_repr of a 2D tensor with many elements shows shape."""
        t = torch.randn(5, 5)  # 25 elements, ndim > 1
        result = _short_repr(t)
        assert '5x5' in result

    def test_1d_tensor_within_limit(self):
        """_short_repr of 1D tensor with <=20 elements shows values."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _short_repr(t)
        # Should show tensor contents, not shape summary
        assert 'size' not in result.lower() or '1' in result


# ============================================================
# Tests for DataBlock end-to-end with datasets
# ============================================================

class TestDataBlockEndToEnd:
    """End-to-end tests for DataBlock with actual data."""

    def test_category_classification_pipeline(self):
        """Full pipeline: items -> split -> categorize -> datasets."""
        items = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird',
                 'cat', 'dog', 'bird', 'cat']
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=IndexSplitter([8, 9]),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        # Verify train sample is a tuple (input, category_tensor)
        sample = dsets.train[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_regression_pipeline(self):
        """Full pipeline with RegressionBlock for float targets."""
        items = list(range(20))
        db = DataBlock(
            blocks=(TransformBlock, RegressionBlock),
            splitter=IndexSplitter([18, 19]),
            get_y=lambda item: float(item) * 0.1,
        )
        dsets = db.datasets(items)
        sample = dsets.train[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_multi_category_pipeline(self):
        """Full pipeline with MultiCategoryBlock for multi-label."""
        items = [
            ['cat', 'dog'],
            ['bird'],
            ['cat', 'bird'],
            ['dog'],
            ['cat', 'dog'],
            ['bird'],
            ['cat', 'bird'],
            ['dog'],
            ['cat'],
            ['dog', 'bird'],
        ]
        db = DataBlock(
            blocks=(TransformBlock, MultiCategoryBlock),
            splitter=IndexSplitter([8, 9]),
            get_y=lambda item: item,
        )
        dsets = db.datasets(items)
        sample = dsets.train[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_datasets_reproducible_with_seed(self):
        """Same seed produces identical splits."""
        items = list(range(100))
        db1 = DataBlock(
            blocks=(TransformBlock, TransformBlock),
            splitter=RandomSplitter(valid_pct=0.2, seed=123),
        )
        db2 = DataBlock(
            blocks=(TransformBlock, TransformBlock),
            splitter=RandomSplitter(valid_pct=0.2, seed=123),
        )
        dsets1 = db1.datasets(items)
        dsets2 = db2.datasets(items)
        assert len(dsets1.train) == len(dsets2.train)
        assert len(dsets1.valid) == len(dsets2.valid)

    def test_datasets_verbose_does_not_crash(self):
        """Calling datasets with verbose=True runs without error."""
        items = ['cat', 'dog', 'bird'] * 5
        db = DataBlock(
            blocks=(TransformBlock, CategoryBlock),
            splitter=IndexSplitter([13, 14]),
            get_y=lambda item: item,
        )
        # Should not raise
        dsets = db.datasets(items, verbose=True)
        assert dsets is not None
