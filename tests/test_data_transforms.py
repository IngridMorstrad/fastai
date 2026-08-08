"""Tests for fastai.data.transforms module.

Covers: RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter, ColSplitter,
RandomSubsetSplitter, parent_label, RegexLabeller, ColReader, CategoryMap,
Categorize, MultiCategorize, OneHotEncode, EncodedMultiCategorize,
RegressionSetup, IntToFloatTensor, broadcast_vec, Normalize,
ItemGetter, AttrGetter, get_files, get_image_files, get_text_files.
"""
import sys
import os
import math
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter,
    ColSplitter, RandomSubsetSplitter,
    parent_label, RegexLabeller, ColReader,
    CategoryMap, Categorize, MultiCategorize, OneHotEncode,
    EncodedMultiCategorize, RegressionSetup,
    IntToFloatTensor, broadcast_vec, Normalize,
    ItemGetter, AttrGetter, get_files, get_image_files, get_text_files,
)
from fastai.torch_basics import TensorImage, TensorCategory, TensorMultiCategory


# ============================================================
# Tests for Splitters
# ============================================================

class TestRandomSplitter:
    """Tests for the RandomSplitter function."""

    def test_basic_split_proportions(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) == 80
        assert len(valid) == 20

    def test_all_indices_covered(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=7)
        train, valid = splitter(items)
        all_idxs = sorted(list(train) + list(valid))
        assert all_idxs == list(range(50))

    def test_no_overlap_between_train_valid(self):
        items = list(range(200))
        splitter = RandomSplitter(valid_pct=0.25, seed=99)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_different_seeds_different_splits(self):
        items = list(range(100))
        s1 = RandomSplitter(valid_pct=0.2, seed=1)
        s2 = RandomSplitter(valid_pct=0.2, seed=2)
        train1, _ = s1(items)
        train2, _ = s2(items)
        # Very unlikely to be the same with different seeds
        assert list(train1) != list(train2)

    def test_valid_pct_zero_point_five(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.5, seed=42)
        train, valid = splitter(items)
        assert len(train) == 50
        assert len(valid) == 50


class TestTrainTestSplitter:
    """Tests for the TrainTestSplitter function."""

    def test_basic_split(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train, valid = splitter(items)
        assert len(train) == 80
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(50))
        splitter = TrainTestSplitter(test_size=0.3, random_state=7)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_all_indices_covered(self):
        items = list(range(50))
        splitter = TrainTestSplitter(test_size=0.3, random_state=7)
        train, valid = splitter(items)
        all_idxs = sorted(list(train) + list(valid))
        assert all_idxs == list(range(50))

    def test_reproducibility(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)


class TestIndexSplitter:
    """Tests for the IndexSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        splitter = IndexSplitter([7, 8, 9])
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 1, 2, 3, 4, 5, 6]
        assert sorted(list(valid)) == [7, 8, 9]

    def test_empty_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([])
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 1, 2, 3, 4]
        assert list(valid) == []

    def test_single_valid_item(self):
        items = list(range(10))
        splitter = IndexSplitter([5])
        train, valid = splitter(items)
        assert 5 not in list(train)
        assert list(valid) == [5]

    def test_non_contiguous_valid(self):
        items = list(range(10))
        splitter = IndexSplitter([1, 3, 5, 7])
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 2, 4, 6, 8, 9]
        assert sorted(list(valid)) == [1, 3, 5, 7]


class TestEndSplitter:
    """Tests for the EndSplitter function."""

    def test_valid_last(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.2, valid_last=True)
        train, valid = splitter(items)
        assert list(train) == list(range(8))
        assert list(valid) == [8, 9]

    def test_valid_first(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.2, valid_last=False)
        train, valid = splitter(items)
        assert list(valid) == [0, 1]
        assert list(train) == list(range(2, 10))

    def test_half_split(self):
        items = list(range(20))
        splitter = EndSplitter(valid_pct=0.5, valid_last=True)
        train, valid = splitter(items)
        assert len(train) == 10
        assert len(valid) == 10
        assert list(train) == list(range(10))
        assert list(valid) == list(range(10, 20))

    def test_preserves_order(self):
        items = list(range(100))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # Verify ordering is preserved
        assert list(train) == sorted(list(train))
        assert list(valid) == sorted(list(valid))

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)


class TestGrandparentSplitter:
    """Tests for the GrandparentSplitter function."""

    def test_basic_split(self):
        items = [
            Path('/data/train/cat/img1.jpg'),
            Path('/data/train/dog/img2.jpg'),
            Path('/data/valid/cat/img3.jpg'),
            Path('/data/valid/dog/img4.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 1]
        assert sorted(list(valid)) == [2, 3]

    def test_custom_names(self):
        items = [
            Path('/data/trn/cls/a.jpg'),
            Path('/data/tst/cls/b.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='trn', valid_name='tst')
        train, valid = splitter(items)
        assert list(train) == [0]
        assert list(valid) == [1]

    def test_mixed_items(self):
        items = [
            Path('/root/train/a/1.jpg'),
            Path('/root/valid/b/2.jpg'),
            Path('/root/train/c/3.jpg'),
            Path('/root/other/d/4.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 2]
        assert list(valid) == [1]


class TestFuncSplitter:
    """Tests for the FuncSplitter function."""

    def test_basic_split_by_function(self):
        items = list(range(10))
        # Put even numbers in validation
        splitter = FuncSplitter(lambda x: x % 2 == 0)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4, 6, 8]
        assert sorted(list(train)) == [1, 3, 5, 7, 9]

    def test_all_valid(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: True)
        train, valid = splitter(items)
        assert len(train) == 0
        assert len(valid) == 5

    def test_all_train(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: False)
        train, valid = splitter(items)
        assert len(train) == 5
        assert len(valid) == 0


class TestMaskSplitter:
    """Tests for the MaskSplitter function."""

    def test_basic_mask(self):
        mask = [True, False, True, False, True]
        items = list(range(5))
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4]
        assert sorted(list(train)) == [1, 3]

    def test_all_true_mask(self):
        mask = [True, True, True]
        items = list(range(3))
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 3
        assert len(train) == 0

    def test_all_false_mask(self):
        mask = [False, False, False]
        items = list(range(3))
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 3


class TestFileSplitter:
    """Tests for the FileSplitter function."""

    def test_basic_file_split(self, tmp_path):
        # Create a file with validation filenames
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("img3.jpg\nimg5.jpg\n")

        items = [
            Path('/data/img1.jpg'),
            Path('/data/img2.jpg'),
            Path('/data/img3.jpg'),
            Path('/data/img4.jpg'),
            Path('/data/img5.jpg'),
        ]
        splitter = FileSplitter(str(valid_file))
        train, valid = splitter(items)
        assert sorted(list(valid)) == [2, 4]
        assert sorted(list(train)) == [0, 1, 3]


class TestColSplitter:
    """Tests for the ColSplitter function."""

    def test_basic_bool_column(self):
        import pandas as pd
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'is_valid': [False, True, False, True, False]
        })
        splitter = ColSplitter('is_valid')
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2, 4]

    def test_column_by_index(self):
        import pandas as pd
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'split': [False, False, True]
        })
        splitter = ColSplitter(1)  # second column
        train, valid = splitter(df)
        assert list(valid) == [2]
        assert sorted(list(train)) == [0, 1]

    def test_on_value(self):
        import pandas as pd
        df = pd.DataFrame({
            'x': [10, 20, 30, 40],
            'fold': ['train', 'valid', 'train', 'valid']
        })
        splitter = ColSplitter('fold', on='valid')
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2]

    def test_on_list_of_values(self):
        import pandas as pd
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'fold': [0, 1, 2, 1, 0]
        })
        splitter = ColSplitter('fold', on=[1, 2])
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 2, 3]
        assert sorted(list(train)) == [0, 4]

    def test_non_dataframe_raises(self):
        splitter = ColSplitter('is_valid')
        with pytest.raises(AssertionError):
            splitter([1, 2, 3])


class TestRandomSubsetSplitter:
    """Tests for the RandomSubsetSplitter function."""

    def test_basic_subset(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) == 60
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=7)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.7, valid_sz=0.2, seed=42)
        t1, v1 = splitter(items)
        t2, v2 = splitter(items)
        assert list(t1) == list(t2)
        assert list(v1) == list(v2)

    def test_invalid_sizes_raises(self):
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.0, valid_sz=0.2)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.6, valid_sz=0.0)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.8, valid_sz=0.3)


# ============================================================
# Tests for Labelers
# ============================================================

class TestParentLabel:
    """Tests for the parent_label function."""

    def test_basic_label(self):
        assert parent_label(Path('/data/cats/img1.jpg')) == 'cats'

    def test_nested_path(self):
        assert parent_label(Path('/a/b/c/dogs/photo.png')) == 'dogs'

    def test_string_input(self):
        assert parent_label('/root/class_a/file.txt') == 'class_a'


class TestRegexLabeller:
    """Tests for the RegexLabeller class."""

    def test_search_mode(self):
        labeller = RegexLabeller(r'/(\w+)/\w+\.\w+$')
        result = labeller(Path('/data/cats/img1.jpg'))
        assert result == 'cats'

    def test_match_mode(self):
        labeller = RegexLabeller(r'(\w+)_\d+\.jpg', match=True)
        result = labeller('cat_001.jpg')
        assert result == 'cat'

    def test_pattern_not_found_raises(self):
        labeller = RegexLabeller(r'(\d+)_class')
        with pytest.raises(AssertionError):
            labeller('no_match_here.jpg')

    def test_extract_number(self):
        labeller = RegexLabeller(r'img_(\d+)')
        result = labeller('img_042_train.jpg')
        assert result == '042'


class TestColReader:
    """Tests for the ColReader class."""

    def test_single_column_read(self):
        Row = namedtuple('Row', ['name', 'label'])
        row = Row(name='img.jpg', label='cat')
        reader = ColReader('label')
        assert reader(row) == 'cat'

    def test_prefix_suffix(self):
        Row = namedtuple('Row', ['fname', 'label'])
        row = Row(fname='img.jpg', label='cat')
        reader = ColReader('fname', pref='data/', suff='.bak')
        assert reader(row) == 'data/img.jpg.bak'

    def test_label_delim(self):
        Row = namedtuple('Row', ['tags'])
        row = Row(tags='cat dog bird')
        reader = ColReader('tags', label_delim=' ')
        result = reader(row)
        assert list(result) == ['cat', 'dog', 'bird']

    def test_label_delim_empty_string(self):
        Row = namedtuple('Row', ['tags'])
        row = Row(tags='')
        reader = ColReader('tags', label_delim=' ')
        result = reader(row)
        assert list(result) == []

    def test_multiple_columns(self):
        Row = namedtuple('Row', ['a', 'b', 'c'])
        row = Row(a='x', b='y', c='z')
        reader = ColReader(['a', 'c'])
        result = reader(row)
        assert list(result) == ['x', 'z']

    def test_integer_column_index(self):
        import pandas as pd
        row = pd.Series({'col0': 'hello', 'col1': 'world'})
        reader = ColReader(0)
        assert reader(row) == 'hello'


# ============================================================
# Tests for CategoryMap and Categorize
# ============================================================

class TestCategoryMap:
    """Tests for the CategoryMap class."""

    def test_basic_creation(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_o2i_mapping(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        assert cm.o2i['bird'] == 0
        assert cm.o2i['cat'] == 1
        assert cm.o2i['dog'] == 2

    def test_unsorted(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=False)
        # Should preserve first-seen order (unique)
        assert 'cat' in list(cm.items)
        assert 'dog' in list(cm.items)
        assert 'bird' in list(cm.items)

    def test_add_na(self):
        cm = CategoryMap(['cat', 'dog'], sort=True, add_na=True)
        assert cm.items[0] == '#na#'
        assert cm.o2i['#na#'] == 0

    def test_map_objs(self):
        cm = CategoryMap(['a', 'b', 'c'], sort=True)
        ids = cm.map_objs(['b', 'a', 'c'])
        assert list(ids) == [1, 0, 2]

    def test_map_ids(self):
        cm = CategoryMap(['a', 'b', 'c'], sort=True)
        objs = cm.map_ids([2, 0, 1])
        assert list(objs) == ['c', 'a', 'b']

    def test_duplicates_handled(self):
        cm = CategoryMap(['cat', 'cat', 'dog', 'dog', 'bird'], sort=True)
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_equality(self):
        cm1 = CategoryMap(['a', 'b', 'c'], sort=True)
        cm2 = CategoryMap(['a', 'b', 'c'], sort=True)
        assert cm1 == cm2


class TestCategorize:
    """Tests for the Categorize transform."""

    def test_encode_with_vocab(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'], sort=True)
        result = cat.encodes('cat')
        assert isinstance(result, TensorCategory)
        assert int(result) == cat.vocab.o2i['cat']

    def test_decode(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'], sort=True)
        encoded = cat.encodes('dog')
        decoded = cat.decodes(encoded)
        assert str(decoded) == 'dog'

    def test_roundtrip(self):
        cat = Categorize(vocab=['a', 'b', 'c'], sort=True)
        for label in ['a', 'b', 'c']:
            encoded = cat.encodes(label)
            decoded = cat.decodes(encoded)
            assert str(decoded) == label

    def test_unknown_label_raises(self):
        cat = Categorize(vocab=['cat', 'dog'], sort=True)
        with pytest.raises(KeyError, match="not included in the training dataset"):
            cat.encodes('bird')

    def test_c_attribute(self):
        cat = Categorize(vocab=['a', 'b', 'c', 'd'], sort=True)
        cat.setups(None)
        assert cat.c == 4


class TestMultiCategorize:
    """Tests for the MultiCategorize transform."""

    def test_encode(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        result = mc.encodes(['cat', 'bird'])
        assert isinstance(result, TensorMultiCategory)
        expected = [mc.vocab.o2i['cat'], mc.vocab.o2i['bird']]
        assert list(result.numpy()) == expected

    def test_decode(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        encoded = mc.encodes(['dog', 'cat'])
        decoded = mc.decodes(encoded)
        assert 'dog' in decoded
        assert 'cat' in decoded

    def test_unknown_labels_raises(self):
        mc = MultiCategorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="not included in the training dataset"):
            mc.encodes(['cat', 'fish'])


class TestOneHotEncode:
    """Tests for the OneHotEncode transform."""

    def test_basic_encoding(self):
        ohe = OneHotEncode(c=4)
        # Input is a tensor of category indices
        inp = TensorMultiCategory([0, 2])
        result = ohe.encodes(inp)
        expected = torch.tensor([1., 0., 1., 0.])
        assert torch.allclose(result, expected)

    def test_decode(self):
        ohe = OneHotEncode(c=3)
        encoded = torch.tensor([1., 0., 1.])
        decoded = ohe.decodes(encoded)
        assert 0 in decoded
        assert 2 in decoded
        assert 1 not in decoded

    def test_all_zeros(self):
        ohe = OneHotEncode(c=5)
        inp = TensorMultiCategory([])
        result = ohe.encodes(inp)
        assert torch.all(result == 0.)


class TestEncodedMultiCategorize:
    """Tests for the EncodedMultiCategorize transform."""

    def test_encode(self):
        emc = EncodedMultiCategorize(vocab=['a', 'b', 'c'])
        inp = [1, 0, 1]  # a=True, b=False, c=True
        result = emc.encodes(inp)
        assert isinstance(result, TensorMultiCategory)
        assert list(result.numpy()) == [1., 0., 1.]

    def test_c_attribute(self):
        emc = EncodedMultiCategorize(vocab=['a', 'b', 'c', 'd'])
        assert emc.c == 4


class TestRegressionSetup:
    """Tests for the RegressionSetup transform."""

    def test_encode_scalar(self):
        rs = RegressionSetup()
        result = rs.encodes(3.14)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert abs(float(result) - 3.14) < 1e-5

    def test_encode_list(self):
        rs = RegressionSetup()
        result = rs.encodes([1.0, 2.0, 3.0])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert list(result.numpy()) == [1.0, 2.0, 3.0]

    def test_setups_infers_c(self):
        rs = RegressionSetup()
        # dsets[0] returns a list of length 3
        dsets = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        rs.setups(dsets)
        assert rs.c == 3

    def test_setups_scalar_c(self):
        rs = RegressionSetup()
        dsets = [1.0, 2.0]
        rs.setups(dsets)
        assert rs.c == 1


# ============================================================
# Tests for Tensor Transforms
# ============================================================

class TestIntToFloatTensor:
    """Tests for the IntToFloatTensor transform."""

    def test_encode_tensor_image(self):
        t = IntToFloatTensor(div=255.)
        img = TensorImage(torch.randint(0, 256, (3, 4, 4), dtype=torch.uint8))
        result = t.encodes(img)
        assert result.dtype == torch.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_decode_tensor_image(self):
        t = IntToFloatTensor(div=255.)
        # Simulate a float image in [0, 1]
        img = TensorImage(torch.rand(3, 4, 4))
        decoded = t.decodes(img)
        assert decoded.dtype == torch.int64
        assert decoded.max() <= 255
        assert decoded.min() >= 0

    def test_custom_div(self):
        t = IntToFloatTensor(div=128.)
        img = TensorImage(torch.tensor([[[128]]], dtype=torch.uint8))
        result = t.encodes(img)
        assert abs(float(result) - 1.0) < 1e-5


class TestBroadcastVec:
    """Tests for the broadcast_vec function."""

    def test_basic_broadcast(self):
        result = broadcast_vec(1, 4, [0.5, 0.5, 0.5], cuda=False)
        assert len(result) == 1
        assert result[0].shape == (1, 3, 1, 1)

    def test_multiple_tensors(self):
        mean, std = broadcast_vec(1, 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], cuda=False)
        assert mean.shape == (1, 3, 1, 1)
        assert std.shape == (1, 3, 1, 1)

    def test_dim_0(self):
        result = broadcast_vec(0, 3, [1.0, 2.0], cuda=False)
        assert result[0].shape == (2, 1, 1)


class TestNormalize:
    """Tests for the Normalize transform."""

    def test_encode_decode_roundtrip(self):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)
        norm = Normalize(mean=mean, std=std)
        img = TensorImage(torch.rand(1, 3, 4, 4))
        encoded = norm.encodes(img)
        decoded = norm.decodes(encoded)
        assert torch.allclose(img, decoded, atol=1e-5)

    def test_encode_normalizes(self):
        mean = torch.tensor([0.5]).view(1, 1, 1, 1)
        std = torch.tensor([0.5]).view(1, 1, 1, 1)
        norm = Normalize(mean=mean, std=std)
        img = TensorImage(torch.ones(1, 1, 2, 2))
        result = norm.encodes(img)
        # (1 - 0.5) / 0.5 = 1.0
        assert torch.allclose(result, torch.ones(1, 1, 2, 2))

    def test_from_stats(self):
        norm = Normalize.from_stats(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            cuda=False
        )
        assert norm.mean.shape == (1, 3, 1, 1)
        assert norm.std.shape == (1, 3, 1, 1)


# ============================================================
# Tests for ItemGetter and AttrGetter
# ============================================================

class TestItemGetter:
    """Tests for the ItemGetter transform."""

    def test_get_item_by_index(self):
        getter = ItemGetter(1)
        result = getter.encodes(['a', 'b', 'c'])
        assert result == 'b'

    def test_get_first_item(self):
        getter = ItemGetter(0)
        result = getter.encodes(('x', 'y'))
        assert result == 'x'

    def test_get_dict_key(self):
        getter = ItemGetter('name')
        result = getter.encodes({'name': 'Alice', 'age': 30})
        assert result == 'Alice'


class TestAttrGetter:
    """Tests for the AttrGetter transform."""

    def test_get_attribute(self):
        getter = AttrGetter('real')
        result = getter.encodes(complex(3, 4))
        assert result == 3.0

    def test_default_value(self):
        getter = AttrGetter('nonexistent', default='fallback')

        class Obj:
            pass

        result = getter.encodes(Obj())
        assert result == 'fallback'

    def test_existing_attribute(self):
        getter = AttrGetter('name')

        class Obj:
            name = 'test'

        result = getter.encodes(Obj())
        assert result == 'test'


# ============================================================
# Tests for get_files functions
# ============================================================

class TestGetFiles:
    """Tests for get_files, get_image_files, get_text_files functions."""

    def test_get_files_no_extension_filter(self, tmp_path):
        (tmp_path / 'a.txt').write_text('hello')
        (tmp_path / 'b.py').write_text('world')
        (tmp_path / 'c.jpg').write_bytes(b'\x00')
        result = get_files(tmp_path, recurse=False)
        assert len(result) == 3

    def test_get_files_with_extension(self, tmp_path):
        (tmp_path / 'a.txt').write_text('hello')
        (tmp_path / 'b.py').write_text('world')
        (tmp_path / 'c.txt').write_text('!')
        result = get_files(tmp_path, extensions=['.txt'], recurse=False)
        names = [p.name for p in result]
        assert 'a.txt' in names
        assert 'c.txt' in names
        assert 'b.py' not in names

    def test_get_files_recursive(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.txt').write_text('hi')
        (sub / 'b.txt').write_text('there')
        result = get_files(tmp_path, extensions=['.txt'], recurse=True)
        assert len(result) == 2

    def test_get_files_no_recurse(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.txt').write_text('hi')
        (sub / 'b.txt').write_text('there')
        result = get_files(tmp_path, extensions=['.txt'], recurse=False)
        assert len(result) == 1

    def test_get_files_hidden_skipped(self, tmp_path):
        (tmp_path / '.hidden').write_text('secret')
        (tmp_path / 'visible.txt').write_text('hello')
        result = get_files(tmp_path, recurse=False)
        names = [p.name for p in result]
        assert '.hidden' not in names
        assert 'visible.txt' in names

    def test_get_text_files(self, tmp_path):
        (tmp_path / 'a.txt').write_text('hello')
        (tmp_path / 'b.jpg').write_bytes(b'\x00')
        result = get_text_files(tmp_path, recurse=False)
        assert len(result) == 1
        assert result[0].name == 'a.txt'

    def test_get_image_files(self, tmp_path):
        (tmp_path / 'a.jpg').write_bytes(b'\x00')
        (tmp_path / 'b.png').write_bytes(b'\x00')
        (tmp_path / 'c.txt').write_text('hello')
        result = get_image_files(tmp_path, recurse=False)
        names = [p.name for p in result]
        assert 'a.jpg' in names
        assert 'b.png' in names
        assert 'c.txt' not in names

    def test_get_files_folders_filter(self, tmp_path):
        sub1 = tmp_path / 'include'
        sub2 = tmp_path / 'exclude'
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / 'a.txt').write_text('in')
        (sub2 / 'b.txt').write_text('out')
        result = get_files(tmp_path, extensions=['.txt'], recurse=True, folders=['include'])
        names = [p.name for p in result]
        assert 'a.txt' in names
        assert 'b.txt' not in names
