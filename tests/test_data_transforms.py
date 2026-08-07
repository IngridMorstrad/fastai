"""Tests for fastai.data.transforms module.

Covers: RandomSplitter, IndexSplitter, EndSplitter, GrandparentSplitter,
FuncSplitter, MaskSplitter, ColSplitter, RandomSubsetSplitter, TrainTestSplitter,
parent_label, RegexLabeller, Categorize, MultiCategorize, OneHotEncode,
RegressionSetup, IntToFloatTensor, Normalize, get_files, get_image_files,
get_text_files, CategoryMap, ItemGetter, AttrGetter.
"""
import sys
import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from fastai.data.transforms import (
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, ColSplitter,
    RandomSubsetSplitter, parent_label, RegexLabeller,
    Categorize, MultiCategorize, OneHotEncode, RegressionSetup,
    IntToFloatTensor, Normalize, get_files, get_image_files, get_text_files,
    CategoryMap, ItemGetter, AttrGetter, broadcast_vec,
)
from fastai.torch_basics import TensorImage, TensorCategory, TensorMultiCategory


# ============================================================
# Tests for RandomSplitter
# ============================================================

class TestRandomSplitter:
    """Tests for RandomSplitter function."""

    def test_returns_two_lists(self):
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100

    def test_valid_pct_respected(self):
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(valid) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        items = list(range(50))
        train, valid = splitter(items)
        train_set = set(train)
        valid_set = set(valid)
        assert len(train_set & valid_set) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=123)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=123)
        train1, valid1 = splitter1(items)
        train2, valid2 = splitter2(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_different_seeds_give_different_splits(self):
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=1)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=2)
        _, valid1 = splitter1(items)
        _, valid2 = splitter2(items)
        assert list(valid1) != list(valid2)

    def test_all_indices_valid(self):
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        items = list(range(50))
        train, valid = splitter(items)
        all_indices = sorted(list(train) + list(valid))
        assert all_indices == list(range(50))

    def test_valid_pct_50(self):
        splitter = RandomSplitter(valid_pct=0.5, seed=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(train) == 50
        assert len(valid) == 50


# ============================================================
# Tests for TrainTestSplitter
# ============================================================

class TestTrainTestSplitter:
    """Tests for TrainTestSplitter using sklearn train_test_split."""

    def test_returns_two_lists(self):
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100

    def test_test_size_respected(self):
        splitter = TrainTestSplitter(test_size=0.3, random_state=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(valid) == 30
        assert len(train) == 70

    def test_reproducibility(self):
        items = list(range(100))
        splitter1 = TrainTestSplitter(test_size=0.2, random_state=42)
        splitter2 = TrainTestSplitter(test_size=0.2, random_state=42)
        train1, valid1 = splitter1(items)
        train2, valid2 = splitter2(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_no_overlap(self):
        splitter = TrainTestSplitter(test_size=0.25, random_state=42)
        items = list(range(80))
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0


# ============================================================
# Tests for IndexSplitter
# ============================================================

class TestIndexSplitter:
    """Tests for IndexSplitter function."""

    def test_specified_indices_in_valid(self):
        splitter = IndexSplitter(valid_idx=[0, 2, 4])
        items = list(range(10))
        train, valid = splitter(items)
        assert list(valid) == [0, 2, 4]

    def test_remaining_in_train(self):
        splitter = IndexSplitter(valid_idx=[0, 2, 4])
        items = list(range(6))
        train, valid = splitter(items)
        assert sorted(list(train)) == [1, 3, 5]

    def test_empty_valid(self):
        splitter = IndexSplitter(valid_idx=[])
        items = list(range(5))
        train, valid = splitter(items)
        assert len(valid) == 0
        assert sorted(list(train)) == list(range(5))

    def test_all_valid(self):
        splitter = IndexSplitter(valid_idx=[0, 1, 2, 3, 4])
        items = list(range(5))
        train, valid = splitter(items)
        assert len(train) == 0
        assert sorted(list(valid)) == list(range(5))

    def test_single_index(self):
        splitter = IndexSplitter(valid_idx=[3])
        items = list(range(5))
        train, valid = splitter(items)
        assert list(valid) == [3]
        assert sorted(list(train)) == [0, 1, 2, 4]


# ============================================================
# Tests for EndSplitter
# ============================================================

class TestEndSplitter:
    """Tests for EndSplitter function."""

    def test_valid_last_default(self):
        splitter = EndSplitter(valid_pct=0.2)
        items = list(range(10))
        train, valid = splitter(items)
        assert list(valid) == [8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_valid_first(self):
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        items = list(range(10))
        train, valid = splitter(items)
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]

    def test_valid_pct_50(self):
        splitter = EndSplitter(valid_pct=0.5)
        items = list(range(10))
        train, valid = splitter(items)
        assert len(train) == 5
        assert len(valid) == 5
        assert list(train) == [0, 1, 2, 3, 4]
        assert list(valid) == [5, 6, 7, 8, 9]

    def test_no_overlap(self):
        splitter = EndSplitter(valid_pct=0.4)
        items = list(range(20))
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0
        assert len(train) + len(valid) == 20

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)


# ============================================================
# Tests for GrandparentSplitter
# ============================================================

class TestGrandparentSplitter:
    """Tests for GrandparentSplitter function."""

    def test_basic_split(self):
        items = [
            Path('/data/train/cats/cat1.jpg'),
            Path('/data/train/dogs/dog1.jpg'),
            Path('/data/valid/cats/cat2.jpg'),
            Path('/data/valid/dogs/dog2.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train, valid = splitter(items)
        assert sorted(train) == [0, 1]
        assert sorted(valid) == [2, 3]

    def test_custom_names(self):
        items = [
            Path('/data/training/cls/img1.jpg'),
            Path('/data/testing/cls/img2.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='training', valid_name='testing')
        train, valid = splitter(items)
        assert list(train) == [0]
        assert list(valid) == [1]

    def test_empty_split(self):
        items = [
            Path('/data/train/cls/img1.jpg'),
            Path('/data/train/cls/img2.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train, valid = splitter(items)
        assert sorted(train) == [0, 1]
        assert list(valid) == []


# ============================================================
# Tests for FuncSplitter
# ============================================================

class TestFuncSplitter:
    """Tests for FuncSplitter function."""

    def test_even_odd_split(self):
        items = list(range(10))
        splitter = FuncSplitter(lambda o: o % 2 == 0)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4, 6, 8]
        assert sorted(list(train)) == [1, 3, 5, 7, 9]

    def test_threshold_split(self):
        items = list(range(10))
        splitter = FuncSplitter(lambda o: o >= 7)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [7, 8, 9]
        assert sorted(list(train)) == [0, 1, 2, 3, 4, 5, 6]

    def test_all_valid(self):
        items = [1, 2, 3]
        splitter = FuncSplitter(lambda o: True)
        train, valid = splitter(items)
        assert len(valid) == 3
        assert len(train) == 0

    def test_none_valid(self):
        items = [1, 2, 3]
        splitter = FuncSplitter(lambda o: False)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 3


# ============================================================
# Tests for MaskSplitter
# ============================================================

class TestMaskSplitter:
    """Tests for MaskSplitter function."""

    def test_basic_mask(self):
        mask = [True, False, True, False, True]
        splitter = MaskSplitter(mask)
        items = list(range(5))
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4]
        assert sorted(list(train)) == [1, 3]

    def test_all_true(self):
        mask = [True, True, True]
        splitter = MaskSplitter(mask)
        items = list(range(3))
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 1, 2]
        assert len(train) == 0

    def test_all_false(self):
        mask = [False, False, False]
        splitter = MaskSplitter(mask)
        items = list(range(3))
        train, valid = splitter(items)
        assert len(valid) == 0
        assert sorted(list(train)) == [0, 1, 2]


# ============================================================
# Tests for ColSplitter
# ============================================================

class TestColSplitter:
    """Tests for ColSplitter with pandas DataFrames."""

    def test_boolean_column(self):
        df = pd.DataFrame({
            'data': [1, 2, 3, 4, 5],
            'is_valid': [False, True, False, True, False]
        })
        splitter = ColSplitter(col='is_valid')
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2, 4]

    def test_integer_column_index(self):
        df = pd.DataFrame({
            'data': [1, 2, 3, 4],
            'split': [False, True, True, False]
        })
        splitter = ColSplitter(col=1)
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 2]
        assert sorted(list(train)) == [0, 3]

    def test_on_parameter_single_value(self):
        df = pd.DataFrame({
            'data': [1, 2, 3, 4],
            'split': ['train', 'valid', 'train', 'valid']
        })
        splitter = ColSplitter(col='split', on='valid')
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2]

    def test_on_parameter_list(self):
        df = pd.DataFrame({
            'data': [1, 2, 3, 4, 5],
            'fold': [0, 1, 2, 1, 0]
        })
        splitter = ColSplitter(col='fold', on=[1, 2])
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 2, 3]
        assert sorted(list(train)) == [0, 4]

    def test_non_dataframe_raises(self):
        splitter = ColSplitter(col='is_valid')
        with pytest.raises(AssertionError):
            splitter([1, 2, 3])


# ============================================================
# Tests for RandomSubsetSplitter
# ============================================================

class TestRandomSubsetSplitter:
    """Tests for RandomSubsetSplitter function."""

    def test_returns_correct_sizes(self):
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(train) == 50
        assert len(valid) == 30

    def test_no_overlap(self):
        splitter = RandomSubsetSplitter(train_sz=0.4, valid_sz=0.4, seed=42)
        items = list(range(100))
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter1 = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=99)
        splitter2 = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=99)
        train1, valid1 = splitter1(items)
        train2, valid2 = splitter2(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_indices_in_range(self):
        splitter = RandomSubsetSplitter(train_sz=0.3, valid_sz=0.2, seed=42)
        items = list(range(50))
        train, valid = splitter(items)
        for idx in train:
            assert 0 <= idx < 50
        for idx in valid:
            assert 0 <= idx < 50


# ============================================================
# Tests for parent_label
# ============================================================

class TestParentLabel:
    """Tests for parent_label function."""

    def test_basic_path(self):
        result = parent_label(Path('/data/cats/image1.jpg'))
        assert result == 'cats'

    def test_string_path(self):
        result = parent_label('/data/dogs/image2.jpg')
        assert result == 'dogs'

    def test_nested_path(self):
        result = parent_label(Path('/root/data/train/cats/img.jpg'))
        assert result == 'cats'

    def test_relative_path(self):
        result = parent_label(Path('data/birds/photo.png'))
        assert result == 'birds'


# ============================================================
# Tests for RegexLabeller
# ============================================================

class TestRegexLabeller:
    """Tests for RegexLabeller class."""

    def test_search_mode(self):
        labeller = RegexLabeller(pat=r'/(\w+)/\w+\.\w+$')
        result = labeller('/data/cats/img001.jpg')
        assert result == 'cats'

    def test_match_mode(self):
        labeller = RegexLabeller(pat=r'(\w+)_\d+\.jpg', match=True)
        result = labeller('cat_001.jpg')
        assert result == 'cat'

    def test_pattern_not_found_raises(self):
        labeller = RegexLabeller(pat=r'(\d+)')
        with pytest.raises(AssertionError):
            labeller('no_numbers_here.txt')

    def test_group_capture(self):
        labeller = RegexLabeller(pat=r'class_(\w+)_\d+')
        result = labeller('class_dog_42.jpg')
        assert result == 'dog'


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for CategoryMap class."""

    def test_basic_creation(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        assert len(cm) == 3
        assert 'cat' in cm.items
        assert 'dog' in cm.items
        assert 'bird' in cm.items

    def test_sorted_by_default(self):
        cm = CategoryMap(['dog', 'cat', 'bird'])
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_unsorted(self):
        cm = CategoryMap(['dog', 'cat', 'bird'], sort=False)
        # Unsorted uses unique order
        assert 'dog' in cm.items
        assert 'cat' in cm.items
        assert 'bird' in cm.items

    def test_o2i_mapping(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        # sorted: bird=0, cat=1, dog=2
        assert cm.o2i['bird'] == 0
        assert cm.o2i['cat'] == 1
        assert cm.o2i['dog'] == 2

    def test_add_na(self):
        cm = CategoryMap(['cat', 'dog'], add_na=True)
        assert '#na#' in cm.items
        assert cm.o2i['#na#'] == 0

    def test_map_objs(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        ids = cm.map_objs(['cat', 'bird'])
        assert list(ids) == [1, 0]

    def test_map_ids(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        objs = cm.map_ids([0, 1, 2])
        assert list(objs) == ['bird', 'cat', 'dog']

    def test_equality(self):
        cm1 = CategoryMap(['cat', 'dog'])
        cm2 = CategoryMap(['cat', 'dog'])
        assert cm1 == cm2


# ============================================================
# Tests for Categorize
# ============================================================

class TestCategorize:
    """Tests for Categorize transform."""

    def test_with_vocab(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        result = cat.encodes('cat')
        assert isinstance(result, TensorCategory)
        # vocab sorted: bird=0, cat=1, dog=2
        assert int(result) == 1

    def test_encode_decode_roundtrip(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        encoded = cat.encodes('dog')
        decoded = cat.decodes(encoded)
        assert str(decoded) == 'dog'

    def test_unknown_label_raises(self):
        cat = Categorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError):
            cat.encodes('fish')

    def test_setups_with_data(self):
        cat = Categorize()
        data = ['cat', 'dog', 'cat', 'bird', 'dog']
        cat.setups(data)
        assert cat.c == 3
        assert cat.vocab is not None

    def test_c_attribute(self):
        cat = Categorize(vocab=['a', 'b', 'c', 'd'])
        cat.setups(None)
        assert cat.c == 4


# ============================================================
# Tests for MultiCategorize
# ============================================================

class TestMultiCategorize:
    """Tests for MultiCategorize transform."""

    def test_encode_multiple_labels(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        result = mc.encodes(['cat', 'dog'])
        assert isinstance(result, TensorMultiCategory)
        assert len(result) == 2

    def test_decode_multiple_labels(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        encoded = mc.encodes(['cat', 'bird'])
        decoded = mc.decodes(encoded)
        assert 'cat' in decoded
        assert 'bird' in decoded

    def test_unknown_label_raises(self):
        mc = MultiCategorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError):
            mc.encodes(['cat', 'fish'])

    def test_setups_with_data(self):
        mc = MultiCategorize()
        data = [['cat', 'dog'], ['bird'], ['cat', 'bird']]
        mc.setups(data)
        assert mc.vocab is not None
        assert len(mc.vocab) == 3

    def test_empty_input(self):
        mc = MultiCategorize(vocab=['cat', 'dog'])
        result = mc.encodes([])
        assert len(result) == 0


# ============================================================
# Tests for OneHotEncode
# ============================================================

class TestOneHotEncode:
    """Tests for OneHotEncode transform."""

    def test_basic_encoding(self):
        ohe = OneHotEncode(c=4)
        inp = TensorMultiCategory([0, 2])
        result = ohe.encodes(inp)
        assert result.shape == (4,)
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 1.0
        assert result[3] == 0.0

    def test_single_category(self):
        ohe = OneHotEncode(c=3)
        inp = TensorMultiCategory([1])
        result = ohe.encodes(inp)
        assert result.shape == (3,)
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 0.0

    def test_all_categories(self):
        ohe = OneHotEncode(c=3)
        inp = TensorMultiCategory([0, 1, 2])
        result = ohe.encodes(inp)
        assert result.sum() == 3.0

    def test_decode(self):
        ohe = OneHotEncode(c=4)
        inp = TensorMultiCategory([0, 2])
        encoded = ohe.encodes(inp)
        decoded = ohe.decodes(encoded)
        assert 0 in decoded
        assert 2 in decoded


# ============================================================
# Tests for RegressionSetup
# ============================================================

class TestRegressionSetup:
    """Tests for RegressionSetup transform."""

    def test_floatifies_scalar(self):
        rs = RegressionSetup()
        result = rs.encodes(3)
        assert result.dtype == torch.float32
        assert float(result) == 3.0

    def test_floatifies_list(self):
        rs = RegressionSetup()
        result = rs.encodes([1.0, 2.0, 3.0])
        assert result.dtype == torch.float32
        assert list(result) == [1.0, 2.0, 3.0]

    def test_setups_scalar(self):
        rs = RegressionSetup()
        data = [1.0, 2.0, 3.0]
        rs.setups(data)
        assert rs.c == 1

    def test_setups_multi_output(self):
        rs = RegressionSetup()
        data = [[1.0, 2.0], [3.0, 4.0]]
        rs.setups(data)
        assert rs.c == 2

    def test_c_override(self):
        rs = RegressionSetup(c=5)
        data = [1.0, 2.0]
        rs.setups(data)
        assert rs.c == 5


# ============================================================
# Tests for IntToFloatTensor
# ============================================================

class TestIntToFloatTensor:
    """Tests for IntToFloatTensor transform."""

    def test_converts_to_float(self):
        t = IntToFloatTensor()
        inp = TensorImage(torch.tensor([[[0, 128, 255]]], dtype=torch.uint8))
        result = t.encodes(inp)
        assert result.dtype == torch.float32

    def test_divides_by_255(self):
        t = IntToFloatTensor(div=255.)
        inp = TensorImage(torch.tensor([[[0, 128, 255]]], dtype=torch.uint8))
        result = t.encodes(inp)
        assert abs(float(result[0, 0, 0]) - 0.0) < 1e-6
        assert abs(float(result[0, 0, 2]) - 1.0) < 1e-6

    def test_custom_div(self):
        t = IntToFloatTensor(div=128.)
        inp = TensorImage(torch.tensor([[[128]]], dtype=torch.uint8))
        result = t.encodes(inp)
        assert abs(float(result[0, 0, 0]) - 1.0) < 1e-6

    def test_decode_roundtrip(self):
        t = IntToFloatTensor(div=255.)
        inp = TensorImage(torch.tensor([[[0, 127, 255]]], dtype=torch.uint8))
        encoded = t.encodes(inp)
        decoded = t.decodes(encoded)
        # Should be close to original (integer rounding)
        assert decoded.dtype == torch.int64


# ============================================================
# Tests for Normalize
# ============================================================

class TestNormalize:
    """Tests for Normalize transform."""

    def test_from_stats(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm = Normalize.from_stats(mean, std, cuda=False)
        assert norm.mean is not None
        assert norm.std is not None

    def test_encode_decode_roundtrip(self):
        mean = [0.5]
        std = [0.5]
        norm = Normalize.from_stats(mean, std, dim=1, ndim=4, cuda=False)
        x = TensorImage(torch.rand(1, 1, 4, 4))
        encoded = norm.encodes(x)
        decoded = norm.decodes(encoded)
        assert torch.allclose(x, decoded, atol=1e-5)

    def test_normalizes_values(self):
        mean = [0.5]
        std = [0.25]
        norm = Normalize.from_stats(mean, std, dim=1, ndim=4, cuda=False)
        # A value of 0.5 should become 0 after normalization
        x = TensorImage(torch.full((1, 1, 2, 2), 0.5))
        result = norm.encodes(x)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_denormalize_on_cpu(self):
        mean = [0.5]
        std = [0.25]
        norm = Normalize.from_stats(mean, std, dim=1, ndim=4, cuda=False)
        x = TensorImage(torch.full((1, 1, 2, 2), 0.75))
        encoded = norm.encodes(x)
        decoded = norm.decodes(encoded)
        assert torch.allclose(decoded, x, atol=1e-5)


# ============================================================
# Tests for broadcast_vec
# ============================================================

class TestBroadcastVec:
    """Tests for broadcast_vec utility function."""

    def test_basic_broadcast(self):
        result = broadcast_vec(1, 4, [0.5, 0.5, 0.5], cuda=False)
        assert result[0].shape == (1, 3, 1, 1)

    def test_dim_0(self):
        result = broadcast_vec(0, 3, [1.0, 2.0], cuda=False)
        assert result[0].shape == (2, 1, 1)

    def test_multiple_tensors(self):
        result = broadcast_vec(1, 4, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], cuda=False)
        assert len(result) == 2
        assert result[0].shape == (1, 3, 1, 1)
        assert result[1].shape == (1, 3, 1, 1)


# ============================================================
# Tests for ItemGetter
# ============================================================

class TestItemGetter:
    """Tests for ItemGetter transform."""

    def test_get_item_by_index(self):
        ig = ItemGetter(1)
        result = ig.encodes(['a', 'b', 'c'])
        assert result == 'b'

    def test_get_first_item(self):
        ig = ItemGetter(0)
        result = ig.encodes(('first', 'second'))
        assert result == 'first'

    def test_get_from_dict(self):
        ig = ItemGetter('key')
        result = ig.encodes({'key': 'value', 'other': 'data'})
        assert result == 'value'


# ============================================================
# Tests for AttrGetter
# ============================================================

class TestAttrGetter:
    """Tests for AttrGetter transform."""

    def test_get_attribute(self):
        class Obj:
            name = 'test'
        ag = AttrGetter('name')
        result = ag.encodes(Obj())
        assert result == 'test'

    def test_default_value(self):
        class Obj:
            pass
        ag = AttrGetter('missing', default='default_val')
        result = ag.encodes(Obj())
        assert result == 'default_val'

    def test_missing_no_default(self):
        class Obj:
            pass
        ag = AttrGetter('missing')
        result = ag.encodes(Obj())
        assert result is None


# ============================================================
# Tests for get_files, get_image_files, get_text_files
# ============================================================

class TestGetFiles:
    """Tests for file getter functions."""

    def test_get_files_basic(self, tmp_path):
        (tmp_path / 'a.txt').write_text('hello')
        (tmp_path / 'b.py').write_text('world')
        result = get_files(tmp_path)
        assert len(result) == 2

    def test_get_files_with_extension(self, tmp_path):
        (tmp_path / 'a.txt').write_text('hello')
        (tmp_path / 'b.py').write_text('world')
        result = get_files(tmp_path, extensions=['.txt'])
        assert len(result) == 1
        assert result[0].name == 'a.txt'

    def test_get_files_recursive(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.txt').write_text('hello')
        (sub / 'b.txt').write_text('world')
        result = get_files(tmp_path, extensions=['.txt'], recurse=True)
        assert len(result) == 2

    def test_get_files_non_recursive(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'a.txt').write_text('hello')
        (sub / 'b.txt').write_text('world')
        result = get_files(tmp_path, extensions=['.txt'], recurse=False)
        assert len(result) == 1

    def test_get_files_folders_filter(self, tmp_path):
        sub1 = tmp_path / 'include'
        sub2 = tmp_path / 'exclude'
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / 'a.txt').write_text('hello')
        (sub2 / 'b.txt').write_text('world')
        result = get_files(tmp_path, extensions=['.txt'], recurse=True, folders=['include'])
        assert len(result) == 1
        assert 'include' in str(result[0])

    def test_get_files_ignores_dotfiles(self, tmp_path):
        (tmp_path / '.hidden').write_text('secret')
        (tmp_path / 'visible.txt').write_text('hello')
        result = get_files(tmp_path)
        names = [r.name for r in result]
        assert '.hidden' not in names
        assert 'visible.txt' in names

    def test_get_image_files(self, tmp_path):
        (tmp_path / 'photo.jpg').write_text('fake image')
        (tmp_path / 'photo.png').write_text('fake image')
        (tmp_path / 'data.txt').write_text('not image')
        result = get_image_files(tmp_path)
        assert len(result) == 2
        names = [r.name for r in result]
        assert 'photo.jpg' in names
        assert 'photo.png' in names

    def test_get_text_files(self, tmp_path):
        (tmp_path / 'doc.txt').write_text('text content')
        (tmp_path / 'image.jpg').write_text('fake image')
        result = get_text_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == 'doc.txt'

    def test_get_files_empty_directory(self, tmp_path):
        result = get_files(tmp_path)
        assert len(result) == 0
