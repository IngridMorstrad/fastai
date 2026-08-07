"""Tests for fastai/data/transforms.py

Covers: get_files, get_image_files, get_text_files, FileGetter, ImageGetter,
ItemGetter, AttrGetter, RandomSplitter, TrainTestSplitter, IndexSplitter,
EndSplitter, GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter,
RandomSubsetSplitter, parent_label, RegexLabeller, CategoryMap, Categorize,
MultiCategorize, OneHotEncode, RegressionSetup, IntToFloatTensor,
broadcast_vec, Normalize.
"""
import sys
import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files, get_image_files, get_text_files, FileGetter, ImageGetter,
    ItemGetter, AttrGetter,
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter,
    RandomSubsetSplitter,
    parent_label, RegexLabeller,
    CategoryMap, Categorize, MultiCategorize, OneHotEncode,
    RegressionSetup, IntToFloatTensor, broadcast_vec, Normalize,
)
from fastcore.foundation import L
from fastai.torch_core import TensorImage, TensorMask


# ============================================================
# Helper: create a temp directory structure for file tests
# ============================================================


@pytest.fixture
def file_tree(tmp_path):
    """Create a directory tree for testing file-getting functions.

    Structure:
        tmp_path/
            a.jpg
            b.png
            c.txt
            d.csv
            .hidden.jpg
            sub/
                e.jpg
                f.txt
            sub2/
                g.png
    """
    (tmp_path / "a.jpg").write_text("img")
    (tmp_path / "b.png").write_text("img")
    (tmp_path / "c.txt").write_text("text")
    (tmp_path / "d.csv").write_text("data")
    (tmp_path / ".hidden.jpg").write_text("hidden")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "e.jpg").write_text("img")
    (sub / "f.txt").write_text("text")
    sub2 = tmp_path / "sub2"
    sub2.mkdir()
    (sub2 / "g.png").write_text("img")
    return tmp_path


# ============================================================
# Tests for get_files
# ============================================================


class TestGetFiles:
    """Tests for get_files utility."""

    def test_get_all_files_recursive(self, file_tree):
        res = get_files(file_tree)
        names = sorted([f.name for f in res])
        # Should not include hidden files (starting with .)
        assert ".hidden.jpg" not in names
        # Should include all non-hidden files recursively
        assert "a.jpg" in names
        assert "e.jpg" in names
        assert "g.png" in names
        assert "c.txt" in names

    def test_get_files_with_extension_filter(self, file_tree):
        res = get_files(file_tree, extensions=['.jpg'])
        names = sorted([f.name for f in res])
        assert "a.jpg" in names
        assert "e.jpg" in names
        assert "b.png" not in names
        assert "c.txt" not in names

    def test_get_files_no_recurse(self, file_tree):
        res = get_files(file_tree, recurse=False)
        names = sorted([f.name for f in res])
        assert "a.jpg" in names
        assert "b.png" in names
        # Files in subdirectories should NOT be present
        assert "e.jpg" not in names
        assert "g.png" not in names

    def test_get_files_specific_folders(self, file_tree):
        res = get_files(file_tree, folders=['sub'])
        names = sorted([f.name for f in res])
        assert "e.jpg" in names
        assert "f.txt" in names
        # Should NOT include files from other folders
        assert "g.png" not in names

    def test_get_files_extension_case_insensitive(self, file_tree):
        # Create a file with uppercase extension
        (file_tree / "upper.JPG").write_text("img")
        res = get_files(file_tree, extensions=['.jpg'])
        names = [f.name for f in res]
        assert "upper.JPG" in names

    def test_get_files_returns_L(self, file_tree):
        res = get_files(file_tree)
        assert isinstance(res, L)

    def test_get_files_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        res = get_files(empty)
        assert len(res) == 0


# ============================================================
# Tests for get_image_files
# ============================================================


class TestGetImageFiles:
    """Tests for get_image_files."""

    def test_returns_only_image_files(self, file_tree):
        res = get_image_files(file_tree)
        names = [f.name for f in res]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "e.jpg" in names
        assert "g.png" in names
        # Non-image files should be excluded
        assert "c.txt" not in names
        assert "d.csv" not in names

    def test_no_recurse(self, file_tree):
        res = get_image_files(file_tree, recurse=False)
        names = [f.name for f in res]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "e.jpg" not in names


# ============================================================
# Tests for get_text_files
# ============================================================


class TestGetTextFiles:
    """Tests for get_text_files."""

    def test_returns_only_txt_files(self, file_tree):
        res = get_text_files(file_tree)
        names = [f.name for f in res]
        assert "c.txt" in names
        assert "f.txt" in names
        assert "a.jpg" not in names
        assert "d.csv" not in names


# ============================================================
# Tests for FileGetter and ImageGetter
# ============================================================


class TestFileGetter:
    """Tests for FileGetter factory."""

    def test_basic_usage(self, file_tree):
        getter = FileGetter(extensions=['.txt'])
        res = getter(file_tree)
        names = [f.name for f in res]
        assert "c.txt" in names
        assert "f.txt" in names

    def test_with_suffix(self, file_tree):
        getter = FileGetter(suf='sub', extensions=['.jpg'])
        res = getter(file_tree)
        names = [f.name for f in res]
        assert "e.jpg" in names
        assert "a.jpg" not in names


class TestImageGetter:
    """Tests for ImageGetter factory."""

    def test_basic_usage(self, file_tree):
        getter = ImageGetter()
        res = getter(file_tree)
        names = [f.name for f in res]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names


# ============================================================
# Tests for ItemGetter and AttrGetter
# ============================================================


class TestItemGetter:
    """Tests for ItemGetter transform."""

    def test_index_into_list(self):
        ig = ItemGetter(1)
        result = ig([10, 20, 30])
        assert result == 20

    def test_index_into_tuple(self):
        ig = ItemGetter(0)
        result = ig((42, 99))
        assert result == 42

    def test_negative_index(self):
        ig = ItemGetter(-1)
        result = ig([1, 2, 3])
        assert result == 3


class TestAttrGetter:
    """Tests for AttrGetter transform."""

    def test_get_existing_attr(self):
        ag = AttrGetter('x')
        obj = SimpleNamespace(x=42, y=99)
        assert ag(obj) == 42

    def test_get_missing_attr_returns_default(self):
        ag = AttrGetter('z', default='fallback')
        obj = SimpleNamespace(x=42)
        assert ag(obj) == 'fallback'

    def test_default_is_none(self):
        ag = AttrGetter('missing')
        obj = SimpleNamespace(x=1)
        assert ag(obj) is None


# ============================================================
# Tests for RandomSplitter
# ============================================================


class TestRandomSplitter:
    """Tests for RandomSplitter."""

    def test_split_sizes(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=7)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_reproducible_with_seed(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=123)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_covers_all_indices(self):
        items = list(range(30))
        splitter = RandomSplitter(valid_pct=0.2, seed=0)
        train, valid = splitter(items)
        combined = sorted(list(train) + list(valid))
        assert combined == list(range(30))

    def test_valid_pct_zero_point_five(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.5, seed=99)
        train, valid = splitter(items)
        assert len(train) == 50
        assert len(valid) == 50


# ============================================================
# Tests for TrainTestSplitter
# ============================================================


class TestTrainTestSplitter:
    """Tests for TrainTestSplitter (sklearn-based)."""

    def test_basic_split(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(50))
        splitter = TrainTestSplitter(test_size=0.3, random_state=7)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_reproducible(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.25, random_state=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)


# ============================================================
# Tests for IndexSplitter
# ============================================================


class TestIndexSplitter:
    """Tests for IndexSplitter."""

    def test_basic_split(self):
        items = list(range(10))
        splitter = IndexSplitter([2, 5, 7])
        train, valid = splitter(items)
        assert sorted(list(valid)) == [2, 5, 7]
        assert sorted(list(train)) == [0, 1, 3, 4, 6, 8, 9]

    def test_empty_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([])
        train, valid = splitter(items)
        assert sorted(list(train)) == [0, 1, 2, 3, 4]
        assert list(valid) == []

    def test_all_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train, valid = splitter(items)
        assert list(train) == []
        assert sorted(list(valid)) == [0, 1, 2, 3, 4]


# ============================================================
# Tests for EndSplitter
# ============================================================


class TestEndSplitter:
    """Tests for EndSplitter."""

    def test_valid_last(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # Last 30% of 10 items = last 3 items
        assert list(valid) == [7, 8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6]

    def test_valid_first(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train, valid = splitter(items)
        # First 30% of 10 items = first 3 items
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)

    def test_no_overlap(self):
        items = list(range(20))
        splitter = EndSplitter(valid_pct=0.25)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0
        assert sorted(list(train) + list(valid)) == list(range(20))


# ============================================================
# Tests for GrandparentSplitter
# ============================================================


class TestGrandparentSplitter:
    """Tests for GrandparentSplitter."""

    def test_basic_split(self, tmp_path):
        # Create structure: train/cls/img1.jpg, valid/cls/img2.jpg
        train_dir = tmp_path / "train" / "cat"
        train_dir.mkdir(parents=True)
        valid_dir = tmp_path / "valid" / "cat"
        valid_dir.mkdir(parents=True)
        (train_dir / "img1.jpg").write_text("")
        (train_dir / "img2.jpg").write_text("")
        (valid_dir / "img3.jpg").write_text("")

        items = [
            train_dir / "img1.jpg",
            train_dir / "img2.jpg",
            valid_dir / "img3.jpg",
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train_idx, valid_idx = splitter(items)
        assert sorted(list(train_idx)) == [0, 1]
        assert list(valid_idx) == [2]

    def test_custom_names(self, tmp_path):
        trn = tmp_path / "trn" / "cls"
        trn.mkdir(parents=True)
        val = tmp_path / "val" / "cls"
        val.mkdir(parents=True)
        (trn / "a.jpg").write_text("")
        (val / "b.jpg").write_text("")

        items = [trn / "a.jpg", val / "b.jpg"]
        splitter = GrandparentSplitter(train_name='trn', valid_name='val')
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0]
        assert list(valid_idx) == [1]


# ============================================================
# Tests for FuncSplitter
# ============================================================


class TestFuncSplitter:
    """Tests for FuncSplitter."""

    def test_basic_func_split(self):
        items = list(range(10))
        # Validation: even numbers
        splitter = FuncSplitter(lambda x: x % 2 == 0)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4, 6, 8]
        assert sorted(list(train)) == [1, 3, 5, 7, 9]


# ============================================================
# Tests for MaskSplitter
# ============================================================


class TestMaskSplitter:
    """Tests for MaskSplitter."""

    def test_basic_mask_split(self):
        items = list(range(5))
        mask = [True, False, True, False, True]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [0, 2, 4]
        assert sorted(list(train)) == [1, 3]


# ============================================================
# Tests for FileSplitter
# ============================================================


class TestFileSplitter:
    """Tests for FileSplitter."""

    def test_basic_file_split(self, tmp_path):
        # Create validation file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("b.jpg\nd.jpg\n")

        # Create items as Path objects with .name property
        items = [
            tmp_path / "a.jpg",
            tmp_path / "b.jpg",
            tmp_path / "c.jpg",
            tmp_path / "d.jpg",
        ]
        splitter = FileSplitter(valid_file)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2]


# ============================================================
# Tests for RandomSubsetSplitter
# ============================================================


class TestRandomSubsetSplitter:
    """Tests for RandomSubsetSplitter."""

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

    def test_invalid_sizes_raise(self):
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.7, valid_sz=0.5)  # sum > 1

    def test_reproducible_with_seed(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=99)
        t1, v1 = splitter(items)
        t2, v2 = splitter(items)
        assert list(t1) == list(t2)
        assert list(v1) == list(v2)


# ============================================================
# Tests for parent_label
# ============================================================


class TestParentLabel:
    """Tests for parent_label."""

    def test_basic(self, tmp_path):
        f = tmp_path / "cats" / "img.jpg"
        f.parent.mkdir(parents=True, exist_ok=True)
        assert parent_label(f) == "cats"

    def test_string_path(self):
        assert parent_label("/data/dogs/img.png") == "dogs"

    def test_nested_path(self):
        assert parent_label("/a/b/c/file.txt") == "c"


# ============================================================
# Tests for RegexLabeller
# ============================================================


class TestRegexLabeller:
    """Tests for RegexLabeller."""

    def test_search_mode(self):
        labeller = RegexLabeller(r'/(\w+)/\w+\.\w+$')
        result = labeller(Path("/data/cats/img001.jpg"))
        assert result == "cats"

    def test_match_mode(self):
        labeller = RegexLabeller(r'(\w+)_\d+', match=True)
        result = labeller("cat_001")
        assert result == "cat"

    def test_no_match_raises(self):
        labeller = RegexLabeller(r'(\d+)')
        with pytest.raises(AssertionError):
            labeller("no_digits_here")


# ============================================================
# Tests for CategoryMap
# ============================================================


class TestCategoryMap:
    """Tests for CategoryMap."""

    def test_basic_creation(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        assert len(cm) == 3
        assert 'cat' in cm.items
        assert 'dog' in cm.items
        assert 'bird' in cm.items

    def test_sorted_by_default(self):
        cm = CategoryMap(['dog', 'cat', 'bird'])
        # Sorted alphabetically
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_o2i_mapping(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        # Sorted: bird=0, cat=1, dog=2
        assert cm.o2i['bird'] == 0
        assert cm.o2i['cat'] == 1
        assert cm.o2i['dog'] == 2

    def test_add_na(self):
        cm = CategoryMap(['a', 'b'], add_na=True)
        assert cm.items[0] == '#na#'
        assert len(cm) == 3

    def test_map_objs(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        ids = cm.map_objs(['dog', 'bird'])
        assert list(ids) == [2, 0]

    def test_map_ids(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        objs = cm.map_ids([0, 2])
        assert list(objs) == ['bird', 'dog']

    def test_equality(self):
        cm1 = CategoryMap(['a', 'b', 'c'])
        cm2 = CategoryMap(['a', 'b', 'c'])
        assert cm1 == cm2

    def test_unsorted(self):
        cm = CategoryMap(['dog', 'cat', 'bird'], sort=False)
        assert list(cm.items) == ['dog', 'cat', 'bird']


# ============================================================
# Tests for Categorize
# ============================================================


class TestCategorize:
    """Tests for Categorize transform."""

    def test_encode_with_vocab(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        result = cat.encodes('dog')
        assert int(result) == cat.vocab.o2i['dog']

    def test_decode_with_vocab(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        idx = cat.vocab.o2i['cat']
        result = cat.decodes(idx)
        assert str(result) == 'cat'

    def test_unknown_label_raises(self):
        cat = Categorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="was not included"):
            cat.encodes('fish')

    def test_setups_builds_vocab(self):
        cat = Categorize()
        data = L(['dog', 'cat', 'dog', 'bird', 'cat'])
        cat.setups(data)
        assert cat.c == 3
        assert 'dog' in cat.vocab.o2i
        assert 'cat' in cat.vocab.o2i
        assert 'bird' in cat.vocab.o2i


# ============================================================
# Tests for MultiCategorize
# ============================================================


class TestMultiCategorize:
    """Tests for MultiCategorize transform."""

    def test_encode(self):
        mc = MultiCategorize(vocab=['a', 'b', 'c', 'd'])
        result = mc.encodes(['b', 'd'])
        expected = [mc.vocab.o2i['b'], mc.vocab.o2i['d']]
        assert list(result.numpy()) == expected

    def test_unknown_label_raises(self):
        mc = MultiCategorize(vocab=['a', 'b', 'c'])
        with pytest.raises(KeyError, match="were not included"):
            mc.encodes(['a', 'x'])

    def test_setups_builds_vocab(self):
        mc = MultiCategorize()
        data = [['a', 'b'], ['b', 'c'], ['a', 'c']]
        mc.setups(data)
        assert len(mc.vocab) == 3


# ============================================================
# Tests for OneHotEncode
# ============================================================


class TestOneHotEncode:
    """Tests for OneHotEncode transform."""

    def test_encode(self):
        ohe = OneHotEncode(c=4)
        # Input is a tensor of category indices
        inp = torch.tensor([0, 2])
        result = ohe.encodes(inp)
        expected = torch.tensor([1., 0., 1., 0.])
        assert torch.allclose(result, expected)

    def test_decode(self):
        ohe = OneHotEncode(c=4)
        inp = torch.tensor([1., 0., 1., 0.])
        result = ohe.decodes(inp)
        assert sorted(result) == [0, 2]


# ============================================================
# Tests for RegressionSetup
# ============================================================


class TestRegressionSetup:
    """Tests for RegressionSetup transform."""

    def test_encodes_to_float_tensor(self):
        rs = RegressionSetup(c=1)
        result = rs.encodes(3.14)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert abs(float(result) - 3.14) < 1e-5

    def test_encodes_list(self):
        rs = RegressionSetup(c=3)
        result = rs.encodes([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        assert result.dtype == torch.float32


# ============================================================
# Tests for IntToFloatTensor
# ============================================================


class TestIntToFloatTensor:
    """Tests for IntToFloatTensor transform."""

    def test_tensor_image_divided_by_255(self):
        t = IntToFloatTensor()
        img = TensorImage(torch.tensor([[[0, 127, 255]]], dtype=torch.uint8))
        result = t.encodes(img)
        assert result.dtype == torch.float32
        assert abs(float(result[0, 0, 0]) - 0.0) < 1e-5
        assert abs(float(result[0, 0, 2]) - 1.0) < 1e-5

    def test_custom_div(self):
        t = IntToFloatTensor(div=128.)
        img = TensorImage(torch.tensor([[[128]]], dtype=torch.uint8))
        result = t.encodes(img)
        assert abs(float(result[0, 0, 0]) - 1.0) < 1e-5

    def test_decode_reverses(self):
        t = IntToFloatTensor()
        img = TensorImage(torch.tensor([[[0.0, 0.5, 1.0]]]))
        decoded = t.decodes(img)
        assert int(decoded[0, 0, 0]) == 0
        assert int(decoded[0, 0, 2]) == 255

    def test_tensor_mask_long(self):
        t = IntToFloatTensor()
        mask = TensorMask(torch.tensor([[1, 2, 3]], dtype=torch.uint8))
        result = t.encodes(mask)
        assert result.dtype == torch.int64


# ============================================================
# Tests for broadcast_vec
# ============================================================


class TestBroadcastVec:
    """Tests for broadcast_vec utility."""

    def test_shape_dim1_ndim4(self):
        mean = [0.485, 0.456, 0.406]
        [result] = broadcast_vec(1, 4, mean, cuda=False)
        assert result.shape == (1, 3, 1, 1)

    def test_shape_dim0_ndim3(self):
        vals = [1.0, 2.0]
        [result] = broadcast_vec(0, 3, vals, cuda=False)
        assert result.shape == (2, 1, 1)

    def test_multiple_tensors(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        m, s = broadcast_vec(1, 4, mean, std, cuda=False)
        assert m.shape == (1, 3, 1, 1)
        assert s.shape == (1, 3, 1, 1)


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
        mean = torch.tensor([0.5]).reshape(1, 1, 1, 1)
        std = torch.tensor([0.25]).reshape(1, 1, 1, 1)
        norm = Normalize(mean=mean, std=std)
        # Create a sample image-like tensor
        x = TensorImage(torch.tensor([[[[0.5, 0.75, 1.0]]]]))
        encoded = norm.encodes(x)
        # (0.5 - 0.5) / 0.25 = 0.0
        assert abs(float(encoded[0, 0, 0, 0]) - 0.0) < 1e-5
        # (0.75 - 0.5) / 0.25 = 1.0
        assert abs(float(encoded[0, 0, 0, 1]) - 1.0) < 1e-5
        # Decode should reverse
        decoded = norm.decodes(encoded)
        assert torch.allclose(decoded, x, atol=1e-5)

    def test_encodes_normalizes(self):
        mean = torch.tensor([2.0]).reshape(1, 1, 1, 1)
        std = torch.tensor([4.0]).reshape(1, 1, 1, 1)
        norm = Normalize(mean=mean, std=std)
        x = TensorImage(torch.tensor([[[[10.0]]]]))
        result = norm.encodes(x)
        # (10 - 2) / 4 = 2.0
        assert abs(float(result[0, 0, 0, 0]) - 2.0) < 1e-5
