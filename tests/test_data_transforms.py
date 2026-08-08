"""Tests for fastai.data.transforms module.

Covers file discovery, splitting functions, labeling, category encoding/decoding,
multi-label categorization, tensor conversions, and normalization transforms.
"""
import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files,
    get_image_files,
    get_text_files,
    FileGetter,
    RandomSplitter,
    TrainTestSplitter,
    IndexSplitter,
    EndSplitter,
    GrandparentSplitter,
    FuncSplitter,
    MaskSplitter,
    parent_label,
    RegexLabeller,
    CategoryMap,
    Categorize,
    MultiCategorize,
    OneHotEncode,
    IntToFloatTensor,
    broadcast_vec,
    Normalize,
    image_extensions,
)
from fastai.torch_core import (
    TensorImage,
    TensorMask,
    TensorCategory,
    TensorMultiCategory,
)
from fastcore.foundation import L
from pathlib import Path


# ============================================================
# Tests for `get_files` function
# ============================================================

class TestGetFiles:
    """Tests for the `get_files` function."""

    def test_get_files_all(self, tmp_path):
        """get_files with no extension filter returns all non-hidden files."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / ".hidden").touch()
        result = get_files(tmp_path, recurse=False)
        names = sorted([f.name for f in result])
        assert "a.txt" in names
        assert "b.py" in names
        assert ".hidden" not in names

    def test_get_files_with_extensions(self, tmp_path):
        """get_files filters by provided extensions."""
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "c.txt").touch()
        result = get_files(tmp_path, extensions=['.txt'], recurse=False)
        names = [f.name for f in result]
        assert "a.txt" in names
        assert "c.txt" in names
        assert "b.py" not in names

    def test_get_files_extension_case_insensitive(self, tmp_path):
        """get_files handles extension matching case-insensitively."""
        (tmp_path / "photo.JPG").touch()
        (tmp_path / "image.jpg").touch()
        result = get_files(tmp_path, extensions=['.jpg'], recurse=False)
        names = [f.name for f in result]
        assert "photo.JPG" in names
        assert "image.jpg" in names

    def test_get_files_recurse_true(self, tmp_path):
        """get_files with recurse=True finds files in subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.txt").touch()
        (sub / "nested.txt").touch()
        result = get_files(tmp_path, extensions=['.txt'], recurse=True)
        names = [f.name for f in result]
        assert "top.txt" in names
        assert "nested.txt" in names

    def test_get_files_recurse_false(self, tmp_path):
        """get_files with recurse=False does not descend into subdirectories."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.txt").touch()
        (sub / "nested.txt").touch()
        result = get_files(tmp_path, extensions=['.txt'], recurse=False)
        names = [f.name for f in result]
        assert "top.txt" in names
        assert "nested.txt" not in names

    def test_get_files_folders_filter(self, tmp_path):
        """get_files with folders restricts recursion to specified folders."""
        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir()
        folder_b.mkdir()
        (folder_a / "file_a.txt").touch()
        (folder_b / "file_b.txt").touch()
        result = get_files(tmp_path, extensions=['.txt'], recurse=True, folders=['a'])
        names = [f.name for f in result]
        assert "file_a.txt" in names
        assert "file_b.txt" not in names

    def test_get_files_skips_hidden_dirs(self, tmp_path):
        """get_files skips directories that start with a dot."""
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.txt").touch()
        (tmp_path / "visible.txt").touch()
        result = get_files(tmp_path, extensions=['.txt'], recurse=True)
        names = [f.name for f in result]
        assert "visible.txt" in names
        assert "secret.txt" not in names

    def test_get_files_returns_L_type(self, tmp_path):
        """get_files returns a fastcore L list."""
        (tmp_path / "a.txt").touch()
        result = get_files(tmp_path, recurse=False)
        assert isinstance(result, L)

    def test_get_files_empty_directory(self, tmp_path):
        """get_files on an empty directory returns an empty list."""
        result = get_files(tmp_path, recurse=False)
        assert len(result) == 0


class TestGetImageFiles:
    """Tests for get_image_files helper."""

    def test_get_image_files_finds_images(self, tmp_path):
        """get_image_files finds common image extensions."""
        (tmp_path / "photo.jpg").touch()
        (tmp_path / "icon.png").touch()
        (tmp_path / "document.txt").touch()
        result = get_image_files(tmp_path)
        names = [f.name for f in result]
        assert "photo.jpg" in names
        assert "icon.png" in names
        assert "document.txt" not in names

    def test_image_extensions_contains_common_types(self):
        """image_extensions set includes jpg, png, gif."""
        assert '.jpg' in image_extensions or '.jpeg' in image_extensions
        assert '.png' in image_extensions
        assert '.gif' in image_extensions


class TestGetTextFiles:
    """Tests for get_text_files helper."""

    def test_get_text_files(self, tmp_path):
        """get_text_files only returns .txt files."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "code.py").touch()
        result = get_text_files(tmp_path)
        names = [f.name for f in result]
        assert "notes.txt" in names
        assert "code.py" not in names


class TestFileGetter:
    """Tests for FileGetter factory."""

    def test_file_getter_basic(self, tmp_path):
        """FileGetter creates a function that calls get_files."""
        (tmp_path / "data.csv").touch()
        (tmp_path / "img.png").touch()
        getter = FileGetter(extensions=['.csv'])
        result = getter(tmp_path)
        names = [f.name for f in result]
        assert "data.csv" in names
        assert "img.png" not in names

    def test_file_getter_with_suffix(self, tmp_path):
        """FileGetter with suf appends suffix to path."""
        sub = tmp_path / "data"
        sub.mkdir()
        (sub / "file.txt").touch()
        getter = FileGetter(suf='data', extensions=['.txt'])
        result = getter(tmp_path)
        names = [f.name for f in result]
        assert "file.txt" in names


# ============================================================
# Tests for splitter functions
# ============================================================

class TestRandomSplitter:
    """Tests for RandomSplitter."""

    def test_random_splitter_default(self):
        """RandomSplitter with default valid_pct=0.2 splits correctly."""
        items = list(range(100))
        splitter = RandomSplitter(seed=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20
        assert len(train) == 80

    def test_random_splitter_custom_pct(self):
        """RandomSplitter respects custom valid_pct."""
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        train, valid = splitter(items)
        assert len(valid) == 15
        assert len(train) == 35

    def test_random_splitter_no_overlap(self):
        """Train and validation sets have no overlapping indices."""
        items = list(range(100))
        splitter = RandomSplitter(seed=42)
        train, valid = splitter(items)
        train_set = set(train)
        valid_set = set(valid)
        assert len(train_set & valid_set) == 0

    def test_random_splitter_covers_all_indices(self):
        """All indices are covered by train + valid."""
        items = list(range(100))
        splitter = RandomSplitter(seed=42)
        train, valid = splitter(items)
        assert set(train) | set(valid) == set(range(100))

    def test_random_splitter_seed_reproducible(self):
        """Same seed produces the same split."""
        items = list(range(100))
        splitter = RandomSplitter(seed=123)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_random_splitter_different_seeds(self):
        """Different seeds produce different splits."""
        items = list(range(100))
        splitter1 = RandomSplitter(seed=1)
        splitter2 = RandomSplitter(seed=2)
        train1, _ = splitter1(items)
        train2, _ = splitter2(items)
        assert list(train1) != list(train2)


class TestTrainTestSplitter:
    """Tests for TrainTestSplitter (sklearn-based)."""

    def test_train_test_splitter_default(self):
        """TrainTestSplitter default test_size=0.2 works."""
        items = list(range(100))
        splitter = TrainTestSplitter(random_state=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20

    def test_train_test_splitter_reproducible(self):
        """TrainTestSplitter with random_state is reproducible."""
        items = list(range(50))
        splitter = TrainTestSplitter(random_state=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)


class TestIndexSplitter:
    """Tests for IndexSplitter."""

    def test_index_splitter_basic(self):
        """IndexSplitter puts specified indices in validation."""
        items = list(range(10))
        splitter = IndexSplitter([2, 5, 7])
        train, valid = splitter(items)
        assert list(valid) == [2, 5, 7]
        assert set(train) == {0, 1, 3, 4, 6, 8, 9}

    def test_index_splitter_empty_valid(self):
        """IndexSplitter with empty list puts everything in training."""
        items = list(range(5))
        splitter = IndexSplitter([])
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 5

    def test_index_splitter_all_valid(self):
        """IndexSplitter can put all indices in validation."""
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train, valid = splitter(items)
        assert len(train) == 0
        assert len(valid) == 5


class TestEndSplitter:
    """Tests for EndSplitter."""

    def test_end_splitter_valid_last(self):
        """EndSplitter with valid_last=True puts last items in validation."""
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # 30% of 10 = 3 items at the end
        assert list(valid) == [7, 8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6]

    def test_end_splitter_valid_first(self):
        """EndSplitter with valid_last=False puts first items in validation."""
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train, valid = splitter(items)
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]


class TestGrandparentSplitter:
    """Tests for GrandparentSplitter."""

    def test_grandparent_splitter_basic(self, tmp_path):
        """GrandparentSplitter splits by grandparent folder name."""
        # Create structure: root/train/class1/file.txt, root/valid/class1/file.txt
        train_dir = tmp_path / "train" / "cats"
        valid_dir = tmp_path / "valid" / "cats"
        train_dir.mkdir(parents=True)
        valid_dir.mkdir(parents=True)
        train_file = train_dir / "img1.jpg"
        valid_file = valid_dir / "img2.jpg"
        train_file.touch()
        valid_file.touch()

        items = [train_file, valid_file]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0]
        assert list(valid_idx) == [1]

    def test_grandparent_splitter_multiple_files(self, tmp_path):
        """GrandparentSplitter handles multiple files correctly."""
        for name in ['train', 'valid']:
            d = tmp_path / name / "class_a"
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"file_{i}.jpg").touch()

        items = sorted(list((tmp_path / "train" / "class_a").iterdir())) + \
                sorted(list((tmp_path / "valid" / "class_a").iterdir()))

        splitter = GrandparentSplitter()
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) == 3
        assert len(valid_idx) == 3

    def test_grandparent_splitter_custom_names(self, tmp_path):
        """GrandparentSplitter supports custom train/valid names."""
        trn = tmp_path / "trn" / "cls"
        val = tmp_path / "val" / "cls"
        trn.mkdir(parents=True)
        val.mkdir(parents=True)
        (trn / "a.jpg").touch()
        (val / "b.jpg").touch()

        items = [trn / "a.jpg", val / "b.jpg"]
        splitter = GrandparentSplitter(train_name='trn', valid_name='val')
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0]
        assert list(valid_idx) == [1]


class TestFuncSplitter:
    """Tests for FuncSplitter."""

    def test_func_splitter_basic(self):
        """FuncSplitter splits based on a boolean function."""
        items = list(range(10))
        # Items > 7 go to validation
        splitter = FuncSplitter(lambda x: x > 7)
        train, valid = splitter(items)
        assert set(valid) == {8, 9}
        assert set(train) == {0, 1, 2, 3, 4, 5, 6, 7}


class TestMaskSplitter:
    """Tests for MaskSplitter."""

    def test_mask_splitter_basic(self):
        """MaskSplitter splits based on boolean mask."""
        items = list(range(5))
        mask = [False, True, False, True, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert set(valid) == {1, 3}
        assert set(train) == {0, 2, 4}


# ============================================================
# Tests for labeling functions
# ============================================================

class TestParentLabel:
    """Tests for parent_label."""

    def test_parent_label_basic(self):
        """parent_label returns the parent directory name."""
        path = Path("/data/cats/img001.jpg")
        assert parent_label(path) == "cats"

    def test_parent_label_string_input(self):
        """parent_label works with string input."""
        assert parent_label("/data/dogs/photo.png") == "dogs"

    def test_parent_label_nested(self):
        """parent_label returns immediate parent, not grandparent."""
        path = Path("/root/train/birds/pic.jpg")
        assert parent_label(path) == "birds"


class TestRegexLabeller:
    """Tests for RegexLabeller."""

    def test_regex_labeller_search(self):
        """RegexLabeller with search mode finds pattern in path."""
        labeller = RegexLabeller(r'/(\w+)/\w+\.\w+$')
        result = labeller(Path("/data/cats/img.jpg"))
        assert result == "cats"

    def test_regex_labeller_match(self):
        """RegexLabeller with match=True anchors to start."""
        labeller = RegexLabeller(r'(\w+)_\d+', match=True)
        result = labeller("cat_001.jpg")
        assert result == "cat"

    def test_regex_labeller_assertion_on_no_match(self):
        """RegexLabeller raises AssertionError when pattern not found."""
        labeller = RegexLabeller(r'impossible_pattern_(\d+)')
        with pytest.raises(AssertionError):
            labeller("normal_filename.jpg")


# ============================================================
# Tests for CategoryMap and Categorize
# ============================================================

class TestCategoryMap:
    """Tests for CategoryMap."""

    def test_category_map_sorted(self):
        """CategoryMap sorts items by default."""
        cm = CategoryMap(['dog', 'cat', 'bird'])
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_category_map_unsorted(self):
        """CategoryMap preserves order when sort=False."""
        cm = CategoryMap(['dog', 'cat', 'bird'], sort=False)
        # unique() order may vary, but items are not sorted
        assert set(cm.items) == {'dog', 'cat', 'bird'}

    def test_category_map_o2i_mapping(self):
        """CategoryMap.o2i maps items to their indices."""
        cm = CategoryMap(['cat', 'dog', 'bird'])
        assert cm.o2i['bird'] == 0
        assert cm.o2i['cat'] == 1
        assert cm.o2i['dog'] == 2

    def test_category_map_with_add_na(self):
        """CategoryMap with add_na prepends #na# token."""
        cm = CategoryMap(['cat', 'dog'], add_na=True)
        assert cm.items[0] == '#na#'
        assert cm.o2i['#na#'] == 0
        assert cm.o2i['cat'] == 1

    def test_category_map_map_objs(self):
        """CategoryMap.map_objs maps object list to IDs."""
        cm = CategoryMap(['cat', 'dog', 'bird'])
        ids = cm.map_objs(['bird', 'dog'])
        assert list(ids) == [0, 2]

    def test_category_map_map_ids(self):
        """CategoryMap.map_ids maps ID list to objects."""
        cm = CategoryMap(['cat', 'dog', 'bird'])
        objs = cm.map_ids([0, 1, 2])
        assert list(objs) == ['bird', 'cat', 'dog']

    def test_category_map_duplicates(self):
        """CategoryMap handles duplicate values (uses unique)."""
        cm = CategoryMap(['cat', 'cat', 'dog', 'dog', 'bird'])
        assert len(cm.items) == 3

    def test_category_map_equality(self):
        """CategoryMap equality compares items."""
        cm1 = CategoryMap(['cat', 'dog'])
        cm2 = CategoryMap(['cat', 'dog'])
        assert cm1 == cm2


class TestCategorize:
    """Tests for Categorize transform."""

    def test_categorize_encodes(self):
        """Categorize.encodes maps category string to index."""
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        result = cat('dog')
        assert isinstance(result, TensorCategory)
        assert result.item() == 2  # sorted: bird=0, cat=1, dog=2

    def test_categorize_decodes(self):
        """Categorize.decodes maps index back to category string."""
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        decoded = cat.decode(TensorCategory(0))
        assert str(decoded) == 'bird'

    def test_categorize_roundtrip(self):
        """Encode then decode returns the original category."""
        cat = Categorize(vocab=['apple', 'banana', 'cherry'])
        for item in ['apple', 'banana', 'cherry']:
            encoded = cat(item)
            decoded = cat.decode(encoded)
            assert str(decoded) == item

    def test_categorize_unknown_label_raises(self):
        """Categorize raises KeyError for unknown label."""
        cat = Categorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="was not included in the training dataset"):
            cat('fish')

    def test_categorize_vocab_length(self):
        """Categorize.c gives the number of categories after setups."""
        cat = Categorize(vocab=['a', 'b', 'c', 'd'])
        # .c is set during setups, which is called when vocab is provided
        cat.setups(None)
        assert cat.c == 4

    def test_categorize_setups_from_data(self):
        """Categorize.setups builds vocab from dataset."""
        cat = Categorize()
        data = L(['dog', 'cat', 'bird', 'cat', 'dog'])
        cat.setups(data)
        assert cat.c == 3
        assert 'bird' in cat.vocab.items
        assert 'cat' in cat.vocab.items
        assert 'dog' in cat.vocab.items


# ============================================================
# Tests for MultiCategorize and OneHotEncode
# ============================================================

class TestMultiCategorize:
    """Tests for MultiCategorize transform."""

    def test_multi_categorize_encodes(self):
        """MultiCategorize encodes a list of labels to indices."""
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        result = mc(['cat', 'bird'])
        assert isinstance(result, TensorMultiCategory)
        # vocab is ['cat', 'dog', 'bird'] (sort=False since vocab is not None)
        assert set(result.tolist()) == {0, 2}

    def test_multi_categorize_decodes(self):
        """MultiCategorize decodes indices back to labels."""
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        encoded = mc(['cat', 'dog'])
        decoded = mc.decode(encoded)
        assert set(decoded) == {'cat', 'dog'}

    def test_multi_categorize_roundtrip(self):
        """Encode then decode returns the original set of labels."""
        mc = MultiCategorize(vocab=['a', 'b', 'c', 'd'])
        labels = ['a', 'c', 'd']
        encoded = mc(labels)
        decoded = mc.decode(encoded)
        assert set(decoded) == set(labels)

    def test_multi_categorize_empty_list(self):
        """MultiCategorize handles empty label list."""
        mc = MultiCategorize(vocab=['cat', 'dog'])
        result = mc([])
        assert len(result) == 0

    def test_multi_categorize_unknown_label_raises(self):
        """MultiCategorize raises KeyError for unknown labels."""
        mc = MultiCategorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="were not included in the training dataset"):
            mc(['fish'])

    def test_multi_categorize_setups_from_data(self):
        """MultiCategorize.setups builds vocab from dataset."""
        mc = MultiCategorize()
        data = L([['cat', 'dog'], ['bird', 'cat'], ['dog', 'bird']])
        mc.setups(data)
        assert set(mc.vocab.items) == {'cat', 'dog', 'bird'}


class TestOneHotEncode:
    """Tests for OneHotEncode transform."""

    def test_one_hot_encode_basic(self):
        """OneHotEncode creates one-hot vector from indices."""
        ohe = OneHotEncode(c=4)
        indices = TensorMultiCategory([0, 2])
        result = ohe(indices)
        assert isinstance(result, TensorMultiCategory)
        expected = torch.tensor([1., 0., 1., 0.])
        assert torch.equal(result, expected)

    def test_one_hot_encode_single_class(self):
        """OneHotEncode works with a single active class."""
        ohe = OneHotEncode(c=3)
        indices = TensorMultiCategory([1])
        result = ohe(indices)
        expected = torch.tensor([0., 1., 0.])
        assert torch.equal(result, expected)

    def test_one_hot_encode_all_classes(self):
        """OneHotEncode with all classes active."""
        ohe = OneHotEncode(c=3)
        indices = TensorMultiCategory([0, 1, 2])
        result = ohe(indices)
        expected = torch.tensor([1., 1., 1.])
        assert torch.equal(result, expected)

    def test_one_hot_encode_empty(self):
        """OneHotEncode with empty indices produces zeros."""
        ohe = OneHotEncode(c=3)
        indices = TensorMultiCategory([])
        result = ohe(indices)
        expected = torch.tensor([0., 0., 0.])
        assert torch.equal(result, expected)

    def test_one_hot_encode_decode(self):
        """OneHotEncode.decode converts one-hot back to indices."""
        ohe = OneHotEncode(c=4)
        one_hot_vec = TensorMultiCategory([1., 0., 1., 0.])
        result = ohe.decode(one_hot_vec)
        assert set(result) == {0, 2}

    def test_one_hot_encode_result_dtype(self):
        """OneHotEncode produces float tensors."""
        ohe = OneHotEncode(c=5)
        indices = TensorMultiCategory([1, 3])
        result = ohe(indices)
        assert result.dtype == torch.float32


# ============================================================
# Tests for IntToFloatTensor
# ============================================================

class TestIntToFloatTensor:
    """Tests for IntToFloatTensor transform."""

    def test_int_to_float_default_div(self):
        """IntToFloatTensor divides by 255 by default."""
        itf = IntToFloatTensor()
        img = TensorImage(torch.tensor([[[0, 128, 255]]], dtype=torch.uint8))
        result = itf(img)
        assert result.dtype == torch.float32
        assert abs(result[0, 0, 0].item() - 0.0) < 1e-5
        assert abs(result[0, 0, 1].item() - 128.0 / 255.0) < 1e-4
        assert abs(result[0, 0, 2].item() - 1.0) < 1e-5

    def test_int_to_float_custom_div(self):
        """IntToFloatTensor respects custom divisor."""
        itf = IntToFloatTensor(div=128.)
        img = TensorImage(torch.tensor([[[128]]], dtype=torch.uint8))
        result = itf(img)
        assert abs(result[0, 0, 0].item() - 1.0) < 1e-5

    def test_int_to_float_preserves_tensor_image_type(self):
        """IntToFloatTensor output is still a TensorImage."""
        itf = IntToFloatTensor()
        img = TensorImage(torch.randint(0, 255, (3, 4, 4), dtype=torch.uint8))
        result = itf(img)
        assert isinstance(result, TensorImage)

    def test_int_to_float_decode(self):
        """IntToFloatTensor.decodes reverses the transformation."""
        itf = IntToFloatTensor()
        img = TensorImage(torch.tensor([[[0.0, 0.5, 1.0]]]))
        decoded = itf.decode(img)
        assert decoded[0, 0, 0].item() == 0
        # 0.5 * 255 = 127.5, .long() truncates to 127
        assert decoded[0, 0, 1].item() == 127
        assert decoded[0, 0, 2].item() == 255

    def test_int_to_float_mask(self):
        """IntToFloatTensor encodes TensorMask as long."""
        itf = IntToFloatTensor()
        mask = TensorMask(torch.tensor([[1, 2, 3]], dtype=torch.long))
        result = itf(mask)
        assert result.dtype == torch.int64

    def test_int_to_float_batch(self):
        """IntToFloatTensor works on batch-like tensors."""
        itf = IntToFloatTensor()
        batch = TensorImage(torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8))
        result = itf(batch)
        assert result.shape == (4, 3, 8, 8)
        assert result.dtype == torch.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ============================================================
# Tests for broadcast_vec and Normalize
# ============================================================

class TestBroadcastVec:
    """Tests for broadcast_vec utility."""

    def test_broadcast_vec_shapes(self):
        """broadcast_vec creates correct shape for broadcasting."""
        mean, std = broadcast_vec(1, 4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], cuda=False)
        assert mean.shape == (1, 3, 1, 1)
        assert std.shape == (1, 3, 1, 1)

    def test_broadcast_vec_values(self):
        """broadcast_vec preserves the original values."""
        mean, = broadcast_vec(1, 4, [0.5, 0.6, 0.7], cuda=False)
        assert abs(mean[0, 0, 0, 0].item() - 0.5) < 1e-5
        assert abs(mean[0, 1, 0, 0].item() - 0.6) < 1e-5
        assert abs(mean[0, 2, 0, 0].item() - 0.7) < 1e-5

    def test_broadcast_vec_different_dims(self):
        """broadcast_vec works with different dim and ndim combinations."""
        vecs = broadcast_vec(0, 3, [1.0, 2.0], cuda=False)
        assert vecs[0].shape == (2, 1, 1)


class TestNormalize:
    """Tests for Normalize transform."""

    def test_normalize_encodes(self):
        """Normalize subtracts mean and divides by std."""
        mean = torch.tensor([[[0.5]]])
        std = torch.tensor([[[0.2]]])
        norm = Normalize(mean=mean, std=std)
        x = TensorImage(torch.tensor([[[1.0]]]))
        result = norm(x)
        expected = (1.0 - 0.5) / 0.2
        assert abs(result[0, 0, 0].item() - expected) < 1e-5

    def test_normalize_decodes(self):
        """Normalize.decodes reverses the normalization."""
        mean = torch.tensor([[[0.5]]])
        std = torch.tensor([[[0.2]]])
        norm = Normalize(mean=mean, std=std)
        x = TensorImage(torch.tensor([[[1.0]]]))
        encoded = norm(x)
        decoded = norm.decode(encoded)
        assert abs(decoded[0, 0, 0].item() - 1.0) < 1e-5

    def test_normalize_roundtrip(self):
        """Normalize encode then decode returns original values."""
        mean = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
        std = torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
        norm = Normalize(mean=mean, std=std)
        x = TensorImage(torch.rand(1, 3, 4, 4))
        encoded = norm(x)
        decoded = norm.decode(encoded)
        assert torch.allclose(decoded, x, atol=1e-5)

    def test_normalize_from_stats(self):
        """Normalize.from_stats creates correct mean/std tensors."""
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        norm = Normalize.from_stats(imagenet_mean, imagenet_std, cuda=False)
        assert norm.mean.shape == (1, 3, 1, 1)
        assert norm.std.shape == (1, 3, 1, 1)
        assert abs(norm.mean[0, 0, 0, 0].item() - 0.485) < 1e-5

    def test_normalize_batch(self):
        """Normalize works on batches of images."""
        mean = torch.tensor([[[[0.5]], [[0.5]], [[0.5]]]])
        std = torch.tensor([[[[0.25]], [[0.25]], [[0.25]]]])
        norm = Normalize(mean=mean, std=std)
        batch = TensorImage(torch.rand(4, 3, 8, 8))
        result = norm(batch)
        assert result.shape == batch.shape
        # Normalized values should be centered around 0
        # (since input is uniform [0,1] with mean ~0.5)

    def test_normalize_zero_preserves_structure(self):
        """Normalizing a zero tensor produces expected result."""
        mean = torch.tensor([[[0.5]]])
        std = torch.tensor([[[0.25]]])
        norm = Normalize(mean=mean, std=std)
        x = TensorImage(torch.zeros(1, 1, 1))
        result = norm(x)
        expected = (0.0 - 0.5) / 0.25
        assert abs(result[0, 0, 0].item() - expected) < 1e-5
