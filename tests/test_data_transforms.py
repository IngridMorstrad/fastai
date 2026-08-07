"""Tests for fastai.data.transforms module.

Covers: get_files, get_image_files, get_text_files, FileGetter, ImageGetter,
ItemGetter, AttrGetter, RandomSplitter, TrainTestSplitter, IndexSplitter,
EndSplitter, GrandparentSplitter, FuncSplitter, MaskSplitter, ColSplitter,
RandomSubsetSplitter, parent_label, RegexLabeller, CategoryMap, Categorize,
MultiCategorize, OneHotEncode, RegressionSetup, IntToFloatTensor, Normalize,
broadcast_vec.
"""
import sys
import os
import tempfile
import re
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files, get_image_files, get_text_files, FileGetter, ImageGetter,
    ItemGetter, AttrGetter,
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, ColSplitter,
    RandomSubsetSplitter,
    parent_label, RegexLabeller,
    CategoryMap, Categorize, MultiCategorize, OneHotEncode,
    RegressionSetup, IntToFloatTensor, Normalize, broadcast_vec,
)
from fastcore.foundation import L


# ============================================================
# Helper: create a temporary directory tree for file-getter tests
# ============================================================

@pytest.fixture
def tmp_file_tree(tmp_path):
    """Create a temp directory with files of various types."""
    # Root files
    (tmp_path / "readme.txt").write_text("hello")
    (tmp_path / "photo.jpg").write_text("img")
    (tmp_path / "data.csv").write_text("a,b")
    # Hidden file (should be skipped)
    (tmp_path / ".hidden").write_text("secret")

    # Subdirectory
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "notes.txt").write_text("note")
    (sub / "icon.png").write_text("png")
    (sub / "archive.zip").write_text("zip")

    # Another subdirectory
    sub2 = tmp_path / "other"
    sub2.mkdir()
    (sub2 / "image.jpeg").write_text("jpeg")
    (sub2 / "doc.txt").write_text("doc")

    return tmp_path


# ============================================================
# Tests for get_files
# ============================================================

class TestGetFiles:
    """Tests for the get_files function."""

    def test_get_all_files_recursively(self, tmp_file_tree):
        files = get_files(tmp_file_tree)
        # Should find files in root and subdirs, excluding hidden
        names = [f.name for f in files]
        assert "readme.txt" in names
        assert "photo.jpg" in names
        assert "notes.txt" in names
        assert "icon.png" in names
        assert ".hidden" not in names

    def test_filter_by_extension(self, tmp_file_tree):
        files = get_files(tmp_file_tree, extensions=[".txt"])
        names = [f.name for f in files]
        assert "readme.txt" in names
        assert "notes.txt" in names
        assert "doc.txt" in names
        assert "photo.jpg" not in names
        assert "icon.png" not in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_files(tmp_file_tree, recurse=False)
        names = [f.name for f in files]
        assert "readme.txt" in names
        assert "photo.jpg" in names
        # Files from subdirectories should not appear
        assert "notes.txt" not in names
        assert "icon.png" not in names

    def test_specific_folders(self, tmp_file_tree):
        files = get_files(tmp_file_tree, folders=["subdir"])
        names = [f.name for f in files]
        assert "notes.txt" in names
        assert "icon.png" in names
        # Files from root and other folder should not appear
        assert "readme.txt" not in names
        assert "image.jpeg" not in names

    def test_extension_case_insensitive(self, tmp_file_tree):
        # Create a file with uppercase extension
        (tmp_file_tree / "UPPER.TXT").write_text("upper")
        files = get_files(tmp_file_tree, extensions=[".txt"])
        names = [f.name for f in files]
        assert "UPPER.TXT" in names

    def test_returns_L_type(self, tmp_file_tree):
        files = get_files(tmp_file_tree)
        assert isinstance(files, L)

    def test_empty_directory(self, tmp_path):
        files = get_files(tmp_path)
        assert len(files) == 0


# ============================================================
# Tests for get_image_files
# ============================================================

class TestGetImageFiles:
    """Tests for the get_image_files function."""

    def test_finds_image_files(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree)
        names = [f.name for f in files]
        assert "photo.jpg" in names
        assert "icon.png" in names
        assert "image.jpeg" in names
        # Non-image files excluded
        assert "readme.txt" not in names
        assert "data.csv" not in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree, recurse=False)
        names = [f.name for f in files]
        assert "photo.jpg" in names
        assert "icon.png" not in names

    def test_specific_folders(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree, folders=["other"])
        names = [f.name for f in files]
        assert "image.jpeg" in names
        assert "photo.jpg" not in names


# ============================================================
# Tests for get_text_files
# ============================================================

class TestGetTextFiles:
    """Tests for the get_text_files function."""

    def test_finds_text_files(self, tmp_file_tree):
        files = get_text_files(tmp_file_tree)
        names = [f.name for f in files]
        assert "readme.txt" in names
        assert "notes.txt" in names
        assert "doc.txt" in names
        # Non-text files excluded
        assert "photo.jpg" not in names
        assert "data.csv" not in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_text_files(tmp_file_tree, recurse=False)
        names = [f.name for f in files]
        assert "readme.txt" in names
        assert "notes.txt" not in names


# ============================================================
# Tests for FileGetter and ImageGetter
# ============================================================

class TestFileGetter:
    """Tests for the FileGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = FileGetter()
        files = getter(tmp_file_tree)
        assert len(files) > 0
        assert all(isinstance(f, Path) for f in files)

    def test_with_extension_filter(self, tmp_file_tree):
        getter = FileGetter(extensions=[".txt"])
        files = getter(tmp_file_tree)
        assert all(f.suffix == ".txt" for f in files)

    def test_with_suffix_path(self, tmp_file_tree):
        getter = FileGetter(suf="subdir")
        files = getter(tmp_file_tree)
        names = [f.name for f in files]
        assert "notes.txt" in names
        assert "readme.txt" not in names


class TestImageGetter:
    """Tests for the ImageGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = ImageGetter()
        files = getter(tmp_file_tree)
        names = [f.name for f in files]
        assert "photo.jpg" in names
        assert "readme.txt" not in names


# ============================================================
# Tests for ItemGetter and AttrGetter
# ============================================================

class TestItemGetter:
    """Tests for the ItemGetter transform."""

    def test_get_from_list(self):
        ig = ItemGetter(1)
        result = ig.encodes(["a", "b", "c"])
        assert result == "b"

    def test_get_from_tuple(self):
        ig = ItemGetter(0)
        result = ig.encodes(("first", "second"))
        assert result == "first"

    def test_get_from_dict(self):
        ig = ItemGetter("key")
        result = ig.encodes({"key": "value", "other": "x"})
        assert result == "value"


class TestAttrGetter:
    """Tests for the AttrGetter transform."""

    def test_get_existing_attribute(self):
        ag = AttrGetter("real")
        result = ag.encodes(complex(3, 4))
        assert result == 3.0

    def test_get_with_default(self):
        ag = AttrGetter("nonexistent", default="fallback")

        class Dummy:
            pass

        result = ag.encodes(Dummy())
        assert result == "fallback"


# ============================================================
# Tests for RandomSplitter
# ============================================================

class TestRandomSplitter:
    """Tests for the RandomSplitter function."""

    def test_correct_split_sizes(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20
        assert len(train) == 80

    def test_no_overlap(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_all_indices_covered(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        train, valid = splitter(items)
        assert sorted(list(train) + list(valid)) == list(range(50))

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=123)
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
        # With different seeds, splits should differ (extremely unlikely to match)
        assert list(train1) != list(train2)

    def test_valid_pct_boundary(self):
        items = list(range(10))
        # 10% of 10 = 1 item in valid
        splitter = RandomSplitter(valid_pct=0.1, seed=42)
        train, valid = splitter(items)
        assert len(valid) == 1
        assert len(train) == 9


# ============================================================
# Tests for TrainTestSplitter
# ============================================================

class TestTrainTestSplitter:
    """Tests for the TrainTestSplitter function."""

    def test_correct_split_sizes(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train, valid = splitter(items)
        assert len(train) == 80
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(50))
        splitter = TrainTestSplitter(test_size=0.3, random_state=42)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_reproducibility(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=99)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)


# ============================================================
# Tests for IndexSplitter
# ============================================================

class TestIndexSplitter:
    """Tests for the IndexSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        splitter = IndexSplitter([2, 4, 6])
        train, valid = splitter(items)
        assert sorted(list(valid)) == [2, 4, 6]
        assert sorted(list(train)) == [0, 1, 3, 5, 7, 8, 9]

    def test_empty_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([])
        train, valid = splitter(items)
        assert len(valid) == 0
        assert sorted(list(train)) == [0, 1, 2, 3, 4]

    def test_all_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train, valid = splitter(items)
        assert len(train) == 0
        assert sorted(list(valid)) == [0, 1, 2, 3, 4]


# ============================================================
# Tests for EndSplitter
# ============================================================

class TestEndSplitter:
    """Tests for the EndSplitter function."""

    def test_valid_last(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # Last 3 items should be valid
        assert list(valid) == [7, 8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6]

    def test_valid_first(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train, valid = splitter(items)
        # First 3 items should be valid
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]

    def test_correct_sizes(self):
        items = list(range(20))
        splitter = EndSplitter(valid_pct=0.25)
        train, valid = splitter(items)
        assert len(valid) == 5
        assert len(train) == 15

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)


# ============================================================
# Tests for GrandparentSplitter
# ============================================================

class TestGrandparentSplitter:
    """Tests for the GrandparentSplitter function."""

    def test_basic_split(self, tmp_path):
        # Create train/class/file and valid/class/file structure
        train_dir = tmp_path / "train" / "cats"
        train_dir.mkdir(parents=True)
        valid_dir = tmp_path / "valid" / "cats"
        valid_dir.mkdir(parents=True)

        (train_dir / "cat1.jpg").write_text("")
        (train_dir / "cat2.jpg").write_text("")
        (valid_dir / "cat3.jpg").write_text("")

        items = [
            train_dir / "cat1.jpg",
            train_dir / "cat2.jpg",
            valid_dir / "cat3.jpg",
        ]

        splitter = GrandparentSplitter(train_name="train", valid_name="valid")
        train_idx, valid_idx = splitter(items)
        assert sorted(list(train_idx)) == [0, 1]
        assert list(valid_idx) == [2]

    def test_custom_names(self, tmp_path):
        tr = tmp_path / "tr" / "cls"
        tr.mkdir(parents=True)
        va = tmp_path / "va" / "cls"
        va.mkdir(parents=True)

        (tr / "a.jpg").write_text("")
        (va / "b.jpg").write_text("")

        items = [tr / "a.jpg", va / "b.jpg"]
        splitter = GrandparentSplitter(train_name="tr", valid_name="va")
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0]
        assert list(valid_idx) == [1]


# ============================================================
# Tests for FuncSplitter
# ============================================================

class TestFuncSplitter:
    """Tests for the FuncSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        # Items > 7 go to validation
        splitter = FuncSplitter(lambda x: x > 7)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [8, 9]
        assert sorted(list(train)) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_all_train(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: False)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 5

    def test_all_valid(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: True)
        train, valid = splitter(items)
        assert len(valid) == 5
        assert len(train) == 0


# ============================================================
# Tests for MaskSplitter
# ============================================================

class TestMaskSplitter:
    """Tests for the MaskSplitter function."""

    def test_basic_mask(self):
        items = list(range(5))
        mask = [False, True, False, True, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2, 4]

    def test_all_false_mask(self):
        items = list(range(3))
        mask = [False, False, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 3


# ============================================================
# Tests for ColSplitter
# ============================================================

class TestColSplitter:
    """Tests for the ColSplitter function."""

    def test_boolean_column(self):
        df = pd.DataFrame({
            "data": [1, 2, 3, 4, 5],
            "is_valid": [False, True, False, True, False]
        })
        splitter = ColSplitter(col="is_valid")
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2, 4]

    def test_integer_column_index(self):
        df = pd.DataFrame({
            "data": [1, 2, 3],
            "split": [False, True, True]
        })
        # Column index 1 is 'split'
        splitter = ColSplitter(col=1)
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 2]
        assert list(train) == [0]

    def test_with_on_value(self):
        df = pd.DataFrame({
            "data": [1, 2, 3, 4],
            "fold": ["train", "valid", "train", "valid"]
        })
        splitter = ColSplitter(col="fold", on="valid")
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2]

    def test_with_on_list(self):
        df = pd.DataFrame({
            "data": [1, 2, 3, 4, 5],
            "fold": [0, 1, 2, 1, 0]
        })
        splitter = ColSplitter(col="fold", on=[1, 2])
        train, valid = splitter(df)
        assert sorted(list(valid)) == [1, 2, 3]
        assert sorted(list(train)) == [0, 4]

    def test_non_dataframe_raises(self):
        splitter = ColSplitter(col="x")
        with pytest.raises(AssertionError):
            splitter([1, 2, 3])


# ============================================================
# Tests for RandomSubsetSplitter
# ============================================================

class TestRandomSubsetSplitter:
    """Tests for the RandomSubsetSplitter function."""

    def test_correct_sizes(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) == 60
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=42)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=7)
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
            RandomSubsetSplitter(train_sz=0.8, valid_sz=0.5)


# ============================================================
# Tests for parent_label
# ============================================================

class TestParentLabel:
    """Tests for the parent_label function."""

    def test_basic_path(self):
        assert parent_label("/data/cats/image1.jpg") == "cats"

    def test_nested_path(self):
        assert parent_label("/data/train/dogs/photo.png") == "dogs"

    def test_path_object(self):
        p = Path("/dataset/class_a/file.txt")
        assert parent_label(p) == "class_a"


# ============================================================
# Tests for RegexLabeller
# ============================================================

class TestRegexLabeller:
    """Tests for the RegexLabeller class."""

    def test_search_mode(self):
        labeller = RegexLabeller(r"/(\w+)/[^/]+$")
        result = labeller("/data/cats/img.jpg")
        assert result == "cats"

    def test_match_mode(self):
        labeller = RegexLabeller(r"(\w+)_\d+\.jpg", match=True)
        result = labeller("cat_001.jpg")
        assert result == "cat"

    def test_no_match_raises(self):
        labeller = RegexLabeller(r"NOMATCH_(\w+)")
        with pytest.raises(AssertionError):
            labeller("something_else.txt")

    def test_with_path_separators(self):
        labeller = RegexLabeller(r"(\w+)/[^/]+$")
        # On all platforms, paths are converted to posix sep
        result = labeller(Path("/train/dogs/img.jpg"))
        assert result == "dogs"


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for the CategoryMap class."""

    def test_basic_creation(self):
        cats = CategoryMap(["cat", "dog", "bird"])
        assert len(cats) == 3
        assert "cat" in cats.items
        assert "dog" in cats.items
        assert "bird" in cats.items

    def test_sorted_by_default(self):
        cats = CategoryMap(["dog", "cat", "bird"])
        assert list(cats.items) == ["bird", "cat", "dog"]

    def test_unsorted(self):
        cats = CategoryMap(["dog", "cat", "bird"], sort=False)
        # Unique preserves first-seen order
        assert list(cats.items) == ["dog", "cat", "bird"]

    def test_o2i_mapping(self):
        cats = CategoryMap(["cat", "dog", "bird"], sort=True)
        # Sorted: bird=0, cat=1, dog=2
        assert cats.o2i["bird"] == 0
        assert cats.o2i["cat"] == 1
        assert cats.o2i["dog"] == 2

    def test_add_na(self):
        cats = CategoryMap(["cat", "dog"], add_na=True)
        assert "#na#" in cats.items
        assert cats.o2i["#na#"] == 0

    def test_map_objs(self):
        cats = CategoryMap(["cat", "dog", "bird"], sort=True)
        ids = cats.map_objs(["dog", "bird"])
        assert list(ids) == [2, 0]

    def test_map_ids(self):
        cats = CategoryMap(["cat", "dog", "bird"], sort=True)
        objs = cats.map_ids([0, 1, 2])
        assert list(objs) == ["bird", "cat", "dog"]

    def test_deduplication(self):
        cats = CategoryMap(["cat", "dog", "cat", "bird", "dog"])
        assert len(cats) == 3


# ============================================================
# Tests for Categorize
# ============================================================

class TestCategorize:
    """Tests for the Categorize transform."""

    def test_encodes_with_vocab(self):
        cat = Categorize(vocab=["cat", "dog", "bird"])
        result = cat.encodes("dog")
        # Sorted vocab: bird=0, cat=1, dog=2
        assert int(result) == 2

    def test_decodes(self):
        cat = Categorize(vocab=["cat", "dog", "bird"])
        encoded = cat.encodes("cat")
        decoded = cat.decodes(encoded)
        assert str(decoded) == "cat"

    def test_unknown_label_raises(self):
        cat = Categorize(vocab=["cat", "dog"])
        with pytest.raises(KeyError):
            cat.encodes("bird")

    def test_c_attribute(self):
        cat = Categorize(vocab=["a", "b", "c", "d"])
        # c is set during setups (called after vocab is created in __init__)
        cat.setups(None)
        assert cat.c == 4

    def test_setups_from_data(self):
        cat = Categorize()
        data = L(["cat", "dog", "cat", "bird"])
        cat.setups(data)
        assert cat.c == 3
        assert int(cat.encodes("cat")) in [0, 1, 2]


# ============================================================
# Tests for MultiCategorize
# ============================================================

class TestMultiCategorize:
    """Tests for the MultiCategorize transform."""

    def test_encodes(self):
        mc = MultiCategorize(vocab=["cat", "dog", "bird"])
        result = mc.encodes(["cat", "bird"])
        assert len(result) == 2

    def test_unknown_label_raises(self):
        mc = MultiCategorize(vocab=["cat", "dog"])
        with pytest.raises(KeyError):
            mc.encodes(["cat", "bird"])

    def test_decodes(self):
        mc = MultiCategorize(vocab=["cat", "dog", "bird"])
        encoded = mc.encodes(["dog", "bird"])
        decoded = mc.decodes(encoded)
        assert "dog" in decoded
        assert "bird" in decoded

    def test_setups_from_data(self):
        mc = MultiCategorize()
        data = L([["cat", "dog"], ["bird", "cat"], ["dog"]])
        mc.setups(data)
        # Should have all 3 categories
        assert len(mc.vocab) == 3


# ============================================================
# Tests for OneHotEncode
# ============================================================

class TestOneHotEncode:
    """Tests for the OneHotEncode transform."""

    def test_encodes(self):
        ohe = OneHotEncode(c=4)
        # Input is a tensor of indices
        inp = torch.tensor([0, 2])
        result = ohe.encodes(inp)
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(result, expected)

    def test_decodes(self):
        ohe = OneHotEncode(c=4)
        encoded = torch.tensor([1.0, 0.0, 1.0, 0.0])
        decoded = ohe.decodes(encoded)
        assert 0 in decoded
        assert 2 in decoded
        assert 1 not in decoded

    def test_c_from_parameter(self):
        ohe = OneHotEncode(c=5)
        inp = torch.tensor([1, 3])
        result = ohe.encodes(inp)
        assert len(result) == 5


# ============================================================
# Tests for RegressionSetup
# ============================================================

class TestRegressionSetup:
    """Tests for the RegressionSetup transform."""

    def test_encodes_scalar(self):
        rs = RegressionSetup()
        result = rs.encodes(3.14)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert abs(float(result) - 3.14) < 1e-5

    def test_encodes_list(self):
        rs = RegressionSetup()
        result = rs.encodes([1.0, 2.0, 3.0])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert len(result) == 3

    def test_decodes_scalar(self):
        rs = RegressionSetup()
        t = torch.tensor(2.5)
        decoded = rs.decodes(t)
        assert abs(float(decoded) - 2.5) < 1e-5


# ============================================================
# Tests for IntToFloatTensor
# ============================================================

class TestIntToFloatTensor:
    """Tests for the IntToFloatTensor transform."""

    def test_encodes_image_tensor(self):
        from fastai.torch_basics import TensorImage
        itf = IntToFloatTensor(div=255.0)
        # Simulate a uint8-like image tensor
        img = TensorImage(torch.randint(0, 256, (3, 4, 4)))
        result = itf.encodes(img)
        assert result.dtype == torch.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_custom_div(self):
        from fastai.torch_basics import TensorImage
        itf = IntToFloatTensor(div=128.0)
        img = TensorImage(torch.tensor([[[128]]]))
        result = itf.encodes(img)
        assert abs(float(result) - 1.0) < 1e-5

    def test_decodes_image_tensor(self):
        from fastai.torch_basics import TensorImage
        itf = IntToFloatTensor(div=255.0)
        # A float tensor with values in [0, 1]
        img = TensorImage(torch.tensor([[[0.5]]]))
        result = itf.decodes(img)
        # Should be approximately 127 or 128
        assert abs(int(result) - 127) <= 1


# ============================================================
# Tests for broadcast_vec
# ============================================================

class TestBroadcastVec:
    """Tests for the broadcast_vec function."""

    def test_basic_broadcast(self):
        # 4D tensor, broadcast over dim 1
        result = broadcast_vec(1, 4, [0.5, 0.5, 0.5], cuda=False)
        assert len(result) == 1
        t = result[0]
        assert t.shape == (1, 3, 1, 1)

    def test_dim_0(self):
        result = broadcast_vec(0, 3, [1.0, 2.0], cuda=False)
        t = result[0]
        assert t.shape == (2, 1, 1)

    def test_multiple_tensors(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        results = broadcast_vec(1, 4, mean, std, cuda=False)
        assert len(results) == 2
        assert results[0].shape == (1, 3, 1, 1)
        assert results[1].shape == (1, 3, 1, 1)


# ============================================================
# Tests for Normalize
# ============================================================

class TestNormalize:
    """Tests for the Normalize transform."""

    def test_from_stats(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm = Normalize.from_stats(mean, std, cuda=False)
        assert norm.mean is not None
        assert norm.std is not None

    def test_encodes_decodes_roundtrip(self):
        from fastai.torch_basics import TensorImage
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        norm = Normalize.from_stats(mean, std, cuda=False)

        # Create a batch of images: (batch, channels, height, width)
        img = TensorImage(torch.rand(2, 3, 4, 4))
        encoded = norm.encodes(img)
        decoded = norm.decodes(encoded)

        # Round-trip should approximately recover original
        assert torch.allclose(img, decoded, atol=1e-5)

    def test_encodes_normalizes(self):
        from fastai.torch_basics import TensorImage
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        norm = Normalize.from_stats(mean, std, cuda=False)

        img = TensorImage(torch.ones(1, 3, 2, 2) * 0.5)
        result = norm.encodes(img)
        # (0.5 - 0) / 1 = 0.5
        assert torch.allclose(result, img, atol=1e-5)

    def test_encodes_with_nonzero_mean(self):
        from fastai.torch_basics import TensorImage
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        norm = Normalize.from_stats(mean, std, cuda=False)

        img = TensorImage(torch.ones(1, 3, 2, 2) * 0.5)
        result = norm.encodes(img)
        # (0.5 - 0.5) / 0.5 = 0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)
