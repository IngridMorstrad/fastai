"""Tests for fastai.data.transforms module.

Covers: get_files, get_image_files, get_text_files, FileGetter, ImageGetter,
RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter, ColSplitter,
RandomSubsetSplitter, parent_label, RegexLabeller, CategoryMap, Categorize,
MultiCategorize, OneHotEncode, IntToFloatTensor, Normalize, ItemGetter,
AttrGetter, broadcast_vec.
"""
import sys
import os
import tempfile
import re
import pytest
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files, get_image_files, get_text_files,
    FileGetter, ImageGetter,
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter,
    ColSplitter, RandomSubsetSplitter,
    parent_label, RegexLabeller,
    CategoryMap, Categorize, MultiCategorize, OneHotEncode,
    IntToFloatTensor, Normalize, broadcast_vec,
    ItemGetter, AttrGetter, image_extensions,
)
from fastcore.foundation import L


# ============================================================
# Helper to create a temporary directory structure for file tests
# ============================================================

@pytest.fixture
def tmp_file_tree(tmp_path):
    """Create a temporary file tree for testing file retrieval functions.

    Structure:
        tmp_path/
            file1.txt
            file2.py
            image1.jpg
            image2.png
            .hidden_file.txt
            subdir/
                file3.txt
                image3.gif
                nested/
                    file4.txt
                    deep.jpg
            other/
                code.py
    """
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.py").write_text("print('hi')")
    (tmp_path / "image1.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "image2.png").write_bytes(b"\x89PNG")
    (tmp_path / ".hidden_file.txt").write_text("secret")

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("world")
    (subdir / "image3.gif").write_bytes(b"GIF89a")
    nested = subdir / "nested"
    nested.mkdir()
    (nested / "file4.txt").write_text("deep")
    (nested / "deep.jpg").write_bytes(b"\xff\xd8\xff")

    other = tmp_path / "other"
    other.mkdir()
    (other / "code.py").write_text("pass")

    return tmp_path


# ============================================================
# Tests for get_files
# ============================================================

class TestGetFiles:
    """Tests for the get_files function."""

    def test_get_all_files_recursive(self, tmp_file_tree):
        result = get_files(tmp_file_tree, recurse=True)
        # Should not include hidden files (starting with .)
        names = [f.name for f in result]
        assert ".hidden_file.txt" not in names
        # Should include regular files at all levels
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "file4.txt" in names

    def test_get_files_non_recursive(self, tmp_file_tree):
        result = get_files(tmp_file_tree, recurse=False)
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "image1.jpg" in names
        # Should NOT include files from subdirectories
        assert "file3.txt" not in names
        assert "file4.txt" not in names

    def test_get_files_with_extensions(self, tmp_file_tree):
        result = get_files(tmp_file_tree, extensions=['.txt'], recurse=True)
        for f in result:
            assert f.suffix == '.txt'
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "file2.py" not in names

    def test_get_files_extensions_case_insensitive(self, tmp_file_tree):
        # Create a file with uppercase extension
        (tmp_file_tree / "upper.TXT").write_text("upper")
        result = get_files(tmp_file_tree, extensions=['.txt'], recurse=True)
        names = [f.name for f in result]
        assert "upper.TXT" in names
        assert "file1.txt" in names

    def test_get_files_specific_folders(self, tmp_file_tree):
        result = get_files(tmp_file_tree, recurse=True, folders=['subdir'])
        # Should only get files from the subdir folder and its children
        for f in result:
            assert 'subdir' in str(f)

    def test_get_files_returns_L(self, tmp_file_tree):
        result = get_files(tmp_file_tree, recurse=True)
        assert isinstance(result, L)

    def test_get_files_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = get_files(empty_dir, recurse=True)
        assert len(result) == 0

    def test_get_files_no_matching_extension(self, tmp_file_tree):
        result = get_files(tmp_file_tree, extensions=['.xyz'], recurse=True)
        assert len(result) == 0


# ============================================================
# Tests for get_image_files
# ============================================================

class TestGetImageFiles:
    """Tests for the get_image_files function."""

    def test_finds_image_files(self, tmp_file_tree):
        result = get_image_files(tmp_file_tree, recurse=True)
        names = [f.name for f in result]
        assert "image1.jpg" in names
        assert "image2.png" in names
        assert "image3.gif" in names
        assert "deep.jpg" in names

    def test_excludes_non_image_files(self, tmp_file_tree):
        result = get_image_files(tmp_file_tree, recurse=True)
        names = [f.name for f in result]
        assert "file1.txt" not in names
        assert "file2.py" not in names

    def test_non_recursive(self, tmp_file_tree):
        result = get_image_files(tmp_file_tree, recurse=False)
        names = [f.name for f in result]
        assert "image1.jpg" in names
        assert "image2.png" in names
        # Should not include images from subdirectories
        assert "image3.gif" not in names
        assert "deep.jpg" not in names

    def test_specific_folders(self, tmp_file_tree):
        result = get_image_files(tmp_file_tree, recurse=True, folders=['subdir'])
        names = [f.name for f in result]
        assert "image3.gif" in names
        assert "deep.jpg" in names
        assert "image1.jpg" not in names


# ============================================================
# Tests for get_text_files
# ============================================================

class TestGetTextFiles:
    """Tests for the get_text_files function."""

    def test_finds_text_files(self, tmp_file_tree):
        result = get_text_files(tmp_file_tree, recurse=True)
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "file4.txt" in names

    def test_excludes_non_text_files(self, tmp_file_tree):
        result = get_text_files(tmp_file_tree, recurse=True)
        names = [f.name for f in result]
        assert "file2.py" not in names
        assert "image1.jpg" not in names

    def test_non_recursive(self, tmp_file_tree):
        result = get_text_files(tmp_file_tree, recurse=False)
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "file3.txt" not in names


# ============================================================
# Tests for FileGetter and ImageGetter
# ============================================================

class TestFileGetter:
    """Tests for the FileGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = FileGetter(extensions=['.txt'])
        result = getter(tmp_file_tree)
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "file3.txt" in names

    def test_with_suffix(self, tmp_file_tree):
        getter = FileGetter(suf='subdir', extensions=['.txt'])
        result = getter(tmp_file_tree)
        names = [f.name for f in result]
        assert "file3.txt" in names
        assert "file4.txt" in names
        assert "file1.txt" not in names

    def test_non_recursive(self, tmp_file_tree):
        getter = FileGetter(extensions=['.txt'], recurse=False)
        result = getter(tmp_file_tree)
        names = [f.name for f in result]
        assert "file1.txt" in names
        assert "file3.txt" not in names


class TestImageGetter:
    """Tests for the ImageGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = ImageGetter()
        result = getter(tmp_file_tree)
        names = [f.name for f in result]
        assert "image1.jpg" in names
        assert "image2.png" in names

    def test_with_suffix(self, tmp_file_tree):
        getter = ImageGetter(suf='subdir')
        result = getter(tmp_file_tree)
        names = [f.name for f in result]
        assert "image3.gif" in names
        assert "deep.jpg" in names
        assert "image1.jpg" not in names


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
        train_set = set(train)
        valid_set = set(valid)
        assert len(train_set & valid_set) == 0

    def test_all_indices_covered(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        train, valid = splitter(items)
        all_indices = set(train) | set(valid)
        assert all_indices == set(range(50))

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_different_seeds_produce_different_splits(self):
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=42)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=99)
        _, valid1 = splitter1(items)
        _, valid2 = splitter2(items)
        # Extremely unlikely to be identical with different seeds
        assert list(valid1) != list(valid2)

    def test_returns_L_instances(self):
        items = list(range(20))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train, valid = splitter(items)
        assert isinstance(train, L)
        assert isinstance(valid, L)


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
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_returns_L_instances(self):
        items = list(range(20))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train, valid = splitter(items)
        assert isinstance(train, L)
        assert isinstance(valid, L)


# ============================================================
# Tests for IndexSplitter
# ============================================================

class TestIndexSplitter:
    """Tests for the IndexSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        splitter = IndexSplitter([2, 5, 7])
        train, valid = splitter(items)
        assert set(valid) == {2, 5, 7}
        assert set(train) == {0, 1, 3, 4, 6, 8, 9}

    def test_empty_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([])
        train, valid = splitter(items)
        assert len(valid) == 0
        assert set(train) == {0, 1, 2, 3, 4}

    def test_all_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train, valid = splitter(items)
        assert len(train) == 0
        assert set(valid) == {0, 1, 2, 3, 4}

    def test_returns_L_instances(self):
        items = list(range(10))
        splitter = IndexSplitter([1, 3])
        train, valid = splitter(items)
        assert isinstance(train, L)
        assert isinstance(valid, L)


# ============================================================
# Tests for EndSplitter
# ============================================================

class TestEndSplitter:
    """Tests for the EndSplitter function."""

    def test_valid_at_end(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # 30% of 10 = 3 items at end
        assert list(valid) == [7, 8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6]

    def test_valid_at_start(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train, valid = splitter(items)
        # 30% of 10 = 3 items at start
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]

    def test_deterministic(self):
        items = list(range(100))
        splitter = EndSplitter(valid_pct=0.2)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

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
        # Create train/class1/file and valid/class1/file structure
        train_dir = tmp_path / "train" / "class1"
        train_dir.mkdir(parents=True)
        valid_dir = tmp_path / "valid" / "class1"
        valid_dir.mkdir(parents=True)

        (train_dir / "img1.jpg").write_bytes(b"img")
        (train_dir / "img2.jpg").write_bytes(b"img")
        (valid_dir / "img3.jpg").write_bytes(b"img")

        items = [
            train_dir / "img1.jpg",
            train_dir / "img2.jpg",
            valid_dir / "img3.jpg",
        ]

        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train_idx, valid_idx = splitter(items)
        assert set(train_idx) == {0, 1}
        assert set(valid_idx) == {2}

    def test_custom_names(self, tmp_path):
        trn = tmp_path / "trn" / "cls"
        trn.mkdir(parents=True)
        val = tmp_path / "val" / "cls"
        val.mkdir(parents=True)

        (trn / "a.jpg").write_bytes(b"a")
        (val / "b.jpg").write_bytes(b"b")

        items = [trn / "a.jpg", val / "b.jpg"]
        splitter = GrandparentSplitter(train_name='trn', valid_name='val')
        train_idx, valid_idx = splitter(items)
        assert set(train_idx) == {0}
        assert set(valid_idx) == {1}


# ============================================================
# Tests for FuncSplitter
# ============================================================

class TestFuncSplitter:
    """Tests for the FuncSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        # Even numbers go to validation
        splitter = FuncSplitter(lambda x: x % 2 == 0)
        train, valid = splitter(items)
        assert set(valid) == {0, 2, 4, 6, 8}
        assert set(train) == {1, 3, 5, 7, 9}

    def test_no_valid_items(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: False)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert set(train) == {0, 1, 2, 3, 4}

    def test_all_valid_items(self):
        items = list(range(5))
        splitter = FuncSplitter(lambda x: True)
        train, valid = splitter(items)
        assert set(valid) == {0, 1, 2, 3, 4}
        assert len(train) == 0


# ============================================================
# Tests for MaskSplitter
# ============================================================

class TestMaskSplitter:
    """Tests for the MaskSplitter function."""

    def test_basic_split(self):
        items = list(range(5))
        mask = [True, False, True, False, True]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert set(valid) == {0, 2, 4}
        assert set(train) == {1, 3}

    def test_all_false(self):
        items = list(range(5))
        mask = [False, False, False, False, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert set(train) == {0, 1, 2, 3, 4}

    def test_all_true(self):
        items = list(range(5))
        mask = [True, True, True, True, True]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert set(valid) == {0, 1, 2, 3, 4}
        assert len(train) == 0


# ============================================================
# Tests for FileSplitter
# ============================================================

class TestFileSplitter:
    """Tests for the FileSplitter function."""

    def test_basic_split(self, tmp_path):
        # Create a file listing valid filenames
        valid_file = tmp_path / "valid_files.txt"
        valid_file.write_text("img2.jpg\nimg3.jpg\n")

        # Create items as Path objects
        items = [
            tmp_path / "data" / "img1.jpg",
            tmp_path / "data" / "img2.jpg",
            tmp_path / "data" / "img3.jpg",
            tmp_path / "data" / "img4.jpg",
        ]

        splitter = FileSplitter(valid_file)
        train, valid = splitter(items)
        assert set(valid) == {1, 2}
        assert set(train) == {0, 3}

    def test_no_matches(self, tmp_path):
        valid_file = tmp_path / "valid_files.txt"
        valid_file.write_text("nonexistent.jpg\n")

        items = [tmp_path / "img1.jpg", tmp_path / "img2.jpg"]
        splitter = FileSplitter(valid_file)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert set(train) == {0, 1}


# ============================================================
# Tests for ColSplitter
# ============================================================

class TestColSplitter:
    """Tests for the ColSplitter function."""

    def test_basic_bool_column(self):
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd'],
            'is_valid': [False, True, False, True],
        })
        splitter = ColSplitter('is_valid')
        train, valid = splitter(df)
        assert set(valid) == {1, 3}
        assert set(train) == {0, 2}

    def test_integer_column_index(self):
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c'],
            'is_valid': [False, True, False],
        })
        splitter = ColSplitter(1)  # Column index 1
        train, valid = splitter(df)
        assert set(valid) == {1}
        assert set(train) == {0, 2}

    def test_with_on_parameter(self):
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd'],
            'split': ['train', 'valid', 'train', 'valid'],
        })
        splitter = ColSplitter('split', on='valid')
        train, valid = splitter(df)
        assert set(valid) == {1, 3}
        assert set(train) == {0, 2}

    def test_with_on_list(self):
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd'],
            'fold': [1, 2, 3, 2],
        })
        splitter = ColSplitter('fold', on=[2, 3])
        train, valid = splitter(df)
        assert set(valid) == {1, 2, 3}
        assert set(train) == {0}

    def test_non_dataframe_raises(self):
        splitter = ColSplitter('is_valid')
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
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=42)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=123)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_invalid_sizes_raise(self):
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.0, valid_sz=0.2)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.5, valid_sz=0.0)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.8, valid_sz=0.5)


# ============================================================
# Tests for parent_label
# ============================================================

class TestParentLabel:
    """Tests for the parent_label function."""

    def test_basic(self):
        assert parent_label(Path("/data/cats/img001.jpg")) == "cats"
        assert parent_label(Path("/data/dogs/img002.jpg")) == "dogs"

    def test_string_input(self):
        assert parent_label("/data/train/cats/img.jpg") == "cats"

    def test_nested_path(self):
        assert parent_label("/a/b/c/d/file.txt") == "d"


# ============================================================
# Tests for RegexLabeller
# ============================================================

class TestRegexLabeller:
    """Tests for the RegexLabeller class."""

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
            labeller("no_numbers_here")

    def test_group_capture(self):
        labeller = RegexLabeller(r'class_(\w+)_img')
        result = labeller("/path/class_dog_img_001.jpg")
        assert result == "dog"


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for the CategoryMap class."""

    def test_basic_creation(self):
        cats = CategoryMap(['cat', 'dog', 'bird'])
        assert len(cats) == 3
        assert 'cat' in cats.items
        assert 'dog' in cats.items
        assert 'bird' in cats.items

    def test_sorted_by_default(self):
        cats = CategoryMap(['dog', 'cat', 'bird'])
        # Should be sorted alphabetically
        assert list(cats.items) == ['bird', 'cat', 'dog']

    def test_unsorted(self):
        cats = CategoryMap(['dog', 'cat', 'bird'], sort=False)
        # unique() order may differ, but items should reflect unique values
        assert set(cats.items) == {'bird', 'cat', 'dog'}

    def test_o2i_mapping(self):
        cats = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        # Sorted: bird=0, cat=1, dog=2
        assert cats.o2i['bird'] == 0
        assert cats.o2i['cat'] == 1
        assert cats.o2i['dog'] == 2

    def test_add_na(self):
        cats = CategoryMap(['cat', 'dog'], add_na=True)
        assert '#na#' in cats.items
        # '#na#' should be first
        assert cats.items[0] == '#na#'

    def test_map_objs(self):
        cats = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        ids = cats.map_objs(['bird', 'dog'])
        assert list(ids) == [0, 2]

    def test_map_ids(self):
        cats = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        objs = cats.map_ids([0, 1, 2])
        assert list(objs) == ['bird', 'cat', 'dog']

    def test_equality(self):
        cats1 = CategoryMap(['cat', 'dog', 'bird'])
        cats2 = CategoryMap(['cat', 'dog', 'bird'])
        assert cats1 == cats2

    def test_duplicates_handled(self):
        cats = CategoryMap(['cat', 'cat', 'dog', 'dog', 'bird'])
        assert len(cats) == 3


# ============================================================
# Tests for Categorize
# ============================================================

class TestCategorize:
    """Tests for the Categorize transform."""

    def test_with_vocab(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        result = cat.encodes('cat')
        # bird=0, cat=1, dog=2 (sorted)
        assert int(result) == 1

    def test_encodes_decodes_roundtrip(self):
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        encoded = cat.encodes('dog')
        decoded = cat.decodes(encoded)
        assert str(decoded) == 'dog'

    def test_unknown_category_raises(self):
        cat = Categorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError):
            cat.encodes('bird')

    def test_setups_with_data(self):
        cat = Categorize()
        data = L(['cat', 'dog', 'cat', 'bird', 'dog'])
        cat.setups(data)
        assert cat.c == 3
        assert cat.vocab is not None
        result = cat.encodes('cat')
        assert isinstance(result, torch.Tensor)


# ============================================================
# Tests for MultiCategorize
# ============================================================

class TestMultiCategorize:
    """Tests for the MultiCategorize transform."""

    def test_with_vocab(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        result = mc.encodes(['cat', 'bird'])
        assert isinstance(result, torch.Tensor)
        assert len(result) == 2

    def test_encodes_decodes_roundtrip(self):
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        encoded = mc.encodes(['dog', 'bird'])
        decoded = mc.decodes(encoded)
        assert 'dog' in decoded
        assert 'bird' in decoded

    def test_unknown_category_raises(self):
        mc = MultiCategorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError):
            mc.encodes(['cat', 'elephant'])

    def test_setups_creates_vocab(self):
        mc = MultiCategorize()
        data = L([['cat', 'dog'], ['dog', 'bird'], ['cat']])
        mc.setups(data)
        assert 'cat' in mc.vocab.items
        assert 'dog' in mc.vocab.items
        assert 'bird' in mc.vocab.items


# ============================================================
# Tests for OneHotEncode
# ============================================================

class TestOneHotEncode:
    """Tests for the OneHotEncode transform."""

    def test_basic_encoding(self):
        ohe = OneHotEncode(c=4)
        # Input: tensor of category indices
        inp = torch.tensor([0, 2])
        result = ohe.encodes(inp)
        expected = torch.tensor([1., 0., 1., 0.])
        assert torch.allclose(result, expected)

    def test_single_category(self):
        ohe = OneHotEncode(c=3)
        inp = torch.tensor([1])
        result = ohe.encodes(inp)
        expected = torch.tensor([0., 1., 0.])
        assert torch.allclose(result, expected)

    def test_decodes(self):
        ohe = OneHotEncode(c=4)
        encoded = torch.tensor([1., 0., 1., 0.])
        decoded = ohe.decodes(encoded)
        assert 0 in decoded
        assert 2 in decoded


# ============================================================
# Tests for ItemGetter
# ============================================================

class TestItemGetter:
    """Tests for the ItemGetter transform."""

    def test_list_index(self):
        getter = ItemGetter(1)
        result = getter.encodes(['a', 'b', 'c'])
        assert result == 'b'

    def test_dict_key(self):
        getter = ItemGetter('name')
        result = getter.encodes({'name': 'Alice', 'age': 30})
        assert result == 'Alice'

    def test_tuple_index(self):
        getter = ItemGetter(0)
        result = getter.encodes(('first', 'second'))
        assert result == 'first'

    def test_negative_index(self):
        getter = ItemGetter(-1)
        result = getter.encodes([10, 20, 30])
        assert result == 30


# ============================================================
# Tests for AttrGetter
# ============================================================

class TestAttrGetter:
    """Tests for the AttrGetter transform."""

    def test_basic_attribute(self):
        getter = AttrGetter('x')
        obj = SimpleNamespace(x=42, y=99)
        result = getter.encodes(obj)
        assert result == 42

    def test_default_value(self):
        getter = AttrGetter('missing', default='N/A')
        obj = SimpleNamespace(x=42)
        result = getter.encodes(obj)
        assert result == 'N/A'

    def test_none_default(self):
        getter = AttrGetter('missing')
        obj = SimpleNamespace(x=42)
        result = getter.encodes(obj)
        assert result is None


# ============================================================
# Tests for IntToFloatTensor
# ============================================================

class TestIntToFloatTensor:
    """Tests for the IntToFloatTensor transform."""

    def test_default_div_255(self):
        from fastai.torch_basics import TensorImage
        transform = IntToFloatTensor()
        inp = TensorImage(torch.tensor([0, 128, 255], dtype=torch.uint8))
        result = transform.encodes(inp)
        assert result.dtype == torch.float32
        assert abs(float(result[0]) - 0.0) < 1e-5
        assert abs(float(result[2]) - 1.0) < 1e-5

    def test_custom_div(self):
        from fastai.torch_basics import TensorImage
        transform = IntToFloatTensor(div=128.)
        inp = TensorImage(torch.tensor([0, 64, 128], dtype=torch.uint8))
        result = transform.encodes(inp)
        assert abs(float(result[1]) - 0.5) < 1e-5
        assert abs(float(result[2]) - 1.0) < 1e-5

    def test_decodes_roundtrip(self):
        from fastai.torch_basics import TensorImage
        transform = IntToFloatTensor()
        inp = TensorImage(torch.tensor([0, 128, 255], dtype=torch.uint8))
        encoded = transform.encodes(inp)
        decoded = transform.decodes(encoded)
        # Should approximately recover original values
        assert abs(int(decoded[0]) - 0) <= 1
        assert abs(int(decoded[2]) - 255) <= 1


# ============================================================
# Tests for broadcast_vec
# ============================================================

class TestBroadcastVec:
    """Tests for the broadcast_vec helper function."""

    def test_basic_shape(self):
        # For a 4D tensor (batch, channel, height, width),
        # broadcasting over dim=1 (channels) should produce shape [1, -1, 1, 1]
        result = broadcast_vec(1, 4, [1.0, 2.0, 3.0], cuda=False)
        assert len(result) == 1
        assert result[0].shape == (1, 3, 1, 1)

    def test_multiple_inputs(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        results = broadcast_vec(1, 4, mean, std, cuda=False)
        assert len(results) == 2
        assert results[0].shape == (1, 3, 1, 1)
        assert results[1].shape == (1, 3, 1, 1)

    def test_dim_0(self):
        result = broadcast_vec(0, 3, [1.0, 2.0], cuda=False)
        # Shape should be [-1, 1, 1] = [2, 1, 1]
        assert result[0].shape == (2, 1, 1)


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
        mean = torch.tensor([0.5]).reshape(1, 1, 1, 1)
        std = torch.tensor([0.25]).reshape(1, 1, 1, 1)
        norm = Normalize(mean=mean, std=std)

        inp = TensorImage(torch.rand(1, 1, 4, 4))
        encoded = norm.encodes(inp)
        decoded = norm.decodes(encoded)
        assert torch.allclose(inp, decoded, atol=1e-5)

    def test_normalized_output_range(self):
        from fastai.torch_basics import TensorImage
        mean = torch.tensor([0.5]).reshape(1, 1, 1, 1)
        std = torch.tensor([0.5]).reshape(1, 1, 1, 1)
        norm = Normalize(mean=mean, std=std)

        # Input uniformly between 0 and 1
        inp = TensorImage(torch.tensor([[[[0.0, 0.5, 1.0]]]]))
        result = norm.encodes(inp)
        # (0-0.5)/0.5 = -1, (0.5-0.5)/0.5 = 0, (1-0.5)/0.5 = 1
        expected = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
        assert torch.allclose(result, expected, atol=1e-5)


# ============================================================
# Tests for image_extensions
# ============================================================

class TestImageExtensions:
    """Tests for the image_extensions constant."""

    def test_contains_common_formats(self):
        assert '.jpg' in image_extensions or '.jpeg' in image_extensions
        assert '.png' in image_extensions
        assert '.gif' in image_extensions

    def test_does_not_contain_non_image(self):
        assert '.txt' not in image_extensions
        assert '.py' not in image_extensions
        assert '.html' not in image_extensions

    def test_is_set(self):
        assert isinstance(image_extensions, set)
