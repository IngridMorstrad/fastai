"""Tests for fastai/data/transforms.py module.

Covers: splitters (RandomSplitter, IndexSplitter, EndSplitter, GrandparentSplitter,
FuncSplitter, MaskSplitter, FileSplitter, ColSplitter, RandomSubsetSplitter,
TrainTestSplitter), labeling functions (parent_label, RegexLabeller),
file getter functions (get_files, get_image_files, get_text_files, FileGetter,
ImageGetter), and utility transforms (ItemGetter, AttrGetter, ColReader,
CategoryMap, Categorize).
"""
import sys
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files, FileGetter, get_image_files, ImageGetter, get_text_files,
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter,
    ColSplitter, RandomSubsetSplitter,
    parent_label, RegexLabeller,
    ItemGetter, AttrGetter, ColReader, CategoryMap,
)


# ============================================================
# Helper fixtures
# ============================================================

@pytest.fixture
def tmp_file_tree(tmp_path):
    """Create a temporary directory with a simple file structure for testing."""
    # Root level files
    (tmp_path / "image1.jpg").touch()
    (tmp_path / "image2.png").touch()
    (tmp_path / "doc.txt").touch()
    (tmp_path / "notes.md").touch()
    (tmp_path / ".hidden.txt").touch()

    # Subdirectory
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "image3.jpeg").touch()
    (sub / "data.txt").touch()
    (sub / "script.py").touch()

    # Nested subdirectory
    nested = sub / "deep"
    nested.mkdir()
    (nested / "image4.gif").touch()
    (nested / "readme.txt").touch()

    return tmp_path


@pytest.fixture
def grandparent_tree(tmp_path):
    """Create a directory structure suitable for GrandparentSplitter tests."""
    # /tmp/train/cats/img1.jpg, /tmp/train/dogs/img2.jpg
    # /tmp/valid/cats/img3.jpg, /tmp/valid/dogs/img4.jpg
    for split in ["train", "valid"]:
        for cls in ["cats", "dogs"]:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            (d / f"{split}_{cls}_img.jpg").touch()
    return tmp_path


# ============================================================
# Tests for get_files
# ============================================================

class TestGetFiles:
    """Tests for the get_files function."""

    def test_gets_all_files_recursively(self, tmp_file_tree):
        files = get_files(tmp_file_tree)
        names = sorted([f.name for f in files])
        # Should not include hidden files (starting with .)
        assert ".hidden.txt" not in names
        # Should include all non-hidden files recursively
        assert "image1.jpg" in names
        assert "image3.jpeg" in names
        assert "image4.gif" in names
        assert "readme.txt" in names

    def test_excludes_hidden_files(self, tmp_file_tree):
        files = get_files(tmp_file_tree)
        names = [f.name for f in files]
        assert ".hidden.txt" not in names

    def test_filters_by_extension(self, tmp_file_tree):
        files = get_files(tmp_file_tree, extensions=['.txt'])
        names = sorted([f.name for f in files])
        assert all(f.endswith('.txt') for f in names)
        assert "doc.txt" in names
        assert "data.txt" in names
        assert "readme.txt" in names
        assert "image1.jpg" not in names

    def test_extension_case_insensitive(self, tmp_file_tree):
        # Create a file with uppercase extension
        (tmp_file_tree / "photo.JPG").touch()
        files = get_files(tmp_file_tree, extensions=['.jpg'])
        names = [f.name for f in files]
        assert "photo.JPG" in names
        assert "image1.jpg" in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_files(tmp_file_tree, recurse=False)
        names = sorted([f.name for f in files])
        # Should only have root-level files (not in subdir)
        assert "image1.jpg" in names
        assert "doc.txt" in names
        # Should not have files from subdirectories
        assert "image3.jpeg" not in names
        assert "image4.gif" not in names

    def test_specific_folders(self, tmp_file_tree):
        files = get_files(tmp_file_tree, folders=["subdir"])
        names = [f.name for f in files]
        # Should include files from subdir and its children
        assert "image3.jpeg" in names
        assert "data.txt" in names
        # Should not include root-level files
        assert "image1.jpg" not in names

    def test_empty_directory(self, tmp_path):
        files = get_files(tmp_path)
        assert len(files) == 0

    def test_returns_L_type(self, tmp_file_tree):
        from fastcore.foundation import L
        files = get_files(tmp_file_tree)
        assert isinstance(files, L)

    def test_multiple_extensions(self, tmp_file_tree):
        files = get_files(tmp_file_tree, extensions=['.txt', '.md'])
        names = [f.name for f in files]
        assert "doc.txt" in names
        assert "notes.md" in names
        assert "image1.jpg" not in names


# ============================================================
# Tests for get_image_files
# ============================================================

class TestGetImageFiles:
    """Tests for the get_image_files function."""

    def test_finds_image_files(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree)
        names = sorted([f.name for f in files])
        assert "image1.jpg" in names
        assert "image2.png" in names
        assert "image3.jpeg" in names
        assert "image4.gif" in names

    def test_excludes_non_image_files(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree)
        names = [f.name for f in files]
        assert "doc.txt" not in names
        assert "script.py" not in names
        assert "notes.md" not in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree, recurse=False)
        names = [f.name for f in files]
        assert "image1.jpg" in names
        assert "image2.png" in names
        assert "image3.jpeg" not in names

    def test_specific_folders(self, tmp_file_tree):
        files = get_image_files(tmp_file_tree, folders=["subdir"])
        names = [f.name for f in files]
        assert "image3.jpeg" in names
        assert "image1.jpg" not in names


# ============================================================
# Tests for get_text_files
# ============================================================

class TestGetTextFiles:
    """Tests for the get_text_files function."""

    def test_finds_txt_files(self, tmp_file_tree):
        files = get_text_files(tmp_file_tree)
        names = sorted([f.name for f in files])
        assert "doc.txt" in names
        assert "data.txt" in names
        assert "readme.txt" in names

    def test_excludes_non_txt_files(self, tmp_file_tree):
        files = get_text_files(tmp_file_tree)
        names = [f.name for f in files]
        assert "image1.jpg" not in names
        assert "notes.md" not in names
        assert "script.py" not in names

    def test_no_recurse(self, tmp_file_tree):
        files = get_text_files(tmp_file_tree, recurse=False)
        names = [f.name for f in files]
        assert "doc.txt" in names
        assert "data.txt" not in names  # in subdir


# ============================================================
# Tests for FileGetter and ImageGetter
# ============================================================

class TestFileGetter:
    """Tests for the FileGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = FileGetter()
        files = getter(tmp_file_tree)
        assert len(files) > 0

    def test_with_suffix(self, tmp_file_tree):
        getter = FileGetter(suf="subdir")
        files = getter(tmp_file_tree)
        names = [f.name for f in files]
        assert "image3.jpeg" in names
        assert "image1.jpg" not in names

    def test_with_extensions(self, tmp_file_tree):
        getter = FileGetter(extensions=['.txt'])
        files = getter(tmp_file_tree)
        assert all(f.suffix == '.txt' for f in files)


class TestImageGetter:
    """Tests for the ImageGetter factory function."""

    def test_basic_usage(self, tmp_file_tree):
        getter = ImageGetter()
        files = getter(tmp_file_tree)
        names = [f.name for f in files]
        assert "image1.jpg" in names
        assert "doc.txt" not in names

    def test_with_suffix(self, tmp_file_tree):
        getter = ImageGetter(suf="subdir")
        files = getter(tmp_file_tree)
        names = [f.name for f in files]
        assert "image3.jpeg" in names
        assert "image1.jpg" not in names


# ============================================================
# Tests for RandomSplitter
# ============================================================

class TestRandomSplitter:
    """Tests for the RandomSplitter function."""

    def test_default_split_ratio(self):
        items = list(range(100))
        train, val = RandomSplitter(seed=42)(items)
        assert len(train) == 80
        assert len(val) == 20

    def test_custom_split_ratio(self):
        items = list(range(100))
        train, val = RandomSplitter(valid_pct=0.3, seed=42)(items)
        assert len(train) == 70
        assert len(val) == 30

    def test_no_overlap_between_splits(self):
        items = list(range(50))
        train, val = RandomSplitter(seed=42)(items)
        train_set = set(int(x) for x in train)
        val_set = set(int(x) for x in val)
        assert train_set.isdisjoint(val_set)

    def test_all_indices_covered(self):
        items = list(range(50))
        train, val = RandomSplitter(seed=42)(items)
        all_idxs = sorted([int(x) for x in train] + [int(x) for x in val])
        assert all_idxs == list(range(50))

    def test_reproducibility_with_seed(self):
        items = list(range(100))
        train1, val1 = RandomSplitter(seed=123)(items)
        train2, val2 = RandomSplitter(seed=123)(items)
        assert list(train1) == list(train2)
        assert list(val1) == list(val2)

    def test_different_seeds_give_different_splits(self):
        items = list(range(100))
        train1, _ = RandomSplitter(seed=1)(items)
        train2, _ = RandomSplitter(seed=2)(items)
        # Very unlikely to be the same
        assert list(train1) != list(train2)

    def test_single_item(self):
        items = [0]
        train, val = RandomSplitter(valid_pct=0.5, seed=42)(items)
        # With 1 item and 0.5, cut=0, so train has all items
        assert len(train) + len(val) == 1


# ============================================================
# Tests for TrainTestSplitter
# ============================================================

class TestTrainTestSplitter:
    """Tests for the TrainTestSplitter function (sklearn-based)."""

    def test_default_split(self):
        items = list(range(100))
        train, val = TrainTestSplitter(random_state=42)(items)
        assert len(train) == 80
        assert len(val) == 20

    def test_custom_test_size(self):
        items = list(range(100))
        train, val = TrainTestSplitter(test_size=0.3, random_state=42)(items)
        assert len(train) == 70
        assert len(val) == 30

    def test_no_overlap(self):
        items = list(range(50))
        train, val = TrainTestSplitter(random_state=42)(items)
        assert set(train).isdisjoint(set(val))

    def test_all_indices_covered(self):
        items = list(range(50))
        train, val = TrainTestSplitter(random_state=42)(items)
        assert sorted(list(train) + list(val)) == list(range(50))

    def test_reproducibility(self):
        items = list(range(100))
        train1, val1 = TrainTestSplitter(random_state=42)(items)
        train2, val2 = TrainTestSplitter(random_state=42)(items)
        assert list(train1) == list(train2)
        assert list(val1) == list(val2)


# ============================================================
# Tests for IndexSplitter
# ============================================================

class TestIndexSplitter:
    """Tests for the IndexSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        train, val = IndexSplitter([2, 5, 7])(items)
        assert sorted(val) == [2, 5, 7]
        assert sorted([int(x) for x in train]) == [0, 1, 3, 4, 6, 8, 9]

    def test_empty_validation(self):
        items = list(range(5))
        train, val = IndexSplitter([])(items)
        assert len(val) == 0
        assert sorted([int(x) for x in train]) == [0, 1, 2, 3, 4]

    def test_all_in_validation(self):
        items = list(range(5))
        train, val = IndexSplitter([0, 1, 2, 3, 4])(items)
        assert sorted(val) == [0, 1, 2, 3, 4]
        assert len(train) == 0

    def test_no_overlap(self):
        items = list(range(20))
        valid_idx = [3, 7, 11, 15]
        train, val = IndexSplitter(valid_idx)(items)
        train_set = set(int(x) for x in train)
        val_set = set(int(x) for x in val)
        assert train_set.isdisjoint(val_set)

    def test_single_validation_index(self):
        items = list(range(10))
        train, val = IndexSplitter([5])(items)
        assert list(val) == [5]
        assert 5 not in [int(x) for x in train]


# ============================================================
# Tests for EndSplitter
# ============================================================

class TestEndSplitter:
    """Tests for the EndSplitter function."""

    def test_valid_last_default(self):
        items = list(range(10))
        train, val = EndSplitter(valid_pct=0.2)(items)
        assert list(train) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert list(val) == [8, 9]

    def test_valid_first(self):
        items = list(range(10))
        train, val = EndSplitter(valid_pct=0.2, valid_last=False)(items)
        assert list(train) == [2, 3, 4, 5, 6, 7, 8, 9]
        assert list(val) == [0, 1]

    def test_custom_pct(self):
        items = list(range(20))
        train, val = EndSplitter(valid_pct=0.5)(items)
        assert len(train) == 10
        assert len(val) == 10

    def test_preserves_order(self):
        items = list(range(10))
        train, val = EndSplitter(valid_pct=0.3)(items)
        # Train indices should be consecutive from start
        assert list(train) == list(range(7))
        # Val indices should be consecutive at end
        assert list(val) == [7, 8, 9]

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)

    def test_no_overlap_and_complete(self):
        items = list(range(50))
        train, val = EndSplitter(valid_pct=0.4)(items)
        all_idxs = list(train) + list(val)
        assert sorted(all_idxs) == list(range(50))


# ============================================================
# Tests for GrandparentSplitter
# ============================================================

class TestGrandparentSplitter:
    """Tests for the GrandparentSplitter function."""

    def test_basic_split(self, grandparent_tree):
        items = sorted(grandparent_tree.rglob("*.jpg"))
        train, val = GrandparentSplitter(train_name='train', valid_name='valid')(items)
        # Items whose grandparent is 'train'
        for idx in train:
            assert items[idx].parent.parent.name == 'train'
        # Items whose grandparent is 'valid'
        for idx in val:
            assert items[idx].parent.parent.name == 'valid'

    def test_correct_counts(self, grandparent_tree):
        items = sorted(grandparent_tree.rglob("*.jpg"))
        train, val = GrandparentSplitter(train_name='train', valid_name='valid')(items)
        # 2 train items (cats + dogs), 2 valid items
        assert len(train) == 2
        assert len(val) == 2

    def test_custom_names(self, tmp_path):
        # Create a structure with custom split names
        for split in ["trn", "tst"]:
            d = tmp_path / split / "class1"
            d.mkdir(parents=True)
            (d / "file.txt").touch()
        items = sorted(tmp_path.rglob("*.txt"))
        train, val = GrandparentSplitter(train_name='trn', valid_name='tst')(items)
        assert len(train) == 1
        assert len(val) == 1


# ============================================================
# Tests for FuncSplitter
# ============================================================

class TestFuncSplitter:
    """Tests for the FuncSplitter function."""

    def test_basic_split(self):
        items = list(range(10))
        # Items >= 8 go to validation
        train, val = FuncSplitter(lambda x: x >= 8)(items)
        assert sorted(val) == [8, 9]
        assert sorted([int(x) for x in train]) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_all_train(self):
        items = list(range(10))
        train, val = FuncSplitter(lambda x: False)(items)
        assert len(val) == 0
        assert len(train) == 10

    def test_all_valid(self):
        items = list(range(10))
        train, val = FuncSplitter(lambda x: True)(items)
        assert len(train) == 0
        assert len(val) == 10

    def test_string_items(self):
        items = ["train_a", "train_b", "valid_c", "valid_d"]
        train, val = FuncSplitter(lambda x: x.startswith("valid"))(items)
        assert sorted(val) == [2, 3]
        assert sorted([int(x) for x in train]) == [0, 1]


# ============================================================
# Tests for MaskSplitter
# ============================================================

class TestMaskSplitter:
    """Tests for the MaskSplitter function."""

    def test_basic_mask(self):
        items = list(range(5))
        mask = [True, False, True, False, True]
        train, val = MaskSplitter(mask)(items)
        # True in mask = validation
        assert sorted(val) == [0, 2, 4]
        assert sorted([int(x) for x in train]) == [1, 3]

    def test_all_false(self):
        items = list(range(5))
        mask = [False, False, False, False, False]
        train, val = MaskSplitter(mask)(items)
        assert len(val) == 0
        assert len(train) == 5

    def test_all_true(self):
        items = list(range(5))
        mask = [True, True, True, True, True]
        train, val = MaskSplitter(mask)(items)
        assert len(val) == 5
        assert len(train) == 0


# ============================================================
# Tests for FileSplitter
# ============================================================

class TestFileSplitter:
    """Tests for the FileSplitter function."""

    def test_basic_split(self, tmp_path):
        # Create item files
        items = [tmp_path / f"file{i}.txt" for i in range(5)]
        for f in items:
            f.touch()

        # Create a file listing valid items
        valid_file = tmp_path / "valid_names.txt"
        valid_file.write_text("file1.txt\nfile3.txt")

        train, val = FileSplitter(valid_file)(items)
        # file1.txt (idx=1) and file3.txt (idx=3) are validation
        val_names = [items[i].name for i in val]
        assert sorted(val_names) == ["file1.txt", "file3.txt"]
        train_names = [items[i].name for i in train]
        assert "file0.txt" in train_names
        assert "file2.txt" in train_names
        assert "file4.txt" in train_names

    def test_empty_valid_file(self, tmp_path):
        items = [tmp_path / f"file{i}.txt" for i in range(3)]
        for f in items:
            f.touch()

        valid_file = tmp_path / "valid_names.txt"
        valid_file.write_text("")

        train, val = FileSplitter(valid_file)(items)
        assert len(val) == 0
        assert len(train) == 3


# ============================================================
# Tests for ColSplitter
# ============================================================

class TestColSplitter:
    """Tests for the ColSplitter function."""

    def test_boolean_column(self):
        df = pd.DataFrame({
            'data': [10, 20, 30, 40, 50],
            'is_valid': [False, False, True, False, True]
        })
        train, val = ColSplitter('is_valid')(df)
        assert sorted(val) == [2, 4]
        assert sorted([int(x) for x in train]) == [0, 1, 3]

    def test_integer_column_index(self):
        df = pd.DataFrame({
            'data': [10, 20, 30, 40, 50],
            'is_valid': [False, True, False, True, False]
        })
        # Column 'is_valid' is at index 1
        train, val = ColSplitter(1)(df)
        assert sorted(val) == [1, 3]
        assert sorted([int(x) for x in train]) == [0, 2, 4]

    def test_with_on_parameter_single_value(self):
        df = pd.DataFrame({
            'data': [10, 20, 30, 40, 50],
            'split': ['train', 'train', 'valid', 'train', 'valid']
        })
        train, val = ColSplitter('split', on='valid')(df)
        assert sorted(val) == [2, 4]
        assert sorted([int(x) for x in train]) == [0, 1, 3]

    def test_with_on_parameter_list(self):
        df = pd.DataFrame({
            'data': list(range(6)),
            'fold': [0, 1, 2, 0, 1, 2]
        })
        # Use folds 1 and 2 as validation
        train, val = ColSplitter('fold', on=[1, 2])(df)
        assert sorted(val) == [1, 2, 4, 5]
        assert sorted([int(x) for x in train]) == [0, 3]

    def test_requires_dataframe(self):
        with pytest.raises(AssertionError, match="ColSplitter only works"):
            ColSplitter('is_valid')([1, 2, 3])


# ============================================================
# Tests for RandomSubsetSplitter
# ============================================================

class TestRandomSubsetSplitter:
    """Tests for the RandomSubsetSplitter function."""

    def test_correct_sizes(self):
        items = list(range(100))
        train, val = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.2, seed=42)(items)
        assert len(train) == 50
        assert len(val) == 20

    def test_no_overlap(self):
        items = list(range(100))
        train, val = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.2, seed=42)(items)
        train_set = set(int(x) for x in train)
        val_set = set(int(x) for x in val)
        assert train_set.isdisjoint(val_set)

    def test_reproducibility_with_seed(self):
        items = list(range(100))
        train1, val1 = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=99)(items)
        train2, val2 = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=99)(items)
        assert list(train1) == list(train2)
        assert list(val1) == list(val2)

    def test_invalid_sizes_raise(self):
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.0, valid_sz=0.2)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.5, valid_sz=0.0)
        with pytest.raises(AssertionError):
            RandomSubsetSplitter(train_sz=0.8, valid_sz=0.3)  # sum > 1

    def test_subset_not_covering_all(self):
        # Train + valid < 1.0, so some items are not in either set
        items = list(range(100))
        train, val = RandomSubsetSplitter(train_sz=0.3, valid_sz=0.2, seed=42)(items)
        assert len(train) + len(val) < len(items)


# ============================================================
# Tests for parent_label
# ============================================================

class TestParentLabel:
    """Tests for the parent_label function."""

    def test_basic_labeling(self):
        assert parent_label(Path("/data/cats/img001.jpg")) == "cats"
        assert parent_label(Path("/data/dogs/img002.jpg")) == "dogs"

    def test_string_input(self):
        assert parent_label("/data/cats/img001.jpg") == "cats"

    def test_nested_path(self):
        assert parent_label(Path("/root/train/birds/pic.png")) == "birds"

    def test_relative_path(self):
        assert parent_label(Path("train/cats/image.jpg")) == "cats"


# ============================================================
# Tests for RegexLabeller
# ============================================================

class TestRegexLabeller:
    """Tests for the RegexLabeller class."""

    def test_search_mode(self):
        labeller = RegexLabeller(r'/([^/]+)/[^/]+$')
        assert labeller(Path("/data/cats/img001.jpg")) == "cats"
        assert labeller(Path("/data/dogs/img002.jpg")) == "dogs"

    def test_match_mode(self):
        labeller = RegexLabeller(r'(\w+)_\d+', match=True)
        assert labeller("cat_001") == "cat"
        assert labeller("dog_123") == "dog"

    def test_no_match_raises(self):
        labeller = RegexLabeller(r'impossible_pattern_(\d+)')
        with pytest.raises(AssertionError):
            labeller("no_match_here.jpg")

    def test_captures_group(self):
        labeller = RegexLabeller(r'class_(\d+)')
        assert labeller("file_class_42_img.jpg") == "42"

    def test_path_separator_normalization(self):
        # On all platforms, the path separator is normalized to /
        labeller = RegexLabeller(r'/([^/]+)/[^/]+$')
        # Even with Path (which uses OS-specific separator), regex should work
        result = labeller(Path("/data/cats/img.jpg"))
        assert result == "cats"


# ============================================================
# Tests for ItemGetter
# ============================================================

class TestItemGetter:
    """Tests for the ItemGetter transform."""

    def test_list_indexing(self):
        getter = ItemGetter(1)
        assert getter(['a', 'b', 'c']) == 'b'

    def test_tuple_indexing(self):
        getter = ItemGetter(0)
        assert getter(('first', 'second')) == 'first'

    def test_dict_indexing(self):
        getter = ItemGetter('key')
        assert getter({'key': 'value', 'other': 'data'}) == 'value'

    def test_negative_index(self):
        getter = ItemGetter(-1)
        assert getter([1, 2, 3]) == 3


# ============================================================
# Tests for AttrGetter
# ============================================================

class TestAttrGetter:
    """Tests for the AttrGetter transform."""

    def test_basic_attr(self):
        obj = SimpleNamespace(name="test", value=42)
        getter = AttrGetter('name')
        assert getter(obj) == "test"

    def test_missing_attr_returns_default(self):
        obj = SimpleNamespace(name="test")
        getter = AttrGetter('missing', default='default_val')
        assert getter(obj) == 'default_val'

    def test_missing_attr_no_default(self):
        obj = SimpleNamespace(name="test")
        getter = AttrGetter('missing')
        assert getter(obj) is None


# ============================================================
# Tests for ColReader
# ============================================================

class TestColReader:
    """Tests for the ColReader transform."""

    def test_single_column_by_name(self):
        row = pd.Series({'fname': 'img.jpg', 'label': 'cat'})
        reader = ColReader('fname')
        assert reader(row) == 'img.jpg'

    def test_with_prefix_and_suffix(self):
        row = pd.Series({'fname': 'img', 'label': 'cat'})
        reader = ColReader('fname', pref='data/', suff='.jpg')
        assert reader(row) == 'data/img.jpg'

    def test_with_label_delim(self):
        row = pd.Series({'labels': 'cat;dog;bird', 'fname': 'img.jpg'})
        reader = ColReader('labels', label_delim=';')
        result = reader(row)
        assert result == ['cat', 'dog', 'bird']

    def test_empty_string_with_label_delim(self):
        row = pd.Series({'labels': '', 'fname': 'img.jpg'})
        reader = ColReader('labels', label_delim=';')
        result = reader(row)
        assert result == []

    def test_multiple_columns(self):
        row = pd.Series({'a': 1, 'b': 2, 'c': 3})
        reader = ColReader(['a', 'c'])
        result = reader(row)
        assert list(result) == [1, 3]


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for the CategoryMap collection."""

    def test_basic_creation(self):
        cm = CategoryMap(['cat', 'dog', 'bird'])
        assert len(cm) == 3
        assert 'cat' in cm.items
        assert 'dog' in cm.items
        assert 'bird' in cm.items

    def test_sorted_by_default(self):
        cm = CategoryMap(['dog', 'cat', 'bird'])
        # Items should be sorted
        assert list(cm.items) == ['bird', 'cat', 'dog']

    def test_o2i_mapping(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        # Sorted: bird=0, cat=1, dog=2
        assert cm.o2i['bird'] == 0
        assert cm.o2i['cat'] == 1
        assert cm.o2i['dog'] == 2

    def test_add_na(self):
        cm = CategoryMap(['cat', 'dog'], add_na=True)
        assert '#na#' in cm.items
        # '#na#' should be at index 0
        assert cm.o2i['#na#'] == 0

    def test_map_objs(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        ids = cm.map_objs(['cat', 'bird', 'dog'])
        assert list(ids) == [1, 0, 2]

    def test_map_ids(self):
        cm = CategoryMap(['cat', 'dog', 'bird'], sort=True)
        objs = cm.map_ids([0, 1, 2])
        assert list(objs) == ['bird', 'cat', 'dog']

    def test_with_duplicates(self):
        cm = CategoryMap(['cat', 'dog', 'cat', 'bird', 'dog'])
        # unique items only
        assert len(cm) == 3

    def test_equality(self):
        cm1 = CategoryMap(['cat', 'dog', 'bird'])
        cm2 = CategoryMap(['cat', 'dog', 'bird'])
        assert cm1 == cm2

    def test_from_pandas_series(self):
        s = pd.Series(['cat', 'dog', 'cat', 'bird'])
        cm = CategoryMap(s)
        assert len(cm) == 3
        assert 'cat' in cm.items
