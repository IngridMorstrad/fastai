"""Tests for fastai.data.transforms module.

Covers file discovery (get_files, get_image_files, get_text_files), splitters
(RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter, GrandparentSplitter,
FuncSplitter, MaskSplitter, FileSplitter, ColSplitter, RandomSubsetSplitter),
labelling (parent_label, RegexLabeller, ColReader), category transforms
(CategoryMap, Categorize, MultiCategorize, OneHotEncode), tensor transforms
(IntToFloatTensor, broadcast_vec, Normalize), and helpers (ItemGetter, AttrGetter).
"""
import sys
import os
import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.transforms import (
    get_files, get_image_files, get_text_files,
    RandomSplitter, TrainTestSplitter, IndexSplitter, EndSplitter,
    GrandparentSplitter, FuncSplitter, MaskSplitter, FileSplitter, ColSplitter,
    RandomSubsetSplitter,
    parent_label, RegexLabeller, ColReader,
    CategoryMap, Categorize, MultiCategorize, OneHotEncode,
    IntToFloatTensor, broadcast_vec, Normalize,
    ItemGetter, AttrGetter, FileGetter, ImageGetter,
    RegressionSetup, get_c, ToTensor,
)
from fastai.torch_core import TensorImage, TensorMask, TensorCategory, TensorMultiCategory


# ============================================================
# Tests for file discovery functions
# ============================================================

class TestGetFiles:
    """Tests for get_files, get_image_files, get_text_files."""

    @pytest.fixture
    def file_tree(self, tmp_path):
        """Create a temporary directory tree with various files."""
        # Root files
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.py").write_text("code")
        (tmp_path / "image1.jpg").write_text("img")
        (tmp_path / "image2.png").write_text("img")
        # Subdirectory
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "file3.txt").write_text("world")
        (sub / "image3.jpeg").write_text("img")
        (sub / "data.csv").write_text("a,b")
        # Hidden file (should be excluded)
        (tmp_path / ".hidden").write_text("hidden")
        # Hidden directory
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "secret.txt").write_text("secret")
        return tmp_path

    def test_get_files_no_extensions(self, file_tree):
        """get_files without extensions returns all non-hidden files."""
        files = get_files(file_tree, recurse=True)
        names = {f.name for f in files}
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "image1.jpg" in names
        assert "file3.txt" in names
        assert ".hidden" not in names
        assert "secret.txt" not in names

    def test_get_files_with_extension_filter(self, file_tree):
        """get_files with extensions filters correctly."""
        files = get_files(file_tree, extensions=['.txt'], recurse=True)
        names = {f.name for f in files}
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "file2.py" not in names
        assert "image1.jpg" not in names

    def test_get_files_no_recurse(self, file_tree):
        """get_files without recursion only gets top-level files."""
        files = get_files(file_tree, recurse=False)
        names = {f.name for f in files}
        assert "file1.txt" in names
        assert "file3.txt" not in names  # in subdirectory

    def test_get_files_specific_folders(self, file_tree):
        """get_files with folders parameter limits to specific subdirectories."""
        files = get_files(file_tree, recurse=True, folders=['subdir'])
        names = {f.name for f in files}
        assert "file3.txt" in names
        assert "image3.jpeg" in names
        # Files in root should not be included
        assert "file1.txt" not in names

    def test_get_files_empty_directory(self, tmp_path):
        """get_files on empty directory returns empty list."""
        empty = tmp_path / "empty"
        empty.mkdir()
        files = get_files(empty, recurse=True)
        assert len(files) == 0

    def test_get_image_files(self, file_tree):
        """get_image_files returns only image files."""
        files = get_image_files(file_tree, recurse=True)
        names = {f.name for f in files}
        assert "image1.jpg" in names
        assert "image2.png" in names
        assert "image3.jpeg" in names
        assert "file1.txt" not in names
        assert "file2.py" not in names

    def test_get_text_files(self, file_tree):
        """get_text_files returns only .txt files."""
        files = get_text_files(file_tree, recurse=True)
        names = {f.name for f in files}
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "image1.jpg" not in names
        assert "data.csv" not in names

    def test_get_files_case_insensitive_extensions(self, tmp_path):
        """Extensions should be matched case-insensitively."""
        (tmp_path / "upper.TXT").write_text("data")
        (tmp_path / "mixed.Txt").write_text("data")
        (tmp_path / "lower.txt").write_text("data")
        files = get_files(tmp_path, extensions=['.txt'], recurse=False)
        names = {f.name for f in files}
        assert "upper.TXT" in names
        assert "mixed.Txt" in names
        assert "lower.txt" in names


class TestFileGetter:
    """Tests for FileGetter and ImageGetter."""

    def test_file_getter_basic(self, tmp_path):
        """FileGetter creates a callable that gets files."""
        (tmp_path / "a.txt").write_text("hi")
        (tmp_path / "b.py").write_text("code")
        getter = FileGetter(extensions=['.txt'])
        files = getter(tmp_path)
        names = {f.name for f in files}
        assert "a.txt" in names
        assert "b.py" not in names

    def test_file_getter_with_suffix(self, tmp_path):
        """FileGetter with suf appends suffix to path."""
        sub = tmp_path / "data"
        sub.mkdir()
        (sub / "file.txt").write_text("hi")
        getter = FileGetter(suf='data', extensions=['.txt'])
        files = getter(tmp_path)
        assert len(files) == 1
        assert files[0].name == "file.txt"

    def test_image_getter(self, tmp_path):
        """ImageGetter finds only image files."""
        (tmp_path / "pic.jpg").write_text("img")
        (tmp_path / "doc.txt").write_text("text")
        getter = ImageGetter()
        files = getter(tmp_path)
        names = {f.name for f in files}
        assert "pic.jpg" in names
        assert "doc.txt" not in names


# ============================================================
# Tests for splitter functions
# ============================================================

class TestRandomSplitter:
    """Tests for RandomSplitter."""

    def test_basic_split(self):
        """RandomSplitter splits items into train and validation sets."""
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) + len(valid_idx) == 100
        assert len(valid_idx) == 20
        assert len(train_idx) == 80

    def test_no_overlap(self):
        """Train and validation indices should not overlap."""
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.3, seed=42)
        train_idx, valid_idx = splitter(items)
        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_reproducible_with_seed(self):
        """Same seed produces same split."""
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=42)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=42)
        train1, valid1 = splitter1(items)
        train2, valid2 = splitter2(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should produce different splits."""
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=42)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=99)
        train1, _ = splitter1(items)
        train2, _ = splitter2(items)
        assert list(train1) != list(train2)

    def test_valid_pct_bounds(self):
        """valid_pct controls the proportion of validation data."""
        items = list(range(1000))
        splitter = RandomSplitter(valid_pct=0.5, seed=42)
        train_idx, valid_idx = splitter(items)
        assert len(valid_idx) == 500
        assert len(train_idx) == 500


class TestTrainTestSplitter:
    """Tests for TrainTestSplitter (sklearn-based)."""

    def test_basic_split(self):
        """TrainTestSplitter creates train/test split."""
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) + len(valid_idx) == 100
        assert len(valid_idx) == 20

    def test_reproducible(self):
        """Same random_state produces same split."""
        items = list(range(100))
        splitter1 = TrainTestSplitter(test_size=0.3, random_state=42)
        splitter2 = TrainTestSplitter(test_size=0.3, random_state=42)
        train1, valid1 = splitter1(items)
        train2, valid2 = splitter2(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)


class TestIndexSplitter:
    """Tests for IndexSplitter."""

    def test_basic_split(self):
        """IndexSplitter puts specified indices in validation."""
        items = list(range(10))
        splitter = IndexSplitter([2, 5, 7])
        train_idx, valid_idx = splitter(items)
        assert list(valid_idx) == [2, 5, 7]
        assert set(train_idx) == {0, 1, 3, 4, 6, 8, 9}

    def test_empty_valid(self):
        """IndexSplitter with no valid indices puts all in training."""
        items = list(range(5))
        splitter = IndexSplitter([])
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) == 5
        assert len(valid_idx) == 0

    def test_all_valid(self):
        """IndexSplitter can put all in validation."""
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) == 0
        assert len(valid_idx) == 5


class TestEndSplitter:
    """Tests for EndSplitter."""

    def test_split_at_end(self):
        """EndSplitter with valid_last=True puts last items in validation."""
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train_idx, valid_idx = splitter(items)
        assert list(valid_idx) == [7, 8, 9]
        assert list(train_idx) == [0, 1, 2, 3, 4, 5, 6]

    def test_split_at_start(self):
        """EndSplitter with valid_last=False puts first items in validation."""
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train_idx, valid_idx = splitter(items)
        assert list(valid_idx) == [0, 1, 2]
        assert list(train_idx) == [3, 4, 5, 6, 7, 8, 9]

    def test_invalid_pct_raises(self):
        """EndSplitter raises for invalid valid_pct."""
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)


class TestGrandparentSplitter:
    """Tests for GrandparentSplitter."""

    def test_basic_split(self):
        """GrandparentSplitter splits based on grandparent directory name."""
        items = [
            Path('/data/train/cats/img1.jpg'),
            Path('/data/train/dogs/img2.jpg'),
            Path('/data/valid/cats/img3.jpg'),
            Path('/data/valid/dogs/img4.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='train', valid_name='valid')
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0, 1]
        assert list(valid_idx) == [2, 3]

    def test_custom_names(self):
        """GrandparentSplitter works with custom train/valid folder names."""
        items = [
            Path('/data/training/a/img1.jpg'),
            Path('/data/testing/b/img2.jpg'),
        ]
        splitter = GrandparentSplitter(train_name='training', valid_name='testing')
        train_idx, valid_idx = splitter(items)
        assert list(train_idx) == [0]
        assert list(valid_idx) == [1]


class TestFuncSplitter:
    """Tests for FuncSplitter."""

    def test_basic_split(self):
        """FuncSplitter splits based on function result (True = validation)."""
        items = list(range(10))
        # Even numbers go to validation
        splitter = FuncSplitter(lambda x: x % 2 == 0)
        train_idx, valid_idx = splitter(items)
        assert set(valid_idx) == {0, 2, 4, 6, 8}
        assert set(train_idx) == {1, 3, 5, 7, 9}


class TestMaskSplitter:
    """Tests for MaskSplitter."""

    def test_basic_split(self):
        """MaskSplitter splits based on boolean mask."""
        items = list(range(5))
        mask = [True, False, True, False, True]
        splitter = MaskSplitter(mask)
        train_idx, valid_idx = splitter(items)
        assert set(valid_idx) == {0, 2, 4}
        assert set(train_idx) == {1, 3}


class TestFileSplitter:
    """Tests for FileSplitter."""

    def test_basic_split(self, tmp_path):
        """FileSplitter reads valid filenames from a file."""
        # Create the file listing valid items
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("img2.jpg\nimg4.jpg\n")

        items = [
            Path('/data/img1.jpg'),
            Path('/data/img2.jpg'),
            Path('/data/img3.jpg'),
            Path('/data/img4.jpg'),
        ]
        splitter = FileSplitter(valid_file)
        train_idx, valid_idx = splitter(items)
        assert set(valid_idx) == {1, 3}
        assert set(train_idx) == {0, 2}


class TestColSplitter:
    """Tests for ColSplitter."""

    def test_basic_split_bool_column(self):
        """ColSplitter splits DataFrame by boolean column."""
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd', 'e'],
            'is_valid': [False, True, False, True, False]
        })
        splitter = ColSplitter(col='is_valid')
        train_idx, valid_idx = splitter(df)
        assert set(valid_idx) == {1, 3}
        assert set(train_idx) == {0, 2, 4}

    def test_split_by_integer_column(self):
        """ColSplitter works with integer column index."""
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c'],
            'split': [True, False, True]
        })
        splitter = ColSplitter(col=1)  # column index 1
        train_idx, valid_idx = splitter(df)
        assert set(valid_idx) == {0, 2}
        assert set(train_idx) == {1}

    def test_split_with_on_parameter(self):
        """ColSplitter with 'on' parameter filters by specific value."""
        import pandas as pd
        df = pd.DataFrame({
            'data': ['a', 'b', 'c', 'd'],
            'fold': [1, 2, 1, 2]
        })
        splitter = ColSplitter(col='fold', on=2)
        train_idx, valid_idx = splitter(df)
        assert set(valid_idx) == {1, 3}
        assert set(train_idx) == {0, 2}

    def test_non_dataframe_raises(self):
        """ColSplitter raises assertion on non-DataFrame input."""
        splitter = ColSplitter(col='is_valid')
        with pytest.raises(AssertionError):
            splitter([1, 2, 3])


class TestRandomSubsetSplitter:
    """Tests for RandomSubsetSplitter."""

    def test_basic_split(self):
        """RandomSubsetSplitter returns subsets of specified sizes."""
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.2, seed=42)
        train_idx, valid_idx = splitter(items)
        assert len(train_idx) == 50
        assert len(valid_idx) == 20

    def test_no_overlap(self):
        """Train and valid subsets should not overlap."""
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.3, seed=42)
        train_idx, valid_idx = splitter(items)
        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_reproducible_with_seed(self):
        """Same seed produces same split."""
        items = list(range(100))
        s1 = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.2, seed=42)
        s2 = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.2, seed=42)
        t1, v1 = s1(items)
        t2, v2 = s2(items)
        assert list(t1) == list(t2)
        assert list(v1) == list(v2)


# ============================================================
# Tests for labelling functions
# ============================================================

class TestParentLabel:
    """Tests for parent_label function."""

    def test_basic(self):
        """parent_label returns the parent directory name."""
        assert parent_label(Path('/data/cats/img1.jpg')) == 'cats'
        assert parent_label(Path('/data/dogs/img2.jpg')) == 'dogs'

    def test_string_input(self):
        """parent_label works with string input."""
        assert parent_label('/data/train/img.jpg') == 'train'

    def test_nested_path(self):
        """parent_label returns immediate parent."""
        assert parent_label(Path('/a/b/c/d/file.txt')) == 'd'


class TestRegexLabeller:
    """Tests for RegexLabeller."""

    def test_basic_search(self):
        """RegexLabeller extracts label using regex search."""
        labeller = RegexLabeller(r'/(\w+)/\w+\.\w+$')
        result = labeller(Path('/data/cats/img1.jpg'))
        assert result == 'cats'

    def test_match_mode(self):
        """RegexLabeller with match=True anchors at start."""
        labeller = RegexLabeller(r'([a-z]+)', match=True)
        result = labeller('hello_world')
        assert result == 'hello'

    def test_no_match_raises(self):
        """RegexLabeller raises if pattern not found."""
        labeller = RegexLabeller(r'(\d+)')
        with pytest.raises(AssertionError):
            labeller('no_numbers_here')


class TestColReader:
    """Tests for ColReader."""

    def test_single_column(self):
        """ColReader reads a single column from a row."""
        import pandas as pd
        df = pd.DataFrame({'fname': ['img1.jpg', 'img2.jpg'], 'label': ['cat', 'dog']})
        reader = ColReader('label')
        assert reader(df.iloc[0]) == 'cat'
        assert reader(df.iloc[1]) == 'dog'

    def test_with_prefix_suffix(self):
        """ColReader applies prefix and suffix."""
        import pandas as pd
        df = pd.DataFrame({'fname': ['img1', 'img2']})
        reader = ColReader('fname', pref='data/', suff='.jpg')
        assert reader(df.iloc[0]) == 'data/img1.jpg'

    def test_multiple_columns(self):
        """ColReader reads multiple columns."""
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        reader = ColReader(['a', 'b'])
        result = reader(df.iloc[0])
        assert list(result) == [1, 3]

    def test_label_delim(self):
        """ColReader splits by label_delim."""
        import pandas as pd
        df = pd.DataFrame({'labels': ['cat dog', 'bird fish']})
        reader = ColReader('labels', label_delim=' ')
        result = reader(df.iloc[0])
        assert result == ['cat', 'dog']

    def test_label_delim_empty_string(self):
        """ColReader with label_delim on empty string returns empty list."""
        import pandas as pd
        df = pd.DataFrame({'labels': ['', 'a b']})
        reader = ColReader('labels', label_delim=' ')
        result = reader(df.iloc[0])
        assert result == []


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for CategoryMap."""

    def test_basic_creation(self):
        """CategoryMap creates sorted vocabulary from items."""
        cmap = CategoryMap(['dog', 'cat', 'bird', 'cat', 'dog'])
        # Should be sorted and unique
        assert list(cmap.items) == ['bird', 'cat', 'dog']

    def test_o2i_mapping(self):
        """CategoryMap provides object-to-index mapping."""
        cmap = CategoryMap(['dog', 'cat', 'bird'])
        assert cmap.o2i['bird'] == 0
        assert cmap.o2i['cat'] == 1
        assert cmap.o2i['dog'] == 2

    def test_unsorted(self):
        """CategoryMap without sorting preserves unique order."""
        cmap = CategoryMap(['dog', 'cat', 'bird'], sort=False)
        # Maintains order of first appearance (unique)
        items = list(cmap.items)
        assert 'dog' in items
        assert 'cat' in items
        assert 'bird' in items

    def test_add_na(self):
        """CategoryMap with add_na prepends #na# to vocab."""
        cmap = CategoryMap(['a', 'b', 'c'], add_na=True)
        assert cmap.items[0] == '#na#'
        # Unknown keys map to 0 (the na index)
        assert cmap.o2i['unknown_key'] == 0

    def test_map_objs(self):
        """CategoryMap.map_objs converts objects to indices."""
        cmap = CategoryMap(['cat', 'dog', 'bird'])
        # Sorted vocab: bird=0, cat=1, dog=2
        indices = cmap.map_objs(['dog', 'cat', 'bird'])
        assert list(indices) == [2, 1, 0]

    def test_map_ids(self):
        """CategoryMap.map_ids converts indices to objects."""
        cmap = CategoryMap(['cat', 'dog', 'bird'])
        objs = cmap.map_ids([0, 1, 2])
        assert list(objs) == ['bird', 'cat', 'dog']

    def test_equality(self):
        """CategoryMap equality check."""
        cmap1 = CategoryMap(['a', 'b', 'c'])
        cmap2 = CategoryMap(['a', 'b', 'c'])
        assert cmap1 == cmap2

    def test_nan_excluded(self):
        """CategoryMap excludes NaN values."""
        cmap = CategoryMap(['a', float('nan'), 'b', float('nan')])
        items = list(cmap.items)
        assert 'a' in items
        assert 'b' in items
        assert len(items) == 2


# ============================================================
# Tests for Categorize
# ============================================================

class TestCategorize:
    """Tests for Categorize transform."""

    def test_encode_with_vocab(self):
        """Categorize encodes category string to index."""
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        result = cat.encodes('dog')
        assert isinstance(result, TensorCategory)
        assert result.item() == cat.vocab.o2i['dog']

    def test_decode_with_vocab(self):
        """Categorize decodes index back to category string."""
        cat = Categorize(vocab=['cat', 'dog', 'bird'])
        idx = cat.vocab.o2i['dog']
        result = cat.decodes(idx)
        assert str(result) == 'dog'

    def test_unknown_label_raises(self):
        """Categorize raises KeyError for unknown labels."""
        cat = Categorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="not included in the training dataset"):
            cat.encodes('fish')

    def test_setups_creates_vocab(self):
        """Categorize.setups creates vocabulary from data."""
        cat = Categorize()
        cat.setups(['cat', 'dog', 'bird', 'cat'])
        assert len(cat.vocab) == 3
        assert cat.c == 3


# ============================================================
# Tests for MultiCategorize
# ============================================================

class TestMultiCategorize:
    """Tests for MultiCategorize transform."""

    def test_encode(self):
        """MultiCategorize encodes list of categories to indices."""
        mc = MultiCategorize(vocab=['cat', 'dog', 'bird'])
        result = mc.encodes(['cat', 'bird'])
        assert isinstance(result, TensorMultiCategory)
        assert len(result) == 2

    def test_setups_from_data(self):
        """MultiCategorize.setups builds vocab from multi-label data."""
        mc = MultiCategorize()
        data = [['cat', 'dog'], ['bird'], ['cat', 'bird']]
        mc.setups(data)
        assert len(mc.vocab) == 3

    def test_unknown_label_raises(self):
        """MultiCategorize raises KeyError for unknown labels."""
        mc = MultiCategorize(vocab=['cat', 'dog'])
        with pytest.raises(KeyError, match="not included in the training dataset"):
            mc.encodes(['fish'])


# ============================================================
# Tests for OneHotEncode
# ============================================================

class TestOneHotEncode:
    """Tests for OneHotEncode transform."""

    def test_basic_encode(self):
        """OneHotEncode creates one-hot vector from indices."""
        ohe = OneHotEncode(c=5)
        result = ohe.encodes(torch.tensor([0, 2, 4]))
        assert result.shape == (5,)
        assert result[0].item() == 1.0
        assert result[1].item() == 0.0
        assert result[2].item() == 1.0
        assert result[3].item() == 0.0
        assert result[4].item() == 1.0

    def test_decode(self):
        """OneHotEncode decodes back to indices."""
        ohe = OneHotEncode(c=5)
        encoded = ohe.encodes(torch.tensor([1, 3]))
        decoded = ohe.decodes(encoded)
        assert 1 in decoded
        assert 3 in decoded

    def test_output_type(self):
        """OneHotEncode outputs TensorMultiCategory."""
        ohe = OneHotEncode(c=3)
        result = ohe.encodes(torch.tensor([0, 2]))
        assert isinstance(result, TensorMultiCategory)


# ============================================================
# Tests for IntToFloatTensor
# ============================================================

class TestIntToFloatTensor:
    """Tests for IntToFloatTensor transform."""

    def test_image_to_float(self):
        """IntToFloatTensor converts integer image tensor to float [0,1]."""
        t = IntToFloatTensor()
        img = TensorImage(torch.randint(0, 256, (3, 32, 32)))
        result = t.encodes(img)
        assert result.dtype == torch.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_custom_div(self):
        """IntToFloatTensor with custom div value."""
        t = IntToFloatTensor(div=128.)
        img = TensorImage(torch.tensor([[[128]]]))
        result = t.encodes(img)
        assert abs(result.item() - 1.0) < 1e-5

    def test_decode_restores(self):
        """IntToFloatTensor.decodes restores to integer range."""
        t = IntToFloatTensor(div=255.)
        img = TensorImage(torch.tensor([[[0.5]]]))
        result = t.decodes(img)
        # 0.5 * 255 = 127 (rounded to long)
        assert result.item() == 127

    def test_mask_to_long(self):
        """IntToFloatTensor converts mask to long."""
        t = IntToFloatTensor()
        mask = TensorMask(torch.tensor([[1, 2, 3]]))
        result = t.encodes(mask)
        assert result.dtype == torch.int64


# ============================================================
# Tests for broadcast_vec
# ============================================================

class TestBroadcastVec:
    """Tests for broadcast_vec function."""

    def test_basic_broadcast(self):
        """broadcast_vec reshapes vector for broadcasting."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        result = broadcast_vec(1, 4, mean, std, cuda=False)
        assert len(result) == 2
        # For dim=1, ndim=4: shape should be [1, -1, 1, 1] -> [1, 3, 1, 1]
        assert result[0].shape == (1, 3, 1, 1)
        assert result[1].shape == (1, 3, 1, 1)

    def test_dim_zero(self):
        """broadcast_vec with dim=0."""
        vals = [1.0, 2.0, 3.0]
        result = broadcast_vec(0, 3, vals, cuda=False)
        assert result[0].shape == (3, 1, 1)

    def test_values_preserved(self):
        """broadcast_vec preserves the values."""
        vals = [0.5, 1.0, 1.5]
        result = broadcast_vec(1, 4, vals, cuda=False)
        expected = torch.tensor(vals).view(1, 3, 1, 1)
        assert torch.allclose(result[0], expected)


# ============================================================
# Tests for Normalize
# ============================================================

class TestNormalize:
    """Tests for Normalize transform."""

    def test_from_stats(self):
        """Normalize.from_stats creates normalizer from mean/std."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm = Normalize.from_stats(mean, std, cuda=False)
        assert norm.mean is not None
        assert norm.std is not None

    def test_encode_normalizes(self):
        """Normalize.encodes normalizes the input."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        norm = Normalize.from_stats(mean, std, cuda=False)
        img = TensorImage(torch.ones(1, 3, 4, 4) * 0.5)
        result = norm.encodes(img)
        # (0.5 - 0.5) / 0.5 = 0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_decode_denormalizes(self):
        """Normalize.decodes reverses normalization."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        norm = Normalize.from_stats(mean, std, cuda=False)
        img = TensorImage(torch.ones(1, 3, 4, 4) * 0.75)
        encoded = norm.encodes(img)
        decoded = norm.decodes(encoded)
        assert torch.allclose(decoded, img, atol=1e-5)

    def test_normalize_different_values(self):
        """Normalize correctly transforms with non-trivial stats."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm = Normalize.from_stats(mean, std, cuda=False)
        # Create an image with known values
        img = TensorImage(torch.zeros(1, 3, 2, 2))
        result = norm.encodes(img)
        # (0 - mean) / std should give negative values
        assert result.mean().item() < 0


# ============================================================
# Tests for ItemGetter and AttrGetter
# ============================================================

class TestItemGetter:
    """Tests for ItemGetter transform."""

    def test_basic_indexing(self):
        """ItemGetter retrieves item by index."""
        getter = ItemGetter(1)
        result = getter.encodes(['a', 'b', 'c'])
        assert result == 'b'

    def test_tuple_indexing(self):
        """ItemGetter works on tuples."""
        getter = ItemGetter(0)
        result = getter.encodes(('first', 'second'))
        assert result == 'first'

    def test_dict_indexing(self):
        """ItemGetter works on dicts with key."""
        getter = ItemGetter('key')
        result = getter.encodes({'key': 'value', 'other': 'data'})
        assert result == 'value'


class TestAttrGetter:
    """Tests for AttrGetter transform."""

    def test_basic_attr(self):
        """AttrGetter retrieves attribute by name."""
        obj = SimpleNamespace(name='test', value=42)
        getter = AttrGetter('name')
        result = getter.encodes(obj)
        assert result == 'test'

    def test_default_value(self):
        """AttrGetter returns default for missing attribute."""
        obj = SimpleNamespace(name='test')
        getter = AttrGetter('missing', default='fallback')
        result = getter.encodes(obj)
        assert result == 'fallback'

    def test_missing_attr_no_default(self):
        """AttrGetter returns None for missing attribute with no default."""
        obj = SimpleNamespace(name='test')
        getter = AttrGetter('missing')
        result = getter.encodes(obj)
        assert result is None


# ============================================================
# Tests for RegressionSetup
# ============================================================

class TestRegressionSetup:
    """Tests for RegressionSetup transform."""

    def test_encode_float(self):
        """RegressionSetup converts target to float tensor."""
        reg = RegressionSetup()
        result = reg.encodes(3.14)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert abs(result.item() - 3.14) < 1e-5

    def test_encode_int(self):
        """RegressionSetup converts integer target to float tensor."""
        reg = RegressionSetup()
        result = reg.encodes(5)
        assert result.dtype == torch.float32
        assert result.item() == 5.0

    def test_encode_list(self):
        """RegressionSetup converts list target to float tensor."""
        reg = RegressionSetup()
        result = reg.encodes([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        assert result.dtype == torch.float32


# ============================================================
# Tests for get_c utility
# ============================================================

class TestGetC:
    """Tests for get_c function."""

    def test_with_c_attribute(self):
        """get_c returns dls.c if present."""
        dls = SimpleNamespace(c=10)
        assert get_c(dls) == 10

    def test_with_vocab(self):
        """get_c returns len(vocab) if no c attribute."""
        dls = SimpleNamespace(c=False, vocab=['a', 'b', 'c'])
        # Patch nested_attr to return False
        dls.train = None
        assert get_c(dls) == 3


# Needed for SimpleNamespace imports
from types import SimpleNamespace
