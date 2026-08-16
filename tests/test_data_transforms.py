"""Tests for fastai.data.transforms module.

Covers: _get_files, get_files, get_image_files, get_text_files, FileGetter,
ImageGetter, RandomSplitter, IndexSplitter, EndSplitter, GrandparentSplitter,
FuncSplitter, MaskSplitter, TrainTestSplitter, RandomSubsetSplitter,
parent_label, RegexLabeller, CategoryMap, and Categorize.
"""
import tempfile
import pytest
from pathlib import Path

from fastai.data.transforms import (
    _get_files,
    get_files,
    get_image_files,
    get_text_files,
    FileGetter,
    ImageGetter,
    RandomSplitter,
    TrainTestSplitter,
    IndexSplitter,
    EndSplitter,
    GrandparentSplitter,
    FuncSplitter,
    MaskSplitter,
    RandomSubsetSplitter,
    parent_label,
    RegexLabeller,
    CategoryMap,
    Categorize,
)


# ============================================================
# Fixtures for file-based tests
# ============================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory with a known file structure for testing."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        # Create subdirectories
        (base / "train" / "cats").mkdir(parents=True)
        (base / "train" / "dogs").mkdir(parents=True)
        (base / "valid" / "cats").mkdir(parents=True)
        (base / "valid" / "dogs").mkdir(parents=True)
        (base / ".hidden").mkdir(parents=True)

        # Create image files
        (base / "train" / "cats" / "cat1.jpg").touch()
        (base / "train" / "cats" / "cat2.png").touch()
        (base / "train" / "dogs" / "dog1.jpg").touch()
        (base / "valid" / "cats" / "cat3.jpeg").touch()
        (base / "valid" / "dogs" / "dog2.bmp").touch()

        # Create text files
        (base / "train" / "cats" / "notes.txt").touch()
        (base / "valid" / "dogs" / "info.txt").touch()

        # Create other files
        (base / "train" / "cats" / "data.csv").touch()
        (base / "train" / "readme.md").touch()

        # Create hidden file (should be excluded)
        (base / ".hidden" / "secret.jpg").touch()
        (base / "train" / "cats" / ".hidden_file.jpg").touch()

        yield base


# ============================================================
# Tests for _get_files
# ============================================================

class TestGetFilesInternal:
    """Tests for the internal _get_files helper."""

    def test_basic_filtering(self, tmp_dir):
        files = ["cat1.jpg", "cat2.png", "notes.txt", "data.csv"]
        result = _get_files(tmp_dir / "train" / "cats", files, extensions={".jpg"})
        assert len(result) == 1
        assert result[0].name == "cat1.jpg"

    def test_multiple_extensions(self, tmp_dir):
        files = ["cat1.jpg", "cat2.png", "notes.txt", "data.csv"]
        result = _get_files(tmp_dir / "train" / "cats", files, extensions={".jpg", ".png"})
        names = {r.name for r in result}
        assert names == {"cat1.jpg", "cat2.png"}

    def test_no_extensions_returns_all(self, tmp_dir):
        files = ["cat1.jpg", "cat2.png", "notes.txt"]
        result = _get_files(tmp_dir / "train" / "cats", files, extensions=None)
        assert len(result) == 3

    def test_hidden_files_excluded(self, tmp_dir):
        files = [".hidden_file.jpg", "cat1.jpg"]
        result = _get_files(tmp_dir / "train" / "cats", files, extensions={".jpg"})
        names = [r.name for r in result]
        assert ".hidden_file.jpg" not in names
        assert "cat1.jpg" in names

    def test_case_insensitive_extensions(self, tmp_dir):
        files = ["photo.JPG", "image.Png"]
        result = _get_files(tmp_dir, files, extensions={".jpg", ".png"})
        assert len(result) == 2

    def test_empty_file_list(self, tmp_dir):
        result = _get_files(tmp_dir, [], extensions={".jpg"})
        assert len(result) == 0


# ============================================================
# Tests for get_files
# ============================================================

class TestGetFiles:
    """Tests for get_files function."""

    def test_recursive_finds_all_images(self, tmp_dir):
        result = get_files(tmp_dir, extensions=[".jpg"], recurse=True)
        # Should find cat1.jpg, dog1.jpg, cat3.jpeg (not .jpeg unless specified)
        names = {r.name for r in result}
        assert "cat1.jpg" in names
        assert "dog1.jpg" in names

    def test_non_recursive(self, tmp_dir):
        result = get_files(tmp_dir / "train" / "cats", extensions=[".jpg"], recurse=False)
        names = {r.name for r in result}
        assert "cat1.jpg" in names
        # Should not find files in subdirectories (there are none deeper in this case)

    def test_folder_filter(self, tmp_dir):
        result = get_files(tmp_dir, extensions=[".jpg", ".png", ".jpeg", ".bmp"], recurse=True, folders=["train"])
        # Should only include files under the "train" folder
        for r in result:
            assert "train" in str(r)

    def test_hidden_directories_excluded(self, tmp_dir):
        result = get_files(tmp_dir, extensions=[".jpg"], recurse=True)
        for r in result:
            assert ".hidden" not in str(r)

    def test_no_extensions_returns_all_non_hidden(self, tmp_dir):
        result = get_files(tmp_dir, recurse=True)
        # Should return all non-hidden files
        assert len(result) > 0
        for r in result:
            assert not r.name.startswith(".")
            assert ".hidden" not in str(r)

    def test_returns_L_type(self, tmp_dir):
        from fastcore.foundation import L
        result = get_files(tmp_dir, extensions=[".txt"], recurse=True)
        assert isinstance(result, L)


# ============================================================
# Tests for get_image_files
# ============================================================

class TestGetImageFiles:
    """Tests for get_image_files function."""

    def test_finds_common_image_formats(self, tmp_dir):
        result = get_image_files(tmp_dir)
        names = {r.name for r in result}
        # jpg and png should always be recognized as image extensions
        assert "cat1.jpg" in names
        assert "cat2.png" in names

    def test_excludes_non_image_files(self, tmp_dir):
        result = get_image_files(tmp_dir)
        names = {r.name for r in result}
        assert "notes.txt" not in names
        assert "data.csv" not in names
        assert "readme.md" not in names

    def test_folder_restriction(self, tmp_dir):
        result = get_image_files(tmp_dir, folders=["valid"])
        for r in result:
            assert "valid" in str(r)


# ============================================================
# Tests for get_text_files
# ============================================================

class TestGetTextFiles:
    """Tests for get_text_files function."""

    def test_finds_txt_files(self, tmp_dir):
        result = get_text_files(tmp_dir)
        names = {r.name for r in result}
        assert "notes.txt" in names
        assert "info.txt" in names

    def test_excludes_non_txt(self, tmp_dir):
        result = get_text_files(tmp_dir)
        names = {r.name for r in result}
        assert "cat1.jpg" not in names
        assert "data.csv" not in names


# ============================================================
# Tests for FileGetter and ImageGetter
# ============================================================

class TestFileGetter:
    """Tests for FileGetter factory function."""

    def test_basic_usage(self, tmp_dir):
        getter = FileGetter(extensions=[".txt"], recurse=True)
        result = getter(tmp_dir)
        names = {r.name for r in result}
        assert "notes.txt" in names
        assert "info.txt" in names

    def test_with_suffix(self, tmp_dir):
        getter = FileGetter(suf="train", extensions=[".jpg"], recurse=True)
        result = getter(tmp_dir)
        names = {r.name for r in result}
        assert "cat1.jpg" in names
        assert "dog1.jpg" in names


class TestImageGetter:
    """Tests for ImageGetter factory function."""

    def test_basic_usage(self, tmp_dir):
        getter = ImageGetter(recurse=True)
        result = getter(tmp_dir)
        names = {r.name for r in result}
        assert "cat1.jpg" in names
        assert "cat2.png" in names

    def test_with_suffix(self, tmp_dir):
        getter = ImageGetter(suf="valid", recurse=True)
        result = getter(tmp_dir)
        for r in result:
            assert "valid" in str(r)


# ============================================================
# Tests for RandomSplitter
# ============================================================

class TestRandomSplitter:
    """Tests for RandomSplitter function."""

    def test_split_sizes(self):
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
        train_set = set(train.items if hasattr(train, 'items') else list(train))
        valid_set = set(valid.items if hasattr(valid, 'items') else list(valid))
        assert len(train_set & valid_set) == 0

    def test_covers_all_indices(self):
        items = list(range(50))
        splitter = RandomSplitter(valid_pct=0.2, seed=42)
        train, valid = splitter(items)
        all_idxs = sorted(list(train) + list(valid))
        assert all_idxs == list(range(50))

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSplitter(valid_pct=0.2, seed=123)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_different_seeds_give_different_splits(self):
        items = list(range(100))
        splitter1 = RandomSplitter(valid_pct=0.2, seed=1)
        splitter2 = RandomSplitter(valid_pct=0.2, seed=2)
        _, valid1 = splitter1(items)
        _, valid2 = splitter2(items)
        # Very unlikely to be identical with different seeds
        assert list(valid1) != list(valid2)

    def test_valid_pct_boundary(self):
        items = list(range(10))
        splitter = RandomSplitter(valid_pct=0.1, seed=42)
        train, valid = splitter(items)
        assert len(valid) == 1
        assert len(train) == 9


# ============================================================
# Tests for TrainTestSplitter
# ============================================================

class TestTrainTestSplitter:
    """Tests for TrainTestSplitter function (sklearn-based)."""

    def test_split_sizes(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train, valid = splitter(items)
        assert len(train) + len(valid) == 100
        assert len(valid) == 20

    def test_reproducibility(self):
        items = list(range(100))
        splitter = TrainTestSplitter(test_size=0.2, random_state=42)
        train1, valid1 = splitter(items)
        train2, valid2 = splitter(items)
        assert list(train1) == list(train2)
        assert list(valid1) == list(valid2)

    def test_no_overlap(self):
        items = list(range(50))
        splitter = TrainTestSplitter(test_size=0.3, random_state=42)
        train, valid = splitter(items)
        assert len(set(train) & set(valid)) == 0


# ============================================================
# Tests for IndexSplitter
# ============================================================

class TestIndexSplitter:
    """Tests for IndexSplitter function."""

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
        assert len(valid) == 0
        assert sorted(list(train)) == [0, 1, 2, 3, 4]

    def test_all_valid(self):
        items = list(range(5))
        splitter = IndexSplitter([0, 1, 2, 3, 4])
        train, valid = splitter(items)
        assert len(train) == 0
        assert sorted(list(valid)) == [0, 1, 2, 3, 4]

    def test_single_item_valid(self):
        items = list(range(10))
        splitter = IndexSplitter([3])
        train, valid = splitter(items)
        assert list(valid) == [3]
        assert 3 not in list(train)


# ============================================================
# Tests for EndSplitter
# ============================================================

class TestEndSplitter:
    """Tests for EndSplitter function."""

    def test_valid_last(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=True)
        train, valid = splitter(items)
        # Last 30% should be validation
        assert list(valid) == [7, 8, 9]
        assert list(train) == [0, 1, 2, 3, 4, 5, 6]

    def test_valid_first(self):
        items = list(range(10))
        splitter = EndSplitter(valid_pct=0.3, valid_last=False)
        train, valid = splitter(items)
        # First 30% should be validation
        assert list(valid) == [0, 1, 2]
        assert list(train) == [3, 4, 5, 6, 7, 8, 9]

    def test_split_sizes(self):
        items = list(range(100))
        splitter = EndSplitter(valid_pct=0.2)
        train, valid = splitter(items)
        assert len(valid) == 20
        assert len(train) == 80

    def test_ordered_indices(self):
        items = list(range(20))
        splitter = EndSplitter(valid_pct=0.25, valid_last=True)
        train, valid = splitter(items)
        # Both should be contiguous and ordered
        assert list(train) == list(range(15))
        assert list(valid) == list(range(15, 20))

    def test_invalid_pct_raises(self):
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=0.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=1.0)
        with pytest.raises(AssertionError):
            EndSplitter(valid_pct=-0.1)


# ============================================================
# Tests for GrandparentSplitter
# ============================================================

class TestGrandparentSplitter:
    """Tests for GrandparentSplitter function."""

    def test_basic_split(self, tmp_dir):
        items = [
            tmp_dir / "train" / "cats" / "cat1.jpg",
            tmp_dir / "train" / "dogs" / "dog1.jpg",
            tmp_dir / "valid" / "cats" / "cat3.jpeg",
            tmp_dir / "valid" / "dogs" / "dog2.bmp",
        ]
        splitter = GrandparentSplitter(train_name="train", valid_name="valid")
        train_idxs, valid_idxs = splitter(items)
        assert sorted(list(train_idxs)) == [0, 1]
        assert sorted(list(valid_idxs)) == [2, 3]

    def test_custom_names(self, tmp_dir):
        # Create custom directory structure
        (tmp_dir / "trn" / "cls").mkdir(parents=True)
        (tmp_dir / "val" / "cls").mkdir(parents=True)
        (tmp_dir / "trn" / "cls" / "a.jpg").touch()
        (tmp_dir / "val" / "cls" / "b.jpg").touch()

        items = [
            tmp_dir / "trn" / "cls" / "a.jpg",
            tmp_dir / "val" / "cls" / "b.jpg",
        ]
        splitter = GrandparentSplitter(train_name="trn", valid_name="val")
        train_idxs, valid_idxs = splitter(items)
        assert list(train_idxs) == [0]
        assert list(valid_idxs) == [1]


# ============================================================
# Tests for FuncSplitter
# ============================================================

class TestFuncSplitter:
    """Tests for FuncSplitter function."""

    def test_even_odd_split(self):
        items = list(range(10))
        # Odd numbers go to validation
        splitter = FuncSplitter(lambda x: x % 2 == 1)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [1, 3, 5, 7, 9]
        assert sorted(list(train)) == [0, 2, 4, 6, 8]

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

    def test_threshold_split(self):
        items = list(range(10))
        splitter = FuncSplitter(lambda x: x >= 7)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [7, 8, 9]
        assert sorted(list(train)) == [0, 1, 2, 3, 4, 5, 6]


# ============================================================
# Tests for MaskSplitter
# ============================================================

class TestMaskSplitter:
    """Tests for MaskSplitter function."""

    def test_basic_mask(self):
        items = list(range(5))
        mask = [False, True, False, True, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert sorted(list(valid)) == [1, 3]
        assert sorted(list(train)) == [0, 2, 4]

    def test_all_true(self):
        items = list(range(4))
        mask = [True, True, True, True]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 4
        assert len(train) == 0

    def test_all_false(self):
        items = list(range(4))
        mask = [False, False, False, False]
        splitter = MaskSplitter(mask)
        train, valid = splitter(items)
        assert len(valid) == 0
        assert len(train) == 4


# ============================================================
# Tests for RandomSubsetSplitter
# ============================================================

class TestRandomSubsetSplitter:
    """Tests for RandomSubsetSplitter function."""

    def test_split_sizes(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=42)
        train, valid = splitter(items)
        assert len(train) == 60
        assert len(valid) == 20

    def test_no_overlap(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.5, valid_sz=0.3, seed=42)
        train, valid = splitter(items)
        train_set = set(list(train))
        valid_set = set(list(valid))
        assert len(train_set & valid_set) == 0

    def test_seed_reproducibility(self):
        items = list(range(100))
        splitter = RandomSubsetSplitter(train_sz=0.6, valid_sz=0.2, seed=99)
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
            RandomSubsetSplitter(train_sz=0.6, valid_sz=0.5)  # sum > 1


# ============================================================
# Tests for parent_label
# ============================================================

class TestParentLabel:
    """Tests for parent_label function."""

    def test_basic_path(self):
        assert parent_label(Path("/data/cats/image1.jpg")) == "cats"

    def test_nested_path(self):
        assert parent_label(Path("/data/train/dogs/img.png")) == "dogs"

    def test_string_input(self):
        assert parent_label("/data/birds/photo.jpg") == "birds"

    def test_relative_path(self):
        assert parent_label(Path("animals/cats/pic.jpg")) == "cats"


# ============================================================
# Tests for RegexLabeller
# ============================================================

class TestRegexLabeller:
    """Tests for RegexLabeller class."""

    def test_search_mode(self):
        labeller = RegexLabeller(r"/([^/]+)/[^/]+$")
        result = labeller(Path("/data/cats/image1.jpg"))
        assert result == "cats"

    def test_match_mode(self):
        labeller = RegexLabeller(r"(\w+)_\d+\.jpg", match=True)
        result = labeller("cat_001.jpg")
        assert result == "cat"

    def test_no_match_raises(self):
        labeller = RegexLabeller(r"(\d+)")
        with pytest.raises(AssertionError):
            labeller("no_numbers_here.jpg")

    def test_complex_pattern(self):
        labeller = RegexLabeller(r"/(\w+)/\w+/[^/]+$")
        result = labeller("/dataset/train/cats/img.jpg")
        assert result == "train"


# ============================================================
# Tests for CategoryMap
# ============================================================

class TestCategoryMap:
    """Tests for CategoryMap class."""

    def test_basic_creation(self):
        cm = CategoryMap(["cat", "dog", "bird"])
        assert len(cm) == 3
        assert "cat" in cm.items
        assert "dog" in cm.items
        assert "bird" in cm.items

    def test_sorted_by_default(self):
        cm = CategoryMap(["dog", "cat", "bird"])
        # Should be alphabetically sorted
        assert list(cm.items) == ["bird", "cat", "dog"]

    def test_unsorted(self):
        cm = CategoryMap(["dog", "cat", "bird"], sort=False)
        # Order depends on unique() behavior, but should contain all items
        assert set(cm.items) == {"bird", "cat", "dog"}

    def test_o2i_mapping(self):
        cm = CategoryMap(["cat", "dog", "bird"], sort=True)
        # Sorted: bird=0, cat=1, dog=2
        assert cm.o2i["bird"] == 0
        assert cm.o2i["cat"] == 1
        assert cm.o2i["dog"] == 2

    def test_add_na(self):
        cm = CategoryMap(["cat", "dog"], sort=True, add_na=True)
        assert cm.items[0] == "#na#"
        assert cm.o2i["#na#"] == 0

    def test_map_objs(self):
        cm = CategoryMap(["cat", "dog", "bird"], sort=True)
        result = cm.map_objs(["dog", "cat", "bird"])
        assert list(result) == [2, 1, 0]

    def test_map_ids(self):
        cm = CategoryMap(["cat", "dog", "bird"], sort=True)
        result = cm.map_ids([0, 1, 2])
        assert list(result) == ["bird", "cat", "dog"]

    def test_duplicates_handled(self):
        cm = CategoryMap(["cat", "cat", "dog", "dog", "bird"])
        # Should only have unique values
        assert len(cm) == 3

    def test_equality(self):
        cm1 = CategoryMap(["cat", "dog"], sort=True)
        cm2 = CategoryMap(["cat", "dog"], sort=True)
        assert cm1 == cm2


# ============================================================
# Tests for Categorize
# ============================================================

class TestCategorize:
    """Tests for Categorize transform."""

    def test_encode_with_vocab(self):
        cat = Categorize(vocab=["bird", "cat", "dog"], sort=True)
        # After sorting: bird=0, cat=1, dog=2
        result = cat.encodes("cat")
        assert int(result) == 1

    def test_decode(self):
        cat = Categorize(vocab=["bird", "cat", "dog"], sort=True)
        encoded = cat.encodes("dog")
        decoded = cat.decodes(encoded)
        assert str(decoded) == "dog"

    def test_unknown_label_raises(self):
        cat = Categorize(vocab=["cat", "dog"], sort=True)
        with pytest.raises(KeyError, match="not included in the training dataset"):
            cat.encodes("bird")

    def test_vocab_length(self):
        cat = Categorize(vocab=["a", "b", "c"], sort=True)
        # The `c` attribute is set during `setups()`, but vocab is available
        assert len(cat.vocab) == 3

    def test_roundtrip(self):
        labels = ["cat", "dog", "bird", "fish"]
        cat = Categorize(vocab=labels, sort=True)
        for label in labels:
            encoded = cat.encodes(label)
            decoded = cat.decodes(encoded)
            assert str(decoded) == label
