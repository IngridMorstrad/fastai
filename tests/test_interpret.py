"""Tests for fastai.interpret module.

Covers: plot_top_losses dispatch, Interpretation, ClassificationInterpretation,
and SegmentationInterpretation classes.

Since interpret.py relies heavily on Learner, DataLoader, and the full
training pipeline, we use mocks/stubs to isolate the logic under test
and keep the tests fast and unit-level.
"""
import sys
import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastcore.foundation import L
from fastai.torch_core import TensorBase


# ============================================================
# Helpers to create mock Learner/DataLoader objects
# ============================================================

class MockVocab:
    """A vocab-like object that is not listy (mirrors CategoryMap behavior).

    fastai's ClassificationInterpretation checks `is_listy(self.vocab)` to detect
    multi-input scenarios.  A real single-task vocab (CategoryMap) is not listy.
    """

    def __init__(self, items):
        self.items = list(items)
        # o2i is used by print_classification_report
        self.o2i = {v: i for i, v in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def __eq__(self, other):
        if isinstance(other, MockVocab):
            return self.items == other.items
        return NotImplemented

    def __repr__(self):
        return f"MockVocab({self.items})"


def _make_mock_dl(items=None, vocab=None):
    """Create a mock DataLoader with the attributes Interpretation uses."""
    dl = MagicMock()
    dl.items = L(items) if items is not None else L([0, 1, 2, 3, 4])
    if vocab is not None:
        dl.vocab = vocab
    return dl


def _make_mock_learner(dl=None, dls=None):
    """Create a mock Learner with the attributes Interpretation uses."""
    learner = MagicMock()
    if dl is not None:
        if dls is None:
            dls = MagicMock()
            dls.__getitem__ = MagicMock(return_value=dl)
        learner.dls = dls
    return learner


def _make_interpretation(n_samples=5, n_classes=3):
    """Create an Interpretation instance with mock learn/dl and random losses."""
    from fastai.interpret import Interpretation

    dl = _make_mock_dl(items=list(range(n_samples)))
    learner = _make_mock_learner(dl=dl)
    losses = TensorBase(torch.randn(n_samples).abs())
    return Interpretation(learner, dl, losses, act=None)


def _make_classification_interpretation(n_samples=10, vocab=None):
    """Create a ClassificationInterpretation with mock dependencies."""
    from fastai.interpret import ClassificationInterpretation

    if vocab is None:
        vocab = MockVocab(['cat', 'dog', 'bird'])
    n_classes = len(vocab)

    dl = _make_mock_dl(items=list(range(n_samples)), vocab=vocab)
    learner = _make_mock_learner(dl=dl)
    losses = TensorBase(torch.randn(n_samples).abs())
    return ClassificationInterpretation(learner, dl, losses, act=None)


# ============================================================
# Tests for plot_top_losses type dispatch
# ============================================================

class TestPlotTopLossesDispatch:
    """Tests for the plot_top_losses typedispatch function."""

    def test_raises_for_unregistered_types(self):
        """The default plot_top_losses should raise for types without dispatch."""
        from fastai.interpret import plot_top_losses
        with pytest.raises(Exception, match="plot_top_losses is not implemented"):
            plot_top_losses("some_input", "some_target")

    def test_raises_with_informative_type_message(self):
        """Error message should mention the types passed."""
        from fastai.interpret import plot_top_losses
        with pytest.raises(Exception, match="str.*str"):
            plot_top_losses("input", "target")


# ============================================================
# Tests for Interpretation class
# ============================================================

class TestInterpretation:
    """Tests for the Interpretation base class."""

    def test_init_stores_attributes(self):
        """Interpretation.__init__ should store learn, dl, losses, act."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl()
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([1.0, 2.0, 3.0]))
        act = lambda x: x

        interp = Interpretation(learner, dl, losses, act=act)
        assert interp.learn is learner
        assert interp.dl is dl
        assert torch.equal(interp.losses, losses)
        assert interp.act is act

    def test_init_act_default_none(self):
        """Default act should be None."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl()
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([1.0]))

        interp = Interpretation(learner, dl, losses, act=None)
        assert interp.act is None

    def test_top_losses_returns_all_by_default(self):
        """top_losses with k=None should return all losses sorted."""
        interp = _make_interpretation(n_samples=5)

        losses, idx = interp.top_losses()
        assert len(losses) == 5
        assert len(idx) == 5
        # Should be sorted descending by default
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1]

    def test_top_losses_with_k(self):
        """top_losses with k should return only k items."""
        interp = _make_interpretation(n_samples=10)

        losses, idx = interp.top_losses(k=3)
        assert len(losses) == 3
        assert len(idx) == 3

    def test_top_losses_largest_true(self):
        """top_losses with largest=True should return largest first."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl(items=[0, 1, 2, 3])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.9, 0.5, 0.3]))
        interp = Interpretation(learner, dl, losses, act=None)

        top_losses, idx = interp.top_losses(k=2, largest=True)
        assert top_losses[0].item() == pytest.approx(0.9)
        assert idx[0].item() == 1

    def test_top_losses_largest_false(self):
        """top_losses with largest=False should return smallest first."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl(items=[0, 1, 2, 3])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.9, 0.5, 0.3]))
        interp = Interpretation(learner, dl, losses, act=None)

        top_losses, idx = interp.top_losses(k=2, largest=False)
        assert top_losses[0].item() == pytest.approx(0.1)
        assert idx[0].item() == 0

    def test_top_losses_with_items(self):
        """top_losses with items=True should return items as well."""
        from fastai.interpret import Interpretation

        items = ['img_a.jpg', 'img_b.jpg', 'img_c.jpg']
        dl = _make_mock_dl(items=items)
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.9, 0.5]))
        interp = Interpretation(learner, dl, losses, act=None)

        result = interp.top_losses(k=2, largest=True, items=True)
        assert len(result) == 3  # losses, idx, items
        top_losses, idx, top_items = result
        assert len(top_losses) == 2
        assert len(idx) == 2
        # The top item should correspond to the highest loss
        assert top_items[0] == 'img_b.jpg'

    def test_from_learner_classmethod(self):
        """from_learner should call get_preds and construct the object."""
        from fastai.interpret import Interpretation

        mock_dl = _make_mock_dl(items=[0, 1, 2])
        mock_dl.new = MagicMock(return_value=mock_dl)

        mock_dls = MagicMock()
        mock_dls.__getitem__ = MagicMock(return_value=mock_dl)

        learner = MagicMock()
        learner.dls = mock_dls

        mock_losses = TensorBase(torch.tensor([0.5, 0.3, 0.7]))
        learner.get_preds = MagicMock(return_value=(None, None, mock_losses))

        interp = Interpretation.from_learner(learner, ds_idx=1, dl=None, act=None)
        assert isinstance(interp, Interpretation)
        assert torch.equal(interp.losses, mock_losses)
        learner.get_preds.assert_called_once()

    def test_from_learner_with_custom_dl(self):
        """from_learner with a provided dl should use it directly."""
        from fastai.interpret import Interpretation

        custom_dl = _make_mock_dl(items=[10, 20, 30])
        learner = MagicMock()
        mock_losses = TensorBase(torch.tensor([1.0, 2.0, 3.0]))
        learner.get_preds = MagicMock(return_value=(None, None, mock_losses))

        interp = Interpretation.from_learner(learner, dl=custom_dl)
        assert interp.dl is custom_dl

    def test_getitem_calls_learner_methods(self):
        """__getitem__ should call learn.dls.test_dl and learn.get_preds."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl(items=['a', 'b', 'c', 'd', 'e'])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))

        # Mock the test_dl and get_preds returns
        tmp_dl = MagicMock()
        learner.dls.test_dl = MagicMock(return_value=tmp_dl)

        mock_inps = torch.randn(2, 3)
        mock_preds = torch.randn(2, 5)
        mock_targs = torch.tensor([0, 1])
        mock_decoded = torch.tensor([0, 1])
        learner.get_preds = MagicMock(return_value=(mock_inps, mock_preds, mock_targs, mock_decoded))

        interp = Interpretation(learner, dl, losses, act=None)
        result = interp[torch.tensor([0, 1])]

        learner.dls.test_dl.assert_called_once()
        learner.get_preds.assert_called_once()
        # Result should have 5 elements: inps, preds, targs, decoded, losses
        assert len(result) == 5

    def test_getitem_single_index(self):
        """__getitem__ with a single integer index should work."""
        from fastai.interpret import Interpretation

        dl = _make_mock_dl(items=['a', 'b', 'c'])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.2, 0.3]))

        tmp_dl = MagicMock()
        learner.dls.test_dl = MagicMock(return_value=tmp_dl)
        learner.get_preds = MagicMock(return_value=(
            torch.randn(1, 3), torch.randn(1, 5), torch.tensor([0]), torch.tensor([0])
        ))

        interp = Interpretation(learner, dl, losses, act=None)
        result = interp[1]
        assert len(result) == 5
        # The losses returned should be the loss at index 1
        assert result[4].item() == pytest.approx(0.2)


# ============================================================
# Tests for ClassificationInterpretation
# ============================================================

class TestClassificationInterpretation:
    """Tests for ClassificationInterpretation."""

    def test_init_sets_vocab(self):
        """ClassificationInterpretation should store vocab from dl."""
        vocab = MockVocab(['cat', 'dog', 'bird'])
        interp = _make_classification_interpretation(vocab=vocab)
        assert interp.vocab == vocab

    def test_init_with_listy_vocab_takes_last(self):
        """If vocab is a list of vocabs, should use the last one."""
        from fastai.interpret import ClassificationInterpretation

        vocab_list = [MockVocab(['input_a', 'input_b']), MockVocab(['cat', 'dog'])]
        dl = _make_mock_dl(items=[0, 1, 2])
        dl.vocab = vocab_list
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.2, 0.3]))

        interp = ClassificationInterpretation(learner, dl, losses, act=None)
        assert interp.vocab == MockVocab(['cat', 'dog'])

    def test_confusion_matrix_shape(self):
        """confusion_matrix should return array of shape (n_classes, n_classes)."""
        vocab = MockVocab(['cat', 'dog', 'bird'])
        interp = _make_classification_interpretation(vocab=vocab)

        # Mock get_preds to return decoded predictions and targets
        n_samples = 10
        decoded = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        targs = torch.tensor([0, 1, 2, 0, 1, 0, 2, 1, 1, 0])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        cm = interp.confusion_matrix()
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_values(self):
        """confusion_matrix should correctly count classifications."""
        vocab = MockVocab(['cat', 'dog'])
        interp = _make_classification_interpretation(n_samples=4, vocab=vocab)

        # 4 samples: all predicted cat, but 2 are actually dog
        decoded = torch.tensor([0, 0, 0, 0])  # all predicted as cat
        targs = torch.tensor([0, 0, 1, 1])    # first 2 cat, last 2 dog
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        cm = interp.confusion_matrix()
        # cm[actual][predicted]
        # cat predicted as cat: 2
        assert cm[0, 0] == 2
        # cat predicted as dog: 0
        assert cm[0, 1] == 0
        # dog predicted as cat: 2
        assert cm[1, 0] == 2
        # dog predicted as dog: 0
        assert cm[1, 1] == 0

    def test_confusion_matrix_perfect_predictions(self):
        """With perfect predictions, diagonal should match counts."""
        vocab = MockVocab(['a', 'b', 'c'])
        interp = _make_classification_interpretation(n_samples=9, vocab=vocab)

        decoded = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        targs = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        cm = interp.confusion_matrix()
        assert cm[0, 0] == 3
        assert cm[1, 1] == 3
        assert cm[2, 2] == 3
        # Off-diagonal should be zero
        assert cm[0, 1] == 0
        assert cm[0, 2] == 0
        assert cm[1, 0] == 0
        assert cm[1, 2] == 0
        assert cm[2, 0] == 0
        assert cm[2, 1] == 0

    def test_most_confused_returns_sorted_descending(self):
        """most_confused should return (actual, predicted, count) sorted descending."""
        vocab = MockVocab(['cat', 'dog', 'bird'])
        interp = _make_classification_interpretation(vocab=vocab)

        decoded = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        targs = torch.tensor([1, 1, 0, 0, 0, 0, 0, 1, 2, 2])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        confused = interp.most_confused(min_val=1)
        # Should be a list of tuples
        assert isinstance(confused, list)
        # Each entry is (actual_name, predicted_name, count)
        for entry in confused:
            assert len(entry) == 3
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], str)
            assert isinstance(entry[2], (int, np.integer))
        # Should be sorted descending by count
        for i in range(len(confused) - 1):
            assert confused[i][2] >= confused[i + 1][2]

    def test_most_confused_min_val_filters(self):
        """most_confused with min_val should filter out low-count confusions."""
        vocab = MockVocab(['cat', 'dog', 'bird'])
        interp = _make_classification_interpretation(vocab=vocab)

        # Create scenario: dog confused as cat 3 times, bird as cat 1 time
        decoded = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        targs = torch.tensor([1, 1, 1, 2, 0, 1, 1, 1, 2, 2])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        confused_all = interp.most_confused(min_val=1)
        confused_high = interp.most_confused(min_val=3)

        # High threshold should have fewer entries
        assert len(confused_high) <= len(confused_all)
        # All entries in high threshold should have count >= 3
        for entry in confused_high:
            assert entry[2] >= 3

    def test_most_confused_excludes_diagonal(self):
        """most_confused should not include correct predictions (diagonal)."""
        vocab = MockVocab(['cat', 'dog'])
        interp = _make_classification_interpretation(n_samples=4, vocab=vocab)

        # All correct predictions
        decoded = torch.tensor([0, 0, 1, 1])
        targs = torch.tensor([0, 0, 1, 1])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        confused = interp.most_confused(min_val=1)
        assert confused == []

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.yticks')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.text')
    def test_plot_confusion_matrix_runs(self, *mocks):
        """plot_confusion_matrix should execute without error."""
        vocab = MockVocab(['cat', 'dog'])
        interp = _make_classification_interpretation(n_samples=4, vocab=vocab)

        decoded = torch.tensor([0, 0, 1, 1])
        targs = torch.tensor([0, 1, 0, 1])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        # Mock the figure's gca
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_fig.gca = MagicMock(return_value=mock_ax)
        mocks[-1].return_value = mock_fig  # plt.text is first in reversed order

        with patch('matplotlib.pyplot.figure', return_value=mock_fig):
            interp.plot_confusion_matrix()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.yticks')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.text')
    def test_plot_confusion_matrix_normalize(self, *mocks):
        """plot_confusion_matrix with normalize=True should not error."""
        vocab = MockVocab(['cat', 'dog'])
        interp = _make_classification_interpretation(n_samples=4, vocab=vocab)

        decoded = torch.tensor([0, 0, 1, 1])
        targs = torch.tensor([0, 1, 0, 1])
        interp.learn.get_preds = MagicMock(return_value=(None, targs, decoded))

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_fig.gca = MagicMock(return_value=mock_ax)

        with patch('matplotlib.pyplot.figure', return_value=mock_fig):
            interp.plot_confusion_matrix(normalize=True)


# ============================================================
# Tests for SegmentationInterpretation
# ============================================================

class TestSegmentationInterpretation:
    """Tests for SegmentationInterpretation."""

    def test_inherits_from_interpretation(self):
        """SegmentationInterpretation should be a subclass of Interpretation."""
        from fastai.interpret import SegmentationInterpretation, Interpretation
        assert issubclass(SegmentationInterpretation, Interpretation)

    def test_instantiation(self):
        """SegmentationInterpretation should instantiate like Interpretation."""
        from fastai.interpret import SegmentationInterpretation

        dl = _make_mock_dl(items=[0, 1, 2])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.5, 0.3, 0.7]))

        interp = SegmentationInterpretation(learner, dl, losses, act=None)
        assert interp.learn is learner
        assert interp.dl is dl
        assert torch.equal(interp.losses, losses)

    def test_top_losses_inherited(self):
        """SegmentationInterpretation should inherit top_losses from Interpretation."""
        from fastai.interpret import SegmentationInterpretation

        dl = _make_mock_dl(items=[0, 1, 2, 3, 4])
        learner = _make_mock_learner(dl=dl)
        losses = TensorBase(torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7]))

        interp = SegmentationInterpretation(learner, dl, losses, act=None)
        top_losses, idx = interp.top_losses(k=3, largest=True)
        assert len(top_losses) == 3
        assert top_losses[0].item() == pytest.approx(0.9)


# ============================================================
# Tests for module-level attributes
# ============================================================

class TestModuleAttributes:
    """Tests for module-level exports and attributes."""

    def test_all_exports(self):
        """__all__ should contain expected names."""
        from fastai import interpret
        expected = ['plot_top_losses', 'Interpretation',
                    'ClassificationInterpretation', 'SegmentationInterpretation']
        for name in expected:
            assert name in interpret.__all__

    def test_classes_are_importable(self):
        """All exported classes should be importable."""
        from fastai.interpret import (
            Interpretation, ClassificationInterpretation,
            SegmentationInterpretation, plot_top_losses
        )
        assert Interpretation is not None
        assert ClassificationInterpretation is not None
        assert SegmentationInterpretation is not None
        assert callable(plot_top_losses)
