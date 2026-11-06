"""Tests for the extractive-summarization core backing `TextLearner.summarize`.

These exercise the frequency-based extractive fallback (`_summarize_extractive`),
which needs no trained model, network, or optional abstractive backend. Importing
`fastai.text.learner` pulls in torch like the rest of the suite, so these run in a
torch-enabled environment (the CI nbdev-sync job does not execute pytest).
"""

import re

import pytest

from fastai.text.learner import _summarize_extractive, _truncate_to_words

_WORD = re.compile(r"[A-Za-z0-9']+")

_TEXT = (
    "Cats are small animals. Cats are popular pets around the world. "
    "Dogs are loyal animals. Many people keep dogs as pets. "
    "Cats and dogs are the most common household pets."
)


class TestExtractiveSummarize:
    """Unit tests for the extractive summarization fallback."""

    def test_returns_non_empty_within_budget(self):
        """A multi-sentence input yields a non-empty summary within the word budget."""
        summary = _summarize_extractive(_TEXT, max_length=12)
        assert summary.strip()
        assert len(_WORD.findall(summary)) <= 12

    def test_budget_is_enforced_hard(self):
        """Even a single over-long sentence is truncated to the word budget."""
        long_sentence = " ".join(f"word{i}" for i in range(40)) + "."
        summary = _summarize_extractive(long_sentence, max_length=5)
        assert 0 < len(_WORD.findall(summary)) <= 5

    def test_single_sentence_returns_something(self):
        """A single-sentence input still returns a non-empty summary."""
        assert _summarize_extractive("One lonely sentence here.", max_length=5).strip()

    def test_preserves_original_order(self):
        """Selected sentences appear in their original document order."""
        summary = _summarize_extractive(_TEXT, max_length=30)
        assert summary.index("Cats are small") < summary.index("Dogs are loyal")

    def test_truncation_retains_sentence_terminator(self):
        """When truncation lands on a word followed by a terminator, the terminator is kept."""
        summary = _summarize_extractive(_TEXT, max_length=12)
        assert summary.endswith(".")

    def test_truncation_does_not_fabricate_terminator(self):
        """No terminator is added when truncation lands mid-sentence."""
        long_sentence = " ".join(f"word{i}" for i in range(40)) + "."
        summary = _summarize_extractive(long_sentence, max_length=5)
        assert not summary.endswith(".")


class TestTruncateToWords:
    """Unit tests for the shared word-budget truncation helper."""

    def test_keeps_terminator_at_boundary(self):
        """A trailing '.'/'!'/'?' immediately after the last kept word is retained."""
        assert _truncate_to_words("a b c.", 3) == "a b c."
        assert _truncate_to_words("a b c!", 3) == "a b c!"
        assert _truncate_to_words("a b c?", 3) == "a b c?"

    def test_truncates_to_word_budget(self):
        """Only `max_length` words are kept and no stray terminator is invented."""
        assert _truncate_to_words("a b c d", 2) == "a b"

    def test_under_budget_is_unchanged(self):
        """Text already within budget is returned intact (modulo surrounding whitespace)."""
        assert _truncate_to_words("hello world.", 5) == "hello world."

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_empty_text_raises(self, text):
        """Empty or whitespace-only text fails loudly instead of returning empty output."""
        with pytest.raises(ValueError):
            _summarize_extractive(text, max_length=10)

    @pytest.mark.parametrize("max_length", [0, -1, -10])
    def test_non_positive_max_length_raises(self, max_length):
        """A non-positive max_length fails loudly."""
        with pytest.raises(ValueError):
            _summarize_extractive(_TEXT, max_length=max_length)
