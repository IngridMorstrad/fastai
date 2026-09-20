"""Tests for fastai.text.vocab_inspector.

Covers VocabInspector: token frequency counting, vocabulary coverage,
OOV rate calculation, top-k token listing, human-readable summaries,
and custom tokenizer support.

All tests use pure Python with no heavy dependencies (no torch, spacy, etc.).
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.text.vocab_inspector import VocabInspector, _SpaceSplitTokenizer


# ============================================================
# Helper fixtures
# ============================================================

SIMPLE_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog",
]

VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "log", "and"]

OOV_CORPUS = [
    "the cat sat on the mat",
    "a zebra jumped over the fence",
]


# ============================================================
# Tests for _SpaceSplitTokenizer
# ============================================================

class TestSpaceSplitTokenizer:
    """Tests for the default fallback tokenizer."""

    def test_splits_on_whitespace(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok(["hello world"]))
        assert result == [["hello", "world"]]

    def test_handles_multiple_strings(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok(["a b", "c d e"]))
        assert result == [["a", "b"], ["c", "d", "e"]]

    def test_handles_empty_string(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok([""]))
        assert result == [[]]

    def test_handles_empty_corpus(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok([]))
        assert result == []

    def test_handles_multiple_spaces(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok(["a  b   c"]))
        # str.split() without args splits on any whitespace and drops empties
        assert result == [["a", "b", "c"]]

    def test_handles_tabs_and_newlines(self):
        tok = _SpaceSplitTokenizer()
        result = list(tok(["a\tb\nc"]))
        assert result == [["a", "b", "c"]]


# ============================================================
# Tests for VocabInspector construction
# ============================================================

class TestVocabInspectorInit:
    """Tests for VocabInspector initialization."""

    def test_default_tokenizer(self):
        vi = VocabInspector(vocab=["a", "b"])
        assert isinstance(vi.tok, _SpaceSplitTokenizer)

    def test_custom_tokenizer(self):
        class MyTok:
            def __call__(self, items): return (t.lower().split() for t in items)
        tok = MyTok()
        vi = VocabInspector(vocab=["a"], tok=tok)
        assert vi.tok is tok

    def test_default_top_k(self):
        vi = VocabInspector(vocab=["a"])
        assert vi.top_k == 20

    def test_custom_top_k(self):
        vi = VocabInspector(vocab=["a"], top_k=5)
        assert vi.top_k == 5

    def test_vocab_stored_as_set(self):
        vi = VocabInspector(vocab=["a", "b", "a"])
        assert vi.vocab == {"a", "b"}

    def test_empty_vocab(self):
        vi = VocabInspector(vocab=[])
        assert vi.vocab == set()


# ============================================================
# Tests for inspect() -- token_freq
# ============================================================

class TestInspectTokenFreq:
    """Tests for the token_freq field of the inspect report."""

    def test_token_freq_type(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert isinstance(rpt['token_freq'], Counter)

    def test_token_freq_counts(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['token_freq']['the'] == 6
        assert rpt['token_freq']['cat'] == 2
        assert rpt['token_freq']['sat'] == 2
        assert rpt['token_freq']['dog'] == 2
        assert rpt['token_freq']['on'] == 2
        assert rpt['token_freq']['mat'] == 1
        assert rpt['token_freq']['log'] == 1
        assert rpt['token_freq']['and'] == 1

    def test_token_freq_empty_corpus(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect([])
        assert len(rpt['token_freq']) == 0

    def test_token_freq_single_doc(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(["hello hello hello"])
        assert rpt['token_freq']['hello'] == 3


# ============================================================
# Tests for inspect() -- vocab_size
# ============================================================

class TestInspectVocabSize:
    """Tests for the vocab_size field (unique tokens in corpus)."""

    def test_vocab_size_simple(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        # "the cat sat on the mat" => the, cat, sat, on, mat
        # "the dog sat on the log" => dog, log (new)
        # "the cat and the dog"    => and (new)
        # unique: the, cat, sat, on, mat, dog, log, and = 8
        assert rpt['vocab_size'] == 8

    def test_vocab_size_empty_corpus(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect([])
        assert rpt['vocab_size'] == 0

    def test_vocab_size_with_oov(self):
        vi = VocabInspector(vocab=["the"])
        rpt = vi.inspect(["the cat the dog"])
        assert rpt['vocab_size'] == 3  # the, cat, dog


# ============================================================
# Tests for inspect() -- corpus_size
# ============================================================

class TestInspectCorpusSize:
    """Tests for the corpus_size field (total token count)."""

    def test_corpus_size_simple(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        # 6 + 6 + 5 = 17
        assert rpt['corpus_size'] == 17

    def test_corpus_size_empty(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect([])
        assert rpt['corpus_size'] == 0

    def test_corpus_size_single_token(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(["hello"])
        assert rpt['corpus_size'] == 1


# ============================================================
# Tests for inspect() -- coverage and oov_rate
# ============================================================

class TestInspectCoverage:
    """Tests for coverage and oov_rate fields."""

    def test_full_coverage(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['coverage'] == 1.0
        assert rpt['oov_rate'] == 0.0

    def test_partial_coverage(self):
        vi = VocabInspector(vocab=["the", "cat", "sat", "on", "mat"])
        rpt = vi.inspect(SIMPLE_CORPUS)
        # Known: the(6), cat(2), sat(2), on(2), mat(1) = 13 of 17
        assert abs(rpt['coverage'] - 13 / 17) < 1e-9
        assert abs(rpt['oov_rate'] - 4 / 17) < 1e-9

    def test_zero_coverage(self):
        vi = VocabInspector(vocab=["zzz"])
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['coverage'] == 0.0
        assert rpt['oov_rate'] == 1.0

    def test_empty_corpus_coverage(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect([])
        assert rpt['coverage'] == 0.0
        assert rpt['oov_rate'] == 0.0

    def test_empty_vocab_coverage(self):
        vi = VocabInspector(vocab=[])
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['coverage'] == 0.0
        assert rpt['oov_rate'] == 1.0

    def test_coverage_plus_oov_equals_one(self):
        vi = VocabInspector(vocab=["the", "dog"])
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert abs(rpt['coverage'] + rpt['oov_rate'] - 1.0) < 1e-9

    def test_coverage_with_mixed_oov(self):
        vi = VocabInspector(vocab=["the", "cat", "sat", "on", "mat"])
        rpt = vi.inspect(OOV_CORPUS)
        # doc1: the(2), cat(1), sat(1), on(1), mat(1) = 6 in vocab of 6 tokens
        # doc2: the(1) in vocab; a, zebra, jumped, over, fence = 5 OOV of 6 tokens
        # total: 7 in vocab, 5 OOV, 12 total
        assert rpt['corpus_size'] == 12
        assert abs(rpt['coverage'] - 7 / 12) < 1e-9


# ============================================================
# Tests for inspect() -- oov_tokens
# ============================================================

class TestInspectOovTokens:
    """Tests for the oov_tokens field."""

    def test_oov_tokens_type(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert isinstance(rpt['oov_tokens'], Counter)

    def test_no_oov_tokens_when_full_coverage(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert len(rpt['oov_tokens']) == 0

    def test_oov_tokens_identified(self):
        vi = VocabInspector(vocab=["the", "cat", "sat", "on", "mat"])
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert "dog" in rpt['oov_tokens']
        assert "log" in rpt['oov_tokens']
        assert "and" in rpt['oov_tokens']

    def test_oov_tokens_counts(self):
        vi = VocabInspector(vocab=["the", "cat", "sat", "on", "mat"])
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['oov_tokens']['dog'] == 2
        assert rpt['oov_tokens']['log'] == 1
        assert rpt['oov_tokens']['and'] == 1

    def test_oov_tokens_empty_vocab(self):
        vi = VocabInspector(vocab=[])
        rpt = vi.inspect(["hello world"])
        assert rpt['oov_tokens']['hello'] == 1
        assert rpt['oov_tokens']['world'] == 1


# ============================================================
# Tests for inspect() -- top_tokens
# ============================================================

class TestInspectTopTokens:
    """Tests for the top_tokens field."""

    def test_top_tokens_type(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert isinstance(rpt['top_tokens'], list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in rpt['top_tokens'])

    def test_top_tokens_ordered_by_frequency(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        counts = [c for _, c in rpt['top_tokens']]
        assert counts == sorted(counts, reverse=True)

    def test_top_tokens_first_is_most_frequent(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert rpt['top_tokens'][0] == ('the', 6)

    def test_top_k_default(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        # corpus has 8 unique tokens, default top_k=20, so all 8 returned
        assert len(rpt['top_tokens']) == 8

    def test_top_k_limits_output(self):
        vi = VocabInspector(vocab=VOCAB, top_k=3)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert len(rpt['top_tokens']) == 3

    def test_top_k_override_in_inspect(self):
        vi = VocabInspector(vocab=VOCAB, top_k=3)
        rpt = vi.inspect(SIMPLE_CORPUS, top_k=2)
        assert len(rpt['top_tokens']) == 2

    def test_top_k_larger_than_vocab(self):
        vi = VocabInspector(vocab=VOCAB, top_k=100)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert len(rpt['top_tokens']) == 8  # only 8 unique tokens exist


# ============================================================
# Tests for summary()
# ============================================================

class TestSummary:
    """Tests for the human-readable summary output."""

    def test_summary_is_string(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert isinstance(s, str)

    def test_summary_contains_header(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert 'Vocabulary Inspection Report' in s

    def test_summary_contains_corpus_size(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert '17' in s

    def test_summary_contains_unique_tokens(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert '8' in s  # 8 unique tokens

    def test_summary_contains_coverage(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert '100.00%' in s

    def test_summary_contains_oov_rate(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert '0.00%' in s

    def test_summary_shows_top_tokens(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        assert 'the' in s

    def test_summary_shows_oov_section_when_oov_exists(self):
        vi = VocabInspector(vocab=["the"])
        s = vi.summary(SIMPLE_CORPUS)
        assert 'OOV tokens' in s

    def test_summary_no_oov_section_when_full_coverage(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary(SIMPLE_CORPUS)
        # "OOV tokens:" header should not appear when there are zero OOV tokens
        assert 'OOV tokens:' not in s

    def test_summary_top_k_override(self):
        vi = VocabInspector(vocab=VOCAB, top_k=20)
        s = vi.summary(SIMPLE_CORPUS, top_k=2)
        assert 'Top 2 tokens' in s

    def test_summary_partial_coverage_values(self):
        vi = VocabInspector(vocab=["the", "cat"])
        s = vi.summary(SIMPLE_CORPUS)
        # coverage = 8/17 ~ 47.06%
        assert '47.06%' in s


# ============================================================
# Tests with custom tokenizer
# ============================================================

class TestCustomTokenizer:
    """Tests using a custom tokenizer to verify tokenizer-agnostic design."""

    def test_lowercase_tokenizer(self):
        class LowerTok:
            def __call__(self, items): return (t.lower().split() for t in items)

        vi = VocabInspector(vocab=["the", "cat"], tok=LowerTok())
        rpt = vi.inspect(["THE CAT", "The Cat"])
        assert rpt['token_freq']['the'] == 2
        assert rpt['token_freq']['cat'] == 2
        assert rpt['coverage'] == 1.0

    def test_char_tokenizer(self):
        class CharTok:
            def __call__(self, items): return (list(t) for t in items)

        vi = VocabInspector(vocab=list("abcdefghijklmnopqrstuvwxyz "), tok=CharTok())
        rpt = vi.inspect(["hello"])
        assert rpt['corpus_size'] == 5
        assert rpt['vocab_size'] == 4  # h, e, l, o
        assert rpt['token_freq']['l'] == 2
        assert rpt['coverage'] == 1.0

    def test_generator_based_tokenizer(self):
        """Tokenizer __call__ can return a generator (like BaseTokenizer)."""
        class GenTok:
            def __call__(self, items):
                return (t.split('-') for t in items)

        vi = VocabInspector(vocab=["a", "b", "c"], tok=GenTok())
        rpt = vi.inspect(["a-b-c", "a-d"])
        assert rpt['corpus_size'] == 5
        assert rpt['token_freq']['a'] == 2
        assert rpt['oov_tokens']['d'] == 1

    def test_tokenizer_returning_list(self):
        """Tokenizer __call__ can return a list of lists."""
        class ListTok:
            def __call__(self, items):
                return [t.split(',') for t in items]

        vi = VocabInspector(vocab=["x", "y"], tok=ListTok())
        rpt = vi.inspect(["x,y,z"])
        assert rpt['corpus_size'] == 3
        assert rpt['oov_tokens']['z'] == 1


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_token_corpus(self):
        vi = VocabInspector(vocab=["a"])
        rpt = vi.inspect(["a"])
        assert rpt['corpus_size'] == 1
        assert rpt['vocab_size'] == 1
        assert rpt['coverage'] == 1.0
        assert rpt['oov_rate'] == 0.0

    def test_single_oov_token(self):
        vi = VocabInspector(vocab=["a"])
        rpt = vi.inspect(["b"])
        assert rpt['coverage'] == 0.0
        assert rpt['oov_rate'] == 1.0
        assert rpt['oov_tokens']['b'] == 1

    def test_empty_strings_in_corpus(self):
        vi = VocabInspector(vocab=["a"])
        rpt = vi.inspect(["", "", ""])
        assert rpt['corpus_size'] == 0
        assert rpt['vocab_size'] == 0

    def test_large_top_k_on_small_corpus(self):
        vi = VocabInspector(vocab=["a"], top_k=1000)
        rpt = vi.inspect(["a b c"])
        assert len(rpt['top_tokens']) == 3

    def test_top_k_zero(self):
        vi = VocabInspector(vocab=VOCAB, top_k=0)
        rpt = vi.inspect(SIMPLE_CORPUS)
        assert len(rpt['top_tokens']) == 0

    def test_duplicate_tokens_in_vocab_list(self):
        vi = VocabInspector(vocab=["a", "a", "b", "b", "b"])
        assert vi.vocab == {"a", "b"}
        rpt = vi.inspect(["a b a"])
        assert rpt['coverage'] == 1.0

    def test_corpus_with_punctuation(self):
        vi = VocabInspector(vocab=["hello", "world"])
        rpt = vi.inspect(["hello, world!"])
        # default tokenizer splits on whitespace: "hello," and "world!" are OOV
        assert rpt['oov_rate'] == 1.0
        assert "hello," in rpt['oov_tokens']
        assert "world!" in rpt['oov_tokens']

    def test_repeated_doc_same_as_concatenated(self):
        vi = VocabInspector(vocab=["a", "b"])
        rpt1 = vi.inspect(["a b", "a b"])
        rpt2 = vi.inspect(["a b a b"])
        assert rpt1['corpus_size'] == rpt2['corpus_size']
        assert rpt1['coverage'] == rpt2['coverage']
        assert rpt1['token_freq'] == rpt2['token_freq']

    def test_report_dict_has_all_keys(self):
        vi = VocabInspector(vocab=VOCAB)
        rpt = vi.inspect(SIMPLE_CORPUS)
        expected_keys = {'token_freq', 'vocab_size', 'corpus_size', 'coverage', 'oov_rate', 'oov_tokens', 'top_tokens'}
        assert set(rpt.keys()) == expected_keys

    def test_summary_on_empty_corpus(self):
        vi = VocabInspector(vocab=VOCAB)
        s = vi.summary([])
        assert 'Vocabulary Inspection Report' in s
        assert '0' in s
