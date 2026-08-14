"""Tests for fastai.text.vocab_inspector module."""

import pytest
from collections import Counter
from fastai.text.vocab_inspector import VocabInspector, VocabReport, vocab_inspect


# -- Fixtures and helpers --

def simple_tokenizer(text):
    """Whitespace tokenizer that lowercases."""
    return text.lower().split()


def identity_tokenizer(text):
    """Splits on whitespace without modification."""
    return text.split()


SAMPLE_CORPUS = [
    "The cat sat on the mat",
    "The dog sat on the log",
]

SAMPLE_VOCAB = {"the", "cat", "sat", "on", "mat", "dog"}


# -- VocabInspector construction tests --

class TestVocabInspectorInit:
    def test_accepts_callable_tokenizer(self):
        inspector = VocabInspector(simple_tokenizer, {"a", "b"})
        assert inspector.tokenizer is simple_tokenizer

    def test_accepts_lambda_tokenizer(self):
        tok = lambda x: x.split()
        inspector = VocabInspector(tok, {"a"})
        assert callable(inspector.tokenizer)

    def test_accepts_list_vocab(self):
        inspector = VocabInspector(simple_tokenizer, ["a", "b", "c"])
        assert inspector.vocab == {"a", "b", "c"}

    def test_accepts_dict_vocab(self):
        inspector = VocabInspector(simple_tokenizer, {"a": 0, "b": 1})
        assert inspector.vocab == {"a", "b"}

    def test_rejects_non_callable_tokenizer(self):
        with pytest.raises(TypeError, match="tokenizer must be callable"):
            VocabInspector("not_callable", {"a"})

    def test_rejects_none_vocab(self):
        with pytest.raises(ValueError, match="vocab must not be None"):
            VocabInspector(simple_tokenizer, None)

    def test_accepts_empty_vocab(self):
        inspector = VocabInspector(simple_tokenizer, set())
        assert inspector.vocab == set()


# -- Inspection result tests --

class TestVocabInspectorInspect:
    def test_total_tokens(self):
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        report = inspector.inspect(SAMPLE_CORPUS)
        # "the cat sat on the mat" -> 6, "the dog sat on the log" -> 6
        assert report.total_tokens == 12

    def test_token_frequency(self):
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        report = inspector.inspect(SAMPLE_CORPUS)
        assert report.token_freq["the"] == 4
        assert report.token_freq["sat"] == 2
        assert report.token_freq["cat"] == 1
        assert report.token_freq["dog"] == 1

    def test_full_coverage(self):
        # All vocab tokens appear in corpus
        corpus = ["a b c"]
        vocab = {"a", "b", "c"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.coverage == 1.0
        assert report.used_vocab_size == 3

    def test_partial_coverage(self):
        corpus = ["a b"]
        vocab = {"a", "b", "c", "d"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.coverage == 0.5
        assert report.used_vocab_size == 2

    def test_zero_coverage_with_empty_vocab(self):
        inspector = VocabInspector(simple_tokenizer, set())
        report = inspector.inspect(["hello world"])
        assert report.coverage == 0.0

    def test_oov_rate_no_oov(self):
        corpus = ["cat sat"]
        vocab = {"cat", "sat"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.oov_rate == 0.0
        assert report.total_oov_tokens == 0

    def test_oov_rate_all_oov(self):
        corpus = ["unknown words here"]
        vocab = {"cat", "dog"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.oov_rate == 1.0
        assert report.total_oov_tokens == 3

    def test_oov_rate_partial(self):
        corpus = ["cat unknown"]
        vocab = {"cat"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.oov_rate == 0.5
        assert report.total_oov_tokens == 1

    def test_oov_token_freq(self):
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        report = inspector.inspect(SAMPLE_CORPUS)
        # "log" and "mat" are in vocab; "the" lowercased is in vocab
        # With simple_tokenizer and SAMPLE_VOCAB = {"the","cat","sat","on","mat","dog"}
        # All tokens: the(4), cat(1), sat(2), on(2), mat(1), dog(1), log(1)
        # OOV: log(1)
        assert report.oov_token_freq["log"] == 1
        assert report.total_oov_tokens == 1

    def test_empty_corpus(self):
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        report = inspector.inspect([])
        assert report.total_tokens == 0
        assert report.oov_rate == 0.0
        assert report.coverage == 0.0

    def test_corpus_with_empty_strings(self):
        inspector = VocabInspector(simple_tokenizer, {"hello"})
        report = inspector.inspect(["", "", "hello"])
        assert report.total_tokens == 1
        assert report.oov_rate == 0.0

    def test_in_vocab_rate(self):
        corpus = ["cat unknown"]
        vocab = {"cat"}
        inspector = VocabInspector(simple_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.in_vocab_rate == 0.5

    def test_case_sensitive_tokenizer(self):
        # identity_tokenizer preserves case
        corpus = ["The CAT"]
        vocab = {"The", "cat"}  # "CAT" not in vocab
        inspector = VocabInspector(identity_tokenizer, vocab)
        report = inspector.inspect(corpus)
        assert report.oov_token_freq["CAT"] == 1
        assert "The" not in report.oov_token_freq


# -- VocabReport methods tests --

class TestVocabReport:
    def setup_method(self):
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        self.report = inspector.inspect(SAMPLE_CORPUS)

    def test_most_common(self):
        mc = self.report.most_common
        # "the" should be first with count 4
        assert mc[0] == ("the", 4)

    def test_top_tokens(self):
        top3 = self.report.top_tokens(3)
        assert len(top3) == 3
        assert top3[0][0] == "the"

    def test_top_oov_tokens(self):
        top = self.report.top_oov_tokens(5)
        assert ("log", 1) in top

    def test_oov_most_common(self):
        oov = self.report.oov_most_common
        assert ("log", 1) in oov

    def test_summary_keys(self):
        s = self.report.summary()
        expected_keys = {
            "total_tokens", "unique_tokens", "vocab_size", "used_vocab_size",
            "coverage", "oov_rate", "in_vocab_rate", "total_oov_tokens",
            "unique_oov_tokens",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_values(self):
        s = self.report.summary()
        assert s["total_tokens"] == 12
        assert s["vocab_size"] == 6
        assert s["unique_oov_tokens"] == 1

    def test_repr_contains_key_info(self):
        r = repr(self.report)
        assert "VocabReport:" in r
        assert "Total tokens:" in r
        assert "coverage" in r.lower()
        assert "OOV rate:" in r


# -- Convenience function tests --

class TestVocabInspectFunction:
    def test_basic_usage(self):
        report = vocab_inspect(simple_tokenizer, SAMPLE_CORPUS, SAMPLE_VOCAB)
        assert isinstance(report, VocabReport)
        assert report.total_tokens == 12

    def test_matches_class_output(self):
        report_func = vocab_inspect(simple_tokenizer, SAMPLE_CORPUS, SAMPLE_VOCAB)
        inspector = VocabInspector(simple_tokenizer, SAMPLE_VOCAB)
        report_class = inspector.inspect(SAMPLE_CORPUS)
        assert report_func.summary() == report_class.summary()


# -- Edge case tests --

class TestEdgeCases:
    def test_single_document_corpus(self):
        report = vocab_inspect(simple_tokenizer, ["hello world"], {"hello", "world"})
        assert report.total_tokens == 2
        assert report.oov_rate == 0.0
        assert report.coverage == 1.0

    def test_large_vocab_small_corpus(self):
        vocab = {f"word{i}" for i in range(1000)}
        report = vocab_inspect(simple_tokenizer, ["word0 word1"], vocab)
        assert report.coverage == 2 / 1000
        assert report.oov_rate == 0.0

    def test_duplicate_tokens_in_vocab(self):
        # list with duplicates - should deduplicate
        vocab = ["a", "a", "b", "b", "c"]
        inspector = VocabInspector(simple_tokenizer, vocab)
        assert inspector.vocab == {"a", "b", "c"}

    def test_tokenizer_returning_empty_list(self):
        empty_tok = lambda x: []
        report = vocab_inspect(empty_tok, ["some text"], {"some", "text"})
        assert report.total_tokens == 0
        assert report.oov_rate == 0.0

    def test_generator_corpus(self):
        def gen_corpus():
            yield "hello world"
            yield "foo bar"

        report = vocab_inspect(simple_tokenizer, gen_corpus(), {"hello", "world", "foo", "bar"})
        assert report.total_tokens == 4
        assert report.oov_rate == 0.0

    def test_special_characters_in_tokens(self):
        vocab = {"hello!", "@world", "#tag"}
        tok = lambda x: x.split()
        report = vocab_inspect(tok, ["hello! @world #tag unknown"], vocab)
        assert report.oov_rate == 0.25  # 1 out of 4
        assert report.oov_token_freq["unknown"] == 1
