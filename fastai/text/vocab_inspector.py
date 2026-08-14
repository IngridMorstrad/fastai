"""Tokenizer-agnostic vocabulary inspector for analyzing token frequency, coverage, and OOV rate.

This module provides utilities to inspect how a vocabulary performs on a given corpus,
helping users make informed text preprocessing decisions. It works with any tokenizer
that implements a callable interface (str -> list of str).
"""

from collections import Counter


class VocabInspector:
    """Analyzes token frequency, vocabulary coverage, and out-of-vocabulary rate for a corpus.

    Works with any tokenizer that accepts a string and returns a list of token strings.

    Usage::

        from fastai.text.vocab_inspector import VocabInspector

        tokenizer = lambda text: text.lower().split()
        corpus = ["The cat sat on the mat", "The dog sat on the log"]
        vocab = {"the", "cat", "sat", "on", "mat", "dog"}

        inspector = VocabInspector(tokenizer, vocab)
        report = inspector.inspect(corpus)
        print(report)
    """

    def __init__(self, tokenizer, vocab):
        """Initialize the VocabInspector.

        Args:
            tokenizer: A callable that takes a string and returns a list of token strings.
                       Can be any function, lambda, or object with a __call__ method.
            vocab: A collection (set, list, or dict) of known vocabulary tokens.
        """
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable (str -> list of str)")
        if vocab is None:
            raise ValueError("vocab must not be None")
        self.tokenizer = tokenizer
        self.vocab = set(vocab)

    def inspect(self, corpus):
        """Analyze the corpus and return a VocabReport with frequency, coverage, and OOV stats.

        Args:
            corpus: An iterable of strings (documents/sentences) to analyze.

        Returns:
            A VocabReport containing the analysis results.
        """
        token_counts = Counter()
        total_tokens = 0
        oov_counts = Counter()
        total_oov = 0

        for doc in corpus:
            tokens = self.tokenizer(doc)
            token_counts.update(tokens)
            total_tokens += len(tokens)
            for tok in tokens:
                if tok not in self.vocab:
                    oov_counts[tok] += 1
                    total_oov += 1

        used_vocab = set(token_counts.keys()) & self.vocab
        coverage = len(used_vocab) / len(self.vocab) if self.vocab else 0.0
        oov_rate = total_oov / total_tokens if total_tokens > 0 else 0.0

        return VocabReport(
            token_freq=token_counts,
            total_tokens=total_tokens,
            vocab_size=len(self.vocab),
            used_vocab_size=len(used_vocab),
            coverage=coverage,
            oov_rate=oov_rate,
            oov_token_freq=oov_counts,
            total_oov_tokens=total_oov,
        )


class VocabReport:
    """Results from a vocabulary inspection analysis.

    Attributes:
        token_freq: Counter of all token frequencies in the corpus.
        total_tokens: Total number of tokens in the corpus.
        vocab_size: Size of the provided vocabulary.
        used_vocab_size: Number of vocab tokens actually seen in the corpus.
        coverage: Fraction of vocabulary used (used_vocab_size / vocab_size).
        oov_rate: Fraction of corpus tokens that are out-of-vocabulary.
        oov_token_freq: Counter of OOV token frequencies.
        total_oov_tokens: Total number of OOV token occurrences.
    """

    def __init__(self, token_freq, total_tokens, vocab_size, used_vocab_size,
                 coverage, oov_rate, oov_token_freq, total_oov_tokens):
        self.token_freq = token_freq
        self.total_tokens = total_tokens
        self.vocab_size = vocab_size
        self.used_vocab_size = used_vocab_size
        self.coverage = coverage
        self.oov_rate = oov_rate
        self.oov_token_freq = oov_token_freq
        self.total_oov_tokens = total_oov_tokens

    @property
    def most_common(self):
        """Return all tokens sorted by frequency (most common first)."""
        return self.token_freq.most_common()

    @property
    def oov_most_common(self):
        """Return OOV tokens sorted by frequency (most common first)."""
        return self.oov_token_freq.most_common()

    @property
    def in_vocab_rate(self):
        """Fraction of corpus tokens that are in-vocabulary (1 - oov_rate)."""
        return 1.0 - self.oov_rate

    def top_tokens(self, n=10):
        """Return the top-n most frequent tokens."""
        return self.token_freq.most_common(n)

    def top_oov_tokens(self, n=10):
        """Return the top-n most frequent OOV tokens."""
        return self.oov_token_freq.most_common(n)

    def summary(self):
        """Return a dict summarizing the key metrics."""
        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": len(self.token_freq),
            "vocab_size": self.vocab_size,
            "used_vocab_size": self.used_vocab_size,
            "coverage": round(self.coverage, 4),
            "oov_rate": round(self.oov_rate, 4),
            "in_vocab_rate": round(self.in_vocab_rate, 4),
            "total_oov_tokens": self.total_oov_tokens,
            "unique_oov_tokens": len(self.oov_token_freq),
        }

    def __repr__(self):
        s = self.summary()
        lines = [
            "VocabReport:",
            f"  Total tokens:      {s['total_tokens']}",
            f"  Unique tokens:     {s['unique_tokens']}",
            f"  Vocab size:        {s['vocab_size']}",
            f"  Vocab used:        {s['used_vocab_size']} ({s['coverage']:.2%} coverage)",
            f"  OOV rate:          {s['oov_rate']:.2%} ({s['total_oov_tokens']} tokens, {s['unique_oov_tokens']} unique)",
            f"  In-vocab rate:     {s['in_vocab_rate']:.2%}",
        ]
        return "\n".join(lines)


def vocab_inspect(tokenizer, corpus, vocab):
    """Convenience function to inspect vocabulary performance on a corpus.

    Args:
        tokenizer: A callable that takes a string and returns a list of token strings.
        corpus: An iterable of strings to analyze.
        vocab: A collection of known vocabulary tokens.

    Returns:
        A VocabReport with the analysis results.
    """
    inspector = VocabInspector(tokenizer, vocab)
    return inspector.inspect(corpus)
