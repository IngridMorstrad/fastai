"""Tokenizer-agnostic vocabulary inspector for analyzing token frequency, coverage, and OOV rates."""

from collections import Counter

__all__ = ['VocabInspector']


class _SpaceSplitTokenizer():
    "Fallback tokenizer that splits on whitespace"
    def __call__(self, items): return (t.split() for t in items)


class VocabInspector():
    "Inspect vocabulary coverage, token frequency, and OOV rate for any tokenizer"
    def __init__(self, vocab, tok=None, top_k=20):
        self.vocab = set(vocab)
        self.vocab_list = list(vocab)
        self.tok = tok or _SpaceSplitTokenizer()
        self.top_k = top_k

    def inspect(self, corpus, top_k=None):
        "Analyze `corpus` (list of strings) and return a report dict with frequency, coverage, and OOV info"
        k = top_k if top_k is not None else self.top_k
        token_freq = Counter()
        for toks in self.tok(corpus):
            token_freq.update(toks)

        corpus_size = sum(token_freq.values())
        vocab_size = len(token_freq)

        oov_tokens = Counter({t: c for t, c in token_freq.items() if t not in self.vocab})
        oov_count = sum(oov_tokens.values())

        coverage = (corpus_size - oov_count) / corpus_size if corpus_size > 0 else 0.0
        oov_rate = oov_count / corpus_size if corpus_size > 0 else 0.0

        return {
            'token_freq':  token_freq,
            'vocab_size':  vocab_size,
            'corpus_size': corpus_size,
            'coverage':    coverage,
            'oov_rate':    oov_rate,
            'oov_tokens':  oov_tokens,
            'top_tokens':  token_freq.most_common(k),
        }

    def summary(self, corpus, top_k=None):
        "Return a human-readable string summary of the vocab inspection report for `corpus`"
        rpt = self.inspect(corpus, top_k=top_k)
        k = top_k if top_k is not None else self.top_k
        lines = [
            'Vocabulary Inspection Report',
            '=' * 40,
            f"Corpus size (tokens): {rpt['corpus_size']}",
            f"Unique tokens:        {rpt['vocab_size']}",
            f"Vocabulary coverage:  {rpt['coverage']:.2%}",
            f"OOV rate:             {rpt['oov_rate']:.2%}",
            f"OOV unique tokens:    {len(rpt['oov_tokens'])}",
            '',
            f'Top {k} tokens:',
        ]
        for tok, cnt in rpt['top_tokens']:
            lines.append(f'  {tok:<20s} {cnt:>8d}')

        if rpt['oov_tokens']:
            top_oov = rpt['oov_tokens'].most_common(k)
            lines.append('')
            lines.append(f'Top {min(k, len(top_oov))} OOV tokens:')
            for tok, cnt in top_oov:
                lines.append(f'  {tok:<20s} {cnt:>8d}')

        return '\n'.join(lines)
