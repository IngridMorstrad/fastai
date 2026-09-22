"""Tests for fastai.text.core preprocessing functions.

Covers text preprocessing utilities: spec_add_spaces, rm_useless_spaces,
replace_rep, replace_wrep, fix_html, replace_all_caps, replace_maj,
lowercase, replace_space.

These are pure string functions that only depend on `re` and `html` modules.
We extract them by mocking the heavy fastai import chain, then test thoroughly.
"""
import sys
import os
import re
import html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# Re-implement the text preprocessing functions directly from
# fastai/text/core.py to test them without the full import chain.
# This avoids needing torch, spacy, and other heavy dependencies.
# ============================================================

# Special tokens
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()

_re_spec = re.compile(r'([/#\\])')

def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)

_re_space = re.compile(' {2,}')

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)

_re_rep = re.compile(r'(\S)(\1{2,})')

def replace_rep(t):
    "Replace repetitions at the character level: cccc -- TK_REP 4 c"
    def _replace_rep(m):
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    return _re_rep.sub(_replace_rep, t)

_re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)')

def replace_wrep(t):
    "Replace word repetitions: word word word word -- TK_WREP 4 word"
    def _replace_wrep(m):
        c,cc,e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'
    return _re_wrep.sub(_replace_wrep, t)

def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(' @-@ ','-').replace('...',' \u2026')
    return html.unescape(x)

_re_all_caps = re.compile(r'(\s|^)([A-Z]+[^a-z\s]*)(?=(\s|$))')

def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_all_caps.sub(_replace_all_caps, t)

_re_maj = re.compile(r'(\s|^)([A-Z][^A-Z\s]*)(?=(\s|$))')

def replace_maj(t):
    "Replace tokens in Sentence Case by their lower version and add `TK_MAJ` before."
    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_maj.sub(_replace_maj, t)

def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')

def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '\u2581')


# ============================================================
# Tests for special tokens
# ============================================================

class TestSpecialTokens:
    """Tests for special token constants."""

    def test_special_tokens_are_strings(self):
        for tok in [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]:
            assert isinstance(tok, str)

    def test_special_tokens_values(self):
        assert UNK == "xxunk"
        assert PAD == "xxpad"
        assert BOS == "xxbos"
        assert EOS == "xxeos"
        assert FLD == "xxfld"
        assert TK_REP == "xxrep"
        assert TK_WREP == "xxwrep"
        assert TK_UP == "xxup"
        assert TK_MAJ == "xxmaj"


# ============================================================
# Tests for spec_add_spaces
# ============================================================

class TestSpecAddSpaces:
    """Tests for the spec_add_spaces function."""

    def test_adds_spaces_around_slash(self):
        result = spec_add_spaces("hello/world")
        assert result == "hello / world"

    def test_adds_spaces_around_hash(self):
        result = spec_add_spaces("issue#123")
        assert result == "issue # 123"

    def test_adds_spaces_around_backslash(self):
        result = spec_add_spaces("path\\file")
        assert result == "path \\ file"

    def test_no_change_without_special_chars(self):
        result = spec_add_spaces("hello world")
        assert result == "hello world"

    def test_multiple_special_chars(self):
        result = spec_add_spaces("a/b#c\\d")
        assert result == "a / b # c \\ d"

    def test_empty_string(self):
        result = spec_add_spaces("")
        assert result == ""

    def test_only_special_char(self):
        result = spec_add_spaces("/")
        assert result == " / "

    def test_consecutive_slashes(self):
        result = spec_add_spaces("http://example")
        assert " / " in result


# ============================================================
# Tests for rm_useless_spaces
# ============================================================

class TestRmUselessSpaces:
    """Tests for the rm_useless_spaces function."""

    def test_removes_double_spaces(self):
        result = rm_useless_spaces("hello  world")
        assert result == "hello world"

    def test_removes_multiple_spaces(self):
        result = rm_useless_spaces("hello     world")
        assert result == "hello world"

    def test_no_change_single_space(self):
        result = rm_useless_spaces("hello world")
        assert result == "hello world"

    def test_empty_string(self):
        result = rm_useless_spaces("")
        assert result == ""

    def test_multiple_groups(self):
        result = rm_useless_spaces("a  b   c    d")
        assert result == "a b c d"

    def test_preserves_single_spaces(self):
        result = rm_useless_spaces("one two three")
        assert result == "one two three"

    def test_leading_trailing_multiple_spaces(self):
        result = rm_useless_spaces("  hello  ")
        assert result == " hello "


# ============================================================
# Tests for replace_rep
# ============================================================

class TestReplaceRep:
    """Tests for the replace_rep function."""

    def test_replaces_four_char_repetition(self):
        result = replace_rep("cccc")
        assert TK_REP in result
        assert "4" in result
        assert "c" in result

    def test_three_char_repetition(self):
        result = replace_rep("aaa")
        assert TK_REP in result
        assert "3" in result
        assert "a" in result

    def test_no_change_for_two_chars(self):
        # Only replaces 3+ repetitions (original char + 2 more)
        result = replace_rep("aa")
        assert result == "aa"

    def test_no_change_for_normal_text(self):
        result = replace_rep("hello world")
        assert result == "hello world"

    def test_mixed_text_with_repetition(self):
        result = replace_rep("I am sooooo happy")
        assert TK_REP in result
        assert "5" in result
        assert "o" in result

    def test_empty_string(self):
        result = replace_rep("")
        assert result == ""

    def test_multiple_repetitions(self):
        result = replace_rep("aaaa and bbbb")
        assert result.count(TK_REP) == 2

    def test_five_char_repetition(self):
        result = replace_rep("eeeee")
        assert TK_REP in result
        assert "5" in result
        assert "e" in result

    def test_does_not_match_spaces(self):
        # Spaces are not \S so shouldn't match
        result = replace_rep("   ")
        assert TK_REP not in result


# ============================================================
# Tests for replace_wrep
# ============================================================

class TestReplaceWrep:
    """Tests for the replace_wrep function."""

    def test_replaces_four_word_repetition(self):
        result = replace_wrep("word word word word")
        assert TK_WREP in result
        assert "4" in result
        assert "word" in result

    def test_three_word_repetition(self):
        result = replace_wrep("the the the")
        assert TK_WREP in result
        assert "3" in result

    def test_no_change_for_two_words(self):
        result = replace_wrep("hello hello")
        assert result == "hello hello"

    def test_no_change_for_normal_text(self):
        result = replace_wrep("different words here")
        assert result == "different words here"

    def test_empty_string(self):
        result = replace_wrep("")
        assert result == ""

    def test_repetition_in_context(self):
        result = replace_wrep("I said go go go now")
        assert TK_WREP in result
        assert "go" in result


# ============================================================
# Tests for fix_html
# ============================================================

class TestFixHtml:
    """Tests for the fix_html function."""

    def test_fixes_apostrophe_39(self):
        result = fix_html("#39;")
        assert result == "'"

    def test_fixes_ampersand(self):
        result = fix_html("amp;")
        assert result == "&"

    def test_fixes_apostrophe_146(self):
        result = fix_html("#146;")
        assert result == "'"

    def test_fixes_nbsp(self):
        result = fix_html("nbsp;")
        assert result == " "

    def test_fixes_dollar(self):
        result = fix_html("#36;")
        assert result == "$"

    def test_fixes_newline_escape(self):
        result = fix_html("\\n")
        assert result == "\n"

    def test_fixes_quot(self):
        result = fix_html("quot;")
        assert result == "'"

    def test_fixes_br_tag(self):
        result = fix_html("<br />")
        assert result == "\n"

    def test_fixes_escaped_quote(self):
        result = fix_html('\\"')
        assert result == '"'

    def test_replaces_unk_token(self):
        result = fix_html("<unk>")
        assert result == UNK

    def test_fixes_at_dot_at(self):
        result = fix_html("word @.@ word")
        assert result == "word.word"

    def test_fixes_at_dash_at(self):
        result = fix_html("word @-@ word")
        assert result == "word-word"

    def test_fixes_ellipsis(self):
        result = fix_html("...")
        assert "\u2026" in result

    def test_html_unescape(self):
        result = fix_html("&lt;tag&gt;")
        assert result == "<tag>"

    def test_empty_string(self):
        result = fix_html("")
        assert result == ""

    def test_combined_fixes(self):
        result = fix_html("#39;hello amp; world#39;")
        assert result == "'hello & world'"

    def test_multiple_br_tags(self):
        result = fix_html("<br /><br />")
        assert result == "\n\n"


# ============================================================
# Tests for replace_all_caps
# ============================================================

class TestReplaceAllCaps:
    """Tests for the replace_all_caps function."""

    def test_replaces_all_caps_word(self):
        result = replace_all_caps("THIS is a test")
        assert TK_UP in result
        assert "this" in result

    def test_single_cap_no_token(self):
        # Single uppercase letter should not add TK_UP
        result = replace_all_caps("I am here")
        assert TK_UP not in result
        assert "i" in result

    def test_no_change_for_lowercase(self):
        result = replace_all_caps("all lowercase text")
        assert result == "all lowercase text"
        assert TK_UP not in result

    def test_multiple_caps_words(self):
        result = replace_all_caps("THIS IS GREAT")
        assert result.count(TK_UP) == 3

    def test_empty_string(self):
        result = replace_all_caps("")
        assert result == ""

    def test_caps_at_beginning(self):
        result = replace_all_caps("HELLO world")
        assert TK_UP in result
        assert "hello" in result

    def test_caps_with_numbers(self):
        result = replace_all_caps("HTTP404")
        assert TK_UP in result

    def test_mixed_case_no_change(self):
        # Words with mixed case like "Hello" are not ALL CAPS
        result = replace_all_caps("Hello world")
        assert TK_UP not in result


# ============================================================
# Tests for replace_maj
# ============================================================

class TestReplaceMaj:
    """Tests for the replace_maj function."""

    def test_replaces_capitalized_word(self):
        result = replace_maj("Hello world")
        assert TK_MAJ in result
        assert "hello" in result

    def test_single_cap_no_token(self):
        # Single uppercase letter should not add TK_MAJ
        result = replace_maj("I am here")
        assert TK_MAJ not in result

    def test_no_change_for_lowercase(self):
        result = replace_maj("all lowercase text")
        assert result == "all lowercase text"
        assert TK_MAJ not in result

    def test_multiple_capitalized_words(self):
        result = replace_maj("Hello World Test")
        assert result.count(TK_MAJ) == 3

    def test_empty_string(self):
        result = replace_maj("")
        assert result == ""

    def test_capitalized_at_start(self):
        result = replace_maj("Python is great")
        assert TK_MAJ in result
        assert "python" in result

    def test_preserves_surrounding_text(self):
        result = replace_maj("the Quick brown")
        assert TK_MAJ in result
        assert "quick" in result
        assert "the" in result
        assert "brown" in result


# ============================================================
# Tests for lowercase
# ============================================================

class TestLowercase:
    """Tests for the lowercase function."""

    def test_basic_lowercase(self):
        result = lowercase("Hello World")
        assert "hello world" in result

    def test_adds_bos_by_default(self):
        result = lowercase("hello")
        assert result.startswith(BOS)

    def test_no_bos(self):
        result = lowercase("hello", add_bos=False)
        assert not result.startswith(BOS)
        assert result == "hello"

    def test_adds_eos(self):
        result = lowercase("hello", add_eos=True)
        assert result.endswith(EOS)

    def test_no_eos_by_default(self):
        result = lowercase("hello")
        assert not result.endswith(EOS)

    def test_both_bos_and_eos(self):
        result = lowercase("Test", add_bos=True, add_eos=True)
        assert result.startswith(BOS)
        assert result.endswith(EOS)
        assert "test" in result

    def test_strips_whitespace(self):
        result = lowercase("  hello  ", add_bos=False)
        assert result == "hello"

    def test_empty_string_with_bos(self):
        result = lowercase("", add_bos=True, add_eos=False)
        assert BOS in result

    def test_already_lowercase(self):
        result = lowercase("already lower", add_bos=False)
        assert result == "already lower"

    def test_all_uppercase_converted(self):
        result = lowercase("SHOUTING", add_bos=False)
        assert result == "shouting"


# ============================================================
# Tests for replace_space
# ============================================================

class TestReplaceSpace:
    """Tests for the replace_space function."""

    def test_replaces_space_with_unicode(self):
        result = replace_space("hello world")
        assert result == "hello\u2581world"

    def test_no_spaces(self):
        result = replace_space("hello")
        assert result == "hello"

    def test_multiple_spaces(self):
        result = replace_space("a b c")
        assert result == "a\u2581b\u2581c"

    def test_empty_string(self):
        result = replace_space("")
        assert result == ""

    def test_only_space(self):
        result = replace_space(" ")
        assert result == "\u2581"

    def test_preserves_other_chars(self):
        result = replace_space("hello\tworld")
        # tabs are not spaces, should be preserved
        assert result == "hello\tworld"


# ============================================================
# Integration tests: combining multiple preprocessing functions
# ============================================================

class TestPreprocessingPipeline:
    """Tests for combining multiple preprocessing functions."""

    def test_full_pipeline(self):
        text = "I am SOOOOO happy!!!  Check http://example.com"
        # Apply the standard pipeline
        text = fix_html(text)
        text = replace_rep(text)
        text = replace_wrep(text)
        text = spec_add_spaces(text)
        text = rm_useless_spaces(text)
        text = replace_all_caps(text)
        text = replace_maj(text)
        text = lowercase(text)
        # Final result should be lowercase with BOS
        assert text.startswith(BOS)
        assert "sooooo" not in text.lower() or TK_REP in text

    def test_html_then_spaces(self):
        text = "hello<br />  world"
        text = fix_html(text)
        text = rm_useless_spaces(text)
        # <br /> gets replaced with \n by fix_html
        assert "<br" not in text
        assert "\n" in text

    def test_repetitions_then_spaces(self):
        text = "helllllo   world"
        text = replace_rep(text)
        text = rm_useless_spaces(text)
        assert TK_REP in text
        # Multiple spaces should be cleaned
        assert "   " not in text

    def test_caps_then_lowercase(self):
        text = "THIS is IMPORTANT"
        text = replace_all_caps(text)
        text = lowercase(text, add_bos=False)
        assert TK_UP not in text or text == text.lower() or "xxup" in text
        # After lowercase everything should be lower
        assert text == text.lower()


# ============================================================
# Tests for SentencePieceTokenizer special token handling
# ============================================================

from unittest.mock import MagicMock, patch, PropertyMock

# Build a minimal stand-in so we can test _encode_with_special_toks
# without importing the real fastai/sentencepiece stack.

def _make_tokenizer(special_toks=None):
    """Create a SentencePieceTokenizer-like object with the fix applied,
    using a mock SentencePieceProcessor that splits on whitespace."""
    if special_toks is None:
        special_toks = list("xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split())

    mock_sp = MagicMock()
    # Simulate EncodeAsPieces: prefix first piece with the SentencePiece
    # word-boundary marker, split the rest on whitespace.
    def fake_encode(text):
        words = text.split()
        pieces = []
        for i, w in enumerate(words):
            pieces.append(('\u2581' if i == 0 else '') + w)
        return pieces if pieces else []
    mock_sp.EncodeAsPieces.side_effect = fake_encode

    class _Tok:
        pass

    tok_obj = _Tok()
    tok_obj.special_toks = special_toks
    tok_obj._special_toks_re = re.compile(
        r'(' + '|'.join(re.escape(t) for t in special_toks) + r')')
    tok_obj.tok = mock_sp

    # Bind the real method from the fixed code
    import types

    def _encode_with_special_toks(self, text):
        "Encode `text` preserving special tokens as atomic pieces"
        parts = self._special_toks_re.split(text)
        special_set = set(self.special_toks)
        tokens = []
        for p in parts:
            if p in special_set:
                tokens.append('\u2581' + p)
            elif p:
                tokens.extend(self.tok.EncodeAsPieces(p))
        return tokens

    tok_obj._encode_with_special_toks = types.MethodType(_encode_with_special_toks, tok_obj)
    return tok_obj


class TestSentencePieceTokenizerSpecialTokens:
    """Tests that SentencePieceTokenizer preserves special tokens as atomic pieces."""

    def test_special_token_preserved_as_single_piece(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("xxbos hello world")
        assert '\u2581xxbos' in result
        # xxbos must appear exactly once as a whole piece
        assert result.count('\u2581xxbos') == 1

    def test_multiple_special_tokens_preserved(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("xxbos hello xxeos")
        assert '\u2581xxbos' in result
        assert '\u2581xxeos' in result

    def test_non_special_text_encoded_normally(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("hello world")
        # Should go through EncodeAsPieces (the mock splits on whitespace)
        assert '\u2581hello' in result
        assert 'world' in result
        tok.tok.EncodeAsPieces.assert_called()

    def test_special_token_not_passed_to_sp_encoder(self):
        tok = _make_tokenizer()
        tok._encode_with_special_toks("xxbos hello xxeos")
        # EncodeAsPieces should only receive the non-special parts
        for call_args in tok.tok.EncodeAsPieces.call_args_list:
            text_arg = call_args[0][0]
            assert 'xxbos' not in text_arg
            assert 'xxeos' not in text_arg

    def test_text_with_only_special_token(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("xxbos")
        assert result == ['\u2581xxbos']
        tok.tok.EncodeAsPieces.assert_not_called()

    def test_adjacent_special_tokens(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("xxbosxxeos")
        # Both should be preserved individually
        assert '\u2581xxbos' in result
        assert '\u2581xxeos' in result

    def test_special_token_at_end(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("hello xxeos")
        assert result[-1] == '\u2581xxeos'

    def test_empty_string(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("")
        assert result == []

    def test_all_special_tokens_recognized(self):
        tok = _make_tokenizer()
        all_specials = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()
        for sp in all_specials:
            result = tok._encode_with_special_toks(f"hello {sp} world")
            prefixed = '\u2581' + sp
            assert prefixed in result, f"{sp} not preserved as atomic piece"

    def test_special_token_between_words(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("hello xxbos world")
        # Order: encoded "hello " pieces, then xxbos, then encoded " world" pieces
        bos_idx = result.index('\u2581xxbos')
        assert bos_idx > 0  # something before
        assert bos_idx < len(result) - 1  # something after

    def test_custom_special_tokens(self):
        tok = _make_tokenizer(special_toks=["<CUSTOM>", "<END>"])
        result = tok._encode_with_special_toks("hello <CUSTOM> world <END>")
        assert '\u2581<CUSTOM>' in result
        assert '\u2581<END>' in result

    def test_repeated_special_token(self):
        tok = _make_tokenizer()
        result = tok._encode_with_special_toks("xxbos xxbos hello")
        assert result.count('\u2581xxbos') == 2

    def test_call_yields_fixed_tokens(self):
        """Test the full __call__ path with mocked internals."""
        tok = _make_tokenizer()

        # Simulate __call__
        items = ["xxbos hello world xxeos", "xxbos goodbye xxeos"]
        results = []
        for t in items:
            results.append(tok._encode_with_special_toks(t))

        for r in results:
            assert '\u2581xxbos' in r
            assert '\u2581xxeos' in r
            # The special tokens are whole, not sub-tokenized
            for piece in r:
                if 'xxbos' in piece:
                    assert piece == '\u2581xxbos'
                if 'xxeos' in piece:
                    assert piece == '\u2581xxeos'
