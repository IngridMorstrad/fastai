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
# Dynamically extract pure-Python text preprocessing functions
# from fastai/text/core.py so we test the real implementations
# without pulling in torch/spacy/fastcore.  Same approach as
# _tracker_test_helpers.py uses for fastai/callback/tracker.py.
# ============================================================

def _extract_text_core_names():
    """Read fastai/text/core.py, strip internal imports, exec() only the
    pure-Python top portion (constants + preprocessing functions), and return
    the resulting namespace dict."""
    src_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'text', 'core.py')
    with open(os.path.abspath(src_path)) as f:
        lines = f.read().split('\n')

    # Only keep lines up through the replace_space function definition.
    # Everything after that depends on the full fastai import chain
    # (defaults, Transform, delegates, spacy, etc.).
    cutoff = None
    for i, line in enumerate(lines):
        if 'def replace_space' in line:
            # Include through the function body (next non-empty, non-comment line
            # after the def that is at top-level indentation, or end of file).
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped and not stripped.startswith('#') and not lines[j].startswith(' '):
                    cutoff = j
                    break
            break
    if cutoff is None:
        cutoff = len(lines)

    filtered = []
    for line in lines[:cutoff]:
        if line.startswith('from ..') or line.startswith('from .'):
            filtered.append('pass  # skipped internal import')
        else:
            filtered.append(line)

    ns = {'re': re, 'html': html, '__builtins__': __builtins__}
    exec(compile('\n'.join(filtered), src_path, 'exec'), ns)
    return ns

_ns = _extract_text_core_names()

# Functions
spec_add_spaces = _ns['spec_add_spaces']
rm_useless_spaces = _ns['rm_useless_spaces']
replace_rep      = _ns['replace_rep']
replace_wrep     = _ns['replace_wrep']
fix_html         = _ns['fix_html']
replace_all_caps = _ns['replace_all_caps']
replace_maj      = _ns['replace_maj']
lowercase        = _ns['lowercase']
replace_space    = _ns['replace_space']

# Constants
UNK, PAD, BOS, EOS, FLD       = _ns['UNK'], _ns['PAD'], _ns['BOS'], _ns['EOS'], _ns['FLD']
TK_REP, TK_WREP, TK_UP, TK_MAJ = _ns['TK_REP'], _ns['TK_WREP'], _ns['TK_UP'], _ns['TK_MAJ']


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
