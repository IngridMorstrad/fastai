"""Tests for fastai.text.core module.

Covers text preprocessing functions: spec_add_spaces, rm_useless_spaces,
replace_rep, replace_wrep, fix_html, replace_all_caps, replace_maj,
lowercase, replace_space, BaseTokenizer, and TokenizeWithRules.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.text.core import (
    spec_add_spaces,
    rm_useless_spaces,
    replace_rep,
    replace_wrep,
    fix_html,
    replace_all_caps,
    replace_maj,
    lowercase,
    replace_space,
    BaseTokenizer,
    TokenizeWithRules,
    UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ,
)


# ============================================================
# Tests for special token constants
# ============================================================

class TestSpecialTokens:
    """Verify that special token constants have expected values."""

    def test_unk(self):
        assert UNK == "xxunk"

    def test_pad(self):
        assert PAD == "xxpad"

    def test_bos(self):
        assert BOS == "xxbos"

    def test_eos(self):
        assert EOS == "xxeos"

    def test_fld(self):
        assert FLD == "xxfld"

    def test_tk_rep(self):
        assert TK_REP == "xxrep"

    def test_tk_wrep(self):
        assert TK_WREP == "xxwrep"

    def test_tk_up(self):
        assert TK_UP == "xxup"

    def test_tk_maj(self):
        assert TK_MAJ == "xxmaj"


# ============================================================
# Tests for spec_add_spaces
# ============================================================

class TestSpecAddSpaces:
    """Tests for spec_add_spaces: adds spaces around / and #."""

    def test_slash(self):
        assert spec_add_spaces("hello/world") == "hello / world"

    def test_hash(self):
        assert spec_add_spaces("hello#world") == "hello # world"

    def test_backslash(self):
        assert spec_add_spaces("hello\\world") == "hello \\ world"

    def test_multiple_special_chars(self):
        assert spec_add_spaces("a/b#c\\d") == "a / b # c \\ d"

    def test_no_special_chars(self):
        assert spec_add_spaces("hello world") == "hello world"

    def test_empty_string(self):
        assert spec_add_spaces("") == ""

    def test_consecutive_slashes(self):
        result = spec_add_spaces("http://example.com")
        assert " / " in result

    def test_special_at_start(self):
        assert spec_add_spaces("/start") == " / start"

    def test_special_at_end(self):
        assert spec_add_spaces("end/") == "end / "


# ============================================================
# Tests for rm_useless_spaces
# ============================================================

class TestRmUselessSpaces:
    """Tests for rm_useless_spaces: collapse multiple spaces to one."""

    def test_double_space(self):
        assert rm_useless_spaces("hello  world") == "hello world"

    def test_triple_space(self):
        assert rm_useless_spaces("hello   world") == "hello world"

    def test_many_spaces(self):
        assert rm_useless_spaces("hello      world") == "hello world"

    def test_no_extra_spaces(self):
        assert rm_useless_spaces("hello world") == "hello world"

    def test_single_space(self):
        assert rm_useless_spaces("a b c") == "a b c"

    def test_spaces_in_multiple_locations(self):
        assert rm_useless_spaces("a  b   c    d") == "a b c d"

    def test_empty_string(self):
        assert rm_useless_spaces("") == ""

    def test_leading_trailing_multiple_spaces(self):
        assert rm_useless_spaces("  hello  ") == " hello "


# ============================================================
# Tests for replace_rep
# ============================================================

class TestReplaceRep:
    """Tests for replace_rep: replaces character repetitions."""

    def test_basic_repetition(self):
        # cccc -> TK_REP 4 c
        result = replace_rep("cccc")
        assert TK_REP in result
        assert "4" in result
        assert "c" in result

    def test_three_chars(self):
        # aaa -> TK_REP 3 a
        result = replace_rep("aaa")
        assert TK_REP in result
        assert "3" in result

    def test_in_context(self):
        result = replace_rep("sooo good")
        assert TK_REP in result
        assert "3" in result
        assert "o" in result

    def test_no_repetition(self):
        result = replace_rep("hello")
        assert TK_REP not in result
        assert result == "hello"

    def test_two_chars_not_replaced(self):
        # Only 3+ repetitions trigger replacement
        result = replace_rep("aa")
        assert TK_REP not in result

    def test_multiple_repetitions(self):
        result = replace_rep("aaaa bbbb")
        assert result.count(TK_REP) == 2

    def test_exclamation_marks(self):
        result = replace_rep("wow!!!")
        assert TK_REP in result
        assert "3" in result
        assert "!" in result

    def test_empty_string(self):
        result = replace_rep("")
        assert result == ""


# ============================================================
# Tests for replace_wrep
# ============================================================

class TestReplaceWrep:
    """Tests for replace_wrep: replaces word repetitions."""

    def test_basic_word_repetition(self):
        result = replace_wrep("word word word word")
        assert TK_WREP in result
        assert "4" in result
        assert "word" in result

    def test_three_word_repetition(self):
        result = replace_wrep("the the the")
        assert TK_WREP in result
        assert "3" in result

    def test_no_repetition(self):
        result = replace_wrep("hello world today")
        assert TK_WREP not in result

    def test_two_words_not_replaced(self):
        # Only 3+ repetitions should be replaced
        result = replace_wrep("go go")
        assert TK_WREP not in result

    def test_repetition_in_sentence(self):
        result = replace_wrep("I am am am happy")
        assert TK_WREP in result
        assert "3" in result
        assert "am" in result

    def test_empty_string(self):
        result = replace_wrep("")
        assert result == ""


# ============================================================
# Tests for fix_html
# ============================================================

class TestFixHtml:
    """Tests for fix_html: cleans messy HTML artifacts."""

    def test_apostrophe_39(self):
        assert "'" in fix_html("#39;")

    def test_ampersand(self):
        assert "&" in fix_html("amp;")

    def test_apostrophe_146(self):
        assert "'" in fix_html("#146;")

    def test_nbsp(self):
        assert " " in fix_html("nbsp;")

    def test_dollar(self):
        assert "$" in fix_html("#36;")

    def test_newline_escaped(self):
        assert "\n" in fix_html("\\n")

    def test_quot(self):
        assert "'" in fix_html("quot;")

    def test_br_tag(self):
        assert "\n" in fix_html("<br />")

    def test_escaped_quote(self):
        assert '"' in fix_html('\\"')

    def test_unk_token(self):
        assert UNK in fix_html("<unk>")

    def test_at_dot(self):
        result = fix_html("hello @.@ world")
        assert "hello.world" in result.replace(" ", "")

    def test_at_dash(self):
        result = fix_html("hello @-@ world")
        assert "hello-world" in result.replace(" ", "")

    def test_ellipsis(self):
        result = fix_html("wait...")
        assert "\u2026" in result

    def test_html_entities(self):
        result = fix_html("&lt;tag&gt;")
        assert "<tag>" in result

    def test_empty_string(self):
        result = fix_html("")
        assert result == ""

    def test_combined(self):
        result = fix_html("#39;hello amp; world\\n")
        assert "'" in result
        assert "&" in result
        assert "\n" in result


# ============================================================
# Tests for replace_all_caps
# ============================================================

class TestReplaceAllCaps:
    """Tests for replace_all_caps: handles ALL CAPS words."""

    def test_single_all_caps_word(self):
        result = replace_all_caps("I am HAPPY today")
        assert TK_UP in result
        assert "happy" in result
        assert "HAPPY" not in result

    def test_multiple_all_caps(self):
        result = replace_all_caps("THIS IS GREAT")
        assert result.count(TK_UP) >= 2

    def test_no_all_caps(self):
        result = replace_all_caps("hello world")
        assert TK_UP not in result
        assert result == "hello world"

    def test_single_capital_letter(self):
        # Single capital letter like "I" should not add TK_UP
        result = replace_all_caps("I am here")
        assert TK_UP not in result

    def test_mixed_case_not_affected(self):
        result = replace_all_caps("Hello World")
        assert TK_UP not in result

    def test_all_caps_with_number(self):
        result = replace_all_caps("HTTP2 is fast")
        assert TK_UP in result
        assert "http2" in result

    def test_empty_string(self):
        result = replace_all_caps("")
        assert result == ""

    def test_at_start_of_string(self):
        result = replace_all_caps("HELLO world")
        assert TK_UP in result
        assert "hello" in result


# ============================================================
# Tests for replace_maj
# ============================================================

class TestReplaceMaj:
    """Tests for replace_maj: handles Capitalized words."""

    def test_capitalized_word(self):
        result = replace_maj("Hello world")
        assert TK_MAJ in result
        assert "hello" in result
        assert "Hello" not in result

    def test_multiple_capitalized(self):
        result = replace_maj("Hello World Today")
        assert result.count(TK_MAJ) == 3

    def test_no_capitalized(self):
        result = replace_maj("hello world")
        assert TK_MAJ not in result
        assert result == "hello world"

    def test_single_capital_letter(self):
        # A single capital letter should not get TK_MAJ prefix
        result = replace_maj("I am here")
        assert TK_MAJ not in result

    def test_all_lowercase(self):
        result = replace_maj("nothing special here")
        assert TK_MAJ not in result

    def test_empty_string(self):
        result = replace_maj("")
        assert result == ""

    def test_at_start_of_string(self):
        result = replace_maj("Start here")
        assert TK_MAJ in result
        assert "start" in result


# ============================================================
# Tests for lowercase
# ============================================================

class TestLowercase:
    """Tests for lowercase: converts to lower and optionally adds BOS/EOS."""

    def test_basic_lowercase(self):
        result = lowercase("Hello World")
        assert "hello world" in result

    def test_adds_bos_by_default(self):
        result = lowercase("test")
        assert result.startswith(BOS)

    def test_no_bos(self):
        result = lowercase("test", add_bos=False)
        assert not result.startswith(BOS)
        assert result == "test"

    def test_add_eos(self):
        result = lowercase("test", add_eos=True)
        assert result.endswith(EOS)

    def test_no_eos_by_default(self):
        result = lowercase("test")
        assert not result.endswith(EOS)

    def test_both_bos_and_eos(self):
        result = lowercase("test", add_bos=True, add_eos=True)
        assert result.startswith(BOS)
        assert result.endswith(EOS)

    def test_strips_whitespace(self):
        result = lowercase("  hello  ", add_bos=False)
        assert result == "hello"

    def test_empty_string(self):
        result = lowercase("", add_bos=False)
        assert result == ""

    def test_already_lowercase(self):
        result = lowercase("already lower", add_bos=False)
        assert result == "already lower"

    def test_mixed_case(self):
        result = lowercase("HeLLo WoRLD", add_bos=False)
        assert result == "hello world"


# ============================================================
# Tests for replace_space
# ============================================================

class TestReplaceSpace:
    """Tests for replace_space: replaces spaces with unicode line char."""

    def test_single_space(self):
        result = replace_space("hello world")
        assert result == "hello\u2581world"

    def test_multiple_spaces(self):
        result = replace_space("a b c")
        assert result == "a\u2581b\u2581c"

    def test_no_space(self):
        result = replace_space("hello")
        assert result == "hello"

    def test_empty_string(self):
        result = replace_space("")
        assert result == ""

    def test_leading_space(self):
        result = replace_space(" hello")
        assert result == "\u2581hello"

    def test_trailing_space(self):
        result = replace_space("hello ")
        assert result == "hello\u2581"


# ============================================================
# Tests for BaseTokenizer
# ============================================================

class TestBaseTokenizer:
    """Tests for BaseTokenizer: basic split-on-space tokenizer."""

    def test_basic_tokenization(self):
        tok = BaseTokenizer()
        result = list(tok(["hello world"]))
        assert result == [["hello", "world"]]

    def test_multiple_inputs(self):
        tok = BaseTokenizer()
        result = list(tok(["hello world", "foo bar baz"]))
        assert result == [["hello", "world"], ["foo", "bar", "baz"]]

    def test_custom_split_char(self):
        tok = BaseTokenizer(split_char=',')
        result = list(tok(["a,b,c"]))
        assert result == [["a", "b", "c"]]

    def test_empty_string(self):
        tok = BaseTokenizer()
        result = list(tok([""]))[0]
        assert result == [""]

    def test_single_word(self):
        tok = BaseTokenizer()
        result = list(tok(["hello"]))
        assert result == [["hello"]]

    def test_multiple_spaces(self):
        tok = BaseTokenizer()
        result = list(tok(["hello  world"]))
        # split on space will produce empty string in the middle
        assert "" in result[0]

    def test_split_char_stored(self):
        tok = BaseTokenizer(split_char='-')
        assert tok.split_char == '-'


# ============================================================
# Tests for TokenizeWithRules
# ============================================================

class TestTokenizeWithRules:
    """Tests for TokenizeWithRules: applies rules + tokenizer + post_rules."""

    def test_basic_usage(self):
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok, rules=[], post_rules=[])
        result = list(twr(["hello world"]))
        # Each result is an L (list-like)
        assert list(result[0]) == ["hello", "world"]

    def test_with_preprocessing_rule(self):
        tok = BaseTokenizer()
        # Use a simple rule that lowercases
        twr = TokenizeWithRules(tok, rules=[str.lower], post_rules=[])
        result = list(twr(["HELLO WORLD"]))
        assert list(result[0]) == ["hello", "world"]

    def test_with_post_rule(self):
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok, rules=[], post_rules=[replace_space])
        result = list(twr(["hello world"]))
        # Post rules are applied to each token, not to a space-containing token
        tokens = list(result[0])
        assert tokens == ["hello", "world"]

    def test_with_multiple_rules(self):
        tok = BaseTokenizer()
        rules = [str.lower, rm_useless_spaces]
        twr = TokenizeWithRules(tok, rules=rules, post_rules=[])
        result = list(twr(["HELLO   WORLD"]))
        assert list(result[0]) == ["hello", "world"]

    def test_multiple_texts(self):
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok, rules=[], post_rules=[])
        result = list(twr(["hello world", "foo bar"]))
        assert list(result[0]) == ["hello", "world"]
        assert list(result[1]) == ["foo", "bar"]

    def test_empty_input(self):
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok, rules=[], post_rules=[])
        result = list(twr([]))
        assert result == []

    def test_default_rules_applied(self):
        """When no rules specified, defaults are used from defaults.text_proc_rules."""
        tok = BaseTokenizer()
        # Uses defaults - includes fix_html, replace_rep, etc.
        twr = TokenizeWithRules(tok)
        result = list(twr(["Hello WORLD"]))
        tokens = list(result[0])
        # Should be lowercased (lowercase is a default rule)
        # BOS should be added by default lowercase rule
        assert BOS in tokens


# ============================================================
# End-to-end / integration tests
# ============================================================

class TestTextPreprocessingPipeline:
    """Integration tests combining multiple preprocessing steps."""

    def test_full_pipeline_simple(self):
        """Test the default pipeline on simple text."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["Hello World"]))[0]
        tokens = list(result)
        # Should contain BOS at start and lowercased words
        assert tokens[0] == BOS

    def test_full_pipeline_with_caps(self):
        """ALL CAPS words should get TK_UP marker."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["I am EXCITED"]))[0]
        tokens = list(result)
        assert TK_UP in tokens

    def test_full_pipeline_with_repetition(self):
        """Character repetitions should get TK_REP marker."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["sooooo good"]))[0]
        tokens = list(result)
        assert TK_REP in tokens

    def test_full_pipeline_with_html(self):
        """HTML entities should be cleaned up."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["hello &lt;b&gt; world"]))[0]
        tokens = list(result)
        # The <b> should be cleaned by fix_html (html.unescape)
        joined = " ".join(tokens)
        assert "&lt;" not in joined

    def test_full_pipeline_special_chars(self):
        """Slashes and hashes should have spaces added."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["path/to/file"]))[0]
        tokens = list(result)
        # The slash should be its own token
        assert "/" in tokens

    def test_pipeline_preserves_content(self):
        """Normal lowercase text without special features should pass through."""
        tok = BaseTokenizer()
        twr = TokenizeWithRules(tok)
        result = list(twr(["hello world"]))[0]
        tokens = list(result)
        # BOS + hello + world
        assert BOS in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_replace_space_post_rule(self):
        """Post-processing rule replace_space should work on tokens with spaces."""
        # replace_space is a default post rule, applied to each token
        result = replace_space("hello world")
        assert "\u2581" in result
        assert " " not in result
