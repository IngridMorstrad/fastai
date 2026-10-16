"""Tests for fastai.imports module.

Covers utility functions: is_iter, all_equal, noop, noops,
one_is_instance, equals, and pv.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.imports import is_iter, all_equal, noop, noops, one_is_instance, equals, pv


# ============================================================
# Tests for is_iter
# ============================================================

class TestIsIter:
    """Tests for the is_iter function."""

    def test_list_is_iterable(self):
        assert is_iter([1, 2, 3])

    def test_tuple_is_iterable(self):
        assert is_iter((1, 2, 3))

    def test_set_is_iterable(self):
        assert is_iter({1, 2, 3})

    def test_dict_is_iterable(self):
        assert is_iter({'a': 1, 'b': 2})

    def test_string_is_iterable(self):
        assert is_iter("hello")

    def test_generator_is_iterable(self):
        gen = (x for x in range(5))
        assert is_iter(gen)

    def test_range_is_iterable(self):
        assert is_iter(range(10))

    def test_numpy_array_is_iterable(self):
        assert is_iter(np.array([1, 2, 3]))

    def test_int_is_not_iterable(self):
        assert not is_iter(42)

    def test_float_is_not_iterable(self):
        assert not is_iter(3.14)

    def test_none_is_not_iterable(self):
        assert not is_iter(None)

    def test_bool_is_not_iterable(self):
        assert not is_iter(True)

    def test_empty_list_is_iterable(self):
        assert is_iter([])

    def test_empty_dict_is_iterable(self):
        assert is_iter({})

    def test_numpy_scalar_is_not_iterable(self):
        # ndim == 0, so should not be considered iterable
        assert not is_iter(np.array(5))

    def test_numpy_2d_array_is_iterable(self):
        assert is_iter(np.array([[1, 2], [3, 4]]))


# ============================================================
# Tests for noop
# ============================================================

class TestNoop:
    """Tests for the noop function."""

    def test_returns_first_arg(self):
        assert noop(42) == 42

    def test_returns_none_when_no_args(self):
        assert noop() is None

    def test_returns_string(self):
        assert noop("hello") == "hello"

    def test_ignores_extra_args(self):
        assert noop(1, 2, 3) == 1

    def test_ignores_kwargs(self):
        assert noop(10, key="value") == 10

    def test_returns_list(self):
        data = [1, 2, 3]
        assert noop(data) is data

    def test_returns_none_with_explicit_none(self):
        assert noop(None) is None


# ============================================================
# Tests for noops
# ============================================================

class TestNoops:
    """Tests for the noops method-like function."""

    def test_returns_first_arg(self):
        # noops takes self as first arg (method signature)
        assert noops(None, 42) == 42

    def test_returns_none_when_no_x(self):
        assert noops(None) is None

    def test_ignores_extra_args(self):
        assert noops("self", 5, 6, 7) == 5

    def test_ignores_kwargs(self):
        assert noops("self", "result", extra=True) == "result"

    def test_as_bound_method(self):
        class Dummy:
            method = noops

        d = Dummy()
        assert d.method(99) == 99

    def test_as_bound_method_no_args(self):
        class Dummy:
            method = noops

        d = Dummy()
        assert d.method() is None


# ============================================================
# Tests for one_is_instance
# ============================================================

class TestOneIsInstance:
    """Tests for the one_is_instance function."""

    def test_first_is_instance(self):
        assert one_is_instance(42, "hello", int)

    def test_second_is_instance(self):
        assert one_is_instance("hello", 42, int)

    def test_both_are_instance(self):
        assert one_is_instance(1, 2, int)

    def test_neither_is_instance(self):
        assert not one_is_instance("hello", "world", int)

    def test_with_tuple_type(self):
        assert one_is_instance(3.14, "hi", (int, float))

    def test_none_check(self):
        assert not one_is_instance(None, None, int)

    def test_with_numpy_array(self):
        assert one_is_instance(np.array([1]), "test", np.ndarray)


# ============================================================
# Tests for equals
# ============================================================

class TestEquals:
    """Tests for the equals function."""

    def test_equal_integers(self):
        assert equals(1, 1)

    def test_unequal_integers(self):
        assert not equals(1, 2)

    def test_equal_strings(self):
        assert equals("abc", "abc")

    def test_unequal_strings(self):
        assert not equals("abc", "def")

    def test_equal_lists(self):
        assert equals([1, 2, 3], [1, 2, 3])

    def test_unequal_lists(self):
        assert not equals([1, 2, 3], [1, 2, 4])

    def test_lists_different_lengths(self):
        assert not equals([1, 2], [1, 2, 3])

    def test_equal_nested_lists(self):
        assert equals([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    def test_unequal_nested_lists(self):
        assert not equals([[1, 2], [3, 4]], [[1, 2], [3, 5]])

    def test_equal_numpy_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert equals(a, b)

    def test_unequal_numpy_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        assert not equals(a, b)

    def test_equal_dicts(self):
        assert equals({'a': 1, 'b': 2}, {'a': 1, 'b': 2})

    def test_unequal_dicts(self):
        assert not equals({'a': 1}, {'a': 2})

    def test_equal_sets(self):
        assert equals({1, 2, 3}, {1, 2, 3})

    def test_unequal_sets(self):
        assert not equals({1, 2}, {1, 3})

    def test_equal_tuples(self):
        assert equals((1, 2, 3), (1, 2, 3))

    def test_unequal_tuples(self):
        assert not equals((1, 2, 3), (1, 2, 4))

    def test_type_comparison(self):
        assert equals(int, int)

    def test_type_comparison_unequal(self):
        assert not equals(int, str)

    def test_mixed_iter_and_non_iter(self):
        # list vs non-iterable should use all_equal which checks is_iter(b)
        assert not equals([1], 1)

    def test_empty_lists(self):
        assert equals([], [])

    def test_numpy_2d_arrays(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])
        assert equals(a, b)

    def test_numpy_2d_arrays_unequal(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 5]])
        assert not equals(a, b)

    def test_none_values(self):
        assert equals(None, None)

    def test_none_vs_value(self):
        assert not equals(None, 1)

    def test_bool_equality(self):
        assert equals(True, True)
        assert not equals(True, False)


# ============================================================
# Tests for all_equal
# ============================================================

class TestAllEqual:
    """Tests for the all_equal function."""

    def test_equal_lists(self):
        assert all_equal([1, 2, 3], [1, 2, 3])

    def test_unequal_lists(self):
        assert not all_equal([1, 2, 3], [1, 2, 4])

    def test_different_lengths(self):
        assert not all_equal([1, 2], [1, 2, 3])

    def test_empty_lists(self):
        assert all_equal([], [])

    def test_non_iterable_b_returns_false(self):
        assert not all_equal([1, 2], 42)

    def test_equal_strings(self):
        assert all_equal("abc", "abc")

    def test_unequal_strings(self):
        assert not all_equal("abc", "abd")

    def test_nested_lists(self):
        assert all_equal([[1, 2], [3]], [[1, 2], [3]])

    def test_generator_comparison(self):
        gen = (x for x in [1, 2, 3])
        assert all_equal(gen, [1, 2, 3])

    def test_tuples(self):
        assert all_equal((10, 20), (10, 20))

    def test_tuples_unequal(self):
        assert not all_equal((10, 20), (10, 30))


# ============================================================
# Tests for pv
# ============================================================

class TestPv:
    """Tests for the pv (print if verbose) function."""

    def test_prints_when_verbose(self, capsys):
        pv("hello world", verbose=True)
        captured = capsys.readouterr()
        assert captured.out == "hello world\n"

    def test_no_print_when_not_verbose(self, capsys):
        pv("should not appear", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_empty_string_when_verbose(self, capsys):
        pv("", verbose=True)
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_prints_number_when_verbose(self, capsys):
        pv(42, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == "42\n"

    def test_no_output_with_zero_verbose(self, capsys):
        pv("text", verbose=0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_with_truthy_verbose(self, capsys):
        pv("text", verbose=1)
        captured = capsys.readouterr()
        assert captured.out == "text\n"
