"""Tests for fit_sgdr ZeroDivisionError fix when cycle_mult=1.

Validates that the n_epoch calculation in fit_sgdr handles cycle_mult=1
correctly instead of dividing by zero. These tests do NOT require torch
or GPU -- they exercise the pure arithmetic extracted from fit_sgdr.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _compute_n_epoch(cycle_len, cycle_mult, n_cycles):
    """Reproduce the n_epoch formula from fit_sgdr (callback/schedule.py).

    The fix: when cycle_mult == 1 the geometric-series denominator
    (cycle_mult - 1) is zero, so we use the simplified form instead.
    """
    if cycle_mult == 1:
        return cycle_len * n_cycles
    return cycle_len * (cycle_mult ** n_cycles - 1) // (cycle_mult - 1)


# ============================================================
# Tests for n_epoch calculation
# ============================================================

class TestFitSgdrNEpoch:
    """Tests for the n_epoch computation used in fit_sgdr."""

    def test_cycle_mult_1_no_zero_division(self):
        """cycle_mult=1 must not raise ZeroDivisionError."""
        result = _compute_n_epoch(cycle_len=5, cycle_mult=1, n_cycles=3)
        assert result == 15

    def test_cycle_mult_1_single_cycle(self):
        result = _compute_n_epoch(cycle_len=10, cycle_mult=1, n_cycles=1)
        assert result == 10

    def test_cycle_mult_1_many_cycles(self):
        result = _compute_n_epoch(cycle_len=3, cycle_mult=1, n_cycles=10)
        assert result == 30

    def test_cycle_mult_2_standard(self):
        # cycle_len=1, cycle_mult=2, n_cycles=3 => 1*(2**3-1)//(2-1) = 7
        result = _compute_n_epoch(cycle_len=1, cycle_mult=2, n_cycles=3)
        assert result == 7

    def test_cycle_mult_2_longer_cycle(self):
        # cycle_len=5, cycle_mult=2, n_cycles=3 => 5*(8-1)//1 = 35
        result = _compute_n_epoch(cycle_len=5, cycle_mult=2, n_cycles=3)
        assert result == 35

    def test_cycle_mult_3(self):
        # cycle_len=2, cycle_mult=3, n_cycles=2 => 2*(9-1)//(3-1) = 2*8//2 = 8
        result = _compute_n_epoch(cycle_len=2, cycle_mult=3, n_cycles=2)
        assert result == 8

    def test_cycle_mult_1_pcts_sum_to_one(self):
        """With cycle_mult=1, pcts should sum to 1.0 (each cycle is equal)."""
        cycle_len = 4
        cycle_mult = 1
        n_cycles = 5
        n_epoch = _compute_n_epoch(cycle_len, cycle_mult, n_cycles)
        pcts = [cycle_len * cycle_mult ** i / n_epoch for i in range(n_cycles)]
        assert abs(sum(pcts) - 1.0) < 1e-9

    def test_cycle_mult_2_pcts_sum_to_one(self):
        """With cycle_mult=2, pcts should also sum to ~1.0."""
        cycle_len = 1
        cycle_mult = 2
        n_cycles = 4
        n_epoch = _compute_n_epoch(cycle_len, cycle_mult, n_cycles)
        pcts = [cycle_len * cycle_mult ** i / n_epoch for i in range(n_cycles)]
        assert abs(sum(pcts) - 1.0) < 1e-9


class TestFitSgdrSourceVerification:
    """Verify the actual source code in schedule.py contains the fix."""

    def test_schedule_py_has_guard(self):
        """The generated schedule.py must contain the cycle_mult == 1 guard."""
        schedule_path = os.path.join(
            os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py'
        )
        with open(schedule_path) as f:
            source = f.read()
        assert 'cycle_len * n_cycles if cycle_mult == 1 else' in source, (
            'schedule.py is missing the cycle_mult == 1 guard for n_epoch'
        )

    def test_notebook_has_guard(self):
        """The source notebook must also contain the cycle_mult == 1 guard."""
        import json
        nb_path = os.path.join(
            os.path.dirname(__file__), '..', 'nbs', '14_callback.schedule.ipynb'
        )
        with open(nb_path) as f:
            nb = json.load(f)
        cell_source = ''.join(nb['cells'][57]['source'])
        assert 'cycle_len * n_cycles if cycle_mult == 1 else' in cell_source, (
            'Notebook cell 57 is missing the cycle_mult == 1 guard for n_epoch'
        )
