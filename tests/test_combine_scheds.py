"""Tests for combine_scheds floating-point tolerance fix.

Verifies that combine_scheds accepts percentage lists that sum to 1.0
within floating-point tolerance (e.g. [0.3, 0.7] which can't be
represented exactly in IEEE 754).
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from fastai.callback.schedule import combine_scheds, SchedLin


class TestCombineSchedsFloatTolerance:
    """combine_scheds should accept pcts that sum to ~1.0 within FP tolerance."""

    def test_exact_sum(self):
        """Percentages that sum to exactly 1.0 should work."""
        sched = combine_scheds([0.5, 0.5], [SchedLin(0, 1), SchedLin(1, 0)])
        assert sched(0.0) == 0.0
        assert sched(1.0) == 0.0

    def test_inexact_sum_03_07(self):
        """[0.3, 0.7] sums to 0.9999...8 in float64 — must not raise."""
        sched = combine_scheds([0.3, 0.7], [SchedLin(0, 1), SchedLin(1, 0)])
        assert sched(0.0) == 0.0

    def test_inexact_sum_01_repeated(self):
        """[0.1]*10 sums to ~1.0 with rounding — must not raise."""
        scheds = [SchedLin(0, 1)] * 10
        sched = combine_scheds([0.1] * 10, scheds)
        assert sched(0.0) == 0.0

    def test_clearly_wrong_sum_raises(self):
        """Percentages that clearly don't sum to 1.0 should still raise."""
        with pytest.raises(AssertionError):
            combine_scheds([0.3, 0.3], [SchedLin(0, 1), SchedLin(1, 0)])

    def test_zero_sum_raises(self):
        """All-zero percentages should raise."""
        with pytest.raises(AssertionError):
            combine_scheds([0.0, 0.0], [SchedLin(0, 1), SchedLin(1, 0)])

    def test_over_one_raises(self):
        """Percentages summing well above 1.0 should raise."""
        with pytest.raises(AssertionError):
            combine_scheds([0.6, 0.6], [SchedLin(0, 1), SchedLin(1, 0)])
