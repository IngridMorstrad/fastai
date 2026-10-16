"""Tests for fastai/callback/schedule.py module.

Covers scheduling functions: sched_lin, sched_cos, sched_no, sched_exp,
SchedLin, SchedCos, SchedNo, SchedExp, SchedPoly, combine_scheds, combined_cos.
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.callback.schedule import (
    sched_lin, sched_cos, sched_no, sched_exp,
    SchedLin, SchedCos, SchedNo, SchedExp, SchedPoly,
    combine_scheds, combined_cos, annealer,
)


# ============================================================
# Tests for sched_lin
# ============================================================

class TestSchedLin:
    """Tests for linear schedule function."""

    def test_start_position(self):
        """At pos=0, should return start value."""
        assert sched_lin(0.1, 0.5, 0.0) == 0.1

    def test_end_position(self):
        """At pos=1, should return end value."""
        assert sched_lin(0.1, 0.5, 1.0) == 0.5

    def test_midpoint(self):
        """At pos=0.5, should return midpoint between start and end."""
        result = sched_lin(0.0, 1.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_quarter_position(self):
        """At pos=0.25, should be 25% of the way from start to end."""
        result = sched_lin(2.0, 6.0, 0.25)
        assert abs(result - 3.0) < 1e-7

    def test_decreasing(self):
        """Should handle decreasing schedules (start > end)."""
        result = sched_lin(1.0, 0.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_negative_values(self):
        """Should handle negative start and end values."""
        result = sched_lin(-1.0, 1.0, 0.5)
        assert abs(result - 0.0) < 1e-7


# ============================================================
# Tests for sched_cos
# ============================================================

class TestSchedCos:
    """Tests for cosine schedule function."""

    def test_start_position(self):
        """At pos=0, should return start value."""
        result = sched_cos(0.1, 0.5, 0.0)
        assert abs(result - 0.1) < 1e-7

    def test_end_position(self):
        """At pos=1, should return end value."""
        result = sched_cos(0.1, 0.5, 1.0)
        assert abs(result - 0.5) < 1e-7

    def test_midpoint(self):
        """At pos=0.5, should return midpoint (cosine property)."""
        result = sched_cos(0.0, 1.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_slow_start(self):
        """Cosine schedule starts slowly - at pos=0.1 should be less than 0.1 of the range."""
        result = sched_cos(0.0, 1.0, 0.1)
        # Cosine starts slowly so result should be less than linear (0.1)
        assert result < 0.1

    def test_fast_end(self):
        """Cosine schedule ends slowly - at pos=0.9 should be more than 0.9 of the range."""
        result = sched_cos(0.0, 1.0, 0.9)
        # Cosine ends slowly so result should be more than linear (0.9)
        assert result > 0.9

    def test_symmetry(self):
        """Cosine schedule is symmetric around the midpoint."""
        r1 = sched_cos(0.0, 1.0, 0.25)
        r2 = sched_cos(0.0, 1.0, 0.75)
        assert abs(r1 + r2 - 1.0) < 1e-7

    def test_decreasing(self):
        """Should handle decreasing schedules."""
        result = sched_cos(1.0, 0.0, 0.5)
        assert abs(result - 0.5) < 1e-7


# ============================================================
# Tests for sched_no
# ============================================================

class TestSchedNo:
    """Tests for constant (no change) schedule function."""

    def test_start_position(self):
        """At pos=0, should return start value."""
        assert sched_no(0.5, 1.0, 0.0) == 0.5

    def test_end_position(self):
        """At pos=1, should still return start value (no change)."""
        assert sched_no(0.5, 1.0, 1.0) == 0.5

    def test_midpoint(self):
        """At any position, should return start value."""
        assert sched_no(0.5, 1.0, 0.5) == 0.5

    def test_various_positions(self):
        """At any position, the output should always be start."""
        for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert sched_no(3.14, 99.0, pos) == 3.14


# ============================================================
# Tests for sched_exp
# ============================================================

class TestSchedExp:
    """Tests for exponential schedule function."""

    def test_start_position(self):
        """At pos=0, should return start value."""
        result = sched_exp(0.1, 1.0, 0.0)
        assert abs(result - 0.1) < 1e-7

    def test_end_position(self):
        """At pos=1, should return end value."""
        result = sched_exp(0.1, 1.0, 1.0)
        assert abs(result - 1.0) < 1e-7

    def test_midpoint(self):
        """At pos=0.5, should return geometric mean of start and end."""
        result = sched_exp(1.0, 100.0, 0.5)
        expected = math.sqrt(1.0 * 100.0)  # geometric mean = 10
        assert abs(result - expected) < 1e-5

    def test_exponential_growth(self):
        """Exponential schedule grows faster than linear."""
        lin_mid = sched_lin(1.0, 100.0, 0.25)
        exp_mid = sched_exp(1.0, 100.0, 0.25)
        # Exponential starts slower than linear when start < end
        assert exp_mid < lin_mid

    def test_large_range(self):
        """Should handle large exponential ranges."""
        result = sched_exp(1e-7, 10.0, 1.0)
        assert abs(result - 10.0) < 1e-5


# ============================================================
# Tests for SchedLin (Annealer wrapper)
# ============================================================

class TestSchedLinAnnealer:
    """Tests for SchedLin factory function."""

    def test_creation_and_call(self):
        """SchedLin should create a callable that behaves like sched_lin."""
        sched = SchedLin(0.0, 1.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(0.5) - 0.5) < 1e-7
        assert abs(sched(1.0) - 1.0) < 1e-7

    def test_matches_sched_lin(self):
        """SchedLin output should match sched_lin for all positions."""
        sched = SchedLin(2.0, 8.0)
        for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert abs(sched(pos) - sched_lin(2.0, 8.0, pos)) < 1e-7


# ============================================================
# Tests for SchedCos (Annealer wrapper)
# ============================================================

class TestSchedCosAnnealer:
    """Tests for SchedCos factory function."""

    def test_creation_and_call(self):
        """SchedCos should create a callable that behaves like sched_cos."""
        sched = SchedCos(0.0, 1.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(0.5) - 0.5) < 1e-7
        assert abs(sched(1.0) - 1.0) < 1e-7

    def test_matches_sched_cos(self):
        """SchedCos output should match sched_cos for all positions."""
        sched = SchedCos(0.5, 2.0)
        for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert abs(sched(pos) - sched_cos(0.5, 2.0, pos)) < 1e-7


# ============================================================
# Tests for SchedNo (Annealer wrapper)
# ============================================================

class TestSchedNoAnnealer:
    """Tests for SchedNo factory function."""

    def test_creation_and_call(self):
        """SchedNo should always return start value."""
        sched = SchedNo(0.5, 1.0)
        assert sched(0.0) == 0.5
        assert sched(0.5) == 0.5
        assert sched(1.0) == 0.5

    def test_matches_sched_no(self):
        """SchedNo output should match sched_no for all positions."""
        sched = SchedNo(3.0, 7.0)
        for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert sched(pos) == sched_no(3.0, 7.0, pos)


# ============================================================
# Tests for SchedExp (Annealer wrapper)
# ============================================================

class TestSchedExpAnnealer:
    """Tests for SchedExp factory function."""

    def test_creation_and_call(self):
        """SchedExp should create a callable that behaves like sched_exp."""
        sched = SchedExp(1.0, 10.0)
        assert abs(sched(0.0) - 1.0) < 1e-7
        assert abs(sched(1.0) - 10.0) < 1e-5

    def test_matches_sched_exp(self):
        """SchedExp output should match sched_exp for all positions."""
        sched = SchedExp(0.01, 1.0)
        for pos in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert abs(sched(pos) - sched_exp(0.01, 1.0, pos)) < 1e-7


# ============================================================
# Tests for SchedPoly
# ============================================================

class TestSchedPoly:
    """Tests for polynomial schedule function."""

    def test_start_position(self):
        """At pos=0, should return start value."""
        sched = SchedPoly(0.0, 1.0, 2)
        assert abs(sched(0.0) - 0.0) < 1e-7

    def test_end_position(self):
        """At pos=1, should return end value regardless of power."""
        for power in [0.5, 1, 2, 3, 5]:
            sched = SchedPoly(0.0, 1.0, power)
            assert abs(sched(1.0) - 1.0) < 1e-7

    def test_power_1_is_linear(self):
        """With power=1, should behave like linear schedule."""
        sched = SchedPoly(2.0, 6.0, 1)
        for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert abs(sched(pos) - sched_lin(2.0, 6.0, pos)) < 1e-7

    def test_power_2_quadratic(self):
        """With power=2, the schedule should follow a quadratic curve."""
        sched = SchedPoly(0.0, 1.0, 2)
        assert abs(sched(0.5) - 0.25) < 1e-7  # 0.5^2 = 0.25

    def test_power_3_cubic(self):
        """With power=3, the schedule should follow a cubic curve."""
        sched = SchedPoly(0.0, 1.0, 3)
        assert abs(sched(0.5) - 0.125) < 1e-7  # 0.5^3 = 0.125

    def test_fractional_power(self):
        """Should handle fractional powers (e.g., square root)."""
        sched = SchedPoly(0.0, 1.0, 0.5)
        result = sched(0.25)
        expected = 0.25 ** 0.5  # = 0.5
        assert abs(result - expected) < 1e-7

    def test_with_offset(self):
        """Should correctly handle non-zero start values."""
        sched = SchedPoly(1.0, 5.0, 2)
        # At pos=0.5: start + (end-start) * 0.5^2 = 1.0 + 4.0 * 0.25 = 2.0
        assert abs(sched(0.5) - 2.0) < 1e-7


# ============================================================
# Tests for combine_scheds
# ============================================================

class TestCombineScheds:
    """Tests for combine_scheds function."""

    def test_single_schedule(self):
        """With a single schedule taking 100%, should behave like that schedule."""
        combined = combine_scheds([1.0], [SchedLin(0.0, 1.0)])
        assert abs(combined(0.0) - 0.0) < 1e-5
        assert abs(combined(0.5) - 0.5) < 1e-5
        assert abs(combined(1.0) - 1.0) < 1e-5

    def test_two_schedules_equal_split(self):
        """Two schedules split 50/50 should each run over their halves."""
        combined = combine_scheds(
            [0.5, 0.5],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # First half: linear 0->1
        assert abs(combined(0.0) - 0.0) < 1e-5
        assert abs(combined(0.25) - 0.5) < 1e-5
        # At boundary (pos=0.5), second schedule starts at pos=0 -> returns 1.0
        assert abs(combined(0.5) - 1.0) < 1e-5
        # Second half: linear 1->0
        assert abs(combined(0.75) - 0.5) < 1e-5

    def test_three_schedules(self):
        """Three schedules with different proportions."""
        combined = combine_scheds(
            [0.3, 0.4, 0.3],
            [SchedLin(0.0, 1.0), SchedNo(1.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # At start
        assert abs(combined(0.0) - 0.0) < 1e-5
        # In constant region (30%-70%)
        assert abs(combined(0.5) - 1.0) < 1e-5

    def test_unequal_split(self):
        """Schedule with 25/75 split."""
        combined = combine_scheds(
            [0.25, 0.75],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 2.0)]
        )
        # At the boundary
        assert abs(combined(0.25) - 1.0) < 1e-5
        # Midpoint of second phase: 1.0 + 0.5*(2.0-1.0) = 1.5
        assert abs(combined(0.625) - 1.5) < 1e-5

    def test_pcts_must_sum_to_one(self):
        """Should raise assertion error if pcts don't sum to 1."""
        with pytest.raises(AssertionError):
            combine_scheds([0.3, 0.3], [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)])


# ============================================================
# Tests for combined_cos
# ============================================================

class TestCombinedCos:
    """Tests for combined_cos convenience function."""

    def test_start_value(self):
        """At pos=0, should return start value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        assert abs(sched(0.0) - 0.0) < 1e-5

    def test_middle_at_pct(self):
        """At pos=pct, should reach the middle value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        assert abs(sched(0.3) - 1.0) < 1e-5

    def test_end_value(self):
        """At pos=1, should reach the end value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.5)
        assert abs(sched(1.0) - 0.5) < 1e-5

    def test_warmup_and_decay(self):
        """Typical 1cycle pattern: warmup then decay."""
        sched = combined_cos(0.25, 1e-4, 1e-2, 1e-5)
        # Start
        assert abs(sched(0.0) - 1e-4) < 1e-7
        # Peak at 25%
        assert abs(sched(0.25) - 1e-2) < 1e-7
        # End
        assert abs(sched(1.0) - 1e-5) < 1e-7

    def test_monotonic_warmup_phase(self):
        """During warmup (0 to pct), values should be non-decreasing."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        prev = sched(0.0)
        for i in range(1, 11):
            pos = 0.3 * i / 10
            current = sched(pos)
            assert current >= prev - 1e-7
            prev = current

    def test_monotonic_decay_phase(self):
        """During decay (pct to 1), values should be non-increasing when end < middle."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        prev = sched(0.3)
        for i in range(1, 11):
            pos = 0.3 + 0.7 * i / 10
            current = sched(pos)
            assert current <= prev + 1e-7
            prev = current


# ============================================================
# Tests for annealer decorator
# ============================================================

class TestAnnealer:
    """Tests for the annealer decorator."""

    def test_basic_annealer(self):
        """Decorated function should return a callable given start and end."""
        @annealer
        def my_sched(start, end, pos):
            return start + pos * (end - start)

        sched = my_sched(0.0, 1.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(0.5) - 0.5) < 1e-7
        assert abs(sched(1.0) - 1.0) < 1e-7

    def test_annealer_with_custom_function(self):
        """Test annealer with a custom scheduling function."""
        @annealer
        def square_sched(start, end, pos):
            return start + (end - start) * pos ** 2

        sched = square_sched(0.0, 4.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(0.5) - 1.0) < 1e-7  # 4.0 * 0.25
        assert abs(sched(1.0) - 4.0) < 1e-7


# ============================================================
# Integration / End-to-End tests
# ============================================================

class TestScheduleIntegration:
    """End-to-end tests combining multiple schedule components."""

    def test_one_cycle_lr_pattern(self):
        """Simulate a 1cycle learning rate schedule."""
        sched = combined_cos(0.25, 1e-4, 1e-2, 1e-6)
        # Sample at 100 positions
        values = [sched(i / 100.0) for i in range(101)]
        # LR should start low
        assert values[0] < 1e-3
        # Peak near 25%
        assert abs(values[25] - 1e-2) < 1e-7
        # End very low
        assert values[100] < 1e-4
        # Max should be at or near position 25
        max_val = max(values)
        assert abs(max_val - 1e-2) < 1e-7

    def test_warmup_constant_decay_pattern(self):
        """Simulate warmup -> constant -> decay schedule."""
        sched = combine_scheds(
            [0.2, 0.5, 0.3],
            [SchedLin(0.0, 1.0), SchedNo(1.0, 1.0), SchedCos(1.0, 0.0)]
        )
        # Warmup phase
        assert abs(sched(0.0) - 0.0) < 1e-5
        assert abs(sched(0.1) - 0.5) < 1e-5

        # Constant phase
        assert abs(sched(0.3) - 1.0) < 1e-5
        assert abs(sched(0.5) - 1.0) < 1e-5

        # Decay phase ends at 0
        assert abs(sched(1.0) - 0.0) < 1e-5

    def test_sgdr_style_schedule(self):
        """Simulate SGDR-style cosine decay over multiple cycles."""
        # Two equal cycles of cosine decay from 1.0 to 0.0
        sched = combine_scheds(
            [0.5, 0.5],
            [SchedCos(1.0, 0.0), SchedCos(1.0, 0.0)]
        )
        # Start of first cycle
        assert abs(sched(0.0) - 1.0) < 1e-5
        # End of first cycle / start of second
        assert abs(sched(0.5) - 1.0) < 1e-5
        # End of second cycle
        assert abs(sched(1.0) - 0.0) < 1e-5

    def test_polynomial_warmup_with_cosine_decay(self):
        """Polynomial warmup followed by cosine decay."""
        sched = combine_scheds(
            [0.3, 0.7],
            [SchedPoly(0.0, 1.0, 2), SchedCos(1.0, 0.0)]
        )
        # Start
        assert abs(sched(0.0) - 0.0) < 1e-5
        # End of warmup
        assert abs(sched(0.3) - 1.0) < 1e-5
        # End of decay
        assert abs(sched(1.0) - 0.0) < 1e-5

    def test_all_schedulers_boundary_consistency(self):
        """All scheduler types should respect start/end at boundaries."""
        schedulers = [
            SchedLin(0.1, 0.9),
            SchedCos(0.1, 0.9),
            SchedNo(0.1, 0.9),
            SchedExp(0.1, 0.9),
            SchedPoly(0.1, 0.9, 2),
            SchedPoly(0.1, 0.9, 0.5),
        ]
        for sched in schedulers:
            # All should start at 0.1
            assert abs(sched(0.0) - 0.1) < 1e-5, f"Failed at start for {sched}"
        # Only lin, cos, exp, poly should end at 0.9 (sched_no stays at start)
        for sched in [schedulers[0], schedulers[1], schedulers[3], schedulers[4], schedulers[5]]:
            assert abs(sched(1.0) - 0.9) < 1e-5, f"Failed at end for {sched}"
        # SchedNo should stay at start
        assert abs(schedulers[2](1.0) - 0.1) < 1e-5
