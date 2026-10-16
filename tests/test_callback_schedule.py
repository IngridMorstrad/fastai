"""Tests for fastai/callback/schedule.py module.

Covers: scheduling functions (sched_lin, sched_cos, sched_no, sched_exp),
schedule constructors (SchedLin, SchedCos, SchedNo, SchedExp, SchedPoly),
combine_scheds, combined_cos, ParamScheduler callback, and LR suggestion
methods (valley, slide, minimum, steep).
"""
import sys
import os
import math
import pytest
import torch
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.callback.schedule import (
    sched_lin, sched_cos, sched_no, sched_exp,
    SchedLin, SchedCos, SchedNo, SchedExp, SchedPoly,
    combine_scheds, combined_cos,
    ParamScheduler,
    valley, slide, minimum, steep,
)


# ============================================================
# Tests for sched_lin (linear schedule)
# ============================================================

class TestSchedLin:
    """Tests for the linear schedule function."""

    def test_at_start(self):
        """pos=0 should return start value."""
        assert sched_lin(0.1, 1.0, 0.0) == 0.1

    def test_at_end(self):
        """pos=1 should return end value."""
        assert abs(sched_lin(0.1, 1.0, 1.0) - 1.0) < 1e-7

    def test_midpoint(self):
        """pos=0.5 should return the midpoint."""
        result = sched_lin(0.0, 1.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_quarter(self):
        """pos=0.25 should return 25% of the way."""
        result = sched_lin(2.0, 6.0, 0.25)
        assert abs(result - 3.0) < 1e-7

    def test_decreasing(self):
        """Should handle decreasing schedules (end < start)."""
        result = sched_lin(1.0, 0.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_negative_values(self):
        """Should work with negative start/end."""
        result = sched_lin(-1.0, 1.0, 0.5)
        assert abs(result - 0.0) < 1e-7

    def test_monotonic_increasing(self):
        """When start < end, the schedule should be monotonically increasing."""
        values = [sched_lin(0.0, 1.0, p / 10.0) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


# ============================================================
# Tests for sched_cos (cosine schedule)
# ============================================================

class TestSchedCos:
    """Tests for the cosine schedule function."""

    def test_at_start(self):
        """pos=0 should return start value."""
        assert abs(sched_cos(0.1, 1.0, 0.0) - 0.1) < 1e-7

    def test_at_end(self):
        """pos=1 should return end value."""
        assert abs(sched_cos(0.1, 1.0, 1.0) - 1.0) < 1e-7

    def test_midpoint(self):
        """pos=0.5 should return midpoint (cosine crosses midpoint at 0.5)."""
        result = sched_cos(0.0, 1.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_smooth_start(self):
        """Near the start, cosine schedule increases slowly."""
        val_01 = sched_cos(0.0, 1.0, 0.1)
        # Cosine annealing: near start, values are closer to start
        assert val_01 < 0.1  # slower than linear at start

    def test_smooth_end(self):
        """Near the end, cosine schedule increases slowly toward end."""
        val_09 = sched_cos(0.0, 1.0, 0.9)
        # Cosine annealing: near end, values are closer to end
        assert val_09 > 0.9  # faster approach than linear near end

    def test_decreasing(self):
        """Should handle decreasing schedules."""
        result = sched_cos(1.0, 0.0, 0.5)
        assert abs(result - 0.5) < 1e-7

    def test_monotonic_increasing(self):
        """When start < end, the schedule should be monotonically increasing."""
        values = [sched_cos(0.0, 1.0, p / 100.0) for p in range(101)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10


# ============================================================
# Tests for sched_no (constant schedule)
# ============================================================

class TestSchedNo:
    """Tests for the constant schedule function."""

    def test_at_start(self):
        """pos=0 should return start."""
        assert sched_no(0.5, 1.0, 0.0) == 0.5

    def test_at_end(self):
        """pos=1 should still return start (constant)."""
        assert sched_no(0.5, 1.0, 1.0) == 0.5

    def test_midpoint(self):
        """pos=0.5 should return start (constant)."""
        assert sched_no(0.5, 1.0, 0.5) == 0.5

    def test_always_constant(self):
        """All positions should return the start value."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert sched_no(3.14, 99.0, p) == 3.14


# ============================================================
# Tests for sched_exp (exponential schedule)
# ============================================================

class TestSchedExp:
    """Tests for the exponential schedule function."""

    def test_at_start(self):
        """pos=0 should return start value."""
        assert abs(sched_exp(0.1, 1.0, 0.0) - 0.1) < 1e-7

    def test_at_end(self):
        """pos=1 should return end value."""
        assert abs(sched_exp(0.1, 1.0, 1.0) - 1.0) < 1e-7

    def test_midpoint(self):
        """pos=0.5 should return geometric mean of start and end."""
        result = sched_exp(1.0, 100.0, 0.5)
        expected = math.sqrt(1.0 * 100.0)  # geometric mean = 10
        assert abs(result - expected) < 1e-5

    def test_monotonic_increasing(self):
        """When start < end, should be monotonically increasing."""
        values = [sched_exp(0.01, 10.0, p / 10.0) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_monotonic_decreasing(self):
        """When start > end, should be monotonically decreasing."""
        values = [sched_exp(10.0, 0.01, p / 10.0) for p in range(11)]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1]


# ============================================================
# Tests for SchedLin constructor
# ============================================================

class TestSchedLinConstructor:
    """Tests for the SchedLin constructor wrapper."""

    def test_creates_callable(self):
        sched = SchedLin(0.0, 1.0)
        assert callable(sched)

    def test_at_boundaries(self):
        sched = SchedLin(0.0, 1.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(1.0) - 1.0) < 1e-7

    def test_intermediate(self):
        sched = SchedLin(2.0, 8.0)
        assert abs(sched(0.5) - 5.0) < 1e-7


# ============================================================
# Tests for SchedCos constructor
# ============================================================

class TestSchedCosConstructor:
    """Tests for the SchedCos constructor wrapper."""

    def test_creates_callable(self):
        sched = SchedCos(0.0, 1.0)
        assert callable(sched)

    def test_at_boundaries(self):
        sched = SchedCos(0.0, 1.0)
        assert abs(sched(0.0) - 0.0) < 1e-7
        assert abs(sched(1.0) - 1.0) < 1e-7

    def test_midpoint(self):
        sched = SchedCos(0.0, 2.0)
        assert abs(sched(0.5) - 1.0) < 1e-7


# ============================================================
# Tests for SchedNo constructor
# ============================================================

class TestSchedNoConstructor:
    """Tests for the SchedNo constructor wrapper."""

    def test_creates_callable(self):
        sched = SchedNo(0.5, 1.0)
        assert callable(sched)

    def test_always_returns_start(self):
        sched = SchedNo(0.42, 99.0)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert sched(p) == 0.42


# ============================================================
# Tests for SchedExp constructor
# ============================================================

class TestSchedExpConstructor:
    """Tests for the SchedExp constructor wrapper."""

    def test_creates_callable(self):
        sched = SchedExp(0.01, 10.0)
        assert callable(sched)

    def test_at_boundaries(self):
        sched = SchedExp(0.01, 10.0)
        assert abs(sched(0.0) - 0.01) < 1e-7
        assert abs(sched(1.0) - 10.0) < 1e-5


# ============================================================
# Tests for SchedPoly (polynomial schedule)
# ============================================================

class TestSchedPoly:
    """Tests for the polynomial schedule function."""

    def test_linear_with_power_1(self):
        """Power=1 should behave like linear."""
        sched = SchedPoly(0.0, 1.0, power=1)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            expected = sched_lin(0.0, 1.0, p)
            assert abs(sched(p) - expected) < 1e-7

    def test_at_boundaries(self):
        """pos=0 returns start, pos=1 returns end regardless of power."""
        for power in [0.5, 1, 2, 3]:
            sched = SchedPoly(0.0, 1.0, power=power)
            assert abs(sched(0.0) - 0.0) < 1e-7
            assert abs(sched(1.0) - 1.0) < 1e-7

    def test_quadratic(self):
        """Power=2 should follow quadratic curve."""
        sched = SchedPoly(0.0, 1.0, power=2)
        assert abs(sched(0.5) - 0.25) < 1e-7  # 0.5^2 = 0.25

    def test_cubic(self):
        """Power=3 should follow cubic curve."""
        sched = SchedPoly(0.0, 1.0, power=3)
        assert abs(sched(0.5) - 0.125) < 1e-7  # 0.5^3 = 0.125

    def test_with_offset(self):
        """Should work with non-zero start."""
        sched = SchedPoly(2.0, 6.0, power=2)
        # At pos=0.5: 2.0 + (6.0 - 2.0) * 0.5^2 = 2.0 + 1.0 = 3.0
        assert abs(sched(0.5) - 3.0) < 1e-7

    def test_square_root(self):
        """Power=0.5 gives square root behavior."""
        sched = SchedPoly(0.0, 1.0, power=0.5)
        assert abs(sched(0.25) - 0.5) < 1e-7  # 0.25^0.5 = 0.5


# ============================================================
# Tests for combine_scheds
# ============================================================

class TestCombineScheds:
    """Tests for combining multiple schedule functions."""

    def test_single_schedule(self):
        """Single schedule with pct=1.0 should behave like the original."""
        combined = combine_scheds([1.0], [SchedLin(0.0, 1.0)])
        assert abs(combined(0.0) - 0.0) < 1e-7
        assert abs(combined(0.5) - 0.5) < 1e-7
        assert abs(combined(1.0) - 1.0) < 1e-7

    def test_two_schedules_equal_split(self):
        """Two schedules each covering 50% of training."""
        combined = combine_scheds(
            [0.5, 0.5],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # First half: linear 0 -> 1
        assert abs(combined(0.0) - 0.0) < 1e-6
        assert abs(combined(0.25) - 0.5) < 1e-6
        # Second half: linear 1 -> 0
        assert abs(combined(0.5) - 1.0) < 1e-6
        assert abs(combined(0.75) - 0.5) < 1e-6

    def test_unequal_split(self):
        """Schedules with different percentage allocations."""
        combined = combine_scheds(
            [0.3, 0.7],
            [SchedLin(0.0, 1.0), SchedNo(1.0, 1.0)]
        )
        # First 30%: linear 0 -> 1
        assert abs(combined(0.0) - 0.0) < 1e-6
        assert abs(combined(0.15) - 0.5) < 1e-6
        # Remaining 70%: constant at 1.0
        assert abs(combined(0.5) - 1.0) < 1e-6
        assert abs(combined(0.9) - 1.0) < 1e-6

    def test_boundary_between_schedules(self):
        """Values at the transition point between schedules."""
        combined = combine_scheds(
            [0.5, 0.5],
            [SchedLin(0.0, 2.0), SchedLin(2.0, 4.0)]
        )
        # At pos=0.5, first schedule ends at 2.0, second starts at 2.0
        assert abs(combined(0.5) - 2.0) < 1e-6

    def test_assertion_on_invalid_pcts(self):
        """Should raise if percentages don't sum to 1."""
        with pytest.raises(AssertionError):
            combine_scheds([0.3, 0.3], [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)])


# ============================================================
# Tests for combined_cos
# ============================================================

class TestCombinedCos:
    """Tests for the combined cosine schedule helper."""

    def test_at_boundaries(self):
        """Should start at `start` and end at `end`."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        assert abs(sched(0.0) - 0.0) < 1e-6
        # At end, should be at `end` value (0.0)
        assert abs(sched(1.0) - 0.0) < 1e-6

    def test_peak_at_pct(self):
        """Should reach `middle` value at the pct point."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        # At pos=0.3, first cosine phase ends (goes from 0 -> 1)
        assert abs(sched(0.3) - 1.0) < 1e-6

    def test_warmup_cooldown(self):
        """Typical 1cycle pattern: warmup then cooldown."""
        sched = combined_cos(0.25, 0.001, 0.01, 0.0001)
        # Start at low lr
        assert abs(sched(0.0) - 0.001) < 1e-6
        # Peak at 25%
        assert abs(sched(0.25) - 0.01) < 1e-6
        # End at very low lr
        assert abs(sched(1.0) - 0.0001) < 1e-6

    def test_symmetric(self):
        """With pct=0.5, schedule should be symmetric around middle."""
        sched = combined_cos(0.5, 0.0, 1.0, 0.0)
        assert abs(sched(0.0) - 0.0) < 1e-6
        assert abs(sched(0.5) - 1.0) < 1e-6
        assert abs(sched(1.0) - 0.0) < 1e-6
        # Check symmetry
        assert abs(sched(0.25) - sched(0.75)) < 1e-6


# ============================================================
# Tests for ParamScheduler callback
# ============================================================

class TestParamScheduler:
    """Tests for the ParamScheduler callback."""

    def test_initialization(self):
        """Should store schedules dictionary."""
        scheds = {'lr': SchedLin(0.001, 0.01)}
        ps = ParamScheduler(scheds)
        assert ps.scheds == scheds

    def test_order(self):
        """ParamScheduler should have order 60."""
        ps = ParamScheduler({'lr': SchedLin(0.001, 0.01)})
        assert ps.order == 60

    def test_run_valid_false(self):
        """ParamScheduler should not run during validation."""
        ps = ParamScheduler({'lr': SchedLin(0.001, 0.01)})
        assert ps.run_valid is False

    def test_before_fit_initializes_hps(self):
        """before_fit should initialize empty HP tracking dictionaries."""
        ps = ParamScheduler({'lr': SchedLin(0.001, 0.01), 'mom': SchedCos(0.9, 0.8)})
        ps.hps = None  # simulate uninitialized state
        ps.before_fit()
        assert 'lr' in ps.hps
        assert 'mom' in ps.hps
        assert ps.hps['lr'] == []
        assert ps.hps['mom'] == []


# ============================================================
# Tests for valley suggestion method
# ============================================================

class TestValley:
    """Tests for the valley LR suggestion method."""

    def test_basic_valley(self):
        """Should find a learning rate in a simple descending-then-ascending loss curve."""
        n = 50
        lrs = torch.logspace(-7, 1, n).tolist()
        # Create a loss curve that decreases then increases
        losses = [10.0 - 8.0 * (i / 25.0) if i < 25 else 2.0 + 5.0 * ((i - 25) / 25.0) for i in range(n)]
        lr_val, (lr_point, loss_point) = valley(lrs, losses, num_it=n)
        # Should suggest an lr in the decreasing region
        assert lr_val > 0
        assert lr_val < lrs[-1]

    def test_returns_float(self):
        """The suggested LR should be a float."""
        lrs = torch.logspace(-5, 0, 30).tolist()
        losses = [5.0 - 4.0 * (i / 15.0) if i < 15 else 1.0 + 3.0 * ((i - 15) / 15.0) for i in range(30)]
        lr_val, _ = valley(lrs, losses, num_it=30)
        assert isinstance(lr_val, float)

    def test_monotone_decreasing(self):
        """With purely decreasing losses, valley should still return something sensible."""
        n = 20
        lrs = torch.logspace(-5, 0, n).tolist()
        losses = [10.0 - 9.0 * (i / (n - 1)) for i in range(n)]
        lr_val, _ = valley(lrs, losses, num_it=n)
        assert lr_val > 0


# ============================================================
# Tests for slide suggestion method
# ============================================================

class TestSlide:
    """Tests for the slide LR suggestion method."""

    def test_basic_slide(self):
        """Should suggest a learning rate for a typical loss curve."""
        n = 50
        lrs = torch.logspace(-7, 0, n)
        # Create a smooth loss curve (as tensor so to_np works correctly)
        losses = torch.tensor([5.0 * math.exp(-3.0 * i / n) + 0.5 for i in range(n)])
        lr_val, (lr_point, loss_point) = slide(lrs, losses, num_it=n)
        assert lr_val > 0
        assert isinstance(lr_val, float)

    def test_returns_positive(self):
        """Suggested LR should always be positive."""
        n = 40
        lrs = torch.logspace(-6, -1, n)
        losses = torch.tensor([3.0 - 2.0 * (i / n) for i in range(n)])
        lr_val, _ = slide(lrs, losses, num_it=n)
        assert lr_val > 0


# ============================================================
# Tests for minimum suggestion method
# ============================================================

class TestMinimum:
    """Tests for the minimum LR suggestion method."""

    def test_basic_minimum(self):
        """Should suggest lr_min/10 where loss is minimum."""
        n = 50
        lrs = torch.logspace(-5, 0, n)
        # Loss curve with clear minimum at index 30
        losses = torch.tensor([abs(i - 30) * 0.1 + 0.5 for i in range(n)])
        lr_val, (lr_point, loss_point) = minimum(lrs, losses, num_it=n)
        # Should be lr at min loss divided by 10
        expected_lr_min = lrs[losses.argmin()].item()
        assert abs(lr_val - expected_lr_min / 10) < 1e-7

    def test_returns_one_tenth(self):
        """Should return 1/10th of the lr at the minimum loss."""
        n = 30
        lrs = torch.logspace(-4, 0, n)
        losses = torch.tensor([(i - 15) ** 2 * 0.01 + 1.0 for i in range(n)])
        lr_val, _ = minimum(lrs, losses, num_it=n)
        min_idx = losses.argmin()
        assert abs(lr_val - lrs[min_idx].item() / 10) < 1e-8

    def test_returns_positive(self):
        """Suggested LR should be positive."""
        n = 20
        lrs = torch.logspace(-3, 0, n)
        losses = torch.tensor([2.0 - 1.5 * math.exp(-((i - 10) ** 2) / 20.0) for i in range(n)])
        lr_val, _ = minimum(lrs, losses, num_it=n)
        assert lr_val > 0


# ============================================================
# Tests for steep suggestion method
# ============================================================

class TestSteep:
    """Tests for the steep LR suggestion method."""

    def test_basic_steep(self):
        """Should find the lr where the loss is decreasing fastest."""
        n = 50
        lrs = torch.logspace(-5, 0, n)
        # Create a loss curve with a steep decline region
        losses = torch.tensor([5.0 / (1.0 + math.exp((i - 25) / 3.0)) + 0.5 for i in range(n)])
        lr_val, (lr_point, loss_point) = steep(lrs, losses, num_it=n)
        assert lr_val > 0
        assert isinstance(lr_val, float)

    def test_returns_positive(self):
        """Suggested LR should be positive."""
        n = 30
        lrs = torch.logspace(-4, 0, n)
        losses = torch.tensor([3.0 * math.exp(-2.0 * i / n) + 1.0 for i in range(n)])
        lr_val, _ = steep(lrs, losses, num_it=n)
        assert lr_val > 0

    def test_steepest_point_in_descent(self):
        """Should pick a point in the descending portion of the loss curve."""
        n = 40
        lrs = torch.logspace(-5, 0, n)
        # Loss descends first 20, then ascends
        losses_list = [5.0 - 4.0 * (i / 20.0) if i < 20 else 1.0 + 3.0 * ((i - 20) / 20.0) for i in range(n)]
        losses = torch.tensor(losses_list)
        lr_val, _ = steep(lrs, losses, num_it=n)
        # The steepest descent is in the first half, so lr should be relatively small
        assert lr_val < lrs[30].item()


# ============================================================
# Tests for schedule function properties
# ============================================================

class TestScheduleProperties:
    """Cross-cutting tests for schedule function properties."""

    def test_all_schedules_agree_at_start(self):
        """All schedule types should return start at pos=0."""
        start, end = 0.1, 1.0
        assert abs(sched_lin(start, end, 0.0) - start) < 1e-7
        assert abs(sched_cos(start, end, 0.0) - start) < 1e-7
        assert abs(sched_no(start, end, 0.0) - start) < 1e-7
        assert abs(sched_exp(start, end, 0.0) - start) < 1e-7

    def test_all_schedules_agree_at_end(self):
        """Linear, cosine, and exp should return end at pos=1. No returns start."""
        start, end = 0.1, 1.0
        assert abs(sched_lin(start, end, 1.0) - end) < 1e-7
        assert abs(sched_cos(start, end, 1.0) - end) < 1e-7
        assert abs(sched_exp(start, end, 1.0) - end) < 1e-7
        # sched_no always returns start
        assert abs(sched_no(start, end, 1.0) - start) < 1e-7

    def test_constructors_match_functions(self):
        """SchedLin/Cos/No/Exp constructors should produce same results as raw functions."""
        start, end = 0.01, 10.0
        for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert abs(SchedLin(start, end)(pos) - sched_lin(start, end, pos)) < 1e-7
            assert abs(SchedCos(start, end)(pos) - sched_cos(start, end, pos)) < 1e-7
            assert abs(SchedNo(start, end)(pos) - sched_no(start, end, pos)) < 1e-7
            assert abs(SchedExp(start, end)(pos) - sched_exp(start, end, pos)) < 1e-7

    def test_cosine_between_linear_at_extremes(self):
        """Cosine schedule should be below linear in first half and above in second half
        when going from lower to higher value (due to its S-curve shape)."""
        start, end = 0.0, 1.0
        # In first quarter, cosine < linear
        assert sched_cos(start, end, 0.25) < sched_lin(start, end, 0.25)
        # In last quarter, cosine > linear
        assert sched_cos(start, end, 0.75) > sched_lin(start, end, 0.75)

    def test_exp_vs_linear_ordering(self):
        """Exponential schedule grows slower than linear initially, then faster."""
        start, end = 0.01, 1.0
        # At midpoint, exp gives geometric mean, linear gives arithmetic mean
        exp_mid = sched_exp(start, end, 0.5)
        lin_mid = sched_lin(start, end, 0.5)
        # Geometric mean < arithmetic mean for positive numbers
        assert exp_mid < lin_mid


# ============================================================
# Tests for _Annealer class (via constructors)
# ============================================================

class TestAnnealer:
    """Tests for the _Annealer class underlying schedule constructors."""

    def test_annealer_stores_start_end(self):
        """Annealer should store start and end values."""
        sched = SchedLin(0.1, 0.9)
        assert sched.start == 0.1
        assert sched.end == 0.9

    def test_annealer_callable(self):
        """Annealer should be callable with a position argument."""
        sched = SchedLin(0.0, 1.0)
        result = sched(0.5)
        assert isinstance(result, float)

    def test_annealer_with_zero_range(self):
        """When start == end, all schedule types should return that value."""
        for Sched in [SchedLin, SchedCos, SchedNo, SchedExp]:
            sched = Sched(0.5, 0.5)
            for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
                assert abs(sched(p) - 0.5) < 1e-7


# ============================================================
# Tests for combine_scheds edge cases
# ============================================================

class TestCombineSchedsEdgeCases:
    """Edge cases for combine_scheds."""

    def test_three_phases(self):
        """Three-phase schedule (warmup, constant, cooldown)."""
        combined = combine_scheds(
            [0.2, 0.5, 0.3],
            [SchedLin(0.0, 1.0), SchedNo(1.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # Warmup phase
        assert abs(combined(0.0) - 0.0) < 1e-6
        assert abs(combined(0.1) - 0.5) < 1e-6
        # Constant phase
        assert abs(combined(0.2) - 1.0) < 1e-6
        assert abs(combined(0.5) - 1.0) < 1e-6
        # Cooldown phase
        assert abs(combined(0.7) - 1.0) < 1e-6
        assert abs(combined(1.0) - 0.0) < 1e-6

    def test_very_small_first_phase(self):
        """Very small first phase (like quick warmup)."""
        combined = combine_scheds(
            [0.01, 0.99],
            [SchedLin(0.0, 1.0), SchedCos(1.0, 0.0)]
        )
        assert abs(combined(0.0) - 0.0) < 1e-5
        assert abs(combined(0.01) - 1.0) < 1e-5


# ============================================================
# Tests for combined_cos typical usage
# ============================================================

class TestCombinedCosUsage:
    """Practical usage patterns for combined_cos."""

    def test_one_cycle_lr_shape(self):
        """1cycle LR: warmup from low to high, then anneal to very low."""
        lr_min, lr_max, lr_final = 1e-4, 1e-2, 1e-5
        sched = combined_cos(0.3, lr_min, lr_max, lr_final)
        # Starts at lr_min
        assert abs(sched(0.0) - lr_min) < 1e-8
        # Peaks at lr_max at 30%
        assert abs(sched(0.3) - lr_max) < 1e-8
        # Ends at lr_final
        assert abs(sched(1.0) - lr_final) < 1e-8

    def test_momentum_schedule(self):
        """Momentum goes high -> low -> high (opposite of LR in 1cycle)."""
        mom_high, mom_low = 0.95, 0.85
        sched = combined_cos(0.3, mom_high, mom_low, mom_high)
        assert abs(sched(0.0) - mom_high) < 1e-7
        assert abs(sched(0.3) - mom_low) < 1e-7
        assert abs(sched(1.0) - mom_high) < 1e-7

    def test_values_bounded(self):
        """All intermediate values should stay between start/end extremes."""
        sched = combined_cos(0.25, 0.001, 0.01, 0.0001)
        for p_int in range(101):
            p = p_int / 100.0
            val = sched(p)
            assert val >= 0.0001 - 1e-8
            assert val <= 0.01 + 1e-8
