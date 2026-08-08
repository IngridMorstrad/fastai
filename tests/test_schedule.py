"""Tests for fastai/callback/schedule.py scheduling functions.

These tests validate the pure mathematical scheduling functions used for
learning rate and hyper-parameter scheduling. The functions are tested
by verifying boundary conditions (pos=0 returns start, pos=1 returns end),
midpoint values, and monotonicity properties.

The heavier combine_scheds and combined_cos functions require torch tensors,
so we mock torch and fastai imports to keep the test suite lightweight.
"""

import sys
import types
import math
import unittest
import functools
import os


# ----- Mock Setup -----
# Mock the fastai import chain so schedule.py can be imported without
# requiring the full PyTorch/fastai dependency stack.

def _make_module(name, attrs=None):
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- Minimal tensor/torch mock for combine_scheds ---

class _MockTensor:
    """Minimal tensor-like object supporting operations used in combine_scheds.

    Supports broadcasting: a single-element tensor compared against a
    multi-element tensor broadcasts the scalar to all elements (like PyTorch).
    """
    def __init__(self, data):
        if isinstance(data, _MockTensor):
            self.data = data.data
        elif isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            self.data = [data]

    def __ge__(self, other):
        """Element-wise >= with broadcasting support."""
        if isinstance(other, _MockTensor):
            # Broadcasting: if one side is scalar, broadcast to the other's shape
            if len(self.data) == 1 and len(other.data) > 1:
                return _MockTensor([self.data[0] >= b for b in other.data])
            elif len(other.data) == 1 and len(self.data) > 1:
                return _MockTensor([a >= other.data[0] for a in self.data])
            return _MockTensor([a >= b for a, b in zip(self.data, other.data)])
        return _MockTensor([a >= other for a in self.data])

    def nonzero(self):
        """Return indices where True."""
        return _MockTensor([i for i, v in enumerate(self.data) if v])

    def max(self):
        """Return max value."""
        if not self.data:
            return 0
        return max(self.data)

    def item(self):
        """Return the scalar value."""
        if isinstance(self.data, list) and len(self.data) == 1:
            return self.data[0]
        return self.data

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _MockTensor(self.data[idx])
        return self.data[idx]

    def __sub__(self, other):
        if isinstance(other, _MockTensor):
            if len(self.data) == 1 and len(other.data) > 1:
                return _MockTensor([self.data[0] - b for b in other.data])
            elif len(other.data) == 1 and len(self.data) > 1:
                return _MockTensor([a - other.data[0] for a in self.data])
            return _MockTensor([a - b for a, b in zip(self.data, other.data)])
        return _MockTensor([a - other for a in self.data])

    def __truediv__(self, other):
        if isinstance(other, _MockTensor):
            if len(self.data) == 1 and len(other.data) > 1:
                return _MockTensor([self.data[0] / b for b in other.data])
            elif len(other.data) == 1 and len(self.data) > 1:
                return _MockTensor([a / other.data[0] for a in self.data])
            return _MockTensor([a / b for a, b in zip(self.data, other.data)])
        return _MockTensor([a / other for a in self.data])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"_MockTensor({self.data})"


def _mock_tensor(data):
    """Create a mock tensor from a list."""
    return _MockTensor(data)


def _mock_cumsum(t, dim):
    """Cumulative sum for mock tensor."""
    result = []
    running = 0
    for v in t.data:
        running += v
        result.append(running)
    return _MockTensor(result)


def _mock_all(t):
    """Check if all values are truthy."""
    return all(t.data)


# Mock torch
torch_mock = _make_module('torch', {
    'tensor': _mock_tensor,
    'cumsum': _mock_cumsum,
    'all': _mock_all,
})
torch_mock.Tensor = _MockTensor
_make_module('torch.multiprocessing')
_make_module('torch.nn')

# Create a mock L class (from fastcore)
class _MockL(list):
    def __init__(self, *args):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            super().__init__(args[0])
        else:
            super().__init__(args)


# Mock store_attr
def _store_attr(names, **kwargs):
    """Mock store_attr that sets attributes from the caller's local scope."""
    import inspect
    frame = inspect.currentframe().f_back
    self = frame.f_locals.get('self')
    if self is not None:
        for name in names.split(','):
            name = name.strip()
            if name in frame.f_locals:
                setattr(self, name, frame.f_locals[name])


# Create fastai package hierarchy
fastai_pkg = _make_module('fastai')
fastai_pkg.__path__ = []

basics_mod = _make_module('fastai.basics', {
    'math': math,
    'functools': functools,
    'torch': torch_mock,
    'tensor': _mock_tensor,
    'L': _MockL,
    'store_attr': _store_attr,
})

callback_pkg = _make_module('fastai.callback')
callback_pkg.__path__ = []

# Remove any cached schedule module
if 'fastai.callback.schedule' in sys.modules:
    del sys.modules['fastai.callback.schedule']

# Load the schedule module source and exec it with our mocked namespace
_schedule_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'schedule.py')
_schedule_path = os.path.abspath(_schedule_path)

schedule_module = types.ModuleType('fastai.callback.schedule')
schedule_module.__file__ = _schedule_path
schedule_module.__package__ = 'fastai.callback'

# Populate namespace with what `from ..basics import *` would provide
schedule_module.math = math
schedule_module.functools = functools
schedule_module.torch = torch_mock
schedule_module.tensor = _mock_tensor
schedule_module.L = _MockL
schedule_module.store_attr = _store_attr
schedule_module.__builtins__ = __builtins__

# Read source and filter out problematic imports
with open(_schedule_path, 'r') as f:
    source = f.read()

lines = source.split('\n')
filtered_lines = []
for line in lines:
    if line.startswith('from __future__'):
        filtered_lines.append(line)
    elif line.startswith('from ..') or line.startswith('from .'):
        filtered_lines.append('pass  # skipped import')
    else:
        filtered_lines.append(line)

# We only need the scheduling functions, not the Callback/Learner parts.
# Stop before the ParamScheduler section to avoid needing Callback, docs, patch, etc.
final_lines = []
for line in filtered_lines:
    # The @docs decorator precedes ParamScheduler - stop there
    if line.strip() == '@docs':
        break
    if 'class ParamScheduler' in line:
        break
    final_lines.append(line)

exec(compile('\n'.join(final_lines), _schedule_path, 'exec'), schedule_module.__dict__)
sys.modules['fastai.callback.schedule'] = schedule_module

# Extract the functions we want to test
sched_lin = schedule_module.sched_lin
sched_cos = schedule_module.sched_cos
sched_no = schedule_module.sched_no
sched_exp = schedule_module.sched_exp
SchedLin = schedule_module.SchedLin
SchedCos = schedule_module.SchedCos
SchedNo = schedule_module.SchedNo
SchedExp = schedule_module.SchedExp
SchedPoly = schedule_module.SchedPoly
combine_scheds = schedule_module.combine_scheds
combined_cos = schedule_module.combined_cos


# ----- Tests -----

class TestSchedLin(unittest.TestCase):
    """Test linear schedule function."""

    def test_pos_zero_returns_start(self):
        """At pos=0, should return start value."""
        self.assertAlmostEqual(sched_lin(0.1, 0.9, 0.0), 0.1)

    def test_pos_one_returns_end(self):
        """At pos=1, should return end value."""
        self.assertAlmostEqual(sched_lin(0.1, 0.9, 1.0), 0.9)

    def test_midpoint(self):
        """At pos=0.5, should return midpoint between start and end."""
        self.assertAlmostEqual(sched_lin(0.0, 1.0, 0.5), 0.5)
        self.assertAlmostEqual(sched_lin(2.0, 4.0, 0.5), 3.0)

    def test_quarter_point(self):
        """At pos=0.25, should return 1/4 of the way from start to end."""
        self.assertAlmostEqual(sched_lin(0.0, 1.0, 0.25), 0.25)
        self.assertAlmostEqual(sched_lin(1.0, 5.0, 0.25), 2.0)

    def test_increasing_values(self):
        """Linear schedule from small to large should be monotonically increasing."""
        start, end = 0.01, 0.1
        prev = sched_lin(start, end, 0.0)
        for i in range(1, 11):
            pos = i / 10.0
            val = sched_lin(start, end, pos)
            self.assertGreater(val, prev)
            prev = val

    def test_decreasing_values(self):
        """Linear schedule from large to small should be monotonically decreasing."""
        start, end = 1.0, 0.0
        prev = sched_lin(start, end, 0.0)
        for i in range(1, 11):
            pos = i / 10.0
            val = sched_lin(start, end, pos)
            self.assertLess(val, prev)
            prev = val

    def test_same_start_end(self):
        """When start equals end, should always return that value."""
        self.assertAlmostEqual(sched_lin(0.5, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(sched_lin(0.5, 0.5, 0.5), 0.5)
        self.assertAlmostEqual(sched_lin(0.5, 0.5, 1.0), 0.5)


class TestSchedCos(unittest.TestCase):
    """Test cosine annealing schedule function."""

    def test_pos_zero_returns_start(self):
        """At pos=0, should return start value."""
        self.assertAlmostEqual(sched_cos(0.1, 0.9, 0.0), 0.1)

    def test_pos_one_returns_end(self):
        """At pos=1, should return end value."""
        self.assertAlmostEqual(sched_cos(0.1, 0.9, 1.0), 0.9)

    def test_midpoint(self):
        """At pos=0.5, cosine schedule should return midpoint between start and end."""
        self.assertAlmostEqual(sched_cos(0.0, 1.0, 0.5), 0.5)
        self.assertAlmostEqual(sched_cos(2.0, 4.0, 0.5), 3.0)

    def test_monotonically_increasing(self):
        """Cosine schedule from small to large should be monotonically increasing."""
        start, end = 0.0, 1.0
        prev = sched_cos(start, end, 0.0)
        for i in range(1, 101):
            pos = i / 100.0
            val = sched_cos(start, end, pos)
            self.assertGreaterEqual(val, prev)
            prev = val

    def test_symmetry_around_midpoint(self):
        """Cosine schedule should be symmetric: f(0.25) + f(0.75) = start + end."""
        start, end = 0.0, 1.0
        val_quarter = sched_cos(start, end, 0.25)
        val_three_quarter = sched_cos(start, end, 0.75)
        self.assertAlmostEqual(val_quarter + val_three_quarter, start + end)

    def test_slow_start_fast_middle(self):
        """Cosine should change slowly near boundaries and faster in the middle."""
        start, end = 0.0, 1.0
        # Change from 0 to 0.1 (near start) should be small
        delta_start = sched_cos(start, end, 0.1) - sched_cos(start, end, 0.0)
        # Change from 0.45 to 0.55 (near middle) should be larger
        delta_mid = sched_cos(start, end, 0.55) - sched_cos(start, end, 0.45)
        self.assertGreater(delta_mid, delta_start)

    def test_same_start_end(self):
        """When start equals end, should always return that value."""
        self.assertAlmostEqual(sched_cos(0.5, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(sched_cos(0.5, 0.5, 0.5), 0.5)
        self.assertAlmostEqual(sched_cos(0.5, 0.5, 1.0), 0.5)


class TestSchedNo(unittest.TestCase):
    """Test constant (no change) schedule function."""

    def test_pos_zero_returns_start(self):
        """At pos=0, should return start value."""
        self.assertAlmostEqual(sched_no(0.5, 0.9, 0.0), 0.5)

    def test_pos_one_returns_start(self):
        """At pos=1, should still return start value (constant)."""
        self.assertAlmostEqual(sched_no(0.5, 0.9, 1.0), 0.5)

    def test_midpoint_returns_start(self):
        """At any position, should always return start."""
        self.assertAlmostEqual(sched_no(0.3, 0.9, 0.5), 0.3)

    def test_always_constant(self):
        """Should return the same value regardless of position."""
        start, end = 0.01, 0.1
        for i in range(11):
            pos = i / 10.0
            self.assertAlmostEqual(sched_no(start, end, pos), start)


class TestSchedExp(unittest.TestCase):
    """Test exponential schedule function."""

    def test_pos_zero_returns_start(self):
        """At pos=0, should return start value."""
        self.assertAlmostEqual(sched_exp(0.1, 0.9, 0.0), 0.1)

    def test_pos_one_returns_end(self):
        """At pos=1, should return end value."""
        self.assertAlmostEqual(sched_exp(0.1, 0.9, 1.0), 0.9)

    def test_monotonically_increasing(self):
        """Exponential schedule from small to large should be monotonically increasing."""
        start, end = 0.001, 1.0
        prev = sched_exp(start, end, 0.0)
        for i in range(1, 11):
            pos = i / 10.0
            val = sched_exp(start, end, pos)
            self.assertGreater(val, prev)
            prev = val

    def test_geometric_mean_at_midpoint(self):
        """At pos=0.5, exponential should return geometric mean of start and end."""
        start, end = 0.01, 1.0
        expected = math.sqrt(start * end)  # geometric mean
        self.assertAlmostEqual(sched_exp(start, end, 0.5), expected)

    def test_multiplicative_property(self):
        """Exponential schedule should have constant ratio between equal intervals."""
        start, end = 0.001, 1.0
        # Ratio between pos=0.2 and pos=0.1 should equal ratio between pos=0.3 and pos=0.2
        v1 = sched_exp(start, end, 0.1)
        v2 = sched_exp(start, end, 0.2)
        v3 = sched_exp(start, end, 0.3)
        ratio1 = v2 / v1
        ratio2 = v3 / v2
        self.assertAlmostEqual(ratio1, ratio2, places=10)

    def test_same_start_end(self):
        """When start equals end, should always return that value."""
        self.assertAlmostEqual(sched_exp(0.5, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(sched_exp(0.5, 0.5, 0.5), 0.5)
        self.assertAlmostEqual(sched_exp(0.5, 0.5, 1.0), 0.5)


class TestSchedWrappers(unittest.TestCase):
    """Test the SchedLin, SchedCos, SchedNo, SchedExp wrapper classes."""

    def test_sched_lin_wrapper(self):
        """SchedLin should create an _Annealer wrapping sched_lin."""
        sched = SchedLin(0.0, 1.0)
        self.assertAlmostEqual(sched(0.0), 0.0)
        self.assertAlmostEqual(sched(1.0), 1.0)
        self.assertAlmostEqual(sched(0.5), 0.5)

    def test_sched_cos_wrapper(self):
        """SchedCos should create an _Annealer wrapping sched_cos."""
        sched = SchedCos(0.0, 1.0)
        self.assertAlmostEqual(sched(0.0), 0.0)
        self.assertAlmostEqual(sched(1.0), 1.0)
        self.assertAlmostEqual(sched(0.5), 0.5)

    def test_sched_no_wrapper(self):
        """SchedNo should create an _Annealer wrapping sched_no."""
        sched = SchedNo(0.5, 1.0)
        self.assertAlmostEqual(sched(0.0), 0.5)
        self.assertAlmostEqual(sched(0.5), 0.5)
        self.assertAlmostEqual(sched(1.0), 0.5)

    def test_sched_exp_wrapper(self):
        """SchedExp should create an _Annealer wrapping sched_exp."""
        sched = SchedExp(0.01, 1.0)
        self.assertAlmostEqual(sched(0.0), 0.01)
        self.assertAlmostEqual(sched(1.0), 1.0)
        self.assertAlmostEqual(sched(0.5), math.sqrt(0.01 * 1.0))


class TestSchedPoly(unittest.TestCase):
    """Test polynomial schedule function."""

    def test_pos_zero_returns_start(self):
        """At pos=0, should return start regardless of power."""
        sched = SchedPoly(0.1, 0.9, 2)
        self.assertAlmostEqual(sched(0.0), 0.1)

    def test_pos_one_returns_end(self):
        """At pos=1, should return end regardless of power."""
        sched = SchedPoly(0.1, 0.9, 2)
        self.assertAlmostEqual(sched(1.0), 0.9)

    def test_linear_when_power_one(self):
        """With power=1, SchedPoly should behave identically to sched_lin."""
        sched = SchedPoly(0.0, 1.0, 1)
        for i in range(11):
            pos = i / 10.0
            self.assertAlmostEqual(sched(pos), sched_lin(0.0, 1.0, pos))

    def test_quadratic(self):
        """With power=2, should follow quadratic curve."""
        sched = SchedPoly(0.0, 1.0, 2)
        self.assertAlmostEqual(sched(0.5), 0.25)  # 0.5^2 = 0.25
        self.assertAlmostEqual(sched(0.25), 0.0625)  # 0.25^2 = 0.0625

    def test_cubic(self):
        """With power=3, should follow cubic curve."""
        sched = SchedPoly(0.0, 1.0, 3)
        self.assertAlmostEqual(sched(0.5), 0.125)  # 0.5^3 = 0.125

    def test_higher_power_slower_start(self):
        """Higher power means slower initial progress."""
        sched2 = SchedPoly(0.0, 1.0, 2)
        sched3 = SchedPoly(0.0, 1.0, 3)
        # At pos=0.3, cubic should be slower (further from end)
        self.assertGreater(sched2(0.3), sched3(0.3))

    def test_monotonically_increasing(self):
        """Polynomial schedule with positive power should be monotonically increasing."""
        sched = SchedPoly(0.0, 1.0, 2)
        prev = sched(0.0)
        for i in range(1, 11):
            pos = i / 10.0
            val = sched(pos)
            self.assertGreater(val, prev)
            prev = val

    def test_fractional_power(self):
        """Fractional power (like 0.5 = sqrt) should work correctly."""
        sched = SchedPoly(0.0, 1.0, 0.5)
        self.assertAlmostEqual(sched(0.25), 0.5)  # 0.25^0.5 = 0.5
        self.assertAlmostEqual(sched(1.0), 1.0)


class TestCombineScheds(unittest.TestCase):
    """Test combine_scheds function."""

    def test_single_schedule_full_range(self):
        """A single schedule covering 100% should behave like that schedule."""
        combined = combine_scheds([1.0], [SchedLin(0.0, 1.0)])
        self.assertAlmostEqual(combined(_mock_tensor([0.0])), 0.0)
        self.assertAlmostEqual(combined(_mock_tensor([1.0])), 1.0)
        self.assertAlmostEqual(combined(_mock_tensor([0.5])), 0.5)

    def test_two_schedules_equal_split(self):
        """Two schedules each covering 50% should transition at the midpoint."""
        combined = combine_scheds(
            [0.5, 0.5],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # First half: linear 0 -> 1
        self.assertAlmostEqual(combined(_mock_tensor([0.0])), 0.0)
        self.assertAlmostEqual(combined(_mock_tensor([0.25])), 0.5)
        # Second half: linear 1 -> 0
        self.assertAlmostEqual(combined(_mock_tensor([0.75])), 0.5)
        self.assertAlmostEqual(combined(_mock_tensor([1.0])), 0.0)

    def test_boundary_between_schedules(self):
        """At the boundary between two schedules, the second schedule should start."""
        combined = combine_scheds(
            [0.5, 0.5],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 2.0)]
        )
        # At pos=0.5, should be start of second schedule = 1.0
        self.assertAlmostEqual(combined(_mock_tensor([0.5])), 1.0)

    def test_unequal_splits(self):
        """Unequal percentage splits should work correctly."""
        combined = combine_scheds(
            [0.3, 0.7],
            [SchedLin(0.0, 1.0), SchedLin(1.0, 0.0)]
        )
        # First 30%: pos=0.15 is 50% of first segment => 0.5
        self.assertAlmostEqual(combined(_mock_tensor([0.15])), 0.5)


class TestCombinedCos(unittest.TestCase):
    """Test combined_cos function (cosine warmup + cosine decay)."""

    def test_start_value(self):
        """At pos=0, should return start value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(sched(_mock_tensor([0.0])), 0.0)

    def test_middle_value_at_pct(self):
        """At the pct boundary, should reach middle value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(sched(_mock_tensor([0.3])), 1.0)

    def test_end_value(self):
        """At pos=1, should return end value."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(sched(_mock_tensor([1.0])), 0.0)

    def test_warmup_phase_increases(self):
        """During warmup phase (0 to pct), values should increase toward middle."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        prev = sched(_mock_tensor([0.0]))
        for i in range(1, 10):
            pos = 0.3 * i / 10.0  # positions within warmup phase
            val = sched(_mock_tensor([pos]))
            self.assertGreaterEqual(val, prev)
            prev = val

    def test_decay_phase_decreases(self):
        """During decay phase (pct to 1), values should decrease toward end."""
        sched = combined_cos(0.3, 0.0, 1.0, 0.0)
        prev = sched(_mock_tensor([0.3]))
        for i in range(1, 10):
            pos = 0.3 + 0.7 * i / 10.0  # positions within decay phase
            val = sched(_mock_tensor([pos]))
            self.assertLessEqual(val, prev)
            prev = val

    def test_symmetric_warmup_decay(self):
        """With pct=0.5 and same start/end, the schedule should be symmetric."""
        sched = combined_cos(0.5, 0.0, 1.0, 0.0)
        # Value at 0.25 (midpoint of warmup) should equal value at 0.75 (midpoint of decay)
        val_warmup = sched(_mock_tensor([0.25]))
        val_decay = sched(_mock_tensor([0.75]))
        self.assertAlmostEqual(val_warmup, val_decay)

    def test_different_start_end(self):
        """Start and end can differ - the decay goes from middle to end."""
        sched = combined_cos(0.3, 0.1, 0.5, 0.2)
        self.assertAlmostEqual(sched(_mock_tensor([0.0])), 0.1)
        self.assertAlmostEqual(sched(_mock_tensor([0.3])), 0.5)
        self.assertAlmostEqual(sched(_mock_tensor([1.0])), 0.2)


class TestAnnealerClass(unittest.TestCase):
    """Test the _Annealer class behavior."""

    def test_annealer_stores_attributes(self):
        """_Annealer should store f, start, end."""
        annealer = schedule_module._Annealer(sched_lin, 0.0, 1.0)
        self.assertEqual(annealer.start, 0.0)
        self.assertEqual(annealer.end, 1.0)
        self.assertEqual(annealer.f, sched_lin)

    def test_annealer_callable(self):
        """_Annealer instances should be callable."""
        annealer = schedule_module._Annealer(sched_lin, 0.0, 1.0)
        self.assertAlmostEqual(annealer(0.5), 0.5)


if __name__ == '__main__':
    unittest.main()
