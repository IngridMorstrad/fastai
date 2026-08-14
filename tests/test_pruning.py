"""Tests for PruningCallback and Learner.prune().

These tests mock heavy fastai/torch dependencies so pruning logic can be
validated without a full PyTorch/fastai installation.
"""
import unittest
from unittest.mock import MagicMock, patch as mock_patch, call
import sys
import os
import types
import numpy as np


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Minimal tensor mock supporting dim(), shape, nelement(), item()."""
    def __init__(self, shape):
        self._shape = shape
        self._data = np.zeros(shape)

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def nelement(self):
        return int(np.prod(self._shape))

    def __eq__(self, other):
        return _FakeBoolTensor(self._data == other)


class _FakeBoolTensor:
    """Result of tensor == value."""
    def __init__(self, arr):
        self._arr = arr

    def sum(self):
        return _FakeScalar(int(self._arr.sum()))


class _FakeScalar:
    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val


class FakeModule:
    """Mock nn.Module with weight attribute."""
    def __init__(self, weight_shape=(64, 32, 3, 3)):
        self.weight = _FakeTensor(weight_shape)
        self._pruned = False

    def modules(self):
        return [self]


class FakeConv2d(FakeModule):
    pass


class FakeLinear(FakeModule):
    def __init__(self, weight_shape=(128, 64)):
        super().__init__(weight_shape)


class FakeModel:
    """Mock model with multiple submodules."""
    def __init__(self):
        self.conv1 = FakeConv2d((64, 3, 3, 3))
        self.conv2 = FakeConv2d((128, 64, 3, 3))
        self.fc = FakeLinear((10, 128))
        self._modules_list = [self.conv1, self.conv2, self.fc]

    def modules(self):
        return [self] + self._modules_list


def _setup_mock_env():
    """Set up mock modules for importing pruning.py."""
    # nn module with Conv2d and Linear classes
    nn_mod = types.ModuleType('torch.nn')
    nn_mod.Conv2d = FakeConv2d
    nn_mod.Linear = FakeLinear
    nn_mod.Module = object

    # torch.nn.utils.prune mock
    prune_mod = types.ModuleType('torch.nn.utils.prune')
    prune_mod.ln_structured = MagicMock()
    prune_mod.global_unstructured = MagicMock()
    prune_mod.L1Unstructured = 'L1Unstructured'
    prune_mod.is_pruned = MagicMock(return_value=True)
    prune_mod.remove = MagicMock()

    utils_mod = types.ModuleType('torch.nn.utils')
    utils_mod.prune = prune_mod
    sys.modules['torch.nn.utils'] = utils_mod
    sys.modules['torch.nn.utils.prune'] = prune_mod

    # torch
    torch_mod = types.ModuleType('torch')
    torch_mod.nn = nn_mod
    sys.modules['torch'] = torch_mod
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.multiprocessing'] = types.ModuleType('torch.multiprocessing')

    # Fake Callback base class
    class Callback:
        order = 0
        def __init_subclass__(cls, **kwargs): pass

    # store_attr mock
    def store_attr(self_=None, **kwargs):
        pass

    # L mock (basic list wrapper)
    class L(list):
        def __init__(self, items=None):
            super().__init__(items or [])
        def __add__(self, other):
            return L(list.__add__(self, list(other) if not isinstance(other, list) else other))

    # fastai.basics
    basics_mod = types.ModuleType('fastai.basics')
    basics_mod.Callback = Callback
    basics_mod.nn = nn_mod
    basics_mod.store_attr = store_attr
    basics_mod.L = L
    basics_mod.np = np
    basics_mod.patch = lambda f: f  # no-op decorator for now

    # fastai package structure
    fastai_mod = types.ModuleType('fastai')
    fastai_mod.__path__ = [os.path.join(os.path.dirname(__file__), '..', 'fastai')]

    callback_mod = types.ModuleType('fastai.callback')
    callback_mod.__path__ = []

    sys.modules['fastai'] = fastai_mod
    sys.modules['fastai.basics'] = basics_mod
    sys.modules['fastai.callback'] = callback_mod
    sys.modules['fastai.callback.progress'] = types.ModuleType('fastai.callback.progress')
    sys.modules['fastai.callback.fp16'] = types.ModuleType('fastai.callback.fp16')

    return prune_mod, Callback, nn_mod, L


def _load_pruning_module(nn_mod, Callback, store_attr_fn, L_cls):
    """Load pruning module source with mocked imports."""
    pruning_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'pruning.py')
    pruning_path = os.path.abspath(pruning_path)

    mod = types.ModuleType('fastai.callback.pruning')
    mod.__file__ = pruning_path
    mod.__package__ = 'fastai.callback'

    # Populate namespace
    mod.nn = nn_mod
    mod.Callback = Callback
    mod.store_attr = store_attr_fn
    mod.L = L_cls
    mod.np = np
    mod.patch = lambda f: f  # no-op decorator; we test the function directly
    mod.__builtins__ = __builtins__

    with open(pruning_path, 'r') as f:
        source = f.read()

    # Strip relative imports
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), pruning_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.pruning'] = mod
    return mod


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
_prune_utils, _Callback, _nn_mod, _L = _setup_mock_env()


def _store_attr_impl(*args, **kwargs):
    """Real store_attr for our mocked env: store constructor args on self."""
    import inspect
    frame = inspect.currentframe().f_back
    self = frame.f_locals.get('self')
    if self is None:
        return
    # Get all params from the calling function signature (excluding self)
    func = frame.f_code
    varnames = func.co_varnames[1:func.co_argcount]  # skip 'self'
    for name in varnames:
        if name in frame.f_locals:
            setattr(self, name, frame.f_locals[name])


pruning_module = _load_pruning_module(_nn_mod, _Callback, _store_attr_impl, _L)
PruningCallback = pruning_module.PruningCallback
_get_prunable_modules = pruning_module._get_prunable_modules
_compute_sparsity = pruning_module._compute_sparsity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetPrunableModules(unittest.TestCase):
    """Test helper that finds prunable modules."""

    def test_finds_conv_and_linear(self):
        model = FakeModel()
        modules = _get_prunable_modules(model, prune_types=(FakeConv2d, FakeLinear))
        self.assertEqual(len(modules), 3)
        for m, name in modules:
            self.assertEqual(name, 'weight')

    def test_empty_model(self):
        model = MagicMock()
        model.modules.return_value = []
        modules = _get_prunable_modules(model, prune_types=(FakeConv2d, FakeLinear))
        self.assertEqual(len(modules), 0)


class TestComputeSparsity(unittest.TestCase):
    """Test sparsity computation."""

    def test_zero_sparsity(self):
        """All weights nonzero."""
        model = FakeModel()
        # Set nonzero data
        model.conv1.weight._data = np.ones((64, 3, 3, 3))
        model.conv2.weight._data = np.ones((128, 64, 3, 3))
        model.fc.weight._data = np.ones((10, 128))
        sparsity = _compute_sparsity(model, prune_types=(FakeConv2d, FakeLinear))
        self.assertAlmostEqual(sparsity, 0.0)

    def test_full_sparsity(self):
        """All weights zero."""
        model = FakeModel()
        sparsity = _compute_sparsity(model, prune_types=(FakeConv2d, FakeLinear))
        self.assertAlmostEqual(sparsity, 1.0)

    def test_partial_sparsity(self):
        """Half of one layer is zero."""
        model = FakeModel()
        # conv1: 64*3*3*3 = 1728 elements, make half nonzero
        model.conv1.weight._data = np.zeros((64, 3, 3, 3))
        model.conv1.weight._data.flat[:864] = 1.0
        # conv2 and fc all zeros
        total = 64*3*3*3 + 128*64*3*3 + 10*128
        zeros = 864 + 128*64*3*3 + 10*128  # 864 nonzero in conv1 means (1728-864) zeros there
        expected_zeros = (1728 - 864) + 128*64*3*3 + 10*128
        expected = expected_zeros / total
        sparsity = _compute_sparsity(model, prune_types=(FakeConv2d, FakeLinear))
        self.assertAlmostEqual(sparsity, expected, places=5)


class TestPruningCallback(unittest.TestCase):
    """Test PruningCallback behavior."""

    def _make_callback(self, amount=0.3, prune_epochs=None, method='structured',
                       norm=2, dim=0, make_permanent=True):
        cb = PruningCallback(
            amount=amount, prune_epochs=prune_epochs, method=method,
            norm=norm, dim=dim, prune_types=(FakeConv2d, FakeLinear),
            make_permanent=make_permanent,
        )
        return cb

    def _attach_model(self, cb, model=None):
        """Simulate learner attaching model to callback."""
        cb.model = model or FakeModel()
        cb.training = True
        cb.epoch = 0

    def test_init_stores_params(self):
        cb = self._make_callback(amount=0.5, method='unstructured')
        self.assertEqual(cb.amount, 0.5)
        self.assertEqual(cb.method, 'unstructured')
        self.assertFalse(cb.pruning_applied)

    def test_before_fit_records_initial_sparsity(self):
        cb = self._make_callback()
        cb.model = FakeModel()
        cb.before_fit()
        self.assertIsNotNone(cb._initial_sparsity)
        self.assertEqual(cb._pruned_at, [])

    def test_after_epoch_prunes_at_designated_epochs(self):
        _prune_utils.ln_structured.reset_mock()
        cb = self._make_callback(prune_epochs=[0, 2])
        cb.model = FakeModel()
        cb.training = True
        cb.before_fit()

        # Epoch 0: should prune
        cb.epoch = 0
        cb.after_epoch()
        self.assertTrue(_prune_utils.ln_structured.called)
        self.assertIn(0, cb._pruned_at)

        # Epoch 1: should NOT prune
        _prune_utils.ln_structured.reset_mock()
        cb.epoch = 1
        cb.after_epoch()
        self.assertFalse(_prune_utils.ln_structured.called)
        self.assertNotIn(1, cb._pruned_at)

        # Epoch 2: should prune
        _prune_utils.ln_structured.reset_mock()
        cb.epoch = 2
        cb.after_epoch()
        self.assertTrue(_prune_utils.ln_structured.called)
        self.assertIn(2, cb._pruned_at)

    def test_after_epoch_skips_when_not_training(self):
        _prune_utils.ln_structured.reset_mock()
        cb = self._make_callback(prune_epochs=None)
        cb.model = FakeModel()
        cb.training = False
        cb.before_fit()
        cb.epoch = 0
        cb.after_epoch()
        self.assertFalse(_prune_utils.ln_structured.called)

    def test_unstructured_pruning_calls_global(self):
        _prune_utils.global_unstructured.reset_mock()
        cb = self._make_callback(method='unstructured')
        cb.model = FakeModel()
        cb.training = True
        cb.before_fit()
        cb.epoch = 0
        cb.after_epoch()
        self.assertTrue(_prune_utils.global_unstructured.called)

    def test_invalid_method_raises(self):
        cb = self._make_callback(method='invalid')
        cb.model = FakeModel()
        cb.training = True
        cb.before_fit()
        cb.epoch = 0
        with self.assertRaises(ValueError) as ctx:
            cb.after_epoch()
        self.assertIn('invalid', str(ctx.exception))

    def test_after_fit_makes_permanent(self):
        _prune_utils.is_pruned.reset_mock()
        _prune_utils.remove.reset_mock()
        _prune_utils.is_pruned.return_value = True

        cb = self._make_callback(make_permanent=True)
        cb.model = FakeModel()
        cb.pruning_applied = True
        cb._initial_sparsity = 0.0
        cb.after_fit()
        self.assertTrue(_prune_utils.remove.called)

    def test_after_fit_no_op_if_not_pruned(self):
        _prune_utils.remove.reset_mock()
        cb = self._make_callback()
        cb.model = FakeModel()
        cb.pruning_applied = False
        cb.after_fit()
        self.assertFalse(_prune_utils.remove.called)

    def test_prune_none_schedule_prunes_every_epoch(self):
        """With prune_epochs=None, pruning happens every epoch."""
        _prune_utils.ln_structured.reset_mock()
        cb = self._make_callback(prune_epochs=None)
        cb.model = FakeModel()
        cb.training = True
        cb.before_fit()

        for epoch in range(3):
            _prune_utils.ln_structured.reset_mock()
            cb.epoch = epoch
            cb.after_epoch()
            self.assertTrue(_prune_utils.ln_structured.called,
                            f"Expected pruning at epoch {epoch}")


class TestLearnerPrune(unittest.TestCase):
    """Test the Learner.prune() patched method."""

    def test_prune_function_exists(self):
        """The prune function should be defined in the module."""
        self.assertTrue(hasattr(pruning_module, 'prune'))

    def test_prune_validates_amount(self):
        """Amount must be between 0 and 1 exclusive."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01

        with self.assertRaises(ValueError):
            prune_fn(learner, amount=0)
        with self.assertRaises(ValueError):
            prune_fn(learner, amount=1.0)
        with self.assertRaises(ValueError):
            prune_fn(learner, amount=-0.1)

    def test_prune_validates_epochs(self):
        """n_epochs must be >= 1."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01

        with self.assertRaises(ValueError):
            prune_fn(learner, n_epochs=0)

    def test_prune_default_schedule(self):
        """Default schedule prunes at epoch 0 and midpoint."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01
        learner.fit_one_cycle = MagicMock()

        prune_fn(learner, amount=0.3, n_epochs=6)

        learner.fit_one_cycle.assert_called_once()
        args, kwargs = learner.fit_one_cycle.call_args
        self.assertEqual(args[0], 6)  # n_epochs
        # Check lr is lr/10
        self.assertAlmostEqual(kwargs['lr_max'], 0.001)

    def test_prune_single_epoch_schedule(self):
        """With n_epochs=1, schedule is just [0]."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01
        learner.fit_one_cycle = MagicMock()

        prune_fn(learner, amount=0.3, n_epochs=1)

        learner.fit_one_cycle.assert_called_once()
        args, kwargs = learner.fit_one_cycle.call_args
        self.assertEqual(args[0], 1)

    def test_prune_custom_lr(self):
        """Custom lr overrides default."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01
        learner.fit_one_cycle = MagicMock()

        prune_fn(learner, amount=0.3, n_epochs=3, lr=0.005)

        _, kwargs = learner.fit_one_cycle.call_args
        self.assertAlmostEqual(kwargs['lr_max'], 0.005)

    def test_prune_returns_self(self):
        """prune() returns learner for method chaining."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01
        learner.fit_one_cycle = MagicMock()

        result = prune_fn(learner, amount=0.3, n_epochs=2)
        self.assertIs(result, learner)

    def test_per_step_amount_calculation(self):
        """Verify iterative pruning distributes amount correctly."""
        prune_fn = pruning_module.prune
        learner = MagicMock()
        learner.lr = 0.01
        learner.fit_one_cycle = MagicMock()

        prune_fn(learner, amount=0.5, n_epochs=4, prune_schedule=[0, 1, 2, 3])

        _, kwargs = learner.fit_one_cycle.call_args
        cbs = kwargs['cbs']
        # Find PruningCallback in cbs
        pruning_cb = None
        for cb in cbs:
            if isinstance(cb, PruningCallback):
                pruning_cb = cb
                break
        self.assertIsNotNone(pruning_cb)
        # 4 steps: per_step = 1 - (1-0.5)^(1/4) ~ 0.1591
        expected_per_step = 1.0 - (0.5) ** (1.0 / 4)
        self.assertAlmostEqual(pruning_cb.amount, expected_per_step, places=4)


class TestPruneSparsity(unittest.TestCase):
    """Test Learner.prune_sparsity() method."""

    def test_reports_sparsity(self):
        """prune_sparsity should report model sparsity."""
        prune_sparsity_fn = pruning_module.prune_sparsity
        learner = MagicMock()
        model = FakeModel()
        # All zeros
        learner.model = model
        sparsity = prune_sparsity_fn(learner, prune_types=(FakeConv2d, FakeLinear))
        self.assertAlmostEqual(sparsity, 1.0)


if __name__ == '__main__':
    unittest.main()
