"""Tests for fastai.learner module.

Covers: replacing_yield, mk_metric, _ConstantFunc, Metric, AvgMetric, AvgLoss,
AvgSmoothLoss, ValueMetric, SkipToEpoch, save_model, load_model, Learner
(callback management, fit loop basics), Recorder, CastToTensor, and the
various Cancel*Exception classes.
"""
import sys
import os
import math
import tempfile
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.learner import (
    replacing_yield,
    mk_metric,
    save_model,
    load_model,
    Metric,
    AvgMetric,
    AvgLoss,
    AvgSmoothLoss,
    ValueMetric,
    SkipToEpoch,
    Learner,
    Recorder,
    CastToTensor,
    CancelFitException,
    CancelEpochException,
    CancelTrainException,
    CancelValidException,
    CancelBatchException,
    CancelBackwardException,
    CancelStepException,
    _ConstantFunc,
)
from fastai.callback.core import Callback, TrainEvalCallback, event
from fastcore.foundation import L


# ============================================================
# Helper utilities for tests
# ============================================================

class SimpleDL:
    """A minimal DataLoader-like object for testing."""
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def new(self, **kwargs):
        return self


class SimpleDLS:
    """A minimal DataLoaders-like object for testing."""
    def __init__(self, train_data=None, valid_data=None):
        self.device = torch.device('cpu')
        self.path = '.'
        self.n_inp = 1
        self._train = SimpleDL(train_data or [(torch.randn(2, 3), torch.tensor([0, 1]))])
        self._valid = SimpleDL(valid_data or [(torch.randn(2, 3), torch.tensor([0, 1]))])

    def __getitem__(self, i):
        return self._valid if i == 1 else self._train

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def train_ds(self):
        return SimpleNamespace(loss_func=nn.CrossEntropyLoss())


def make_learner(default_cbs=False, **kwargs):
    """Create a minimal Learner for testing."""
    model = nn.Linear(3, 2)
    dls = SimpleDLS()
    return Learner(dls, model, loss_func=nn.CrossEntropyLoss(), default_cbs=default_cbs, **kwargs)


# ============================================================
# Tests for replacing_yield
# ============================================================

class TestReplacingYield:
    """Tests for the replacing_yield context manager."""

    def test_replaces_attribute(self):
        obj = SimpleNamespace(x=10)
        gen = replacing_yield(obj, 'x', 42)
        next(gen)
        assert obj.x == 42

    def test_restores_attribute(self):
        obj = SimpleNamespace(x=10)
        gen = replacing_yield(obj, 'x', 42)
        next(gen)
        try:
            gen.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        assert obj.x == 10

    def test_used_as_context_manager(self):
        """Test replacing_yield works via contextmanager decorator pattern."""
        obj = SimpleNamespace(value='original')

        @contextmanager
        def replace_value(o, val):
            return replacing_yield(o, 'value', val)

        with replace_value(obj, 'replaced'):
            assert obj.value == 'replaced'
        assert obj.value == 'original'

    def test_restores_on_exception(self):
        obj = SimpleNamespace(x='before')

        @contextmanager
        def replace_x(o, val):
            return replacing_yield(o, 'x', val)

        with pytest.raises(RuntimeError):
            with replace_x(obj, 'during'):
                assert obj.x == 'during'
                raise RuntimeError("test error")
        assert obj.x == 'before'


# ============================================================
# Tests for _ConstantFunc
# ============================================================

class TestConstantFunc:
    """Tests for the _ConstantFunc helper class."""

    def test_returns_stored_value(self):
        f = _ConstantFunc(42)
        assert f() == 42

    def test_ignores_args(self):
        f = _ConstantFunc('hello')
        assert f(1, 2, 3) == 'hello'

    def test_ignores_kwargs(self):
        f = _ConstantFunc([1, 2, 3])
        assert f(key='value') == [1, 2, 3]

    def test_returns_none(self):
        f = _ConstantFunc(None)
        assert f() is None

    def test_returns_tensor(self):
        t = torch.tensor([1.0, 2.0])
        f = _ConstantFunc(t)
        assert torch.equal(f(), t)


# ============================================================
# Tests for mk_metric
# ============================================================

class TestMkMetric:
    """Tests for the mk_metric conversion function."""

    def test_wraps_function_in_avg_metric(self):
        def my_func(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        result = mk_metric(my_func)
        assert isinstance(result, AvgMetric)

    def test_passes_through_metric_instance(self):
        metric = AvgLoss()
        result = mk_metric(metric)
        assert result is metric

    def test_instantiates_metric_class(self):
        result = mk_metric(AvgLoss)
        assert isinstance(result, AvgLoss)

    def test_passes_through_value_metric(self):
        vm = ValueMetric(lambda: 0.5)
        result = mk_metric(vm)
        assert result is vm


# ============================================================
# Tests for Metric base class
# ============================================================

class TestMetric:
    """Tests for the Metric base class."""

    def test_value_raises_not_implemented(self):
        m = Metric()
        with pytest.raises(NotImplementedError):
            _ = m.value

    def test_reset_does_nothing(self):
        m = Metric()
        m.reset()  # should not raise

    def test_accumulate_does_nothing(self):
        m = Metric()
        m.accumulate(None)  # should not raise

    def test_name_property(self):
        m = Metric()
        assert m.name == 'metric'


# ============================================================
# Tests for AvgMetric
# ============================================================

class TestAvgMetric:
    """Tests for the AvgMetric class."""

    def _mock_learn(self, pred, targ):
        return SimpleNamespace(
            pred=pred,
            yb=(targ,),
            to_detach=lambda x, *a, **kw: x
        )

    def test_perfect_accuracy(self):
        def accuracy(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        metric = AvgMetric(accuracy)
        metric.reset()
        learn = self._mock_learn(
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([1, 0])
        )
        metric.accumulate(learn)
        assert float(metric.value) == 1.0

    def test_zero_accuracy(self):
        def accuracy(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        metric = AvgMetric(accuracy)
        metric.reset()
        learn = self._mock_learn(
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([0, 1])  # all wrong
        )
        metric.accumulate(learn)
        assert float(metric.value) == 0.0

    def test_multiple_accumulations(self):
        def accuracy(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        metric = AvgMetric(accuracy)
        metric.reset()

        # First batch: 2 correct out of 2
        learn1 = self._mock_learn(
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([1, 0])
        )
        metric.accumulate(learn1)

        # Second batch: 0 correct out of 2
        learn2 = self._mock_learn(
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([0, 1])
        )
        metric.accumulate(learn2)

        # Overall: 2 correct out of 4
        assert abs(float(metric.value) - 0.5) < 1e-5

    def test_reset_clears_state(self):
        def accuracy(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        metric = AvgMetric(accuracy)
        metric.reset()
        learn = self._mock_learn(
            torch.tensor([[0.1, 0.9]]),
            torch.tensor([1])
        )
        metric.accumulate(learn)
        metric.reset()
        assert metric.total == 0.0
        assert metric.count == 0

    def test_value_none_when_empty(self):
        def accuracy(pred, targ):
            return (pred.argmax(dim=1) == targ).float().mean()

        metric = AvgMetric(accuracy)
        metric.reset()
        assert metric.value is None

    def test_name_from_function(self):
        def my_custom_metric(pred, targ):
            return torch.tensor(0.0)

        metric = AvgMetric(my_custom_metric)
        assert metric.name == 'my_custom_metric'

    def test_name_from_partial(self):
        from functools import partial

        def base_metric(pred, targ, k=5):
            return torch.tensor(0.0)

        metric = AvgMetric(partial(base_metric, k=3))
        assert metric.name == 'base_metric'


# ============================================================
# Tests for AvgLoss
# ============================================================

class TestAvgLoss:
    """Tests for the AvgLoss class."""

    def _mock_learn(self, loss_val, bs):
        return SimpleNamespace(
            loss=torch.tensor(loss_val),
            yb=(torch.zeros(bs),),
            to_detach=lambda x, *a, **kw: x
        )

    def test_single_batch(self):
        metric = AvgLoss()
        metric.reset()
        learn = self._mock_learn(0.5, bs=4)
        metric.accumulate(learn)
        assert abs(float(metric.value) - 0.5) < 1e-5

    def test_multiple_batches_same_size(self):
        metric = AvgLoss()
        metric.reset()
        metric.accumulate(self._mock_learn(1.0, bs=4))
        metric.accumulate(self._mock_learn(2.0, bs=4))
        # Average: (1.0*4 + 2.0*4) / 8 = 1.5
        assert abs(float(metric.value) - 1.5) < 1e-5

    def test_multiple_batches_different_sizes(self):
        metric = AvgLoss()
        metric.reset()
        metric.accumulate(self._mock_learn(1.0, bs=2))
        metric.accumulate(self._mock_learn(3.0, bs=6))
        # Average: (1.0*2 + 3.0*6) / 8 = 20/8 = 2.5
        assert abs(float(metric.value) - 2.5) < 1e-5

    def test_name_is_loss(self):
        metric = AvgLoss()
        assert metric.name == 'loss'

    def test_reset(self):
        metric = AvgLoss()
        metric.reset()
        metric.accumulate(self._mock_learn(1.0, bs=4))
        metric.reset()
        assert metric.total == 0.0
        assert metric.count == 0

    def test_value_none_when_empty(self):
        metric = AvgLoss()
        metric.reset()
        assert metric.value is None


# ============================================================
# Tests for AvgSmoothLoss
# ============================================================

class TestAvgSmoothLoss:
    """Tests for the AvgSmoothLoss class."""

    def _mock_learn(self, loss_val):
        return SimpleNamespace(
            loss=torch.tensor(loss_val)
        )

    def test_single_step(self):
        metric = AvgSmoothLoss(beta=0.98)
        metric.reset()
        metric.accumulate(self._mock_learn(1.0))
        # value = val / (1 - beta^count) = lerp(1.0, 0., 0.98) / (1 - 0.98)
        # lerp(loss, val, beta) = loss + beta * (val - loss) = 1.0 + 0.98*(0-1) = 0.02
        # value = 0.02 / (1 - 0.98) = 0.02/0.02 = 1.0
        assert abs(float(metric.value) - 1.0) < 1e-4

    def test_smooth_converges(self):
        """Smooth loss should converge toward the constant loss value."""
        metric = AvgSmoothLoss(beta=0.9)
        metric.reset()
        for _ in range(100):
            metric.accumulate(self._mock_learn(2.0))
        # Should converge close to 2.0
        assert abs(float(metric.value) - 2.0) < 0.1

    def test_reset_clears(self):
        metric = AvgSmoothLoss(beta=0.98)
        metric.reset()
        metric.accumulate(self._mock_learn(5.0))
        metric.reset()
        assert metric.count == 0
        assert float(metric.val) == 0.0

    def test_custom_beta(self):
        metric = AvgSmoothLoss(beta=0.5)
        metric.reset()
        metric.accumulate(self._mock_learn(4.0))
        # lerp(4.0, 0., 0.5) = 4.0 + 0.5*(0-4.0) = 2.0
        # value = 2.0 / (1 - 0.5^1) = 2.0 / 0.5 = 4.0
        assert abs(float(metric.value) - 4.0) < 1e-4


# ============================================================
# Tests for ValueMetric
# ============================================================

class TestValueMetric:
    """Tests for the ValueMetric class."""

    def test_returns_function_value(self):
        metric = ValueMetric(lambda: 0.95)
        assert metric.value == 0.95

    def test_name_from_function(self):
        def my_metric():
            return 0.5

        metric = ValueMetric(my_metric)
        assert metric.name == 'my_metric'

    def test_custom_name(self):
        metric = ValueMetric(lambda: 0.5, metric_name='custom_score')
        assert metric.name == 'custom_score'

    def test_dynamic_value(self):
        counter = [0]

        def counting_metric():
            counter[0] += 1
            return counter[0]

        metric = ValueMetric(counting_metric)
        assert metric.value == 1
        assert metric.value == 2
        assert metric.value == 3


# ============================================================
# Tests for SkipToEpoch
# ============================================================

class TestSkipToEpoch:
    """Tests for the SkipToEpoch callback."""

    def test_order(self):
        cb = SkipToEpoch(5)
        assert cb.order == 70

    def test_stores_skip_target(self):
        cb = SkipToEpoch(3)
        assert cb._skip_to == 3

    def test_raises_cancel_epoch_before_target(self):
        cb = SkipToEpoch(3)
        cb.learn = SimpleNamespace(epoch=1)
        # The callback accesses self.epoch through GetAttr delegation to learn
        # Let's simulate it directly
        cb.epoch = 1
        with pytest.raises(CancelEpochException):
            cb.before_epoch()

    def test_does_not_raise_at_target(self):
        cb = SkipToEpoch(3)
        cb.epoch = 3
        # Should not raise
        cb.before_epoch()

    def test_does_not_raise_after_target(self):
        cb = SkipToEpoch(3)
        cb.epoch = 5
        # Should not raise
        cb.before_epoch()


# ============================================================
# Tests for save_model and load_model
# ============================================================

class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""

    def test_save_and_load_without_optimizer(self):
        model = nn.Linear(4, 2)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fname = f.name
        try:
            save_model(fname, model, opt=None, with_opt=False)
            model2 = nn.Linear(4, 2)
            load_model(fname, model2, opt=None, with_opt=False)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(fname)

    def test_save_and_load_with_optimizer(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        # Run a step to populate optimizer state
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fname = f.name
        try:
            save_model(fname, model, opt, with_opt=True)

            model2 = nn.Linear(4, 2)
            opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            # Need a step first so state dict keys exist
            loss2 = model2(torch.randn(2, 4)).sum()
            loss2.backward()
            opt2.step()

            load_model(fname, model2, opt2, with_opt=True)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(fname)

    def test_save_without_opt_when_opt_is_none(self):
        model = nn.Linear(4, 2)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fname = f.name
        try:
            # with_opt=True but opt=None should still work (with_opt forced to False)
            save_model(fname, model, opt=None, with_opt=True)
            model2 = nn.Linear(4, 2)
            load_model(fname, model2, opt=None, with_opt=False)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(fname)

    def test_load_to_specific_device(self):
        model = nn.Linear(4, 2)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fname = f.name
        try:
            save_model(fname, model, opt=None, with_opt=False)
            model2 = nn.Linear(4, 2)
            load_model(fname, model2, opt=None, with_opt=False, device='cpu')
            for p in model2.parameters():
                assert p.device == torch.device('cpu')
        finally:
            os.unlink(fname)

    def test_pickle_protocol(self):
        model = nn.Linear(4, 2)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fname = f.name
        try:
            save_model(fname, model, opt=None, with_opt=False, pickle_protocol=2)
            model2 = nn.Linear(4, 2)
            load_model(fname, model2, opt=None, with_opt=False)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(fname)


# ============================================================
# Tests for Learner callback management
# ============================================================

class TestLearnerCallbacks:
    """Tests for Learner callback add/remove operations."""

    def test_add_cb(self):
        learn = make_learner()
        class MyCB(Callback):
            order = 5
        cb = MyCB()
        learn.add_cb(cb)
        assert cb in learn.cbs
        assert hasattr(learn, 'my_cb')
        assert cb.learn is learn

    def test_add_cb_from_class(self):
        learn = make_learner()
        class MyCB(Callback):
            order = 5
        learn.add_cb(MyCB)
        assert len(learn.cbs) == 1
        assert isinstance(learn.cbs[0], MyCB)

    def test_remove_cb(self):
        learn = make_learner()
        class MyCB(Callback):
            order = 5
        cb = MyCB()
        learn.add_cb(cb)
        learn.remove_cb(cb)
        assert cb not in learn.cbs
        assert not hasattr(learn, 'my_cb')
        assert cb.learn is None

    def test_remove_cb_by_class(self):
        learn = make_learner()
        class MyCB(Callback):
            order = 5
        learn.add_cb(MyCB())
        learn.remove_cb(MyCB)
        assert len(learn.cbs) == 0

    def test_add_cbs(self):
        learn = make_learner()
        class CB1(Callback): order = 1
        class CB2(Callback): order = 2
        learn.add_cbs([CB1(), CB2()])
        assert len(learn.cbs) == 2

    def test_remove_cbs(self):
        learn = make_learner()
        class CB1(Callback): order = 1
        class CB2(Callback): order = 2
        cb1, cb2 = CB1(), CB2()
        learn.add_cbs([cb1, cb2])
        learn.remove_cbs([cb1, cb2])
        assert len(learn.cbs) == 0

    def test_added_cbs_context_manager(self):
        learn = make_learner()
        class TempCB(Callback): order = 1
        cb = TempCB()
        with learn.added_cbs([cb]):
            assert cb in learn.cbs
        assert cb not in learn.cbs

    def test_removed_cbs_context_manager(self):
        learn = make_learner()
        class PermanentCB(Callback): order = 1
        cb = PermanentCB()
        learn.add_cb(cb)
        with learn.removed_cbs([cb]):
            assert cb not in learn.cbs
        assert cb in learn.cbs

    def test_ordered_cbs(self):
        learn = make_learner()
        class CB1(Callback):
            order = 10
            def before_fit(self): pass
        class CB2(Callback):
            order = 1
            def before_fit(self): pass
        learn.add_cbs([CB1(), CB2()])
        ordered = learn.ordered_cbs('before_fit')
        assert len(ordered) == 2
        # Lower order should come first
        assert ordered[0].order < ordered[1].order


# ============================================================
# Tests for Learner initialization
# ============================================================

class TestLearnerInit:
    """Tests for Learner initialization."""

    def test_default_attributes(self):
        learn = make_learner(default_cbs=False)
        assert learn.training is False
        assert learn.create_mbar is True
        assert learn.opt is None

    def test_loss_func_from_arg(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        loss = nn.MSELoss()
        learn = Learner(dls, model, loss_func=loss, default_cbs=False)
        assert learn.loss_func is loss

    def test_model_stored(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), default_cbs=False)
        assert learn.model is model

    def test_dls_stored(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), default_cbs=False)
        assert learn.dls is dls

    def test_default_cbs_adds_callbacks(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), default_cbs=True)
        # Should have TrainEvalCallback and Recorder (and CastToTensor) as default cbs
        cb_types = [type(cb) for cb in learn.cbs]
        assert TrainEvalCallback in cb_types

    def test_metrics_converted(self):
        def my_acc(pred, targ):
            return torch.tensor(1.0)

        learn = make_learner(metrics=[my_acc])
        assert len(learn.metrics) == 1
        assert isinstance(learn.metrics[0], AvgMetric)

    def test_path_default(self):
        learn = make_learner()
        # path comes from dls.path which is '.'
        from pathlib import Path
        assert str(learn.path) == '.'

    def test_custom_lr(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(),
                        lr=0.05, default_cbs=False)
        assert learn.lr == 0.05


# ============================================================
# Tests for Learner._split
# ============================================================

class TestLearnerSplit:
    """Tests for the Learner._split method."""

    def test_single_input(self):
        learn = make_learner()
        learn.dls.n_inp = 1
        batch = (torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        learn._split(batch)
        assert len(learn.xb) == 1
        assert torch.equal(learn.xb[0], batch[0])
        assert len(learn.yb) == 1
        assert torch.equal(learn.yb[0], batch[1])

    def test_multiple_inputs(self):
        learn = make_learner()
        learn.dls.n_inp = 2
        batch = (torch.randn(4, 3), torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        learn._split(batch)
        assert len(learn.xb) == 2
        assert len(learn.yb) == 1


# ============================================================
# Tests for Learner._call_one / __call__
# ============================================================

class TestLearnerEvents:
    """Tests for Learner event calling."""

    def test_call_triggers_callback(self):
        learn = make_learner()
        called = []

        class TrackerCB(Callback):
            def before_fit(self): called.append('before_fit')

        learn.add_cb(TrackerCB())
        learn('before_fit')
        assert called == ['before_fit']

    def test_call_multiple_events(self):
        learn = make_learner()
        called = []

        class TrackerCB(Callback):
            def before_fit(self): called.append('before_fit')
            def after_fit(self): called.append('after_fit')

        learn.add_cb(TrackerCB())
        learn(['before_fit', 'after_fit'])
        assert called == ['before_fit', 'after_fit']

    def test_call_unknown_event_raises(self):
        learn = make_learner()
        with pytest.raises(Exception, match='missing'):
            learn('nonexistent_event')

    def test_callbacks_called_in_order(self):
        learn = make_learner()
        called = []

        class FirstCB(Callback):
            order = 1
            def before_fit(self): called.append('first')

        class SecondCB(Callback):
            order = 10
            def before_fit(self): called.append('second')

        learn.add_cbs([SecondCB(), FirstCB()])
        learn('before_fit')
        assert called == ['first', 'second']


# ============================================================
# Tests for Cancel*Exception classes
# ============================================================

class TestCancelExceptions:
    """Tests for various Cancel exception classes."""

    def test_cancel_fit_is_exception(self):
        assert issubclass(CancelFitException, Exception)

    def test_cancel_epoch_is_exception(self):
        assert issubclass(CancelEpochException, Exception)

    def test_cancel_train_is_exception(self):
        assert issubclass(CancelTrainException, Exception)

    def test_cancel_valid_is_exception(self):
        assert issubclass(CancelValidException, Exception)

    def test_cancel_batch_is_exception(self):
        assert issubclass(CancelBatchException, Exception)

    def test_cancel_backward_is_exception(self):
        assert issubclass(CancelBackwardException, Exception)

    def test_cancel_step_is_exception(self):
        assert issubclass(CancelStepException, Exception)

    def test_all_distinct(self):
        exceptions = [
            CancelFitException, CancelEpochException,
            CancelTrainException, CancelValidException,
            CancelBatchException, CancelBackwardException,
            CancelStepException
        ]
        # All should be different types
        assert len(set(exceptions)) == len(exceptions)


# ============================================================
# Tests for CastToTensor
# ============================================================

class TestCastToTensor:
    """Tests for the CastToTensor callback."""

    def test_order(self):
        cb = CastToTensor()
        assert cb.order == 9

    def test_casts_tensor_subclass(self):
        """CastToTensor should cast tensor subclasses to plain Tensor."""
        from fastai.learner import _cast_tensor

        # Regular tensor stays the same type
        t = torch.tensor([1.0, 2.0])
        result = _cast_tensor(t)
        assert isinstance(result, torch.Tensor)

    def test_handles_tuple(self):
        from fastai.learner import _cast_tensor
        t1 = torch.tensor([1.0])
        t2 = torch.tensor([2.0])
        result = _cast_tensor((t1, t2))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_non_tensor_passthrough(self):
        from fastai.learner import _cast_tensor
        result = _cast_tensor("not a tensor")
        assert result == "not a tensor"


# ============================================================
# Tests for Recorder
# ============================================================

class TestRecorder:
    """Tests for the Recorder callback."""

    def test_order(self):
        rec = Recorder()
        assert rec.order == 50

    def test_remove_on_fetch(self):
        rec = Recorder()
        assert rec.remove_on_fetch is True

    def test_initialization(self):
        rec = Recorder(add_time=True, train_metrics=False, valid_metrics=True, beta=0.98)
        assert rec.add_time is True
        assert rec.train_metrics is False
        assert rec.valid_metrics is True
        assert isinstance(rec.loss, AvgLoss)
        assert isinstance(rec.smooth_loss, AvgSmoothLoss)

    def test_has_required_event_methods(self):
        rec = Recorder()
        assert hasattr(rec, 'before_fit')
        assert hasattr(rec, 'after_batch')
        assert hasattr(rec, 'before_epoch')
        assert hasattr(rec, 'after_epoch')
        assert hasattr(rec, 'before_train')
        assert hasattr(rec, 'before_validate')
        assert hasattr(rec, 'after_train')
        assert hasattr(rec, 'after_validate')


# ============================================================
# Tests for Learner.create_opt
# ============================================================

class TestLearnerCreateOpt:
    """Tests for Learner.create_opt method."""

    def test_creates_optimizer(self):
        learn = make_learner(default_cbs=True)
        learn.create_opt()
        assert learn.opt is not None

    def test_uses_default_lr(self):
        model = nn.Linear(3, 2)
        dls = SimpleDLS()
        learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(),
                        lr=0.05, default_cbs=True)
        learn.create_opt()
        # Check hyper-parameter lr
        assert learn.opt.hypers[0]['lr'] == 0.05


# ============================================================
# Tests for Learner serialization (pickling)
# ============================================================

class TestLearnerSerialization:
    """Tests for Learner __getstate__ and __setstate__."""

    def test_getstate_excludes_lock(self):
        learn = make_learner()
        state = learn.__getstate__()
        assert 'lock' not in state

    def test_setstate_creates_lock(self):
        learn = make_learner()
        state = learn.__getstate__()
        learn2 = make_learner()
        learn2.__setstate__(state)
        import threading
        assert hasattr(learn2, 'lock')
        assert type(learn2.lock).__name__ == 'lock'


# ============================================================
# Tests for Learner._end_cleanup
# ============================================================

class TestLearnerEndCleanup:
    """Tests for the _end_cleanup method."""

    def test_clears_attributes(self):
        learn = make_learner()
        learn.dl = "something"
        learn.xb = (torch.tensor([1.0]),)
        learn.yb = (torch.tensor([0]),)
        learn.pred = torch.tensor([0.5])
        learn.loss = torch.tensor(1.0)

        learn._end_cleanup()

        assert learn.dl is None
        assert learn.xb == (None,)
        assert learn.yb == (None,)
        assert learn.pred is None
        assert learn.loss is None


# ============================================================
# Tests for Learner no_logging / no_mbar context managers
# ============================================================

class TestLearnerContextManagers:
    """Tests for Learner context managers."""

    def test_no_logging(self):
        learn = make_learner()
        original_logger = learn.logger
        with learn.no_logging():
            # logger should be noop during this context
            assert learn.logger is not original_logger
        assert learn.logger is original_logger

    def test_no_mbar(self):
        learn = make_learner()
        assert learn.create_mbar is True
        with learn.no_mbar():
            assert learn.create_mbar is False
        assert learn.create_mbar is True
