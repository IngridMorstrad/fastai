"""Tests for fastai.optimizer module.

Covers: Optimizer base class, SGD, Adam, RMSProp, utility functions
(sgd_step, weight_decay, l2_reg, average_grad, average_sqr_grad,
momentum_step, debias, step_stat), freeze/unfreeze, state management.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.optimizer import (
    Optimizer, SGD, Adam, RMSProp, RAdam, QHAdam, Larc, Lamb, Lookahead, ranger,
    sgd_step, weight_decay, l2_reg, average_grad, average_sqr_grad,
    momentum_step, step_stat, debias, rms_prop_step, adam_step,
    OptimWrapper, detuplify_pg, set_item_pg,
)


# ============================================================
# Helper utilities
# ============================================================

def _make_linear_model(in_features=4, out_features=2):
    """Create a simple linear model for optimizer testing."""
    model = nn.Linear(in_features, out_features, bias=True)
    return model


def _set_grad(model, value=1.0):
    """Set all parameter gradients to a constant value."""
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


def _get_params_with_grad(model, grad_value=1.0):
    """Get model params with synthetic gradients set."""
    params = list(model.parameters())
    for p in params:
        p.grad = torch.full_like(p, grad_value)
    return params


# ============================================================
# Tests for utility functions
# ============================================================

class TestDebias:
    """Tests for the debias utility function."""

    def test_debias_step_1(self):
        # debias(mom, damp, step) = damp * (1 - mom**step) / (1 - mom)
        result = debias(0.9, 0.1, 1)
        expected = 0.1 * (1 - 0.9**1) / (1 - 0.9)
        assert abs(result - expected) < 1e-7

    def test_debias_step_10(self):
        result = debias(0.9, 0.1, 10)
        expected = 0.1 * (1 - 0.9**10) / (1 - 0.9)
        assert abs(result - expected) < 1e-7

    def test_debias_converges_to_one(self):
        # As step -> infinity, debias(mom, 1-mom, step) -> 1
        result = debias(0.9, 0.1, 10000)
        assert abs(result - 1.0) < 1e-5

    def test_debias_different_momentum(self):
        result = debias(0.99, 0.01, 1)
        expected = 0.01 * (1 - 0.99) / (1 - 0.99)
        assert abs(result - expected) < 1e-7


class TestStepStat:
    """Tests for step_stat utility."""

    def test_step_stat_initial(self):
        p = torch.randn(3, 4)
        p.grad = torch.randn(3, 4)
        result = step_stat(p, step=0)
        assert result == {'step': 1}

    def test_step_stat_increments(self):
        p = torch.randn(3, 4)
        p.grad = torch.randn(3, 4)
        result = step_stat(p, step=5)
        assert result == {'step': 6}

    def test_step_stat_multiple_calls(self):
        p = torch.randn(3, 4)
        p.grad = torch.randn(3, 4)
        state = step_stat(p, step=0)
        state = step_stat(p, **state)
        assert state == {'step': 2}


class TestWeightDecay:
    """Tests for the weight_decay utility function."""

    def test_weight_decay_basic(self):
        p = torch.ones(3, 4)
        p.grad = torch.zeros(3, 4)
        lr, wd = 0.1, 0.1
        weight_decay(p, lr=lr, wd=wd, do_wd=True)
        # p.data should be multiplied by (1 - lr*wd) = (1 - 0.01) = 0.99
        expected = torch.full((3, 4), 0.99)
        assert torch.allclose(p.data, expected)

    def test_weight_decay_zero_wd(self):
        p = torch.ones(3, 4)
        p.grad = torch.zeros(3, 4)
        original = p.data.clone()
        weight_decay(p, lr=0.1, wd=0.0, do_wd=True)
        assert torch.equal(p.data, original)

    def test_weight_decay_disabled(self):
        p = torch.ones(3, 4)
        p.grad = torch.zeros(3, 4)
        original = p.data.clone()
        weight_decay(p, lr=0.1, wd=0.1, do_wd=False)
        assert torch.equal(p.data, original)

    def test_weight_decay_defaults(self):
        assert weight_decay.defaults == dict(wd=0.)


class TestL2Reg:
    """Tests for the l2_reg utility function."""

    def test_l2_reg_basic(self):
        p = torch.ones(3, 4)
        p.grad = torch.zeros(3, 4)
        l2_reg(p, lr=0.1, wd=0.1, do_wd=True)
        # p.grad should be p.grad + wd * p.data = 0 + 0.1 * 1 = 0.1
        expected_grad = torch.full((3, 4), 0.1)
        assert torch.allclose(p.grad.data, expected_grad)

    def test_l2_reg_zero_wd(self):
        p = torch.ones(3, 4)
        p.grad = torch.zeros(3, 4)
        l2_reg(p, lr=0.1, wd=0.0, do_wd=True)
        assert torch.all(p.grad.data == 0.0)

    def test_l2_reg_disabled(self):
        p = torch.ones(3, 4)
        p.grad = torch.ones(3, 4) * 0.5
        original_grad = p.grad.data.clone()
        l2_reg(p, lr=0.1, wd=0.1, do_wd=False)
        assert torch.equal(p.grad.data, original_grad)

    def test_l2_reg_defaults(self):
        assert l2_reg.defaults == dict(wd=0.)


class TestAverageGrad:
    """Tests for average_grad utility function."""

    def test_initial_call(self):
        p = torch.ones(2, 3)
        p.grad = torch.ones(2, 3)
        result = average_grad(p, mom=0.9, grad_avg=None)
        # grad_avg = 0 * 0.9 + 1.0 * 1 = 1.0 (dampening=False, damp=1)
        assert 'grad_avg' in result
        expected = torch.ones(2, 3)
        assert torch.allclose(result['grad_avg'], expected)

    def test_with_dampening(self):
        p = torch.ones(2, 3)
        p.grad = torch.ones(2, 3)
        result = average_grad(p, mom=0.9, dampening=True, grad_avg=None)
        # grad_avg = 0 * 0.9 + 1.0 * (1-0.9) = 0.1
        expected = torch.full((2, 3), 0.1)
        assert torch.allclose(result['grad_avg'], expected)

    def test_accumulation(self):
        p = torch.ones(2, 3)
        p.grad = torch.ones(2, 3)
        # First call
        result = average_grad(p, mom=0.9, grad_avg=None)
        # Second call
        result = average_grad(p, mom=0.9, grad_avg=result['grad_avg'])
        # grad_avg = 1.0 * 0.9 + 1.0 * 1 = 1.9
        expected = torch.full((2, 3), 1.9)
        assert torch.allclose(result['grad_avg'], expected)

    def test_defaults(self):
        assert average_grad.defaults == dict(mom=0.9)


class TestAverageSqrGrad:
    """Tests for average_sqr_grad utility function."""

    def test_initial_call(self):
        p = torch.ones(2, 3) * 2.0
        p.grad = torch.ones(2, 3) * 2.0
        result = average_sqr_grad(p, sqr_mom=0.99, sqr_avg=None)
        # sqr_avg = 0 * 0.99 + 2^2 * (1-0.99) = 4 * 0.01 = 0.04
        assert 'sqr_avg' in result
        expected = torch.full((2, 3), 0.04)
        assert torch.allclose(result['sqr_avg'], expected)

    def test_no_dampening(self):
        p = torch.ones(2, 3) * 2.0
        p.grad = torch.ones(2, 3) * 2.0
        result = average_sqr_grad(p, sqr_mom=0.99, dampening=False, sqr_avg=None)
        # sqr_avg = 0 * 0.99 + 2^2 * 1 = 4.0
        expected = torch.full((2, 3), 4.0)
        assert torch.allclose(result['sqr_avg'], expected)

    def test_defaults(self):
        assert average_sqr_grad.defaults == dict(sqr_mom=0.99)


class TestSgdStep:
    """Tests for sgd_step utility function."""

    def test_basic_step(self):
        p = torch.ones(3, 4)
        p.grad = torch.ones(3, 4)
        sgd_step(p, lr=0.1)
        # p = p - lr * grad = 1 - 0.1 * 1 = 0.9
        expected = torch.full((3, 4), 0.9)
        assert torch.allclose(p.data, expected)

    def test_zero_lr(self):
        p = torch.ones(3, 4)
        p.grad = torch.ones(3, 4)
        original = p.data.clone()
        sgd_step(p, lr=0.0)
        assert torch.equal(p.data, original)


class TestMomentumStep:
    """Tests for momentum_step utility function."""

    def test_basic_step(self):
        p = torch.ones(3, 4)
        p.grad = torch.ones(3, 4)
        grad_avg = torch.full((3, 4), 0.5)
        momentum_step(p, lr=0.1, grad_avg=grad_avg)
        # p = p - lr * grad_avg = 1 - 0.1 * 0.5 = 0.95
        expected = torch.full((3, 4), 0.95)
        assert torch.allclose(p.data, expected)


# ============================================================
# Tests for SGD optimizer
# ============================================================

class TestSGD:
    """Tests for the SGD optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1)
        original_weight = model.weight.data.clone()
        opt.step()
        # weight should decrease by lr * grad = 0.1
        expected = original_weight - 0.1
        assert torch.allclose(model.weight.data, expected)

    def test_with_momentum(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1, mom=0.9)
        original_weight = model.weight.data.clone()
        opt.step()
        # First step with momentum: grad_avg = 0*0.9 + 1.0*1 = 1.0
        # p = p - lr * grad_avg = p - 0.1 * 1.0
        expected = original_weight - 0.1
        assert torch.allclose(model.weight.data, expected)

    def test_with_momentum_two_steps(self):
        model = _make_linear_model()
        params = list(model.parameters())
        # First step
        for p in params:
            p.grad = torch.ones_like(p)
        opt = SGD(params, lr=0.1, mom=0.9)
        w0 = model.weight.data.clone()
        opt.step()
        # grad_avg after step 1: 0*0.9 + 1 = 1, weight = w0 - 0.1*1
        w1 = w0 - 0.1
        assert torch.allclose(model.weight.data, w1)

        # Second step
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
        # grad_avg after step 2: 1*0.9 + 1 = 1.9, weight = w1 - 0.1*1.9
        w2 = w1 - 0.1 * 1.9
        assert torch.allclose(model.weight.data, w2)

    def test_with_weight_decay(self):
        model = _make_linear_model()
        nn.init.ones_(model.weight)
        nn.init.ones_(model.bias)
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1, wd=0.1)
        # Weight decay is applied first: p = p * (1 - lr*wd) = 1 * 0.99 = 0.99
        # Then SGD step: p = 0.99 - 0.1 * 1 = 0.89
        opt.step()
        expected = torch.full_like(model.weight, 0.89)
        assert torch.allclose(model.weight.data, expected, atol=1e-6)

    def test_with_l2_regularization(self):
        model = _make_linear_model()
        nn.init.ones_(model.weight)
        nn.init.ones_(model.bias)
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1, wd=0.1, decouple_wd=False)
        # L2 reg modifies grad: grad = grad + wd * p = 1 + 0.1 * 1 = 1.1
        # Then SGD step: p = 1 - 0.1 * 1.1 = 0.89
        opt.step()
        expected = torch.full_like(model.weight, 0.89)
        assert torch.allclose(model.weight.data, expected, atol=1e-6)

    def test_zero_grad(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1)
        opt.zero_grad()
        for p in params:
            assert torch.all(p.grad == 0.0)


# ============================================================
# Tests for Adam optimizer
# ============================================================

class TestAdam:
    """Tests for the Adam/AdamW optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Adam(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        # After one step, weights should have changed
        assert not torch.equal(model.weight.data, original_weight)
        # Weights should decrease (grad is positive, so step is negative)
        assert torch.all(model.weight.data < original_weight)

    def test_with_weight_decay(self):
        model = _make_linear_model()
        nn.init.ones_(model.weight)
        nn.init.ones_(model.bias)
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Adam(params, lr=0.01, wd=0.1)
        original_weight = model.weight.data.clone()
        opt.step()
        # Weight decay plus Adam step means weights decrease
        assert torch.all(model.weight.data < original_weight)

    def test_multiple_steps(self):
        model = _make_linear_model()
        params = list(model.parameters())
        opt = Adam(params, lr=0.01)
        # Run multiple steps
        for _ in range(5):
            for p in params:
                p.grad = torch.ones_like(p)
            opt.step()
        # After multiple steps with constant positive gradient, weights should decrease
        # The state should have accumulated step count
        for p in params:
            state = opt.state[p]
            assert state['step'] == 5

    def test_state_tracks_averages(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Adam(params, lr=0.01)
        opt.step()
        # After one step, state should have grad_avg, sqr_avg, and step
        for p in params:
            state = opt.state[p]
            assert 'grad_avg' in state
            assert 'sqr_avg' in state
            assert 'step' in state
            assert state['step'] == 1


# ============================================================
# Tests for RMSProp optimizer
# ============================================================

class TestRMSProp:
    """Tests for the RMSProp optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = RMSProp(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        # Weights should have changed
        assert not torch.equal(model.weight.data, original_weight)

    def test_with_momentum(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = RMSProp(params, lr=0.01, mom=0.9)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)

    def test_state_tracks_sqr_avg(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = RMSProp(params, lr=0.01)
        opt.step()
        for p in params:
            state = opt.state[p]
            assert 'sqr_avg' in state


# ============================================================
# Tests for Optimizer base class
# ============================================================

class TestOptimizer:
    """Tests for the Optimizer base class."""

    def test_zero_grad(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=2.0)
        opt = Optimizer(params, [sgd_step], lr=0.1)
        opt.zero_grad()
        for p in params:
            assert torch.all(p.grad == 0.0)

    def test_state_dict_and_load(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Adam(params, lr=0.01)
        opt.step()

        # Save state dict
        sd = opt.state_dict()
        assert 'state' in sd
        assert 'hypers' in sd
        assert len(sd['state']) == len(params)

        # Create a new optimizer and load state
        model2 = _make_linear_model()
        params2 = list(model2.parameters())
        for p in params2:
            p.grad = torch.ones_like(p)
        opt2 = Adam(params2, lr=0.01)
        opt2.load_state_dict(sd)

        # Verify hypers were loaded
        assert opt2.hypers == sd['hypers']

    def test_clear_state(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Adam(params, lr=0.01)
        opt.step()

        # State should be non-empty
        for p in params:
            assert len(opt.state[p]) > 0

        # Clear state
        opt.clear_state()

        # State should be cleared (only _keep_on_clear keys remain)
        for p in params:
            for k in opt.state[p]:
                assert k in Optimizer._keep_on_clear

    def test_param_groups_property(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1)
        pg = opt.param_groups
        assert isinstance(pg, list)
        assert len(pg) == 1
        assert 'params' in pg[0]
        assert 'lr' in pg[0]


# ============================================================
# Tests for freeze/unfreeze
# ============================================================

class TestFreezeUnfreeze:
    """Tests for freeze and unfreeze functionality."""

    def test_freeze_to(self):
        # Create optimizer with multiple param groups
        model1 = nn.Linear(4, 3)
        model2 = nn.Linear(3, 2)
        params = [list(model1.parameters()), list(model2.parameters())]
        opt = SGD(params, lr=0.1)

        # Freeze first group
        opt.freeze_to(1)
        assert opt.frozen_idx == 1
        # First group params should not require grad
        for p in model1.parameters():
            assert p.requires_grad is False
        # Second group params should require grad
        for p in model2.parameters():
            assert p.requires_grad is True

    def test_unfreeze(self):
        model1 = nn.Linear(4, 3)
        model2 = nn.Linear(3, 2)
        params = [list(model1.parameters()), list(model2.parameters())]
        opt = SGD(params, lr=0.1)

        # Freeze then unfreeze
        opt.freeze_to(1)
        opt.unfreeze()
        assert opt.frozen_idx == 0
        for p in model1.parameters():
            assert p.requires_grad is True
        for p in model2.parameters():
            assert p.requires_grad is True

    def test_freeze(self):
        model1 = nn.Linear(4, 3)
        model2 = nn.Linear(3, 2)
        params = [list(model1.parameters()), list(model2.parameters())]
        opt = SGD(params, lr=0.1)

        # freeze() freezes all but the last group
        opt.freeze()
        assert opt.frozen_idx == 1
        for p in model1.parameters():
            assert p.requires_grad is False
        for p in model2.parameters():
            assert p.requires_grad is True

    def test_frozen_params_not_updated(self):
        model1 = nn.Linear(4, 3)
        model2 = nn.Linear(3, 2)
        params = [list(model1.parameters()), list(model2.parameters())]
        opt = SGD(params, lr=0.1)

        opt.freeze_to(1)

        # Set grads only on unfrozen params
        for p in model2.parameters():
            p.grad = torch.ones_like(p)

        original_model1_weight = model1.weight.data.clone()
        original_model2_weight = model2.weight.data.clone()

        opt.step()

        # model1 should not have changed (no grad)
        assert torch.equal(model1.weight.data, original_model1_weight)
        # model2 should have changed
        assert not torch.equal(model2.weight.data, original_model2_weight)


# ============================================================
# Tests for RAdam optimizer
# ============================================================

class TestRAdam:
    """Tests for the RAdam optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = RAdam(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)

    def test_multiple_steps_converges(self):
        model = _make_linear_model()
        params = list(model.parameters())
        opt = RAdam(params, lr=0.01)
        # After many steps, the variance correction should kick in
        for _ in range(10):
            for p in params:
                p.grad = torch.ones_like(p)
            opt.step()
        # State should track step
        for p in params:
            assert opt.state[p]['step'] == 10


# ============================================================
# Tests for Lookahead optimizer
# ============================================================

class TestLookahead:
    """Tests for the Lookahead optimizer wrapper."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        base_opt = SGD(params, lr=0.1)
        opt = Lookahead(base_opt, k=6, alpha=0.5)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)

    def test_lookahead_step_applied_at_k(self):
        model = _make_linear_model()
        params = list(model.parameters())
        for p in params:
            p.grad = torch.ones_like(p)
        base_opt = SGD(params, lr=0.1)
        opt = Lookahead(base_opt, k=3, alpha=0.5)

        # Do k steps
        for i in range(3):
            for p in params:
                p.grad = torch.ones_like(p)
            opt.step()

        assert opt.count == 3
        # slow_weights should have been initialized and used
        assert opt.slow_weights is not None

    def test_state_dict_and_load(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        base_opt = SGD(params, lr=0.1)
        opt = Lookahead(base_opt, k=6, alpha=0.5)
        opt.step()
        sd = opt.state_dict()
        assert 'count' in sd
        assert 'slow_weights' in sd


# ============================================================
# Tests for ranger
# ============================================================

class TestRanger:
    """Tests for the ranger convenience function."""

    def test_creates_lookahead(self):
        model = _make_linear_model()
        params = list(model.parameters())
        opt = ranger(params, lr=0.01)
        assert isinstance(opt, Lookahead)

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = ranger(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)


# ============================================================
# Tests for QHAdam optimizer
# ============================================================

class TestQHAdam:
    """Tests for the QHAdam optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = QHAdam(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)


# ============================================================
# Tests for Larc optimizer
# ============================================================

class TestLarc:
    """Tests for the LARC/LARS optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Larc(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)

    def test_lars_no_clip(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Larc(params, lr=0.01, clip=False)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)


# ============================================================
# Tests for Lamb optimizer
# ============================================================

class TestLamb:
    """Tests for the LAMB optimizer."""

    def test_basic_step(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = Lamb(params, lr=0.01)
        original_weight = model.weight.data.clone()
        opt.step()
        assert not torch.equal(model.weight.data, original_weight)


# ============================================================
# Tests for detuplify_pg and set_item_pg
# ============================================================

class TestPgUtils:
    """Tests for detuplify_pg and set_item_pg utilities."""

    def test_detuplify_pg_simple(self):
        d = {'params': [1, 2, 3], 'lr': 0.1, 'momentum': 0.9}
        result = detuplify_pg(d)
        assert 'params' not in result
        assert result['lr'] == 0.1
        assert result['momentum'] == 0.9

    def test_detuplify_pg_with_list(self):
        d = {'params': [1], 'betas': (0.9, 0.999)}
        result = detuplify_pg(d)
        assert 'betas__0' in result
        assert 'betas__1' in result
        assert result['betas__0'] == 0.9
        assert result['betas__1'] == 0.999

    def test_set_item_pg_simple(self):
        pg = {'lr': 0.1}
        result = set_item_pg(pg, 'lr', 0.01)
        assert result['lr'] == 0.01

    def test_set_item_pg_tuple(self):
        pg = {'betas': (0.9, 0.999)}
        result = set_item_pg(pg, 'betas__0', 0.8)
        assert result['betas'][0] == 0.8
        assert result['betas'][1] == 0.999


# ============================================================
# Tests for set_hypers
# ============================================================

class TestSetHypers:
    """Tests for setting hyperparameters on optimizer."""

    def test_set_hyper_single_value(self):
        model = _make_linear_model()
        params = _get_params_with_grad(model, grad_value=1.0)
        opt = SGD(params, lr=0.1)
        opt.set_hyper('lr', 0.01)
        assert opt.hypers[0]['lr'] == 0.01

    def test_set_hyper_multiple_groups(self):
        model1 = nn.Linear(4, 3)
        model2 = nn.Linear(3, 2)
        params = [list(model1.parameters()), list(model2.parameters())]
        opt = SGD(params, lr=0.1)
        opt.set_hyper('lr', [0.01, 0.001])
        assert opt.hypers[0]['lr'] == 0.01
        assert opt.hypers[1]['lr'] == 0.001
