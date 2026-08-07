"""Tests for fastai.metrics module.

Covers: accuracy, error_rate, top_k_accuracy, mse, mae, msle, rmse, exp_rmspe,
AccumMetric, accuracy_multi, foreground_acc, Dice, DiceMulti, JaccardCoeff,
JaccardCoeffMulti, Perplexity, LossMetric, LossMetrics, skm_to_fastai,
and sklearn-based metric wrappers (F1Score, Precision, Recall, etc.).
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.metrics import (
    accuracy, error_rate, top_k_accuracy, mse, mae, msle,
    rmse, exp_rmspe, AccumMetric, accuracy_multi, foreground_acc,
    Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti,
    Perplexity, LossMetric, LossMetrics, skm_to_fastai,
    F1Score, Precision, Recall, BalancedAccuracy, HammingLoss,
    CohenKappa, MatthewsCorrCoef, Jaccard,
    F1ScoreMulti, PrecisionMulti, RecallMulti, HammingLossMulti,
    MatthewsCorrCoefMulti, JaccardMulti,
    ExplainedVariance, R2Score, PearsonCorrCoef, SpearmanCorrCoef,
    CorpusBLEUMetric,
)
from fastai.torch_core import TensorBase


# ============================================================
# Helper utilities
# ============================================================

def make_learner_mock(pred, targ):
    """Create a mock learner with pred/y attributes for metric accumulation."""
    learn = MagicMock()
    learn.pred = pred
    learn.y = targ
    learn.yb = (targ,)
    learn.training = False
    learn.to_detach = lambda x: x
    return learn


# ============================================================
# Tests for accuracy
# ============================================================

class TestAccuracy:
    """Tests for the accuracy metric function."""

    def test_perfect_accuracy(self):
        # 4 samples, 3 classes. Predictions strongly favor the correct class.
        pred = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, 10.0],
            [10.0, -10.0, -10.0],
        ])
        targ = torch.tensor([0, 1, 2, 0])
        result = accuracy(pred, targ)
        assert abs(result.item() - 1.0) < 1e-5

    def test_zero_accuracy(self):
        # All predictions are wrong
        pred = torch.tensor([
            [-10.0, 10.0, -10.0],
            [10.0, -10.0, -10.0],
            [10.0, -10.0, -10.0],
        ])
        targ = torch.tensor([0, 1, 2])
        result = accuracy(pred, targ)
        assert abs(result.item()) < 1e-5

    def test_partial_accuracy(self):
        # 2 out of 4 correct
        pred = torch.tensor([
            [10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 1, 0])
        result = accuracy(pred, targ)
        assert abs(result.item() - 0.5) < 1e-5

    def test_single_sample(self):
        pred = torch.tensor([[5.0, 1.0, 2.0]])
        targ = torch.tensor([0])
        result = accuracy(pred, targ)
        assert abs(result.item() - 1.0) < 1e-5


# ============================================================
# Tests for error_rate
# ============================================================

class TestErrorRate:
    """Tests for the error_rate metric function."""

    def test_perfect_predictions(self):
        pred = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
        ])
        targ = torch.tensor([0, 1])
        result = error_rate(pred, targ)
        assert abs(result.item()) < 1e-5

    def test_all_wrong(self):
        pred = torch.tensor([
            [-10.0, 10.0],
            [10.0, -10.0],
        ])
        targ = torch.tensor([0, 1])
        result = error_rate(pred, targ)
        assert abs(result.item() - 1.0) < 1e-5

    def test_complementary_to_accuracy(self):
        pred = torch.randn(10, 5)
        targ = torch.randint(0, 5, (10,))
        acc = accuracy(pred, targ)
        err = error_rate(pred, targ)
        assert abs((acc + err).item() - 1.0) < 1e-5


# ============================================================
# Tests for top_k_accuracy
# ============================================================

class TestTopKAccuracy:
    """Tests for the top_k_accuracy metric function."""

    def test_top1_same_as_accuracy(self):
        pred = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 2])
        top1 = top_k_accuracy(pred, targ, k=1)
        acc = accuracy(pred, targ)
        assert abs(top1.item() - acc.item()) < 1e-5

    def test_top_k_is_higher_or_equal_to_top_1(self):
        pred = torch.randn(20, 10)
        targ = torch.randint(0, 10, (20,))
        top1 = top_k_accuracy(pred, targ, k=1)
        top5 = top_k_accuracy(pred, targ, k=5)
        assert top5.item() >= top1.item() - 1e-5

    def test_top_k_equals_n_classes_is_perfect(self):
        # When k equals number of classes, every target must be in top-k
        pred = torch.randn(10, 5)
        targ = torch.randint(0, 5, (10,))
        result = top_k_accuracy(pred, targ, k=5)
        assert abs(result.item() - 1.0) < 1e-5

    def test_specific_case(self):
        # Manually crafted: target is second-highest prediction
        pred = torch.tensor([
            [5.0, 3.0, 1.0],
            [1.0, 5.0, 3.0],
        ])
        targ = torch.tensor([1, 2])  # second-best predictions
        top1 = top_k_accuracy(pred, targ, k=1)
        top2 = top_k_accuracy(pred, targ, k=2)
        assert abs(top1.item()) < 1e-5  # not in top-1
        assert abs(top2.item() - 1.0) < 1e-5  # in top-2


# ============================================================
# Tests for mse
# ============================================================

class TestMSE:
    """Tests for the mse metric function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(mse(inp, targ).item()) < 1e-7

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 3.0, 4.0])
        # MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
        assert abs(mse(inp, targ).item() - 1.0) < 1e-5

    def test_non_negative(self):
        inp = torch.randn(100)
        targ = torch.randn(100)
        assert mse(inp, targ).item() >= 0


# ============================================================
# Tests for mae
# ============================================================

class TestMAE:
    """Tests for the mae metric function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(mae(inp, targ).item()) < 1e-7

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 4.0, 6.0])
        # MAE = (1 + 2 + 3) / 3 = 2.0
        assert abs(mae(inp, targ).item() - 2.0) < 1e-5

    def test_symmetric(self):
        inp = torch.randn(50)
        targ = torch.randn(50)
        assert abs(mae(inp, targ).item() - mae(targ, inp).item()) < 1e-5


# ============================================================
# Tests for msle
# ============================================================

class TestMSLE:
    """Tests for the msle metric function."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(msle(inp, targ).item()) < 1e-7

    def test_non_negative(self):
        # msle requires non-negative inputs for log to work sensibly
        inp = torch.tensor([0.5, 1.5, 2.5])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert msle(inp, targ).item() >= 0

    def test_known_value(self):
        # msle(inp, targ) = mse(log(1+inp), log(1+targ))
        inp = torch.tensor([0.0])
        targ = torch.tensor([1.0])
        # log(1+0)=0, log(1+1)=log(2)~0.6931
        expected = (0.6931) ** 2
        assert abs(msle(inp, targ).item() - expected) < 1e-3


# ============================================================
# Tests for rmse (AccumMetric-based)
# ============================================================

class TestRMSE:
    """Tests for the rmse metric (AccumMetric wrapping _rmse)."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(inp, targ)
        assert abs(result.item()) < 1e-5

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 3.0, 4.0])
        # RMSE = sqrt(1.0) = 1.0
        result = rmse(inp, targ)
        assert abs(result.item() - 1.0) < 1e-5

    def test_rmse_is_sqrt_of_mse(self):
        inp = torch.randn(50)
        targ = torch.randn(50)
        mse_val = mse(inp, targ)
        rmse_val = rmse(inp, targ)
        assert abs(rmse_val.item() - torch.sqrt(mse_val).item()) < 1e-4


# ============================================================
# Tests for exp_rmspe (AccumMetric-based)
# ============================================================

class TestExpRMSPE:
    """Tests for the exp_rmspe metric."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = exp_rmspe(inp, targ)
        assert abs(result.item()) < 1e-5

    def test_non_negative(self):
        inp = torch.randn(20)
        targ = torch.randn(20)
        result = exp_rmspe(inp, targ)
        assert result.item() >= 0


# ============================================================
# Tests for AccumMetric
# ============================================================

class TestAccumMetric:
    """Tests for the AccumMetric class."""

    def test_basic_callable(self):
        # Use AccumMetric directly with a simple function
        def my_metric(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(my_metric)
        preds = torch.tensor([0, 1, 2, 1])
        targs = torch.tensor([0, 1, 2, 0])
        result = metric(preds, targs)
        assert abs(result.item() - 0.75) < 1e-5

    def test_reset(self):
        def my_metric(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(my_metric)
        metric.reset()
        assert metric.preds == []
        assert metric.targs == []

    def test_name_from_function(self):
        def custom_metric(preds, targs):
            return torch.tensor(0.5)

        metric = AccumMetric(custom_metric)
        assert metric.name == 'custom_metric'

    def test_custom_name(self):
        def custom_metric(preds, targs):
            return torch.tensor(0.5)

        metric = AccumMetric(custom_metric, name='my_name')
        assert metric.name == 'my_name'

    def test_name_setter(self):
        def custom_metric(preds, targs):
            return torch.tensor(0.5)

        metric = AccumMetric(custom_metric)
        metric.name = 'new_name'
        assert metric.name == 'new_name'

    def test_invert_args(self):
        # Test that invert_arg swaps preds and targs in function call
        def ordered_metric(a, b):
            return a.float().mean() - b.float().mean()

        metric_normal = AccumMetric(ordered_metric, invert_arg=False)
        metric_invert = AccumMetric(ordered_metric, invert_arg=True)

        preds = torch.tensor([10.0, 10.0])
        targs = torch.tensor([1.0, 1.0])

        result_normal = metric_normal(preds, targs)
        result_invert = metric_invert(preds, targs)

        # Normal: preds - targs = 9.0
        assert abs(result_normal.item() - 9.0) < 1e-5
        # Inverted: targs - preds = -9.0
        assert abs(result_invert.item() + 9.0) < 1e-5

    def test_to_np(self):
        # When to_np=True, preds and targs should be converted to numpy
        def np_metric(preds, targs):
            assert isinstance(preds, np.ndarray)
            assert isinstance(targs, np.ndarray)
            return np.mean(preds == targs)

        metric = AccumMetric(np_metric, to_np=True)
        preds = torch.tensor([0, 1, 2])
        targs = torch.tensor([0, 1, 2])
        result = metric(preds, targs)
        assert abs(result - 1.0) < 1e-5

    def test_value_empty_preds(self):
        def my_metric(preds, targs):
            return torch.tensor(0.0)

        metric = AccumMetric(my_metric)
        metric.reset()
        assert metric.value is None

    def test_accumulate_with_dim_argmax(self):
        def my_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(my_acc, dim_argmax=-1)
        # Simulates logits for 3 classes
        pred = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0]])
        targ = torch.tensor([0, 1])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        assert abs(metric.value.item() - 1.0) < 1e-5


# ============================================================
# Tests for accuracy_multi
# ============================================================

class TestAccuracyMulti:
    """Tests for the accuracy_multi metric function."""

    def test_perfect_multi_label(self):
        # Logits that are > thresh=0.5 after sigmoid
        # sigmoid(2.0) ~ 0.88 > 0.5, sigmoid(-2.0) ~ 0.12 < 0.5
        inp = torch.tensor([[2.0, -2.0, 2.0], [2.0, 2.0, -2.0]])
        targ = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert abs(result.item() - 1.0) < 1e-5

    def test_all_wrong(self):
        inp = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        targ = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert abs(result.item()) < 1e-5

    def test_no_sigmoid(self):
        # When sigmoid=False, values are compared directly to threshold
        inp = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        targ = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert abs(result.item() - 1.0) < 1e-5


# ============================================================
# Tests for foreground_acc
# ============================================================

class TestForegroundAcc:
    """Tests for the foreground_acc function."""

    def test_perfect_foreground(self):
        # 2 samples, 3 classes, 2x2 spatial
        # Targets: some foreground (non-zero), predictions match
        targ = torch.tensor([[[1, 2], [0, 1]]])  # 1 sample, 2x2
        # Create predictions that argmax to match targ
        pred = torch.zeros(1, 3, 2, 2)
        pred[0, 1, 0, 0] = 10.0  # class 1 at (0,0)
        pred[0, 2, 0, 1] = 10.0  # class 2 at (0,1)
        pred[0, 0, 1, 0] = 10.0  # class 0 at (1,0) - background, excluded
        pred[0, 1, 1, 1] = 10.0  # class 1 at (1,1)
        result = foreground_acc(pred, targ, bkg_idx=0, axis=1)
        assert abs(result.item() - 1.0) < 1e-5

    def test_no_foreground(self):
        # All background - result would be mean of empty tensor
        targ = torch.tensor([[[0, 0], [0, 0]]])
        pred = torch.zeros(1, 3, 2, 2)
        pred[0, 0, :, :] = 10.0
        # When there's no foreground, the result is mean of empty tensor (nan)
        result = foreground_acc(pred, targ, bkg_idx=0, axis=1)
        assert torch.isnan(result)


# ============================================================
# Tests for Dice metric
# ============================================================

class TestDice:
    """Tests for the Dice segmentation metric."""

    def test_perfect_segmentation(self):
        dice = Dice(axis=1)
        dice.reset()
        # Create prediction that matches target exactly
        targ = torch.tensor([1, 0, 1, 0])
        pred = torch.zeros(4, 2)
        pred[0, 1] = 10.0  # predicts class 1
        pred[1, 0] = 10.0  # predicts class 0
        pred[2, 1] = 10.0  # predicts class 1
        pred[3, 0] = 10.0  # predicts class 0
        learn = make_learner_mock(pred, targ)
        dice.accumulate(learn)
        # Dice = 2*inter / union. For class 1: inter=2, union(pred_1+targ_1)=4
        # Actually Dice sums over all: inter = sum(pred*targ), union = sum(pred+targ)
        # pred (argmax) = [1, 0, 1, 0], targ = [1, 0, 1, 0]
        # inter = 1*1 + 0*0 + 1*1 + 0*0 = 2, union = (1+1)+(0+0)+(1+1)+(0+0) = 4
        # Dice = 2*2/4 = 1.0
        assert abs(dice.value - 1.0) < 1e-5

    def test_no_overlap(self):
        dice = Dice(axis=1)
        dice.reset()
        # Predictions are opposite of targets
        targ = torch.tensor([1, 1, 0, 0])
        pred = torch.zeros(4, 2)
        pred[0, 0] = 10.0  # predicts class 0
        pred[1, 0] = 10.0  # predicts class 0
        pred[2, 1] = 10.0  # predicts class 1
        pred[3, 1] = 10.0  # predicts class 1
        learn = make_learner_mock(pred, targ)
        dice.accumulate(learn)
        # pred=[0,0,1,1], targ=[1,1,0,0]
        # inter = 0*1+0*1+1*0+1*0 = 0, union = (0+1)+(0+1)+(1+0)+(1+0)=4
        # Dice = 0/4 = 0
        assert abs(dice.value) < 1e-5

    def test_reset(self):
        dice = Dice(axis=1)
        dice.reset()
        assert dice.inter == 0
        assert dice.union == 0

    def test_empty_returns_none(self):
        dice = Dice(axis=1)
        dice.reset()
        assert dice.value is None


# ============================================================
# Tests for DiceMulti metric
# ============================================================

class TestDiceMulti:
    """Tests for the DiceMulti segmentation metric."""

    def test_perfect_multiclass(self):
        dice = DiceMulti(axis=1)
        dice.reset()
        # 4 pixels, 3 classes
        targ = torch.tensor([0, 1, 2, 1])
        pred = torch.zeros(4, 3)
        pred[0, 0] = 10.0
        pred[1, 1] = 10.0
        pred[2, 2] = 10.0
        pred[3, 1] = 10.0
        learn = make_learner_mock(pred, targ)
        # Need to set pred shape for the accumulate logic
        dice.accumulate(learn)
        assert abs(dice.value - 1.0) < 1e-5

    def test_reset_clears_dicts(self):
        dice = DiceMulti(axis=1)
        dice.reset()
        assert dice.inter == {}
        assert dice.union == {}


# ============================================================
# Tests for JaccardCoeff metric
# ============================================================

class TestJaccardCoeff:
    """Tests for the JaccardCoeff segmentation metric."""

    def test_perfect_segmentation(self):
        jac = JaccardCoeff(axis=1)
        jac.reset()
        targ = torch.tensor([1, 0, 1, 0])
        pred = torch.zeros(4, 2)
        pred[0, 1] = 10.0
        pred[1, 0] = 10.0
        pred[2, 1] = 10.0
        pred[3, 0] = 10.0
        learn = make_learner_mock(pred, targ)
        jac.accumulate(learn)
        # inter=2, union=4, jaccard = 2/(4-2) = 1.0
        assert abs(jac.value - 1.0) < 1e-5

    def test_no_overlap(self):
        jac = JaccardCoeff(axis=1)
        jac.reset()
        targ = torch.tensor([1, 1, 0, 0])
        pred = torch.zeros(4, 2)
        pred[0, 0] = 10.0
        pred[1, 0] = 10.0
        pred[2, 1] = 10.0
        pred[3, 1] = 10.0
        learn = make_learner_mock(pred, targ)
        jac.accumulate(learn)
        # inter=0, union=4, jaccard = 0/(4-0) = 0
        assert abs(jac.value) < 1e-5


# ============================================================
# Tests for JaccardCoeffMulti metric
# ============================================================

class TestJaccardCoeffMulti:
    """Tests for the JaccardCoeffMulti segmentation metric."""

    def test_perfect_multiclass(self):
        jac = JaccardCoeffMulti(axis=1)
        jac.reset()
        targ = torch.tensor([0, 1, 2, 1])
        pred = torch.zeros(4, 3)
        pred[0, 0] = 10.0
        pred[1, 1] = 10.0
        pred[2, 2] = 10.0
        pred[3, 1] = 10.0
        learn = make_learner_mock(pred, targ)
        jac.accumulate(learn)
        assert abs(jac.value - 1.0) < 1e-5


# ============================================================
# Tests for Perplexity metric
# ============================================================

class TestPerplexity:
    """Tests for the Perplexity metric."""

    def test_perplexity_creation(self):
        perp = Perplexity()
        assert perp.name == "perplexity"

    def test_perplexity_computation(self):
        perp = Perplexity()
        perp.reset()
        # Simulate a learner with loss
        learn = MagicMock()
        learn.yb = (torch.tensor([1, 2, 3]),)
        learn.loss = torch.tensor(2.0)  # cross-entropy loss = 2.0
        learn.to_detach = lambda x: x

        perp.accumulate(learn)
        # Perplexity = exp(loss) = exp(2.0) ~ 7.389
        result = perp.value
        assert abs(result.item() - np.exp(2.0)) < 1e-3

    def test_perplexity_empty(self):
        perp = Perplexity()
        perp.reset()
        assert perp.value is None


# ============================================================
# Tests for LossMetric
# ============================================================

class TestLossMetric:
    """Tests for the LossMetric class."""

    def test_basic(self):
        metric = LossMetric('my_loss')
        assert metric.name == 'my_loss'

    def test_custom_name(self):
        metric = LossMetric('attr_loss', nm='custom_name')
        assert metric.name == 'custom_name'

    def test_accumulate(self):
        metric = LossMetric('component_loss')
        metric.reset()
        learn = MagicMock()
        learn.yb = (torch.tensor([1, 2, 3]),)
        learn.loss_func = MagicMock()
        learn.loss_func.component_loss = torch.tensor(0.5)
        learn.to_detach = lambda x: x
        metric.accumulate(learn)
        # total = 0.5 * 3 = 1.5, count = 3
        assert abs(metric.value.item() - 0.5) < 1e-5


# ============================================================
# Tests for LossMetrics
# ============================================================

class TestLossMetrics:
    """Tests for the LossMetrics factory function."""

    def test_from_string(self):
        metrics = LossMetrics('loss_a,loss_b,loss_c')
        assert len(metrics) == 3
        assert metrics[0].name == 'loss_a'
        assert metrics[1].name == 'loss_b'
        assert metrics[2].name == 'loss_c'

    def test_from_list(self):
        metrics = LossMetrics(['loss_x', 'loss_y'])
        assert len(metrics) == 2

    def test_custom_names(self):
        metrics = LossMetrics('attr_a,attr_b', nms='Name A,Name B')
        assert metrics[0].name == 'Name A'
        assert metrics[1].name == 'Name B'


# ============================================================
# Tests for skm_to_fastai (sklearn metric wrapper)
# ============================================================

class TestSkmToFastai:
    """Tests for the skm_to_fastai conversion function.

    These metrics use dim_argmax internally via the accumulate path.
    When called directly via __call__, they need pre-argmaxed predictions
    (already class indices) matching the target shape.
    """

    def test_balanced_accuracy(self):
        metric = BalancedAccuracy()
        # Use the accumulate path which handles argmax internally
        pred = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 2])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5

    def test_f1_score_binary(self):
        metric = F1Score()
        pred = torch.tensor([
            [10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        targ = torch.tensor([0, 0, 1, 1])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5

    def test_precision_binary(self):
        metric = Precision()
        pred = torch.tensor([
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 1, 0])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        # Predicted class 1: samples 1,2,3. True positives: 1,2. FP: 3.
        # Precision = 2/3
        assert abs(result - 2.0/3.0) < 1e-5

    def test_recall_binary(self):
        metric = Recall()
        pred = torch.tensor([
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
        ])
        targ = torch.tensor([0, 1, 1, 1])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        # True class 1: samples 1,2,3. Predicted 1: samples 1,2. Recall = 2/3
        assert abs(result - 2.0/3.0) < 1e-5

    def test_hamming_loss(self):
        metric = HammingLoss()
        pred = torch.tensor([
            [10.0, -10.0],
            [-10.0, 10.0],
        ])
        targ = torch.tensor([0, 1])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        # Perfect predictions -> hamming loss = 0
        assert abs(result) < 1e-5

    def test_cohen_kappa_perfect(self):
        metric = CohenKappa()
        pred = torch.tensor([
            [10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 0, 1])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5


# ============================================================
# Tests for multi-label sklearn metrics
# ============================================================

class TestMultiLabelMetrics:
    """Tests for multi-label sklearn-based metric wrappers.

    These metrics use sigmoid activation and thresholding internally,
    which only happens through the accumulate path.
    """

    def test_f1_score_multi(self):
        metric = F1ScoreMulti(thresh=0.5, sigmoid=True)
        # logits: sigmoid(2.0)~0.88>0.5, sigmoid(-2.0)~0.12<0.5
        pred = torch.tensor([
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
        ])
        targ = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
        ])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5

    def test_precision_multi(self):
        metric = PrecisionMulti(thresh=0.5, sigmoid=True)
        pred = torch.tensor([
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
        ])
        targ = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
        ])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5

    def test_recall_multi(self):
        metric = RecallMulti(thresh=0.5, sigmoid=True)
        pred = torch.tensor([
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
        ])
        targ = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
        ])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result - 1.0) < 1e-5

    def test_hamming_loss_multi(self):
        metric = HammingLossMulti(thresh=0.5, sigmoid=True)
        pred = torch.tensor([
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
        ])
        targ = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
        ])
        learn = make_learner_mock(pred, targ)
        metric.reset()
        metric.accumulate(learn)
        result = metric.value
        assert abs(result) < 1e-5


# ============================================================
# Tests for regression sklearn metrics
# ============================================================

class TestRegressionMetrics:
    """Tests for sklearn-based regression metrics."""

    def test_explained_variance_perfect(self):
        metric = ExplainedVariance()
        pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targ = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        result = metric(pred, targ)
        assert abs(result - 1.0) < 1e-5

    def test_r2_score_perfect(self):
        metric = R2Score()
        pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targ = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        result = metric(pred, targ)
        assert abs(result - 1.0) < 1e-5

    def test_r2_score_bad(self):
        metric = R2Score()
        # Constant prediction should give low/negative R2
        pred = torch.tensor([[2.5], [2.5], [2.5], [2.5]])
        targ = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        result = metric(pred, targ)
        assert result <= 0.0 + 1e-5

    def test_pearson_corr_perfect(self):
        metric = PearsonCorrCoef(to_np=True)
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = metric(pred, targ)
        assert abs(result - 1.0) < 1e-5

    def test_spearman_corr_perfect(self):
        metric = SpearmanCorrCoef(to_np=True)
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = metric(pred, targ)
        assert abs(result - 1.0) < 1e-5

    def test_pearson_negative_corr(self):
        metric = PearsonCorrCoef(to_np=True)
        pred = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = metric(pred, targ)
        assert abs(result + 1.0) < 1e-5  # should be -1.0


# ============================================================
# Tests for CorpusBLEUMetric
# ============================================================

class TestCorpusBLEUMetric:
    """Tests for the CorpusBLEUMetric class."""

    def test_creation(self):
        bleu = CorpusBLEUMetric(vocab_sz=100)
        assert bleu.metric_name == 'CorpusBLEU'

    def test_reset(self):
        bleu = CorpusBLEUMetric(vocab_sz=100)
        bleu.pred_len = 50
        bleu.targ_len = 50
        bleu.reset()
        assert bleu.pred_len == 0
        assert bleu.targ_len == 0
        assert bleu.corrects == [0, 0, 0, 0]
        assert bleu.counts == [0, 0, 0, 0]

    def test_perfect_match(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        # Create a sequence that matches perfectly
        seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # Create pred logits where argmax gives same sequence
        pred = torch.zeros(1, 8, 100)
        for i, token in enumerate(seq[0]):
            pred[0, i, token] = 10.0
        learn = make_learner_mock(pred, seq)
        bleu.accumulate(learn)
        result = bleu.value
        # Perfect match should give a high BLEU score
        assert result > 0.9

    def test_empty_returns_zero(self):
        bleu = CorpusBLEUMetric(vocab_sz=100)
        bleu.reset()
        # After reset, counts are [0,0,0,0] which triggers the zero branch
        assert bleu.value == 0.0
