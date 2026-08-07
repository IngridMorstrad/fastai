"""Tests for fastai.metrics module.

Covers metric functions and classes: accuracy, error_rate, top_k_accuracy,
mse, rmse, mae, msle, accuracy_multi, foreground_acc, Dice, DiceMulti,
JaccardCoeff, JaccardCoeffMulti, AccumMetric, exp_rmspe, and Perplexity.
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.metrics import (
    accuracy, error_rate, top_k_accuracy,
    mse, mae, msle, rmse, exp_rmspe,
    accuracy_multi, foreground_acc,
    Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti,
    AccumMetric, Perplexity,
    skm_to_fastai, ActivationType,
)


# ============================================================
# Tests for accuracy
# ============================================================

class TestAccuracy:
    """Tests for the accuracy metric function."""

    def test_perfect_accuracy(self):
        """All predictions correct gives accuracy of 1.0."""
        # 4 samples, 3 classes; targets are [0, 1, 2, 1]
        inp = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
        ])
        targ = torch.tensor([0, 1, 2, 1])
        result = accuracy(inp, targ)
        assert abs(result.item() - 1.0) < 1e-6

    def test_zero_accuracy(self):
        """All predictions wrong gives accuracy of 0.0."""
        inp = torch.tensor([
            [0.0, 0.0, 10.0],  # predicts 2, target 0
            [10.0, 0.0, 0.0],  # predicts 0, target 1
            [10.0, 0.0, 0.0],  # predicts 0, target 2
        ])
        targ = torch.tensor([0, 1, 2])
        result = accuracy(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_partial_accuracy(self):
        """Some predictions correct gives expected fraction."""
        inp = torch.tensor([
            [10.0, 0.0],  # predicts 0, correct
            [10.0, 0.0],  # predicts 0, wrong
            [0.0, 10.0],  # predicts 1, correct
            [0.0, 10.0],  # predicts 1, wrong
        ])
        targ = torch.tensor([0, 1, 1, 0])
        result = accuracy(inp, targ)
        assert abs(result.item() - 0.5) < 1e-6

    def test_single_sample(self):
        """Works with a single sample."""
        inp = torch.tensor([[0.1, 0.9]])
        targ = torch.tensor([1])
        result = accuracy(inp, targ)
        assert abs(result.item() - 1.0) < 1e-6

    def test_custom_axis(self):
        """Works with a different axis argument."""
        # axis=1 is the default, explicit test
        inp = torch.tensor([[5.0, 1.0, 1.0]])
        targ = torch.tensor([0])
        result = accuracy(inp, targ, axis=-1)
        assert abs(result.item() - 1.0) < 1e-6


# ============================================================
# Tests for error_rate
# ============================================================

class TestErrorRate:
    """Tests for the error_rate metric function."""

    def test_perfect_predictions(self):
        """Perfect predictions give error_rate of 0.0."""
        inp = torch.tensor([
            [10.0, 0.0],
            [0.0, 10.0],
        ])
        targ = torch.tensor([0, 1])
        result = error_rate(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_all_wrong(self):
        """All wrong predictions give error_rate of 1.0."""
        inp = torch.tensor([
            [0.0, 10.0],
            [10.0, 0.0],
        ])
        targ = torch.tensor([0, 1])
        result = error_rate(inp, targ)
        assert abs(result.item() - 1.0) < 1e-6

    def test_complement_of_accuracy(self):
        """error_rate should be exactly 1 - accuracy."""
        inp = torch.randn(16, 5)
        targ = torch.randint(0, 5, (16,))
        acc = accuracy(inp, targ)
        err = error_rate(inp, targ)
        assert abs((acc + err).item() - 1.0) < 1e-6


# ============================================================
# Tests for top_k_accuracy
# ============================================================

class TestTopKAccuracy:
    """Tests for the top_k_accuracy metric function."""

    def test_top_1_same_as_accuracy(self):
        """top_k_accuracy with k=1 should match accuracy."""
        inp = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ])
        targ = torch.tensor([0, 1, 2])
        result = top_k_accuracy(inp, targ, k=1)
        assert abs(result.item() - 1.0) < 1e-6

    def test_top_k_captures_lower_ranked(self):
        """Target in second-best prediction captured by k=2."""
        inp = torch.tensor([
            [5.0, 10.0, 0.0],  # top-2: [1, 0] -> targ=0 in top-2
            [0.0, 5.0, 10.0],  # top-2: [2, 1] -> targ=1 in top-2
        ])
        targ = torch.tensor([0, 1])
        result = top_k_accuracy(inp, targ, k=2)
        assert abs(result.item() - 1.0) < 1e-6

    def test_top_k_misses(self):
        """Target not in top-k gives 0."""
        inp = torch.tensor([
            [0.0, 10.0, 5.0],  # top-1: [1] -> targ=0 not in top-1
        ])
        targ = torch.tensor([0])
        result = top_k_accuracy(inp, targ, k=1)
        assert abs(result.item()) < 1e-6

    def test_top_k_with_k_equals_n_classes(self):
        """k equal to number of classes always gives 1.0."""
        inp = torch.randn(8, 4)
        targ = torch.randint(0, 4, (8,))
        result = top_k_accuracy(inp, targ, k=4)
        assert abs(result.item() - 1.0) < 1e-6


# ============================================================
# Tests for mse (mean squared error)
# ============================================================

class TestMSE:
    """Tests for the mse metric function."""

    def test_zero_error(self):
        """Identical inputs give MSE of 0."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = mse(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        """Known MSE calculation."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 5.0])
        # MSE = ((0)^2 + (0)^2 + (2)^2) / 3 = 4/3
        expected = 4.0 / 3.0
        result = mse(inp, targ)
        assert abs(result.item() - expected) < 1e-5

    def test_batch_dimension(self):
        """Works with 2D tensors (batch of predictions)."""
        inp = torch.ones(4, 3)
        targ = torch.zeros(4, 3)
        # MSE = 1.0
        result = mse(inp, targ)
        assert abs(result.item() - 1.0) < 1e-6

    def test_symmetry(self):
        """MSE is symmetric: mse(a, b) == mse(b, a)."""
        a = torch.randn(10)
        b = torch.randn(10)
        assert abs(mse(a, b).item() - mse(b, a).item()) < 1e-6


# ============================================================
# Tests for rmse (root mean squared error)
# ============================================================

class TestRMSE:
    """Tests for the rmse metric (AccumMetric-based)."""

    def test_zero_error(self):
        """Identical inputs give RMSE of 0."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        """Known RMSE calculation."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 5.0])
        # RMSE = sqrt(4/3) ~ 1.1547
        expected = (4.0 / 3.0) ** 0.5
        result = rmse(inp, targ)
        assert abs(result.item() - expected) < 1e-4

    def test_is_sqrt_of_mse(self):
        """RMSE should equal sqrt(MSE)."""
        inp = torch.randn(20)
        targ = torch.randn(20)
        mse_val = mse(inp, targ)
        rmse_val = rmse(inp, targ)
        assert abs(rmse_val.item() - mse_val.item() ** 0.5) < 1e-5


# ============================================================
# Tests for mae (mean absolute error)
# ============================================================

class TestMAE:
    """Tests for the mae metric function."""

    def test_zero_error(self):
        """Identical inputs give MAE of 0."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = mae(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        """Known MAE calculation."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 2.0, 5.0])
        # MAE = (|1| + |0| + |2|) / 3 = 1.0
        result = mae(inp, targ)
        assert abs(result.item() - 1.0) < 1e-6

    def test_symmetry(self):
        """MAE is symmetric."""
        a = torch.randn(10)
        b = torch.randn(10)
        assert abs(mae(a, b).item() - mae(b, a).item()) < 1e-6

    def test_non_negative(self):
        """MAE is always non-negative."""
        inp = torch.randn(50)
        targ = torch.randn(50)
        assert mae(inp, targ).item() >= 0.0


# ============================================================
# Tests for msle (mean squared logarithmic error)
# ============================================================

class TestMSLE:
    """Tests for the msle metric function."""

    def test_zero_error(self):
        """Identical positive inputs give MSLE of 0."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = msle(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        """Known MSLE calculation."""
        inp = torch.tensor([1.0, 2.0])
        targ = torch.tensor([1.0, 3.0])
        # msle = mean((log(2) - log(2))^2, (log(3) - log(4))^2)
        import math
        expected = ((math.log(2) - math.log(2))**2 + (math.log(3) - math.log(4))**2) / 2.0
        result = msle(inp, targ)
        assert abs(result.item() - expected) < 1e-5

    def test_zero_inputs(self):
        """Works with zero inputs (log(1+0) = 0)."""
        inp = torch.tensor([0.0, 0.0])
        targ = torch.tensor([0.0, 0.0])
        result = msle(inp, targ)
        assert abs(result.item()) < 1e-6


# ============================================================
# Tests for accuracy_multi
# ============================================================

class TestAccuracyMulti:
    """Tests for the accuracy_multi metric function."""

    def test_perfect_accuracy(self):
        """Perfect multi-label predictions."""
        # After sigmoid: [0.73, 0.27, 0.73] -> thresh 0.5 -> [True, False, True]
        inp = torch.tensor([1.0, -1.0, 1.0])
        targ = torch.tensor([1.0, 0.0, 1.0])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert abs(result.item() - 1.0) < 1e-6

    def test_all_wrong(self):
        """All predictions wrong."""
        inp = torch.tensor([-5.0, 5.0, -5.0])
        targ = torch.tensor([1.0, 0.0, 1.0])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert abs(result.item()) < 1e-6

    def test_no_sigmoid(self):
        """Without sigmoid, raw values compared to threshold."""
        inp = torch.tensor([0.8, 0.2, 0.9])
        targ = torch.tensor([1.0, 0.0, 1.0])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert abs(result.item() - 1.0) < 1e-6

    def test_custom_threshold(self):
        """Custom threshold changes predictions."""
        inp = torch.tensor([0.6, 0.6, 0.6])  # all above 0.5 but below 0.7
        targ = torch.tensor([1.0, 0.0, 1.0])
        # With threshold=0.7, all predict False -> matches only middle
        result = accuracy_multi(inp, targ, thresh=0.7, sigmoid=False)
        # pred = [False, False, False], targ = [True, False, True]
        # matches = [False, True, False] -> mean = 1/3
        expected = 1.0 / 3.0
        assert abs(result.item() - expected) < 1e-5


# ============================================================
# Tests for foreground_acc
# ============================================================

class TestForegroundAcc:
    """Tests for the foreground_acc metric function."""

    def test_perfect_foreground(self):
        """Perfect predictions on foreground pixels."""
        # 1 sample, 3 classes, 2x2 spatial
        # Predictions: all class 1 everywhere (foreground)
        inp = torch.zeros(1, 3, 2, 2)
        inp[0, 1, :, :] = 10.0  # class 1 is highest
        # Target: all class 1 (no background)
        targ = torch.ones(1, 1, 2, 2).long()
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        assert abs(result.item() - 1.0) < 1e-6

    def test_all_wrong_foreground(self):
        """All foreground predictions wrong."""
        # Predictions: all class 2
        inp = torch.zeros(1, 3, 2, 2)
        inp[0, 2, :, :] = 10.0  # class 2 is highest
        # Target: all class 1 (foreground)
        targ = torch.ones(1, 1, 2, 2).long()
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        assert abs(result.item()) < 1e-6

    def test_background_ignored(self):
        """Background pixels are excluded from accuracy calculation."""
        # 1 sample, 3 classes, 4 pixels total
        inp = torch.zeros(1, 3, 2, 2)
        # pixel (0,0): predict class 0 (background)
        inp[0, 0, 0, 0] = 10.0
        # pixel (0,1): predict class 1 (correct foreground)
        inp[0, 1, 0, 1] = 10.0
        # pixel (1,0): predict class 1 (correct foreground)
        inp[0, 1, 1, 0] = 10.0
        # pixel (1,1): predict class 2 (wrong foreground)
        inp[0, 2, 1, 1] = 10.0

        # Target: pixel (0,0)=0(bg), (0,1)=1, (1,0)=1, (1,1)=1
        targ = torch.tensor([[[[0, 1], [1, 1]]]])
        result = foreground_acc(inp, targ, bkg_idx=0, axis=1)
        # Only 3 foreground pixels; 2 correct (pixel(0,1) and (1,0)), 1 wrong (pixel(1,1))
        expected = 2.0 / 3.0
        assert abs(result.item() - expected) < 1e-5


# ============================================================
# Tests for Dice metric
# ============================================================

class TestDice:
    """Tests for the Dice coefficient metric."""

    def _make_learn(self, pred, y):
        """Create a simple namespace mimicking learn for accumulate."""
        class FakeLearner:
            pass
        learn = FakeLearner()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect_overlap(self):
        """Perfect overlap gives Dice = 1.0."""
        dice = Dice(axis=1)
        dice.reset()
        # 2 classes, predictions match target exactly
        # pred logits: class 1 is high where target is 1
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, :] = 10.0  # all pixels predicted as class 1
        targ = torch.ones(4).long()  # all pixels are class 1
        learn = self._make_learn(pred, targ)
        dice.accumulate(learn)
        assert abs(dice.value - 1.0) < 1e-6

    def test_no_overlap(self):
        """No overlap gives Dice = 0.0."""
        dice = Dice(axis=1)
        dice.reset()
        # pred: all class 0, target: all class 1
        pred = torch.zeros(1, 2, 4)
        pred[0, 0, :] = 10.0  # all pixels predicted as class 0
        targ = torch.ones(4).long()  # target is class 1
        learn = self._make_learn(pred, targ)
        dice.accumulate(learn)
        assert abs(dice.value) < 1e-6

    def test_partial_overlap(self):
        """Partial overlap gives expected Dice score."""
        dice = Dice(axis=1)
        dice.reset()
        # 4 pixels: pred=[1,1,0,0], targ=[1,1,1,0]
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 0, 2] = 10.0
        pred[0, 0, 3] = 10.0
        targ = torch.tensor([1, 1, 1, 0]).long()
        learn = self._make_learn(pred, targ)
        dice.accumulate(learn)
        # inter = 2 (pred=1 & targ=1 at pixels 0,1)
        # union = pred_sum + targ_sum = 2 + 3 = 5
        # Dice = 2*2/5 = 0.8
        assert abs(dice.value - 0.8) < 1e-6

    def test_accumulate_multiple_batches(self):
        """Accumulate across multiple batches."""
        dice = Dice(axis=1)
        dice.reset()
        # Batch 1: pred=[1,1], targ=[1,1]
        pred1 = torch.zeros(1, 2, 2)
        pred1[0, 1, :] = 10.0
        targ1 = torch.ones(2).long()
        learn1 = self._make_learn(pred1, targ1)
        dice.accumulate(learn1)
        # Batch 2: pred=[0,0], targ=[1,1]
        pred2 = torch.zeros(1, 2, 2)
        pred2[0, 0, :] = 10.0
        targ2 = torch.ones(2).long()
        learn2 = self._make_learn(pred2, targ2)
        dice.accumulate(learn2)
        # Total: inter = 2+0 = 2, union = (2+2) + (0+2) = 6
        # Dice = 2*2/6 = 2/3
        expected = 2.0 * 2.0 / 6.0
        assert abs(dice.value - expected) < 1e-6


# ============================================================
# Tests for DiceMulti metric
# ============================================================

class TestDiceMulti:
    """Tests for the DiceMulti (macro-averaged Dice) metric."""

    def _make_learn(self, pred, y):
        class FakeLearner:
            pass
        learn = FakeLearner()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect_overlap(self):
        """Perfect overlap gives DiceMulti = 1.0."""
        dice = DiceMulti(axis=1)
        dice.reset()
        # 3 classes, 4 pixels, perfect prediction
        pred = torch.zeros(1, 3, 4)
        targ = torch.tensor([0, 1, 2, 1]).long()
        for i in range(4):
            pred[0, targ[i].item(), i] = 10.0
        learn = self._make_learn(pred, targ)
        dice.accumulate(learn)
        assert abs(dice.value - 1.0) < 1e-6

    def test_all_same_class(self):
        """All predictions same class, target varied."""
        dice = DiceMulti(axis=1)
        dice.reset()
        # Predict all class 0, target has classes 0, 1, 2
        pred = torch.zeros(1, 3, 3)
        pred[0, 0, :] = 10.0
        targ = torch.tensor([0, 1, 2]).long()
        learn = self._make_learn(pred, targ)
        dice.accumulate(learn)
        # Class 0: inter=1, union=3+1=4, dice=2/4=0.5 (pred has 3 ones for class 0, targ has 1)
        # Wait, let's recalculate:
        # pred argmax -> [0, 0, 0]
        # Class 0: p=[1,1,1], t=[1,0,0] -> inter=1, union=4, dice=0.5
        # Class 1: p=[0,0,0], t=[0,1,0] -> inter=0, union=1, dice=0
        # Class 2: p=[0,0,0], t=[0,0,1] -> inter=0, union=1, dice=0
        # Mean = (0.5 + 0 + 0) / 3 = 1/6
        expected = (0.5 + 0.0 + 0.0) / 3.0
        assert abs(dice.value - expected) < 1e-5


# ============================================================
# Tests for JaccardCoeff metric
# ============================================================

class TestJaccardCoeff:
    """Tests for the Jaccard coefficient (IoU) metric."""

    def _make_learn(self, pred, y):
        class FakeLearner:
            pass
        learn = FakeLearner()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect_overlap(self):
        """Perfect overlap gives Jaccard = 1.0."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, :] = 10.0
        targ = torch.ones(4).long()
        learn = self._make_learn(pred, targ)
        jac.accumulate(learn)
        # inter = 4, union = 4+4 = 8, jaccard = 4/(8-4) = 1.0
        assert abs(jac.value - 1.0) < 1e-6

    def test_no_overlap(self):
        """No overlap gives Jaccard = 0.0."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 0, :] = 10.0  # predict all class 0
        targ = torch.ones(4).long()  # target all class 1
        learn = self._make_learn(pred, targ)
        jac.accumulate(learn)
        # inter = 0, union = 0+4 = 4, jaccard = 0/(4-0) = 0
        assert abs(jac.value) < 1e-6

    def test_partial_overlap(self):
        """Partial overlap gives expected Jaccard."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        # pred=[1,1,0,0], targ=[1,1,1,0]
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 0, 2] = 10.0
        pred[0, 0, 3] = 10.0
        targ = torch.tensor([1, 1, 1, 0]).long()
        learn = self._make_learn(pred, targ)
        jac.accumulate(learn)
        # inter = 2, union = 2+3 = 5, jaccard = 2/(5-2) = 2/3
        expected = 2.0 / 3.0
        assert abs(jac.value - expected) < 1e-5


# ============================================================
# Tests for AccumMetric
# ============================================================

class TestAccumMetric:
    """Tests for the AccumMetric class."""

    def test_basic_callable(self):
        """AccumMetric can be called directly with preds and targs."""
        def simple_metric(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_metric)
        preds = torch.tensor([0, 1, 1, 0])
        targs = torch.tensor([0, 1, 0, 0])
        result = metric(preds, targs)
        assert abs(result.item() - 0.75) < 1e-6

    def test_name_from_function(self):
        """AccumMetric derives name from the function."""
        def my_custom_metric(p, t):
            return torch.tensor(0.0)

        metric = AccumMetric(my_custom_metric)
        assert metric.name == 'my_custom_metric'

    def test_custom_name(self):
        """AccumMetric uses custom name when provided."""
        def func(p, t):
            return torch.tensor(0.0)

        metric = AccumMetric(func, name='custom_name')
        assert metric.name == 'custom_name'

    def test_reset_clears_state(self):
        """Reset clears accumulated predictions and targets."""
        def func(p, t):
            return torch.tensor(0.0)

        metric = AccumMetric(func)
        metric.reset()
        assert metric.preds == []
        assert metric.targs == []

    def test_invert_args(self):
        """invert_arg=True passes (targs, preds) instead of (preds, targs)."""
        def ordered_metric(first, second):
            # Return 1.0 if first > second elementwise mean
            return (first > second).float().mean()

        metric_normal = AccumMetric(ordered_metric, invert_arg=False)
        metric_invert = AccumMetric(ordered_metric, invert_arg=True)

        preds = torch.tensor([10.0, 10.0])
        targs = torch.tensor([1.0, 1.0])

        result_normal = metric_normal(preds, targs)
        result_invert = metric_invert(preds, targs)

        # Normal: func(preds, targs) -> (10 > 1) = True -> 1.0
        assert abs(result_normal.item() - 1.0) < 1e-6
        # Inverted: func(targs, preds) -> (1 > 10) = False -> 0.0
        assert abs(result_invert.item()) < 1e-6


# ============================================================
# Tests for exp_rmspe
# ============================================================

class TestExpRMSPE:
    """Tests for the exp_rmspe metric."""

    def test_zero_error(self):
        """Identical predictions give exp_rmspe of 0."""
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = exp_rmspe(inp, targ)
        assert abs(result.item()) < 1e-6

    def test_known_value(self):
        """Known exp_rmspe calculation."""
        # inp = log(2), targ = log(2) -> exp(inp) = 2, exp(targ) = 2
        import math
        val = math.log(2.0)
        inp = torch.tensor([val, val])
        targ = torch.tensor([val, val])
        result = exp_rmspe(inp, targ)
        assert abs(result.item()) < 1e-5

    def test_non_zero(self):
        """Different values give non-zero exp_rmspe."""
        inp = torch.tensor([0.0, 1.0])
        targ = torch.tensor([1.0, 0.0])
        result = exp_rmspe(inp, targ)
        assert result.item() > 0


# ============================================================
# Tests for Perplexity
# ============================================================

class TestPerplexity:
    """Tests for the Perplexity metric class."""

    def test_perplexity_init(self):
        """Perplexity can be instantiated."""
        p = Perplexity()
        assert p.name == "perplexity"

    def test_perplexity_value_none_when_empty(self):
        """Perplexity returns None when no data accumulated."""
        p = Perplexity()
        p.reset()
        assert p.value is None


# ============================================================
# Tests for ActivationType
# ============================================================

class TestActivationType:
    """Tests for ActivationType enum-like class."""

    def test_activation_types_exist(self):
        """All activation types are defined."""
        assert ActivationType.No == 'no'
        assert ActivationType.Sigmoid == 'sigmoid'
        assert ActivationType.Softmax == 'softmax'
        assert ActivationType.BinarySoftmax == 'binarysoftmax'


# ============================================================
# Tests for skm_to_fastai
# ============================================================

class TestSkmToFastai:
    """Tests for the skm_to_fastai converter function."""

    def test_returns_accum_metric(self):
        """skm_to_fastai returns an AccumMetric instance."""
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score)
        assert isinstance(metric, AccumMetric)

    def test_sklearn_accuracy(self):
        """skm_to_fastai works with sklearn accuracy_score (non-class mode)."""
        import sklearn.metrics as skm
        # Use is_class=False to skip dim_argmax; pass pre-argmaxed predictions
        metric = skm_to_fastai(skm.accuracy_score, is_class=False)
        # Predictions already in label form (like regression outputs matching targets)
        inp = torch.tensor([0.0, 1.0, 2.0, 1.0])
        targ = torch.tensor([0.0, 1.0, 2.0, 1.0])
        result = metric(inp, targ)
        assert abs(result - 1.0) < 1e-6
