"""Tests for fastai.metrics module.

Covers: AccumMetric, accuracy, error_rate, top_k_accuracy, accuracy_multi,
foreground_acc, mse, mae, msle, rmse, exp_rmspe, Dice, DiceMulti,
JaccardCoeff, JaccardCoeffMulti, CorpusBLEUMetric, Perplexity, LossMetric,
LossMetrics, skm_to_fastai wrappers (F1Score, Precision, Recall,
BalancedAccuracy, CohenKappa, HammingLoss, Jaccard, MatthewsCorrCoef,
RocAucBinary, ExplainedVariance, R2Score, PearsonCorrCoef, SpearmanCorrCoef),
and multi-label metrics (F1ScoreMulti, PrecisionMulti, RecallMulti,
HammingLossMulti, JaccardMulti).
"""
import sys
import os
import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.metrics import (
    AccumMetric, ActivationType, skm_to_fastai, optim_metric,
    accuracy, error_rate, top_k_accuracy, accuracy_multi, foreground_acc,
    mse, mae, msle, rmse, exp_rmspe,
    Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti,
    CorpusBLEUMetric, Perplexity, perplexity, LossMetric, LossMetrics,
    F1Score, Precision, Recall, BalancedAccuracy, CohenKappa, HammingLoss,
    Jaccard, MatthewsCorrCoef, RocAucBinary,
    F1ScoreMulti, PrecisionMulti, RecallMulti, HammingLossMulti, JaccardMulti,
    ExplainedVariance, R2Score, PearsonCorrCoef, SpearmanCorrCoef,
    FBeta, FBetaMulti,
)


# ============================================================
# Tests for accuracy
# ============================================================

class TestAccuracy:
    """Tests for the accuracy metric function."""

    def test_perfect_predictions(self):
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targ = torch.tensor([1, 0, 1])
        result = accuracy(pred, targ)
        assert float(result) == 1.0

    def test_all_wrong_predictions(self):
        pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        targ = torch.tensor([1, 0, 1])
        result = accuracy(pred, targ)
        assert float(result) == 0.0

    def test_partial_predictions(self):
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
        targ = torch.tensor([1, 0, 1, 1])
        # Predictions: 1, 0, 0, 1 -> correct: 1,0,_,1 -> 3/4
        result = accuracy(pred, targ)
        assert abs(float(result) - 0.75) < 1e-5

    def test_multiclass(self):
        pred = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        targ = torch.tensor([2, 0, 1])
        result = accuracy(pred, targ)
        assert float(result) == 1.0

    def test_single_sample(self):
        pred = torch.tensor([[0.3, 0.7]])
        targ = torch.tensor([1])
        result = accuracy(pred, targ)
        assert float(result) == 1.0


# ============================================================
# Tests for error_rate
# ============================================================

class TestErrorRate:
    """Tests for the error_rate metric function."""

    def test_perfect_predictions_zero_error(self):
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targ = torch.tensor([1, 0])
        result = error_rate(pred, targ)
        assert float(result) == 0.0

    def test_all_wrong_predictions(self):
        pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        targ = torch.tensor([1, 0])
        result = error_rate(pred, targ)
        assert float(result) == 1.0

    def test_complementary_to_accuracy(self):
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
        targ = torch.tensor([1, 0, 1, 1])
        acc = accuracy(pred, targ)
        err = error_rate(pred, targ)
        assert abs(float(acc) + float(err) - 1.0) < 1e-6


# ============================================================
# Tests for top_k_accuracy
# ============================================================

class TestTopKAccuracy:
    """Tests for the top_k_accuracy metric function."""

    def test_top1_same_as_accuracy(self):
        pred = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        targ = torch.tensor([1, 0])
        result = top_k_accuracy(pred, targ, k=1)
        acc = accuracy(pred, targ)
        assert abs(float(result) - float(acc)) < 1e-6

    def test_top2_includes_second_best(self):
        # pred argmax gives class 2, second is class 1
        pred = torch.tensor([[0.1, 0.3, 0.6]])
        targ = torch.tensor([1])  # class 1 is in top 2
        result = top_k_accuracy(pred, targ, k=2)
        assert float(result) == 1.0

    def test_top_k_not_in_predictions(self):
        pred = torch.tensor([[0.8, 0.15, 0.05]])
        targ = torch.tensor([2])  # class 2 is last
        result = top_k_accuracy(pred, targ, k=1)
        assert float(result) == 0.0

    def test_top_k_all_correct_with_large_k(self):
        pred = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
        targ = torch.tensor([2, 1])
        # k=3 means all classes are in top-3
        result = top_k_accuracy(pred, targ, k=3)
        assert float(result) == 1.0

    def test_batch_mixed(self):
        pred = torch.tensor([
            [0.1, 0.2, 0.7],  # top 2: [2, 1]
            [0.5, 0.3, 0.2],  # top 2: [0, 1]
            [0.1, 0.8, 0.1],  # top 2: [1, 0] or [1, 2]
        ])
        targ = torch.tensor([1, 0, 2])
        result = top_k_accuracy(pred, targ, k=2)
        # Sample 0: targ=1 in top2 [2,1]? yes
        # Sample 1: targ=0 in top2 [0,1]? yes
        # Sample 2: targ=2 in top2 [1,?]? depends on tie-breaking
        # At least 2/3 should be correct
        assert float(result) >= 2.0 / 3.0 - 1e-6


# ============================================================
# Tests for accuracy_multi (multi-label)
# ============================================================

class TestAccuracyMulti:
    """Tests for multi-label accuracy."""

    def test_perfect_predictions(self):
        # After sigmoid, values > 0.5 are predicted positive
        inp = torch.tensor([[2.0, -2.0, 2.0], [-2.0, 2.0, -2.0]])
        targ = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert float(result) == 1.0

    def test_all_wrong(self):
        inp = torch.tensor([[-2.0, 2.0, -2.0]])
        targ = torch.tensor([[1.0, 0.0, 1.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=True)
        assert float(result) == 0.0

    def test_no_sigmoid(self):
        # When sigmoid=False, raw values are compared to threshold
        inp = torch.tensor([[0.8, 0.2, 0.9]])
        targ = torch.tensor([[1.0, 0.0, 1.0]])
        result = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert float(result) == 1.0

    def test_threshold_effect(self):
        inp = torch.tensor([[0.6, 0.4, 0.6]])
        targ = torch.tensor([[1.0, 0.0, 1.0]])
        # With thresh=0.5, predictions: [1, 0, 1] matches
        result_low = accuracy_multi(inp, targ, thresh=0.5, sigmoid=False)
        assert float(result_low) == 1.0
        # With thresh=0.7, predictions: [0, 0, 0]
        result_high = accuracy_multi(inp, targ, thresh=0.7, sigmoid=False)
        assert float(result_high) < 1.0


# ============================================================
# Tests for foreground_acc
# ============================================================

class TestForegroundAcc:
    """Tests for foreground accuracy in segmentation."""

    def test_perfect_foreground(self):
        # 3 classes, 4 pixels; class 0 is background
        pred = torch.zeros(1, 3, 4)
        pred[0, 1, 0] = 10.0  # predict class 1 at pos 0
        pred[0, 2, 1] = 10.0  # predict class 2 at pos 1
        pred[0, 0, 2] = 10.0  # predict class 0 at pos 2
        pred[0, 1, 3] = 10.0  # predict class 1 at pos 3
        targ = torch.tensor([[1, 2, 0, 1]])
        result = foreground_acc(pred, targ, bkg_idx=0, axis=1)
        assert float(result) == 1.0

    def test_wrong_foreground(self):
        pred = torch.zeros(1, 3, 4)
        pred[0, 2, 0] = 10.0  # predict class 2 at pos 0 (truth: class 1)
        pred[0, 1, 1] = 10.0  # predict class 1 at pos 1 (truth: class 2)
        pred[0, 0, 2] = 10.0  # background, ignored
        pred[0, 1, 3] = 10.0  # predict class 1 at pos 3 (truth: class 1)
        targ = torch.tensor([[1, 2, 0, 1]])
        # Foreground pixels: pos 0 (wrong), pos 1 (wrong), pos 3 (correct) -> 1/3
        result = foreground_acc(pred, targ, bkg_idx=0, axis=1)
        assert abs(float(result) - 1.0 / 3.0) < 1e-5

    def test_all_background(self):
        # If all pixels are background, there are no foreground pixels to evaluate
        pred = torch.zeros(1, 3, 4)
        pred[0, 0, :] = 10.0
        targ = torch.tensor([[0, 0, 0, 0]])
        # No foreground pixels, mean of empty tensor
        # This would cause a runtime warning/nan
        # The function returns mean of an empty selection
        result = foreground_acc(pred, targ, bkg_idx=0, axis=1)
        assert torch.isnan(result) or float(result) == 0.0


# ============================================================
# Tests for regression metrics: mse, mae, msle
# ============================================================

class TestMSE:
    """Tests for mean squared error."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(float(mse(inp, targ))) < 1e-7

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        # MSE = mean((0.5^2 + 0.5^2 + 0.5^2)) = 0.25
        assert abs(float(mse(inp, targ)) - 0.25) < 1e-5

    def test_symmetric(self):
        inp = torch.tensor([1.0, 2.0])
        targ = torch.tensor([3.0, 4.0])
        assert abs(float(mse(inp, targ)) - float(mse(targ, inp))) < 1e-6


class TestMAE:
    """Tests for mean absolute error."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(float(mae(inp, targ))) < 1e-7

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        # MAE = mean(0.5, 0.5, 0.5) = 0.5
        assert abs(float(mae(inp, targ)) - 0.5) < 1e-5

    def test_symmetric(self):
        inp = torch.tensor([1.0, 2.0])
        targ = torch.tensor([3.0, 4.0])
        assert abs(float(mae(inp, targ)) - float(mae(targ, inp))) < 1e-6


class TestMSLE:
    """Tests for mean squared logarithmic error."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        assert abs(float(msle(inp, targ))) < 1e-7

    def test_positive_error(self):
        inp = torch.tensor([1.0, 2.0])
        targ = torch.tensor([2.0, 3.0])
        result = msle(inp, targ)
        assert float(result) > 0

    def test_penalizes_underestimates_differently(self):
        # MSLE penalizes underestimates more than overestimates
        inp_over = torch.tensor([3.0])
        inp_under = torch.tensor([1.0])
        targ = torch.tensor([2.0])
        msle_over = msle(inp_over, targ)
        msle_under = msle(inp_under, targ)
        # For log-based loss, underestimates are penalized more
        assert float(msle_under) > float(msle_over)


# ============================================================
# Tests for rmse (AccumMetric instance)
# ============================================================

class TestRMSE:
    """Tests for root mean squared error."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(inp, targ)
        assert abs(float(result)) < 1e-6

    def test_known_value(self):
        inp = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([2.0, 3.0, 4.0])
        # MSE = 1.0, RMSE = 1.0
        result = rmse(inp, targ)
        assert abs(float(result) - 1.0) < 1e-5

    def test_sqrt_of_mse(self):
        inp = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targ = torch.tensor([1.5, 2.5, 3.5, 4.5])
        mse_val = float(mse(inp, targ))
        rmse_val = float(rmse(inp, targ))
        assert abs(rmse_val - math.sqrt(mse_val)) < 1e-5

    def test_has_name(self):
        assert rmse.name == '_rmse'


# ============================================================
# Tests for exp_rmspe (AccumMetric instance)
# ============================================================

class TestExpRMSPE:
    """Tests for exponential root mean square percentage error."""

    def test_zero_error(self):
        inp = torch.tensor([1.0, 2.0])
        targ = torch.tensor([1.0, 2.0])
        result = exp_rmspe(inp, targ)
        assert abs(float(result)) < 1e-6

    def test_positive_error(self):
        inp = torch.tensor([0.5, 1.0])
        targ = torch.tensor([1.0, 2.0])
        result = exp_rmspe(inp, targ)
        assert float(result) > 0


# ============================================================
# Tests for AccumMetric class
# ============================================================

class TestAccumMetric:
    """Tests for the AccumMetric class."""

    def test_basic_creation(self):
        def my_func(preds, targs):
            return (preds == targs).float().mean()
        metric = AccumMetric(my_func)
        assert metric.name == 'my_func'

    def test_custom_name(self):
        def my_func(preds, targs):
            return 0.0
        metric = AccumMetric(my_func, name='custom_name')
        assert metric.name == 'custom_name'

    def test_name_setter(self):
        def my_func(preds, targs):
            return 0.0
        metric = AccumMetric(my_func)
        metric.name = 'new_name'
        assert metric.name == 'new_name'

    def test_reset(self):
        def my_func(preds, targs):
            return 0.0
        metric = AccumMetric(my_func)
        metric.reset()
        assert metric.preds == []
        assert metric.targs == []

    def test_accumulate_stores_values(self):
        def my_func(preds, targs):
            return (preds == targs).float().mean()
        metric = AccumMetric(my_func, dim_argmax=None, flatten=True)
        metric.reset()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        metric.accum_values(preds, targs)
        assert len(metric.preds) == 1
        assert len(metric.targs) == 1

    def test_value_after_accumulation(self):
        def my_func(preds, targs):
            return (preds == targs).float().mean()
        metric = AccumMetric(my_func, dim_argmax=None, flatten=True)
        metric.reset()
        metric.accum_values(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
        assert float(metric.value) == 1.0

    def test_value_empty_returns_none(self):
        def my_func(preds, targs):
            return 0.0
        metric = AccumMetric(my_func)
        metric.reset()
        assert metric.value is None

    def test_invert_args(self):
        def ordered_func(a, b):
            return a.sum() - b.sum()
        metric_normal = AccumMetric(ordered_func, invert_arg=False, flatten=True)
        metric_invert = AccumMetric(ordered_func, invert_arg=True, flatten=True)
        preds = torch.tensor([3.0])
        targs = torch.tensor([1.0])
        val_normal = metric_normal(preds, targs)
        val_invert = metric_invert(preds, targs)
        # normal: func(preds, targs) = 3-1 = 2
        # invert: func(targs, preds) = 1-3 = -2
        assert abs(float(val_normal) - 2.0) < 1e-6
        assert abs(float(val_invert) - (-2.0)) < 1e-6

    def test_to_np(self):
        def np_func(preds, targs):
            assert isinstance(preds, np.ndarray)
            assert isinstance(targs, np.ndarray)
            return float(np.mean(preds == targs))
        metric = AccumMetric(np_func, to_np=True, flatten=True)
        preds = torch.tensor([1.0, 2.0, 3.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        result = metric(preds, targs)
        assert result == 1.0

    def test_direct_call(self):
        def my_func(preds, targs):
            return (preds == targs).float().mean()
        metric = AccumMetric(my_func, flatten=True)
        preds = torch.tensor([1.0, 2.0, 3.0])
        targs = torch.tensor([1.0, 2.0, 4.0])
        result = metric(preds, targs)
        assert abs(float(result) - 2.0 / 3.0) < 1e-5

    def test_accumulate_with_learn(self):
        """Test accumulate method with a mock learn object."""
        def my_func(preds, targs):
            return (preds == targs).float().mean()
        metric = AccumMetric(my_func, dim_argmax=-1, activation=ActivationType.No, flatten=True)
        metric.reset()

        # Simulate: pred is logits for 2 classes, dim_argmax=-1 applies argmax
        learn = SimpleNamespace(
            pred=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            y=torch.tensor([1, 0]),
            to_detach=lambda x, **kwargs: x
        )
        metric.accumulate(learn)
        assert float(metric.value) == 1.0

    def test_accumulate_with_sigmoid_activation(self):
        def my_func(preds, targs):
            return (preds > 0.5).float().eq(targs).float().mean()
        metric = AccumMetric(my_func, activation=ActivationType.Sigmoid, flatten=True)
        metric.reset()

        learn = SimpleNamespace(
            pred=torch.tensor([5.0, -5.0, 5.0]),  # sigmoid -> ~[1, 0, 1]
            y=torch.tensor([1.0, 0.0, 1.0]),
            to_detach=lambda x, **kwargs: x
        )
        metric.accumulate(learn)
        assert float(metric.value) == 1.0

    def test_accumulate_with_softmax_activation(self):
        def my_func(preds, targs):
            return float(preds.sum())
        # Softmax activation without flatten since softmax output shape differs from target
        metric = AccumMetric(my_func, activation=ActivationType.Softmax,
                             dim_argmax=-1, flatten=False)
        metric.reset()

        learn = SimpleNamespace(
            pred=torch.tensor([[1.0, 2.0, 3.0]]),
            y=torch.tensor([2]),
            to_detach=lambda x, **kwargs: x
        )
        metric.accumulate(learn)
        # After softmax, values should sum to 1 per sample
        assert abs(float(metric.value) - 1.0) < 1e-5

    def test_accumulate_with_thresh(self):
        def my_func(preds, targs):
            return preds.float().eq(targs).float().mean()
        metric = AccumMetric(my_func, activation=ActivationType.Sigmoid,
                             thresh=0.5, flatten=True)
        metric.reset()

        learn = SimpleNamespace(
            pred=torch.tensor([5.0, -5.0]),  # sigmoid -> [~1, ~0], thresh -> [True, False]
            y=torch.tensor([True, False]),
            to_detach=lambda x, **kwargs: x
        )
        metric.accumulate(learn)
        assert float(metric.value) == 1.0


# ============================================================
# Tests for skm_to_fastai
# ============================================================

class TestSkmToFastai:
    """Tests for sklearn metric conversion."""

    def test_basic_conversion(self):
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score)
        # Requires pre-argmaxed inputs due to dim_argmax
        preds = torch.tensor([0, 1, 1, 0])
        targs = torch.tensor([0, 1, 1, 0])
        result = metric(preds, targs)
        assert result == 1.0

    def test_with_threshold(self):
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score, thresh=0.5)
        # With threshold, activation is sigmoid by default
        # But in direct call mode, accum_values doesn't apply activation
        # We pass already-thresholded values
        preds = torch.tensor([True, False, True])
        targs = torch.tensor([1, 0, 1])
        result = metric(preds, targs)
        assert result == 1.0


# ============================================================
# Tests for single-label sklearn metrics
# ============================================================

class TestF1Score:
    """Tests for F1Score (single-label binary classification)."""

    def test_perfect_binary(self):
        f1 = F1Score()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert f1(preds, targs) == 1.0

    def test_zero_f1(self):
        f1 = F1Score()
        preds = torch.tensor([0, 0, 0, 0])
        targs = torch.tensor([1, 1, 1, 1])
        assert f1(preds, targs) == 0.0

    def test_partial_f1(self):
        f1 = F1Score()
        # TP=1, FP=1, FN=1, TN=1 -> P=1/2, R=1/2, F1=1/2
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = f1(preds, targs)
        assert abs(result - 0.5) < 1e-5


class TestPrecision:
    """Tests for Precision metric."""

    def test_perfect_precision(self):
        prec = Precision()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert prec(preds, targs) == 1.0

    def test_half_precision(self):
        prec = Precision()
        # Predict positive for 2, but only 1 is actually positive
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 0, 0])
        # TP=1, FP=1 -> Precision = 1/2
        assert abs(prec(preds, targs) - 0.5) < 1e-5


class TestRecall:
    """Tests for Recall metric."""

    def test_perfect_recall(self):
        rec = Recall()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert rec(preds, targs) == 1.0

    def test_half_recall(self):
        rec = Recall()
        # 2 actual positives, only 1 predicted correctly
        preds = torch.tensor([1, 0, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        # TP=1, FN=1 -> Recall = 1/2
        assert abs(rec(preds, targs) - 0.5) < 1e-5


class TestFBeta:
    """Tests for FBeta metric."""

    def test_fbeta_equals_f1_when_beta_1(self):
        f1 = F1Score()
        fb = FBeta(beta=1)
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert abs(f1(preds, targs) - fb(preds, targs)) < 1e-5

    def test_fbeta_with_beta_2(self):
        fb = FBeta(beta=2)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert fb(preds, targs) == 1.0


class TestBalancedAccuracy:
    """Tests for BalancedAccuracy."""

    def test_perfect(self):
        ba = BalancedAccuracy()
        preds = torch.tensor([0, 0, 1, 1])
        targs = torch.tensor([0, 0, 1, 1])
        assert ba(preds, targs) == 1.0

    def test_imbalanced_correct(self):
        ba = BalancedAccuracy()
        # 1 negative, 3 positives, all correct
        preds = torch.tensor([0, 1, 1, 1])
        targs = torch.tensor([0, 1, 1, 1])
        assert ba(preds, targs) == 1.0


class TestCohenKappa:
    """Tests for CohenKappa."""

    def test_perfect_agreement(self):
        ck = CohenKappa()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([0, 1, 0, 1])
        assert ck(preds, targs) == 1.0

    def test_no_agreement(self):
        ck = CohenKappa()
        preds = torch.tensor([1, 1, 1, 1])
        targs = torch.tensor([0, 0, 0, 0])
        assert ck(preds, targs) <= 0.0


class TestHammingLoss:
    """Tests for HammingLoss."""

    def test_perfect_zero_loss(self):
        hl = HammingLoss()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([0, 1, 0, 1])
        assert hl(preds, targs) == 0.0

    def test_all_wrong(self):
        hl = HammingLoss()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([0, 1, 0, 1])
        assert hl(preds, targs) == 1.0

    def test_half_wrong(self):
        hl = HammingLoss()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 0, 1])
        assert abs(hl(preds, targs) - 0.5) < 1e-5


class TestJaccard:
    """Tests for Jaccard (single-label)."""

    def test_perfect(self):
        jac = Jaccard()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        assert jac(preds, targs) == 1.0

    def test_no_overlap(self):
        jac = Jaccard()
        preds = torch.tensor([0, 0, 0, 0])
        targs = torch.tensor([1, 1, 1, 1])
        assert jac(preds, targs) == 0.0


class TestMatthewsCorrCoef:
    """Tests for Matthews Correlation Coefficient."""

    def test_perfect_correlation(self):
        mcc = MatthewsCorrCoef()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([0, 1, 0, 1])
        assert abs(mcc(preds, targs) - 1.0) < 1e-5

    def test_anti_correlation(self):
        mcc = MatthewsCorrCoef()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([0, 1, 0, 1])
        assert abs(mcc(preds, targs) - (-1.0)) < 1e-5


class TestRocAucBinary:
    """Tests for RocAucBinary."""

    def test_perfect_separation(self):
        roc = RocAucBinary()
        # After BinarySoftmax, second column scores are used
        # We pass pre-processed probabilities for direct call
        preds = torch.tensor([0.9, 0.8, 0.1, 0.2])
        targs = torch.tensor([1, 1, 0, 0])
        result = roc(preds, targs)
        assert result == 1.0

    def test_random_predictions(self):
        roc = RocAucBinary()
        # Predictions that don't discriminate
        preds = torch.tensor([0.5, 0.5, 0.5, 0.5])
        targs = torch.tensor([1, 0, 1, 0])
        result = roc(preds, targs)
        assert abs(result - 0.5) < 1e-5


# ============================================================
# Tests for multi-label sklearn metrics
# ============================================================

class TestF1ScoreMulti:
    """Tests for F1Score in multi-label setting."""

    def test_perfect(self):
        f1m = F1ScoreMulti(sigmoid=False)
        preds = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = f1m(preds, targs)
        assert result == 1.0

    def test_all_wrong(self):
        f1m = F1ScoreMulti(sigmoid=False)
        preds = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = f1m(preds, targs)
        assert result == 0.0


class TestPrecisionMulti:
    """Tests for Precision in multi-label setting."""

    def test_perfect(self):
        pm = PrecisionMulti(sigmoid=False)
        # Use multiple samples so each label has both predictions and true values
        preds = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = pm(preds, targs)
        assert result == 1.0


class TestRecallMulti:
    """Tests for Recall in multi-label setting."""

    def test_perfect(self):
        rm = RecallMulti(sigmoid=False)
        # Use multiple samples so each label has true values
        preds = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = rm(preds, targs)
        assert result == 1.0


class TestHammingLossMulti:
    """Tests for HammingLoss in multi-label setting."""

    def test_perfect_zero_loss(self):
        hlm = HammingLossMulti(sigmoid=False)
        preds = torch.tensor([[1.0, 0.0, 1.0]])
        targs = torch.tensor([[1, 0, 1]])
        result = hlm(preds, targs)
        assert result == 0.0

    def test_all_wrong(self):
        hlm = HammingLossMulti(sigmoid=False)
        preds = torch.tensor([[0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1]])
        result = hlm(preds, targs)
        assert result == 1.0


class TestJaccardMulti:
    """Tests for Jaccard in multi-label setting."""

    def test_perfect(self):
        jm = JaccardMulti(sigmoid=False)
        preds = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = jm(preds, targs)
        assert result == 1.0


class TestFBetaMulti:
    """Tests for FBetaMulti."""

    def test_perfect(self):
        fbm = FBetaMulti(beta=1, sigmoid=False)
        # Use multiple samples so each label has both predictions and true values
        preds = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        targs = torch.tensor([[1, 0, 1], [0, 1, 0]])
        result = fbm(preds, targs)
        assert result == 1.0


# ============================================================
# Tests for regression sklearn metrics
# ============================================================

class TestExplainedVariance:
    """Tests for ExplainedVariance."""

    def test_perfect(self):
        ev = ExplainedVariance()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        assert ev(preds, targs) == 1.0

    def test_constant_offset(self):
        ev = ExplainedVariance()
        # Predictions off by a constant have explained variance < 1
        preds = torch.tensor([2.0, 3.0, 4.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        result = ev(preds, targs)
        # Explained variance considers residual variance vs target variance
        # Residual = preds - targs = [1,1,1], var=0 -> EV = 1.0
        assert abs(result - 1.0) < 1e-5


class TestR2Score:
    """Tests for R2 Score."""

    def test_perfect(self):
        r2 = R2Score()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        assert r2(preds, targs) == 1.0

    def test_negative_r2(self):
        r2 = R2Score()
        # Predictions worse than just predicting the mean
        preds = torch.tensor([10.0, 10.0, 10.0])
        targs = torch.tensor([1.0, 2.0, 3.0])
        result = r2(preds, targs)
        assert result < 0


class TestPearsonCorrCoef:
    """Tests for Pearson Correlation Coefficient."""

    def test_perfect_positive(self):
        pcc = PearsonCorrCoef(to_np=True)
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pcc(preds, targs)
        assert abs(result - 1.0) < 1e-5

    def test_perfect_negative(self):
        pcc = PearsonCorrCoef(to_np=True)
        preds = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pcc(preds, targs)
        assert abs(result - (-1.0)) < 1e-5

    def test_no_correlation(self):
        pcc = PearsonCorrCoef(to_np=True)
        # Constant predictions have no correlation
        preds = torch.tensor([1.0, 1.0, 1.0, 1.0])
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = pcc(preds, targs)
        # Pearson of constant is nan or 0
        assert result != result or abs(result) < 1e-5  # nan or 0


class TestSpearmanCorrCoef:
    """Tests for Spearman Correlation Coefficient."""

    def test_perfect_positive(self):
        scc = SpearmanCorrCoef(to_np=True)
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scc(preds, targs)
        assert abs(result - 1.0) < 1e-5

    def test_perfect_negative(self):
        scc = SpearmanCorrCoef(to_np=True)
        preds = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scc(preds, targs)
        assert abs(result - (-1.0)) < 1e-5

    def test_monotonic_nonlinear(self):
        """Spearman should be 1.0 for any monotonic relationship."""
        scc = SpearmanCorrCoef(to_np=True)
        preds = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])  # x^2
        targs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scc(preds, targs)
        assert abs(result - 1.0) < 1e-5


# ============================================================
# Tests for Dice metric (segmentation)
# ============================================================

class TestDice:
    """Tests for binary Dice coefficient."""

    def _make_learn(self, pred, targ):
        return SimpleNamespace(pred=pred, y=targ)

    def test_perfect_prediction(self):
        dice = Dice(axis=1)
        dice.reset()
        # 2 classes, batch=1, 4 pixels
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 0, 2] = 10.0
        pred[0, 0, 3] = 10.0
        targ = torch.tensor([[1, 1, 0, 0]])
        dice.accumulate(self._make_learn(pred, targ))
        assert dice.value == 1.0

    def test_no_overlap(self):
        dice = Dice(axis=1)
        dice.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 0, :] = 10.0  # predict all class 0
        targ = torch.tensor([[1, 1, 1, 1]])  # all class 1
        dice.accumulate(self._make_learn(pred, targ))
        # inter=0, union=4, dice=0/4=0
        assert dice.value == 0.0

    def test_partial_overlap(self):
        dice = Dice(axis=1)
        dice.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0  # predict class 1 at pos 0
        pred[0, 1, 1] = 10.0  # predict class 1 at pos 1
        pred[0, 0, 2] = 10.0  # predict class 0 at pos 2
        pred[0, 0, 3] = 10.0  # predict class 0 at pos 3
        targ = torch.tensor([[1, 0, 1, 0]])
        # Predicted 1s: {0, 1}, True 1s: {0, 2}
        # inter = 1 (pos 0), union = pred_1_count + targ_1_count = 2 + 2 = 4
        # dice = 2*1/4 = 0.5
        dice.accumulate(self._make_learn(pred, targ))
        assert abs(dice.value - 0.5) < 1e-5

    def test_multi_batch_accumulation(self):
        dice = Dice(axis=1)
        dice.reset()
        # Batch 1: perfect overlap
        pred1 = torch.zeros(1, 2, 2)
        pred1[0, 1, 0] = 10.0
        pred1[0, 1, 1] = 10.0
        targ1 = torch.tensor([[1, 1]])
        dice.accumulate(self._make_learn(pred1, targ1))
        # inter=2, union=4

        # Batch 2: no overlap
        pred2 = torch.zeros(1, 2, 2)
        pred2[0, 0, 0] = 10.0
        pred2[0, 0, 1] = 10.0
        targ2 = torch.tensor([[1, 1]])
        dice.accumulate(self._make_learn(pred2, targ2))
        # inter=2+0=2, union=4+2=6 (pred 0+0=0 for class 1, targ=2)
        # dice = 2*2/6 = 0.6667
        assert abs(dice.value - 2.0 / 3.0) < 1e-5

    def test_no_positives_returns_none(self):
        dice = Dice(axis=1)
        dice.reset()
        pred = torch.zeros(1, 2, 2)
        pred[0, 0, :] = 10.0  # all class 0
        targ = torch.tensor([[0, 0]])  # all class 0
        dice.accumulate(self._make_learn(pred, targ))
        # inter=0, union=0 -> returns None
        assert dice.value is None


# ============================================================
# Tests for DiceMulti metric (segmentation)
# ============================================================

class TestDiceMulti:
    """Tests for multiclass Dice metric."""

    def _make_learn(self, pred, targ):
        return SimpleNamespace(pred=pred, y=targ)

    def test_perfect_prediction(self):
        dice = DiceMulti(axis=1)
        dice.reset()
        # 3 classes, batch=1, 3 pixels
        pred = torch.zeros(1, 3, 3)
        pred[0, 0, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 2, 2] = 10.0
        targ = torch.tensor([[0, 1, 2]])
        dice.accumulate(self._make_learn(pred, targ))
        assert abs(dice.value - 1.0) < 1e-5

    def test_all_same_class(self):
        dice = DiceMulti(axis=1)
        dice.reset()
        pred = torch.zeros(1, 3, 4)
        pred[0, 0, :] = 10.0  # predict all class 0
        targ = torch.tensor([[0, 0, 0, 0]])
        dice.accumulate(self._make_learn(pred, targ))
        # Class 0: dice=1 (perfect), Class 1: 0/0 -> nan, Class 2: 0/0 -> nan
        # nanmean of [1.0, nan, nan] = 1.0
        assert abs(dice.value - 1.0) < 1e-5


# ============================================================
# Tests for JaccardCoeff
# ============================================================

class TestJaccardCoeff:
    """Tests for binary Jaccard coefficient (IoU)."""

    def _make_learn(self, pred, targ):
        return SimpleNamespace(pred=pred, y=targ)

    def test_perfect(self):
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 0, 2] = 10.0
        pred[0, 0, 3] = 10.0
        targ = torch.tensor([[1, 1, 0, 0]])
        jac.accumulate(self._make_learn(pred, targ))
        # inter=2, union=4, jaccard = 2/(4-2) = 1.0
        assert jac.value == 1.0

    def test_partial_overlap(self):
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 4)
        pred[0, 1, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 0, 2] = 10.0
        pred[0, 0, 3] = 10.0
        targ = torch.tensor([[1, 0, 1, 0]])
        # pred 1s: {0,1}, targ 1s: {0,2}
        # inter=1, union=2+2=4, jaccard=1/(4-1)=1/3
        jac.accumulate(self._make_learn(pred, targ))
        assert abs(jac.value - 1.0 / 3.0) < 1e-5

    def test_no_positives_returns_none(self):
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 2)
        pred[0, 0, :] = 10.0
        targ = torch.tensor([[0, 0]])
        jac.accumulate(self._make_learn(pred, targ))
        assert jac.value is None


# ============================================================
# Tests for JaccardCoeffMulti
# ============================================================

class TestJaccardCoeffMulti:
    """Tests for multiclass Jaccard coefficient (mIoU)."""

    def _make_learn(self, pred, targ):
        return SimpleNamespace(pred=pred, y=targ)

    def test_perfect(self):
        jac = JaccardCoeffMulti(axis=1)
        jac.reset()
        pred = torch.zeros(1, 3, 3)
        pred[0, 0, 0] = 10.0
        pred[0, 1, 1] = 10.0
        pred[0, 2, 2] = 10.0
        targ = torch.tensor([[0, 1, 2]])
        jac.accumulate(self._make_learn(pred, targ))
        assert abs(jac.value - 1.0) < 1e-5


# ============================================================
# Tests for CorpusBLEUMetric
# ============================================================

class TestCorpusBLEUMetric:
    """Tests for corpus-level BLEU score."""

    def _make_learn(self, pred, targ, training=False):
        return SimpleNamespace(pred=pred, y=targ, training=training)

    def test_perfect_bleu(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        batch_size, seq_len, vocab_sz = 2, 10, 100
        targ = torch.randint(0, vocab_sz, (batch_size, seq_len))
        pred = torch.zeros(batch_size, seq_len, vocab_sz)
        for i in range(batch_size):
            for j in range(seq_len):
                pred[i, j, targ[i, j]] = 10.0
        bleu.accumulate(self._make_learn(pred, targ))
        assert bleu.value == 1.0

    def test_random_predictions_low_bleu(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        batch_size, seq_len, vocab_sz = 4, 20, 100
        torch.manual_seed(42)
        targ = torch.randint(0, vocab_sz, (batch_size, seq_len))
        pred = torch.randn(batch_size, seq_len, vocab_sz)
        bleu.accumulate(self._make_learn(pred, targ))
        # Random predictions should have very low BLEU
        assert bleu.value < 0.2

    def test_training_mode_skipped(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        targ = torch.randint(0, 100, (2, 10))
        pred = torch.randn(2, 10, 100)
        bleu.accumulate(self._make_learn(pred, targ, training=True))
        # When training=True, accumulate returns None and does not update counts
        assert bleu.pred_len == 0
        assert bleu.targ_len == 0

    def test_reset(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        assert bleu.pred_len == 0
        assert bleu.targ_len == 0
        assert bleu.corrects == [0, 0, 0, 0]
        assert bleu.counts == [0, 0, 0, 0]

    def test_empty_returns_none(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        bleu.reset()
        # With no accumulation, counts are all zero so value is None
        assert bleu.value is None

    def test_ngram_equality(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        ng1 = bleu.NGram([1, 2, 3], max_n=100)
        ng2 = bleu.NGram([1, 2, 3], max_n=100)
        ng3 = bleu.NGram([1, 2, 4], max_n=100)
        assert ng1 == ng2
        assert ng1 != ng3

    def test_ngram_hash(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        ng1 = bleu.NGram([1, 2, 3], max_n=100)
        ng2 = bleu.NGram([1, 2, 3], max_n=100)
        assert hash(ng1) == hash(ng2)

    def test_get_grams_unigram(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        x = [1, 2, 3, 4]
        grams = bleu.get_grams(x, n=1)
        assert grams == x  # unigrams are just the tokens themselves

    def test_get_grams_bigram(self):
        bleu = CorpusBLEUMetric(vocab_sz=100, axis=-1)
        x = [1, 2, 3, 4]
        grams = bleu.get_grams(x, n=2, max_n=100)
        assert len(grams) == 3  # 4-2+1 = 3 bigrams


# ============================================================
# Tests for Perplexity
# ============================================================

class TestPerplexity:
    """Tests for Perplexity metric."""

    def _make_learn(self, loss, batch_size):
        return SimpleNamespace(
            loss=torch.tensor(loss),
            yb=(torch.randn(batch_size, 10),),
            to_detach=lambda x, **kwargs: x
        )

    def test_basic_perplexity(self):
        perp = Perplexity()
        perp.reset()
        perp.accumulate(self._make_learn(2.0, 4))
        expected = math.exp(2.0)
        assert abs(float(perp.value) - expected) < 1e-4

    def test_zero_loss(self):
        perp = Perplexity()
        perp.reset()
        perp.accumulate(self._make_learn(0.0, 4))
        # exp(0) = 1.0
        assert abs(float(perp.value) - 1.0) < 1e-5

    def test_name(self):
        perp = Perplexity()
        assert perp.name == 'perplexity'

    def test_no_accumulation_returns_none(self):
        perp = Perplexity()
        perp.reset()
        assert perp.value is None

    def test_multiple_batches(self):
        perp = Perplexity()
        perp.reset()
        # Batch 1: loss=2.0, bs=4 -> total=8, count=4
        perp.accumulate(self._make_learn(2.0, 4))
        # Batch 2: loss=4.0, bs=4 -> total=8+16=24, count=8
        perp.accumulate(self._make_learn(4.0, 4))
        # Average loss = 24/8 = 3.0, perplexity = exp(3.0)
        expected = math.exp(3.0)
        assert abs(float(perp.value) - expected) < 1e-3

    def test_global_perplexity_instance(self):
        """Test the module-level perplexity instance exists and has correct name."""
        assert perplexity.name == 'perplexity'


# ============================================================
# Tests for LossMetric and LossMetrics
# ============================================================

class TestLossMetric:
    """Tests for LossMetric."""

    def test_basic(self):
        lm = LossMetric('reconstruction_loss')
        assert lm.name == 'reconstruction_loss'

    def test_custom_name(self):
        lm = LossMetric('recon', nm='reconstruction')
        assert lm.name == 'reconstruction'

    def test_accumulate(self):
        lm = LossMetric('my_loss')
        lm.reset()
        learn = SimpleNamespace(
            loss_func=SimpleNamespace(my_loss=torch.tensor(2.5)),
            yb=(torch.randn(8, 5),),
            to_detach=lambda x, **kwargs: x
        )
        lm.accumulate(learn)
        # total = 2.5 * 8 = 20, count = 8, value = 20/8 = 2.5
        assert abs(float(lm.value) - 2.5) < 1e-5


class TestLossMetrics:
    """Tests for LossMetrics factory function."""

    def test_from_string(self):
        lms = LossMetrics('loss1,loss2,loss3')
        assert len(lms) == 3
        assert lms[0].name == 'loss1'
        assert lms[1].name == 'loss2'
        assert lms[2].name == 'loss3'

    def test_from_list(self):
        lms = LossMetrics(['a', 'b'])
        assert len(lms) == 2
        assert lms[0].name == 'a'
        assert lms[1].name == 'b'

    def test_custom_names(self):
        lms = LossMetrics('attr1,attr2', nms='Name1,Name2')
        assert lms[0].name == 'Name1'
        assert lms[1].name == 'Name2'

    def test_single_metric(self):
        lms = LossMetrics('single_loss')
        assert len(lms) == 1
        assert lms[0].name == 'single_loss'


# ============================================================
# Tests for ActivationType
# ============================================================

class TestActivationType:
    """Tests for ActivationType enum-like class."""

    def test_values(self):
        assert ActivationType.No == 'no'
        assert ActivationType.Sigmoid == 'sigmoid'
        assert ActivationType.Softmax == 'softmax'
        assert ActivationType.BinarySoftmax == 'binarysoftmax'
