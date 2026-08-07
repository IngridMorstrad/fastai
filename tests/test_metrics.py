"""Tests for fastai.metrics module.

Covers: accuracy, error_rate, top_k_accuracy, AccumMetric, skm_to_fastai metrics
(F1Score, Precision, Recall, BalancedAccuracy, CohenKappa, HammingLoss, Jaccard,
MatthewsCorrCoef), multi-label metrics (accuracy_multi, F1ScoreMulti, PrecisionMulti,
RecallMulti), regression metrics (mse, mae, msle, rmse, exp_rmspe, R2Score,
ExplainedVariance, PearsonCorrCoef, SpearmanCorrCoef), segmentation metrics
(foreground_acc, Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti), and
sequence metrics (CorpusBLEUMetric, Perplexity, LossMetric, LossMetrics).
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.metrics import (
    accuracy, error_rate, top_k_accuracy,
    AccumMetric, skm_to_fastai,
    F1Score, Precision, Recall, BalancedAccuracy, CohenKappa, HammingLoss,
    Jaccard, MatthewsCorrCoef, FBeta,
    accuracy_multi, F1ScoreMulti, PrecisionMulti, RecallMulti, HammingLossMulti,
    JaccardMulti,
    mse, mae, msle, rmse, exp_rmspe,
    R2Score, ExplainedVariance, PearsonCorrCoef, SpearmanCorrCoef,
    foreground_acc, Dice, DiceMulti, JaccardCoeff, JaccardCoeffMulti,
    CorpusBLEUMetric, Perplexity, LossMetric, LossMetrics,
)


# ============================================================
# Tests for accuracy
# ============================================================

class TestAccuracy:
    """Tests for the accuracy function."""

    def test_perfect_predictions(self):
        """All predictions match targets."""
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targs = torch.tensor([1, 0, 1])
        result = accuracy(preds, targs)
        assert abs(float(result) - 1.0) < 1e-6

    def test_all_wrong(self):
        """No predictions match targets."""
        preds = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]])
        targs = torch.tensor([1, 0, 1])
        result = accuracy(preds, targs)
        assert abs(float(result) - 0.0) < 1e-6

    def test_partial_accuracy(self):
        """Some predictions match, some do not."""
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]])
        targs = torch.tensor([1, 0, 1])  # 2 out of 3 correct
        result = accuracy(preds, targs)
        assert abs(float(result) - 2.0 / 3.0) < 1e-6

    def test_multiclass(self):
        """Accuracy with more than 2 classes."""
        preds = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
        ])
        targs = torch.tensor([0, 1, 2, 2])  # 3 correct out of 4
        result = accuracy(preds, targs)
        assert abs(float(result) - 0.75) < 1e-6

    def test_single_sample(self):
        """Works with a single sample."""
        preds = torch.tensor([[0.3, 0.7]])
        targs = torch.tensor([1])
        result = accuracy(preds, targs)
        assert abs(float(result) - 1.0) < 1e-6

    def test_large_batch(self):
        """Works with a larger batch."""
        torch.manual_seed(42)
        num_classes = 10
        batch_size = 100
        preds = torch.randn(batch_size, num_classes)
        targs = preds.argmax(dim=-1)  # perfect predictions
        result = accuracy(preds, targs)
        assert abs(float(result) - 1.0) < 1e-6


# ============================================================
# Tests for error_rate
# ============================================================

class TestErrorRate:
    """Tests for the error_rate function."""

    def test_perfect_predictions(self):
        """Error rate should be 0 for perfect predictions."""
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targs = torch.tensor([1, 0])
        result = error_rate(preds, targs)
        assert abs(float(result) - 0.0) < 1e-6

    def test_all_wrong(self):
        """Error rate should be 1 when all wrong."""
        preds = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targs = torch.tensor([1, 0])
        result = error_rate(preds, targs)
        assert abs(float(result) - 1.0) < 1e-6

    def test_complementary_to_accuracy(self):
        """error_rate + accuracy should equal 1."""
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
        targs = torch.tensor([1, 0, 0])
        acc = float(accuracy(preds, targs))
        err = float(error_rate(preds, targs))
        assert abs(acc + err - 1.0) < 1e-6


# ============================================================
# Tests for top_k_accuracy
# ============================================================

class TestTopKAccuracy:
    """Tests for top_k_accuracy."""

    def test_top1_equals_accuracy(self):
        """top_k_accuracy with k=1 should match accuracy."""
        preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        targs = torch.tensor([1, 0, 0])
        top1 = float(top_k_accuracy(preds, targs, k=1))
        acc = float(accuracy(preds, targs))
        assert abs(top1 - acc) < 1e-6

    def test_top2_binary(self):
        """With 2 classes, top-2 should always be 1.0."""
        preds = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targs = torch.tensor([1, 0])
        result = top_k_accuracy(preds, targs, k=2)
        assert abs(float(result) - 1.0) < 1e-6

    def test_top2_multiclass(self):
        """Top-2 accuracy in a 3-class problem."""
        preds = torch.tensor([
            [0.6, 0.3, 0.1],  # top-2: [0, 1]
            [0.1, 0.6, 0.3],  # top-2: [1, 2]
            [0.1, 0.3, 0.6],  # top-2: [2, 1]
        ])
        targs = torch.tensor([0, 2, 1])  # All in top-2
        result = top_k_accuracy(preds, targs, k=2)
        assert abs(float(result) - 1.0) < 1e-6

    def test_top3_with_miss(self):
        """Verify correct behavior when target is not in top-k."""
        preds = torch.tensor([
            [0.4, 0.3, 0.2, 0.1],  # top-2: [0, 1] -> target=3, miss
            [0.1, 0.4, 0.3, 0.2],  # top-2: [1, 2] -> target=1, hit
        ])
        targs = torch.tensor([3, 1])
        result = top_k_accuracy(preds, targs, k=2)
        assert abs(float(result) - 0.5) < 1e-6


# ============================================================
# Tests for AccumMetric
# ============================================================

class TestAccumMetric:
    """Tests for the AccumMetric class."""

    def test_basic_metric(self):
        """AccumMetric with a simple function."""
        def simple_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_acc)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 1])
        result = metric(preds, targs)
        assert abs(float(result) - 0.75) < 1e-6

    def test_reset(self):
        """Reset clears accumulated values."""
        def simple_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_acc)
        metric.reset()
        assert metric.preds == []
        assert metric.targs == []

    def test_name_from_function(self):
        """Name should be derived from the function."""
        def my_metric(preds, targs):
            return 0.0

        metric = AccumMetric(my_metric)
        assert metric.name == 'my_metric'

    def test_custom_name(self):
        """Custom name should override function name."""
        def my_metric(preds, targs):
            return 0.0

        metric = AccumMetric(my_metric, name='custom_name')
        assert metric.name == 'custom_name'

    def test_name_setter(self):
        """Name can be set after creation."""
        def my_metric(preds, targs):
            return 0.0

        metric = AccumMetric(my_metric)
        metric.name = 'new_name'
        assert metric.name == 'new_name'

    def test_value_empty(self):
        """Value should be None when no data has been accumulated."""
        def my_metric(preds, targs):
            return 0.0

        metric = AccumMetric(my_metric)
        metric.reset()
        assert metric.value is None

    def test_invert_arg(self):
        """invert_arg should swap preds and targs when calling func."""
        def asymmetric(a, b):
            return float(a.sum() - b.sum())

        metric_normal = AccumMetric(asymmetric, invert_arg=False)
        metric_invert = AccumMetric(asymmetric, invert_arg=True)

        preds = torch.tensor([3.0, 4.0])
        targs = torch.tensor([1.0, 2.0])

        result_normal = metric_normal(preds, targs)
        result_invert = metric_invert(preds, targs)

        # normal: func(preds, targs) = 7 - 3 = 4
        # invert: func(targs, preds) = 3 - 7 = -4
        assert abs(float(result_normal) - 4.0) < 1e-6
        assert abs(float(result_invert) - (-4.0)) < 1e-6

    def test_to_np(self):
        """to_np should convert tensors to numpy before calling func."""
        def numpy_func(preds, targs):
            assert isinstance(preds, np.ndarray)
            assert isinstance(targs, np.ndarray)
            return float(np.mean(preds == targs))

        metric = AccumMetric(numpy_func, to_np=True)
        preds = torch.tensor([1, 0, 1])
        targs = torch.tensor([1, 0, 0])
        result = metric(preds, targs)
        assert abs(result - 2.0 / 3.0) < 1e-6

    def test_accumulate_multiple_batches(self):
        """Accumulate stores data across multiple calls."""
        def simple_acc(preds, targs):
            return (preds == targs).float().mean()

        metric = AccumMetric(simple_acc)
        metric.reset()

        # First batch: 2/2 correct
        preds1 = torch.tensor([1, 0])
        targs1 = torch.tensor([1, 0])
        metric.accum_values(preds1, targs1)

        # Second batch: 0/2 correct
        preds2 = torch.tensor([0, 1])
        targs2 = torch.tensor([1, 0])
        metric.accum_values(preds2, targs2)

        # Overall: 2/4 = 0.5
        assert abs(float(metric.value) - 0.5) < 1e-6


# ============================================================
# Tests for sklearn-based single-label classification metrics
# ============================================================

class TestF1Score:
    """Tests for F1Score metric."""

    def test_perfect_binary(self):
        """Perfect binary classification."""
        f1 = F1Score()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = f1(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_all_wrong_binary(self):
        """All wrong binary classification."""
        f1 = F1Score()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([1, 0, 1, 0])
        result = f1(preds, targs)
        assert abs(result - 0.0) < 1e-6

    def test_partial_binary(self):
        """Partial correct binary classification."""
        f1 = F1Score()
        # pred=[1,1,0,0], targ=[1,0,1,0]
        # TP=1, FP=1, FN=1, TN=1
        # precision=1/2, recall=1/2, f1=0.5
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = f1(preds, targs)
        assert abs(result - 0.5) < 1e-6

    def test_name(self):
        """Name should be f1_score."""
        f1 = F1Score()
        assert f1.name == 'f1_score'


class TestPrecision:
    """Tests for Precision metric."""

    def test_perfect(self):
        """Perfect predictions give precision=1."""
        prec = Precision()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = prec(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_no_true_positives(self):
        """When all positive predictions are wrong, precision=0."""
        prec = Precision()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([0, 0, 1, 1])
        result = prec(preds, targs)
        assert abs(result - 0.0) < 1e-6

    def test_half_precision(self):
        """Half of positive predictions are correct."""
        prec = Precision()
        # pred=[1,1,0,0], targ=[1,0,0,0] -> TP=1, FP=1 -> precision=0.5
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 0, 0])
        result = prec(preds, targs)
        assert abs(result - 0.5) < 1e-6


class TestRecall:
    """Tests for Recall metric."""

    def test_perfect(self):
        """Perfect predictions give recall=1."""
        rec = Recall()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = rec(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_missed_positives(self):
        """When all positives are missed, recall=0."""
        rec = Recall()
        preds = torch.tensor([0, 0, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = rec(preds, targs)
        assert abs(result - 0.0) < 1e-6

    def test_half_recall(self):
        """Half of true positives are detected."""
        rec = Recall()
        # pred=[1,0,0,0], targ=[1,1,0,0] -> TP=1, FN=1 -> recall=0.5
        preds = torch.tensor([1, 0, 0, 0])
        targs = torch.tensor([1, 1, 0, 0])
        result = rec(preds, targs)
        assert abs(result - 0.5) < 1e-6


class TestBalancedAccuracy:
    """Tests for BalancedAccuracy metric."""

    def test_perfect(self):
        """Perfect predictions."""
        ba = BalancedAccuracy()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = ba(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_imbalanced(self):
        """Balanced accuracy accounts for class imbalance."""
        ba = BalancedAccuracy()
        # 3 negatives correct, 0 positives correct
        # recall class 0 = 3/3 = 1, recall class 1 = 0/1 = 0
        # balanced acc = (1+0)/2 = 0.5
        preds = torch.tensor([0, 0, 0, 0])
        targs = torch.tensor([0, 0, 0, 1])
        result = ba(preds, targs)
        assert abs(result - 0.5) < 1e-6


class TestCohenKappa:
    """Tests for CohenKappa metric."""

    def test_perfect_agreement(self):
        """Perfect agreement gives kappa=1."""
        ck = CohenKappa()
        preds = torch.tensor([0, 1, 0, 1, 0])
        targs = torch.tensor([0, 1, 0, 1, 0])
        result = ck(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_random_agreement(self):
        """Random agreement gives kappa near 0."""
        ck = CohenKappa()
        # Predictions are shifted by 1 from targets in a way that reduces kappa
        preds = torch.tensor([1, 0, 1, 0, 1, 0])
        targs = torch.tensor([0, 1, 0, 1, 0, 1])
        result = ck(preds, targs)
        assert result < 0.1  # Should be negative or near zero


class TestHammingLoss:
    """Tests for HammingLoss metric."""

    def test_zero_loss(self):
        """Zero hamming loss for perfect predictions."""
        hl = HammingLoss()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([0, 1, 0, 1])
        result = hl(preds, targs)
        assert abs(result - 0.0) < 1e-6

    def test_half_loss(self):
        """Half wrong gives hamming loss of 0.5."""
        hl = HammingLoss()
        preds = torch.tensor([0, 1, 1, 0])
        targs = torch.tensor([0, 1, 0, 1])
        result = hl(preds, targs)
        assert abs(result - 0.5) < 1e-6


class TestJaccard:
    """Tests for Jaccard score metric."""

    def test_perfect(self):
        """Perfect predictions give Jaccard=1."""
        jac = Jaccard()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = jac(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_partial(self):
        """Partial overlap."""
        jac = Jaccard()
        # pred=[1,1,0,0], targ=[1,0,1,0]
        # TP=1, FP=1, FN=1 -> jaccard = 1/(1+1+1) = 1/3
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = jac(preds, targs)
        assert abs(result - 1.0 / 3.0) < 1e-6


class TestMatthewsCorrCoef:
    """Tests for MatthewsCorrCoef metric."""

    def test_perfect(self):
        """Perfect correlation gives MCC=1."""
        mcc = MatthewsCorrCoef()
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = mcc(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_inverse(self):
        """Perfectly inverse predictions give MCC=-1."""
        mcc = MatthewsCorrCoef()
        preds = torch.tensor([0, 1, 0, 1])
        targs = torch.tensor([1, 0, 1, 0])
        result = mcc(preds, targs)
        assert abs(result - (-1.0)) < 1e-6


class TestFBeta:
    """Tests for FBeta metric."""

    def test_fbeta_1_equals_f1(self):
        """FBeta with beta=1 should equal F1Score."""
        fbeta = FBeta(beta=1)
        f1 = F1Score()
        preds = torch.tensor([1, 1, 0, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result_fbeta = fbeta(preds, targs)
        result_f1 = f1(preds, targs)
        assert abs(result_fbeta - result_f1) < 1e-6

    def test_high_beta_favors_recall(self):
        """Higher beta emphasizes recall more."""
        # pred=[1,0,0,0], targ=[1,1,1,0]
        # precision=1/1=1, recall=1/3
        # F1 = 2*(1*1/3)/(1+1/3) = 0.5
        # F2 (beta=2): (1+4)*p*r / (4*p + r) = 5*1*(1/3) / (4*1 + 1/3) = 5/3 / (13/3) = 5/13
        fbeta_low = FBeta(beta=0.5)
        fbeta_high = FBeta(beta=2)
        preds = torch.tensor([1, 0, 0, 0])
        targs = torch.tensor([1, 1, 1, 0])
        # High recall but low precision: recall=1/3, precision=1.0
        result_low = fbeta_low(preds, targs)
        result_high = fbeta_high(preds, targs)
        # With low precision and lower recall here,
        # beta=0.5 weights precision more -> higher score
        # beta=2 weights recall more -> lower score (since recall is poor)
        assert result_low > result_high


# ============================================================
# Tests for multi-label classification metrics
# ============================================================

class TestAccuracyMulti:
    """Tests for accuracy_multi."""

    def test_perfect(self):
        """Perfect multi-label predictions."""
        preds = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1]])
        targs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = accuracy_multi(preds, targs, thresh=0.5, sigmoid=False)
        assert abs(float(result) - 1.0) < 1e-6

    def test_all_wrong(self):
        """All wrong multi-label predictions."""
        preds = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.9]])
        targs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        result = accuracy_multi(preds, targs, thresh=0.5, sigmoid=False)
        assert abs(float(result) - 0.0) < 1e-6

    def test_with_sigmoid(self):
        """Test with sigmoid activation applied."""
        # Large positive logits -> after sigmoid > 0.5
        preds = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        targs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = accuracy_multi(preds, targs, thresh=0.5, sigmoid=True)
        assert abs(float(result) - 1.0) < 1e-6

    def test_threshold_effect(self):
        """Different thresholds give different results."""
        preds = torch.tensor([[0.6, 0.6, 0.6]])
        targs = torch.tensor([[1.0, 1.0, 1.0]])
        result_low = accuracy_multi(preds, targs, thresh=0.5, sigmoid=False)
        result_high = accuracy_multi(preds, targs, thresh=0.7, sigmoid=False)
        assert float(result_low) == 1.0
        assert float(result_high) == 0.0


class TestMultiLabelSklearnMetrics:
    """Tests for multi-label sklearn-based metrics."""

    def test_f1_multi_perfect(self):
        """Perfect multi-label F1."""
        f1m = F1ScoreMulti(thresh=0.5, sigmoid=False)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = f1m(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_precision_multi_perfect(self):
        """Perfect multi-label precision."""
        pm = PrecisionMulti(thresh=0.5, sigmoid=False)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = pm(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_recall_multi_perfect(self):
        """Perfect multi-label recall."""
        rm = RecallMulti(thresh=0.5, sigmoid=False)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = rm(preds, targs)
        assert abs(result - 1.0) < 1e-6

    def test_hamming_multi_zero(self):
        """Zero hamming loss for perfect multi-label predictions."""
        hm = HammingLossMulti(thresh=0.5, sigmoid=False)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = hm(preds, targs)
        assert abs(result - 0.0) < 1e-6

    def test_jaccard_multi_perfect(self):
        """Perfect multi-label Jaccard."""
        jm = JaccardMulti(thresh=0.5, sigmoid=False)
        preds = torch.tensor([1, 0, 1, 0])
        targs = torch.tensor([1, 0, 1, 0])
        result = jm(preds, targs)
        assert abs(result - 1.0) < 1e-6


# ============================================================
# Tests for regression metrics
# ============================================================

class TestMSE:
    """Tests for mean squared error."""

    def test_zero_error(self):
        """MSE should be 0 for identical predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = mse(pred, targ)
        assert abs(float(result)) < 1e-7

    def test_known_value(self):
        """MSE for known inputs."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        result = mse(pred, targ)
        # Each error is 0.5, squared is 0.25, mean is 0.25
        assert abs(float(result) - 0.25) < 1e-6

    def test_single_value(self):
        """MSE with single element."""
        pred = torch.tensor([2.0])
        targ = torch.tensor([5.0])
        result = mse(pred, targ)
        assert abs(float(result) - 9.0) < 1e-6


class TestMAE:
    """Tests for mean absolute error."""

    def test_zero_error(self):
        """MAE should be 0 for identical predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = mae(pred, targ)
        assert abs(float(result)) < 1e-7

    def test_known_value(self):
        """MAE for known inputs."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        result = mae(pred, targ)
        # Each error is 0.5, mean is 0.5
        assert abs(float(result) - 0.5) < 1e-6

    def test_negative_errors(self):
        """MAE handles negative errors correctly (absolute value)."""
        pred = torch.tensor([3.0, 1.0])
        targ = torch.tensor([1.0, 3.0])
        result = mae(pred, targ)
        # |3-1| + |1-3| / 2 = (2+2)/2 = 2.0
        assert abs(float(result) - 2.0) < 1e-6


class TestMSLE:
    """Tests for mean squared logarithmic error."""

    def test_zero_error(self):
        """MSLE should be 0 for identical predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = msle(pred, targ)
        assert abs(float(result)) < 1e-6

    def test_positive_values(self):
        """MSLE with positive values."""
        pred = torch.tensor([1.0, 2.0])
        targ = torch.tensor([1.0, 2.0])
        result = msle(pred, targ)
        assert abs(float(result)) < 1e-6

    def test_known_value(self):
        """MSLE for known values."""
        pred = torch.tensor([0.0])
        targ = torch.tensor([1.0])
        # msle = (log(1+0) - log(1+1))^2 = (0 - log(2))^2 = log(2)^2
        expected = np.log(2) ** 2
        result = msle(pred, targ)
        assert abs(float(result) - expected) < 1e-5


class TestRMSE:
    """Tests for root mean squared error."""

    def test_zero_error(self):
        """RMSE should be 0 for identical predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = rmse(pred, targ)
        assert abs(float(result)) < 1e-6

    def test_known_value(self):
        """RMSE is sqrt of MSE."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.5, 2.5, 3.5])
        result = rmse(pred, targ)
        # MSE = 0.25, RMSE = 0.5
        assert abs(float(result) - 0.5) < 1e-6

    def test_name(self):
        """RMSE metric should have appropriate name."""
        assert rmse.name == '_rmse'


class TestExpRMSPE:
    """Tests for exponential root mean square percentage error."""

    def test_zero_error(self):
        """exp_rmspe should be 0 when predictions equal targets."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        targ = torch.tensor([1.0, 2.0, 3.0])
        result = exp_rmspe(pred, targ)
        assert abs(float(result)) < 1e-6

    def test_positive_value(self):
        """exp_rmspe should give a positive value for different inputs."""
        pred = torch.tensor([0.5, 1.0, 1.5])
        targ = torch.tensor([0.6, 1.1, 1.6])
        result = exp_rmspe(pred, targ)
        assert float(result) > 0


class TestR2Score:
    """Tests for R2 score."""

    def test_perfect_prediction(self):
        """R2 should be 1.0 for perfect predictions."""
        r2 = R2Score()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = r2(pred, targ)
        assert abs(result - 1.0) < 1e-6

    def test_good_prediction(self):
        """R2 close to 1 for good predictions."""
        r2 = R2Score()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])
        result = r2(pred, targ)
        assert result > 0.9

    def test_mean_prediction(self):
        """Predicting the mean gives R2=0."""
        r2 = R2Score()
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.full((5,), targ.float().mean().item())
        result = r2(pred, targ)
        assert abs(result) < 1e-6


class TestExplainedVariance:
    """Tests for explained variance."""

    def test_perfect_prediction(self):
        """Explained variance is 1 for perfect predictions."""
        ev = ExplainedVariance()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = ev(pred, targ)
        assert abs(result - 1.0) < 1e-6

    def test_bad_prediction(self):
        """Explained variance is low for poor predictions."""
        ev = ExplainedVariance()
        pred = torch.tensor([5.0, 5.0, 5.0, 5.0])
        targ = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = ev(pred, targ)
        assert result < 0.5


class TestPearsonCorrCoef:
    """Tests for Pearson correlation coefficient."""

    def test_perfect_positive(self):
        """Perfect positive correlation."""
        pearson = PearsonCorrCoef()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        result = pearson(pred, targ)
        assert abs(result - 1.0) < 1e-6

    def test_perfect_negative(self):
        """Perfect negative correlation."""
        pearson = PearsonCorrCoef()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        result = pearson(pred, targ)
        assert abs(result - (-1.0)) < 1e-6

    def test_no_correlation(self):
        """Orthogonal data has correlation near 0."""
        pearson = PearsonCorrCoef()
        pred = torch.tensor([1.0, 0.0, -1.0, 0.0])
        targ = torch.tensor([0.0, 1.0, 0.0, -1.0])
        result = pearson(pred, targ)
        assert abs(result) < 1e-6


class TestSpearmanCorrCoef:
    """Tests for Spearman correlation coefficient."""

    def test_perfect_monotonic(self):
        """Perfect monotonic relationship."""
        spearman = SpearmanCorrCoef()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        result = spearman(pred, targ)
        assert abs(result - 1.0) < 1e-6

    def test_perfect_negative_monotonic(self):
        """Perfect negative monotonic relationship."""
        spearman = SpearmanCorrCoef()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        result = spearman(pred, targ)
        assert abs(result - (-1.0)) < 1e-6

    def test_nonlinear_monotonic(self):
        """Spearman captures nonlinear monotonic relationships."""
        spearman = SpearmanCorrCoef()
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targ = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])  # quadratic but monotonic
        result = spearman(pred, targ)
        assert abs(result - 1.0) < 1e-6


# ============================================================
# Tests for segmentation metrics
# ============================================================

class TestForegroundAcc:
    """Tests for foreground_acc."""

    def test_perfect_foreground(self):
        """Perfect foreground accuracy."""
        # 1 sample, 3 classes, 2x2
        preds = torch.zeros(1, 3, 2, 2)
        preds[0, 1, 0, 0] = 10.0
        preds[0, 2, 0, 1] = 10.0
        preds[0, 0, 1, 0] = 10.0  # background
        preds[0, 1, 1, 1] = 10.0
        targs = torch.tensor([[[1, 2], [0, 1]]])
        result = foreground_acc(preds, targs, bkg_idx=0)
        assert abs(float(result) - 1.0) < 1e-6

    def test_all_background(self):
        """When all targets are background, metric is not computed (empty mask)."""
        preds = torch.zeros(1, 3, 2, 2)
        preds[0, 0, :, :] = 10.0  # all background
        targs = torch.zeros(1, 2, 2, dtype=torch.long)  # all background
        # With all background, mask is empty and mean of empty tensor
        # This would produce nan or error, so we just check it doesn't crash
        # Actually foreground_acc will have empty mask, resulting in nan
        result = foreground_acc(preds, targs, bkg_idx=0)
        # With empty foreground, torch returns nan
        assert torch.isnan(result) or float(result) == 0.0

    def test_partial_foreground(self):
        """Partial foreground accuracy."""
        preds = torch.zeros(1, 3, 2, 2)
        preds[0, 1, 0, 0] = 10.0  # predict class 1 at (0,0) -- correct
        preds[0, 1, 0, 1] = 10.0  # predict class 1 at (0,1) -- wrong (target is 2)
        preds[0, 0, 1, 0] = 10.0  # predict bg at (1,0) -- bg
        preds[0, 2, 1, 1] = 10.0  # predict class 2 at (1,1) -- correct
        targs = torch.tensor([[[1, 2], [0, 2]]])
        result = foreground_acc(preds, targs, bkg_idx=0)
        # Foreground pixels: (0,0)->1 correct, (0,1)->2 wrong, (1,1)->2 correct
        # 2/3 correct
        assert abs(float(result) - 2.0 / 3.0) < 1e-6


class TestDice:
    """Tests for Dice coefficient metric."""

    def _make_learn(self, pred, y):
        """Create a mock learn object."""
        learn = Mock()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect_segmentation(self):
        """Perfect binary segmentation gives Dice=1."""
        dice = Dice(axis=1)
        dice.reset()
        # 2 samples, 2 classes, 2x2
        pred = torch.zeros(2, 2, 2, 2)
        pred[:, 1, :, :] = 10.0  # predict class 1 everywhere
        targ = torch.ones(2, 2, 2, dtype=torch.long)  # target is class 1 everywhere
        dice.accumulate(self._make_learn(pred, targ))
        assert abs(dice.value - 1.0) < 1e-6

    def test_no_overlap(self):
        """No overlap gives Dice=0."""
        dice = Dice(axis=1)
        dice.reset()
        pred = torch.zeros(1, 2, 2, 2)
        pred[0, 0, :, :] = 10.0  # predict all class 0
        targ = torch.ones(1, 2, 2, dtype=torch.long)  # target all class 1
        dice.accumulate(self._make_learn(pred, targ))
        assert abs(dice.value - 0.0) < 1e-6

    def test_partial_overlap(self):
        """Partial overlap gives expected Dice."""
        dice = Dice(axis=1)
        dice.reset()
        # 1 sample, 2 classes, 1x4
        pred = torch.zeros(1, 2, 1, 4)
        # Predict class 1 for positions 0,1 and class 0 for 2,3
        pred[0, 1, 0, 0] = 10.0
        pred[0, 1, 0, 1] = 10.0
        pred[0, 0, 0, 2] = 10.0
        pred[0, 0, 0, 3] = 10.0
        # Target class 1 for positions 0,1,2 and class 0 for 3
        targ = torch.tensor([[[1, 1, 1, 0]]])
        dice.accumulate(self._make_learn(pred, targ))
        # pred class 1: [1,1,0,0], targ class 1: [1,1,1,0]
        # inter = 2, union = 2+3 = 5
        # dice = 2*2/5 = 0.8
        assert abs(dice.value - 0.8) < 1e-6

    def test_multiple_accumulations(self):
        """Dice accumulates across multiple batches."""
        dice = Dice(axis=1)
        dice.reset()

        # Batch 1: perfect
        pred1 = torch.zeros(1, 2, 1, 2)
        pred1[0, 1, 0, :] = 10.0
        targ1 = torch.ones(1, 1, 2, dtype=torch.long)
        dice.accumulate(self._make_learn(pred1, targ1))

        # Batch 2: no overlap
        pred2 = torch.zeros(1, 2, 1, 2)
        pred2[0, 0, 0, :] = 10.0
        targ2 = torch.ones(1, 1, 2, dtype=torch.long)
        dice.accumulate(self._make_learn(pred2, targ2))

        # Combined: inter=2, union= (2+2) + (0+2) = 6
        # dice = 2*2/6 = 2/3
        assert abs(dice.value - 2.0 / 3.0) < 1e-6

    def test_empty_prediction(self):
        """When union is 0, value should be None."""
        dice = Dice(axis=1)
        dice.reset()
        pred = torch.zeros(1, 2, 1, 2)
        pred[0, 0, 0, :] = 10.0  # all class 0
        targ = torch.zeros(1, 1, 2, dtype=torch.long)  # all class 0
        dice.accumulate(self._make_learn(pred, targ))
        # inter=0, union=0 -> value is None
        assert dice.value is None


class TestDiceMulti:
    """Tests for DiceMulti metric."""

    def _make_learn(self, pred, y):
        learn = Mock()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect_multiclass(self):
        """Perfect multiclass segmentation."""
        dice_multi = DiceMulti(axis=1)
        dice_multi.reset()
        # 1 sample, 3 classes, 1x3
        pred = torch.zeros(1, 3, 1, 3)
        pred[0, 0, 0, 0] = 10.0  # class 0 at pos 0
        pred[0, 1, 0, 1] = 10.0  # class 1 at pos 1
        pred[0, 2, 0, 2] = 10.0  # class 2 at pos 2
        targ = torch.tensor([[[0, 1, 2]]])
        dice_multi.accumulate(self._make_learn(pred, targ))
        assert abs(dice_multi.value - 1.0) < 1e-6

    def test_partial_multiclass(self):
        """Partial multiclass segmentation."""
        dice_multi = DiceMulti(axis=1)
        dice_multi.reset()
        # 1 sample, 2 classes, 1x4
        pred = torch.zeros(1, 2, 1, 4)
        pred[0, 0, 0, :2] = 10.0  # class 0 at pos 0,1
        pred[0, 1, 0, 2:] = 10.0  # class 1 at pos 2,3
        targ = torch.tensor([[[0, 1, 1, 0]]])  # mixed
        dice_multi.accumulate(self._make_learn(pred, targ))
        # class 0: pred=[1,1,0,0], targ=[1,0,0,1] -> inter=1, union=2+2=4, dice=2/4=0.5
        # class 1: pred=[0,0,1,1], targ=[0,1,1,0] -> inter=1, union=2+2=4, dice=2/4=0.5
        # mean = 0.5
        assert abs(dice_multi.value - 0.5) < 1e-6


class TestJaccardCoeff:
    """Tests for JaccardCoeff metric."""

    def _make_learn(self, pred, y):
        learn = Mock()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect(self):
        """Perfect segmentation gives IoU=1."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 2, 2)
        pred[0, 1, :, :] = 10.0
        targ = torch.ones(1, 2, 2, dtype=torch.long)
        jac.accumulate(self._make_learn(pred, targ))
        assert abs(jac.value - 1.0) < 1e-6

    def test_partial_overlap(self):
        """Partial overlap gives expected IoU."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        # 1 sample, 2 classes, 1x4
        pred = torch.zeros(1, 2, 1, 4)
        pred[0, 1, 0, 0] = 10.0
        pred[0, 1, 0, 1] = 10.0
        pred[0, 0, 0, 2] = 10.0
        pred[0, 0, 0, 3] = 10.0
        targ = torch.tensor([[[1, 1, 1, 0]]])
        jac.accumulate(self._make_learn(pred, targ))
        # pred class 1: [1,1,0,0], targ class 1: [1,1,1,0]
        # inter=2, union=2+3=5, IoU = 2/(5-2) = 2/3
        assert abs(jac.value - 2.0 / 3.0) < 1e-6

    def test_no_overlap(self):
        """No overlap gives IoU=0."""
        jac = JaccardCoeff(axis=1)
        jac.reset()
        pred = torch.zeros(1, 2, 1, 2)
        pred[0, 0, 0, :] = 10.0  # all class 0
        targ = torch.ones(1, 1, 2, dtype=torch.long)  # all class 1
        jac.accumulate(self._make_learn(pred, targ))
        # inter=0, union=0+2=2, IoU = 0/(2-0) = 0
        assert abs(jac.value - 0.0) < 1e-6


class TestJaccardCoeffMulti:
    """Tests for JaccardCoeffMulti metric."""

    def _make_learn(self, pred, y):
        learn = Mock()
        learn.pred = pred
        learn.y = y
        return learn

    def test_perfect(self):
        """Perfect multiclass gives mIoU=1."""
        jac_multi = JaccardCoeffMulti(axis=1)
        jac_multi.reset()
        pred = torch.zeros(1, 3, 1, 3)
        pred[0, 0, 0, 0] = 10.0
        pred[0, 1, 0, 1] = 10.0
        pred[0, 2, 0, 2] = 10.0
        targ = torch.tensor([[[0, 1, 2]]])
        jac_multi.accumulate(self._make_learn(pred, targ))
        assert abs(jac_multi.value - 1.0) < 1e-6


# ============================================================
# Tests for CorpusBLEUMetric
# ============================================================

class TestCorpusBLEUMetric:
    """Tests for CorpusBLEUMetric."""

    def _make_learn(self, pred, y, training=False):
        learn = Mock()
        learn.pred = pred
        learn.y = y
        learn.training = training
        return learn

    def test_perfect_bleu(self):
        """Identical sequences should give BLEU near 1."""
        bleu = CorpusBLEUMetric(vocab_sz=10, axis=-1)
        bleu.reset()
        # Create a sequence: batch=1, seq_len=8, vocab=10
        targ = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # Create logits where argmax matches target
        pred = torch.zeros(1, 8, 10)
        for i in range(8):
            pred[0, i, targ[0, i]] = 10.0
        bleu.accumulate(self._make_learn(pred, targ))
        assert bleu.value > 0.9

    def test_no_match_bleu(self):
        """Completely different sequences give low BLEU."""
        bleu = CorpusBLEUMetric(vocab_sz=10, axis=-1)
        bleu.reset()
        targ = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # Predictions are all wrong
        pred = torch.zeros(1, 8, 10)
        for i in range(8):
            pred[0, i, 9] = 10.0  # always predict token 9
        bleu.accumulate(self._make_learn(pred, targ))
        assert bleu.value < 0.3

    def test_training_skips(self):
        """During training, accumulate is skipped."""
        bleu = CorpusBLEUMetric(vocab_sz=10, axis=-1)
        bleu.reset()
        pred = torch.zeros(1, 8, 10)
        targ = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        bleu.accumulate(self._make_learn(pred, targ, training=True))
        # No data accumulated, value should reflect empty state
        assert bleu.value is None or bleu.value == 0.0

    def test_reset(self):
        """Reset clears accumulated state."""
        bleu = CorpusBLEUMetric(vocab_sz=10, axis=-1)
        bleu.reset()
        assert bleu.pred_len == 0
        assert bleu.targ_len == 0
        assert bleu.corrects == [0, 0, 0, 0]
        assert bleu.counts == [0, 0, 0, 0]


# ============================================================
# Tests for Perplexity
# ============================================================

class TestPerplexity:
    """Tests for Perplexity metric."""

    def test_name(self):
        """Perplexity should have name 'perplexity'."""
        from fastai.metrics import perplexity
        assert perplexity.name == 'perplexity'

    def test_value_from_loss(self):
        """Perplexity is exp of average loss."""
        from fastai.metrics import Perplexity
        ppl = Perplexity()
        ppl.total = torch.tensor(2.0)
        ppl.count = 2
        # exp(2.0/2) = exp(1.0) = e
        expected = float(torch.exp(torch.tensor(1.0)))
        assert abs(float(ppl.value) - expected) < 1e-5

    def test_zero_count(self):
        """When count is 0, value should be None."""
        from fastai.metrics import Perplexity
        ppl = Perplexity()
        ppl.total = torch.tensor(0.0)
        ppl.count = 0
        assert ppl.value is None


# ============================================================
# Tests for LossMetric and LossMetrics
# ============================================================

class TestLossMetric:
    """Tests for LossMetric."""

    def test_name_default(self):
        """Name defaults to attr name."""
        lm = LossMetric('my_loss_attr')
        assert lm.name == 'my_loss_attr'

    def test_name_custom(self):
        """Custom name overrides attr."""
        lm = LossMetric('my_loss_attr', nm='custom_name')
        assert lm.name == 'custom_name'


class TestLossMetrics:
    """Tests for LossMetrics."""

    def test_creates_list(self):
        """LossMetrics creates a list of LossMetric objects."""
        metrics = LossMetrics(['attr1', 'attr2', 'attr3'])
        assert len(metrics) == 3
        assert metrics[0].name == 'attr1'
        assert metrics[1].name == 'attr2'
        assert metrics[2].name == 'attr3'

    def test_string_input(self):
        """LossMetrics accepts comma-separated string."""
        metrics = LossMetrics('attr1,attr2')
        assert len(metrics) == 2
        assert metrics[0].name == 'attr1'
        assert metrics[1].name == 'attr2'

    def test_custom_names(self):
        """LossMetrics uses custom names when provided."""
        metrics = LossMetrics(['a', 'b'], nms=['name_a', 'name_b'])
        assert metrics[0].name == 'name_a'
        assert metrics[1].name == 'name_b'

    def test_custom_names_string(self):
        """LossMetrics accepts comma-separated string for names."""
        metrics = LossMetrics('a,b', nms='na,nb')
        assert metrics[0].name == 'na'
        assert metrics[1].name == 'nb'


# ============================================================
# Tests for skm_to_fastai utility
# ============================================================

class TestSkmToFastai:
    """Tests for skm_to_fastai conversion utility."""

    def test_creates_accum_metric(self):
        """skm_to_fastai returns an AccumMetric instance."""
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score)
        assert isinstance(metric, AccumMetric)

    def test_to_np_enabled(self):
        """skm_to_fastai sets to_np=True for sklearn compatibility."""
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score)
        assert metric.to_np is True

    def test_invert_arg_enabled(self):
        """skm_to_fastai sets invert_arg=True (sklearn uses targ, pred order)."""
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.accuracy_score)
        assert metric.invert_args is True

    def test_classification_with_thresh(self):
        """When thresh is set, activation should be Sigmoid."""
        import sklearn.metrics as skm
        from fastai.metrics import ActivationType
        metric = skm_to_fastai(skm.accuracy_score, thresh=0.5)
        assert metric.activation == ActivationType.Sigmoid

    def test_regression_metric(self):
        """Regression metrics have is_class=False and no dim_argmax."""
        import sklearn.metrics as skm
        metric = skm_to_fastai(skm.r2_score, is_class=False)
        assert metric.dim_argmax is None
