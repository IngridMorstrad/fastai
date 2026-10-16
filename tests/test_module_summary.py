"""Tests for module_summary ParameterModule parameter counting fix.

Verifies that Learner.summary() / module_summary() reports the correct total
parameter count even when the model contains bare nn.Parameter attributes
(which get wrapped in ParameterModule by flatten_model but whose forward()
is never invoked during inference, so hooks never fire for them).
"""

import sys
import os
import re

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.test_utils import synth_learner
from fastai.callback.hook import module_summary
from fastai.layers import ParameterModule, Module


# ------------------------------------------------------------------
# A minimal model whose only trainable storage is a lone nn.Parameter
# wrapped inside a normal Linear layer's output path.  flatten_model
# turns the bare parameter into a ParameterModule, but inference
# never calls ParameterModule.forward(), so hooks miss it.
# ------------------------------------------------------------------
class LinearPlusBias(Module):
    """Linear layer with an extra standalone nn.Parameter bias term."""
    def __init__(self, in_f, out_f):
        self.linear = nn.Linear(in_f, out_f)
        # This bare parameter will become a ParameterModule in flatten_model
        self.extra_bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return self.linear(x) + self.extra_bias


def _extract_total_params(summary_text: str) -> int:
    """Pull the 'Total params: N' integer out of a summary string."""
    m = re.search(r'Total params:\s*([\d,]+)', summary_text)
    assert m, f"Could not find 'Total params' in summary:\n{summary_text}"
    return int(m.group(1).replace(',', ''))


def _extract_trainable_params(summary_text: str) -> int:
    m = re.search(r'Total trainable params:\s*([\d,]+)', summary_text)
    assert m, f"Could not find 'Total trainable params' in summary:\n{summary_text}"
    return int(m.group(1).replace(',', ''))


class TestModuleSummaryParameterModule:
    """module_summary must include ParameterModule params in the totals."""

    def test_extra_parameter_counted_in_summary(self):
        """Total params should equal the true model parameter count."""
        model = LinearPlusBias(1, 1)
        learn = synth_learner(model=model)

        xb = learn.dls.train.one_batch()[:1]
        summary = module_summary(learn, *xb)

        true_params = sum(p.numel() for p in model.parameters())
        reported_params = _extract_total_params(str(summary))

        assert reported_params == true_params, (
            f"Summary reported {reported_params} total params but model "
            f"actually has {true_params}"
        )

    def test_trainable_params_match_model(self):
        """Total trainable params should match model's trainable count."""
        model = LinearPlusBias(1, 1)
        learn = synth_learner(model=model)

        xb = learn.dls.train.one_batch()[:1]
        summary = module_summary(learn, *xb)

        true_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        reported_trainable = _extract_trainable_params(str(summary))

        assert reported_trainable == true_trainable, (
            f"Summary reported {reported_trainable} trainable params but "
            f"model actually has {true_trainable}"
        )

    def test_frozen_parameter_module_counted_as_non_trainable(self):
        """A frozen bare parameter must appear in totals but not trainable."""
        model = LinearPlusBias(1, 1)
        model.extra_bias.requires_grad = False
        learn = synth_learner(model=model)

        xb = learn.dls.train.one_batch()[:1]
        summary = module_summary(learn, *xb)

        total = _extract_total_params(str(summary))
        trainable = _extract_trainable_params(str(summary))

        true_total = sum(p.numel() for p in model.parameters())
        true_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        assert total == true_total
        assert trainable == true_trainable
        assert total > trainable, (
            "Freezing extra_bias should make non-trainable > 0"
        )

    def test_synth_learner_regmodel_counts_match(self):
        """Default synth_learner (RegModel) should also report correct counts.

        RegModel has two bare nn.Parameters (a, b) and no child modules,
        so every parameter is a ParameterModule in flatten_model.
        """
        learn = synth_learner()
        xb = learn.dls.train.one_batch()[:1]
        summary = module_summary(learn, *xb)

        true_params = sum(p.numel() for p in learn.model.parameters())
        reported = _extract_total_params(str(summary))

        assert reported == true_params, (
            f"RegModel: summary reported {reported} but model has {true_params}"
        )
