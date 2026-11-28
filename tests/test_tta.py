"""Tests for the configurable TTA pipeline on fastai.learner.Learner.

Covers Learner.tta_config: the named reduction strategies ('mean', 'max',
'gmean', 'median'), user-supplied callable reductions, the configurable
final combination, invalid-reduction error handling, gmean numerical safety,
and backward compatibility of the existing Learner.tta. CPU only.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.learner import Learner, TTA_REDUCTIONS, _tta_resolve_reduce, _tta_gmean
from fastai.test_utils import synth_learner
from fastai.data.core import TfmdDL, Datasets, DataLoaders
from fastcore.transform import noop


# ============================================================
# Helper: synth learner with a predictable TTA-able DataLoaders
# ============================================================

def _make_learner():
    """Build a synth_learner with a TfmdDL mirroring the notebook's hidden tta test."""
    learn = synth_learner()
    # num_workers=0 keeps torch on its single-process DataLoader iterator, which
    # fastai's _FakeLoader supports across torch versions in this test env.
    dl = TfmdDL(Datasets(torch.arange(50), [noop, noop]), num_workers=0)
    learn.dls = DataLoaders(dl, dl)
    return learn


# ============================================================
# Reduction registry / resolver
# ============================================================

class TestReductionRegistry:
    """Tests for the TTA_REDUCTIONS registry and the resolver helper."""

    def test_registry_has_expected_names(self):
        assert set(TTA_REDUCTIONS) == {'mean', 'max', 'gmean', 'median'}

    def test_named_reducers_shape_over_stack(self):
        # stacked shape (n, batch, k) -> reduced shape (batch, k)
        stacked = torch.rand(4, 6, 3)
        for name, fn in TTA_REDUCTIONS.items():
            out = fn(stacked)
            assert out.shape == (6, 3), name

    def test_mean_matches_torch_mean(self):
        stacked = torch.rand(4, 6, 3)
        assert torch.allclose(TTA_REDUCTIONS['mean'](stacked), stacked.mean(0))

    def test_resolve_accepts_callable(self):
        fn = lambda x: x.sum(0)
        assert _tta_resolve_reduce(fn) is fn

    def test_resolve_accepts_name(self):
        assert _tta_resolve_reduce('median') is TTA_REDUCTIONS['median']

    def test_resolve_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown TTA reduction"):
            _tta_resolve_reduce('not_a_real_reduction')

    def test_gmean_no_nan_on_zeros(self):
        """gmean must not produce NaN/inf even when inputs contain zeros."""
        stacked = torch.zeros(4, 5, 2)
        out = _tta_gmean(stacked)
        assert torch.isfinite(out).all()

    def test_gmean_small_positive_inputs_finite(self):
        stacked = torch.rand(4, 5, 2) * 1e-3
        out = _tta_gmean(stacked)
        assert torch.isfinite(out).all()
        assert (out >= 0).all()


# ============================================================
# Learner.tta_config behavior
# ============================================================

class TestTtaConfig:
    """Tests for the Learner.tta_config method."""

    def test_default_runs_and_returns_aligned_preds_targs(self):
        learn = _make_learner()
        preds, targs = learn.tta_config()
        assert len(preds) == len(targs)

    @pytest.mark.parametrize('reduce', ['mean', 'max', 'gmean', 'median'])
    def test_each_named_reduction_runs(self, reduce):
        learn = _make_learner()
        preds, targs = learn.tta_config(aug_reduce=reduce)
        assert len(preds) == len(targs)

    def test_callable_reduction_runs(self):
        learn = _make_learner()
        preds, targs = learn.tta_config(aug_reduce=lambda x: x.mean(0))
        assert len(preds) == len(targs)

    def test_invalid_reduction_string_raises(self):
        learn = _make_learner()
        with pytest.raises(ValueError, match="Unknown TTA reduction"):
            learn.tta_config(aug_reduce='bogus')

    def test_invalid_final_reduce_string_raises(self):
        learn = _make_learner()
        with pytest.raises(ValueError, match="Unknown TTA reduction"):
            learn.tta_config(final_reduce='bogus')

    def test_final_reduce_path_runs(self):
        learn = _make_learner()
        preds, targs = learn.tta_config(final_reduce='max')
        assert len(preds) == len(targs)

    def test_final_reduce_callable_runs(self):
        learn = _make_learner()
        preds, targs = learn.tta_config(final_reduce=lambda x: x.mean(0))
        assert len(preds) == len(targs)

    def test_beta_none_returns_aug_and_clean_tuple(self):
        learn = _make_learner()
        preds, targs = learn.tta_config(beta=None)
        assert isinstance(preds, tuple) and len(preds) == 2

    def test_backward_compatible_with_tta_defaults(self):
        """mean aug + default beta reproduces the legacy tta() under the same seed."""
        learn = _make_learner()
        torch.manual_seed(0); a, _ = learn.tta()
        torch.manual_seed(0); b, _ = learn.tta_config()
        assert torch.allclose(a, b, atol=1e-5)


# ============================================================
# Backward compatibility of the existing tta
# ============================================================

class TestTtaUnchanged:
    """The original Learner.tta must keep working unchanged."""

    def test_tta_still_runs(self):
        learn = _make_learner()
        preds, targs = learn.tta()
        assert len(preds) == len(targs)

    def test_tta_use_max_runs(self):
        learn = _make_learner()
        preds, targs = learn.tta(use_max=True)
        assert len(preds) == len(targs)

    def test_tta_beta_none_tuple(self):
        learn = _make_learner()
        preds, targs = learn.tta(beta=None)
        assert isinstance(preds, tuple) and len(preds) == 2
