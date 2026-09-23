from __future__ import annotations

"""Tests for TensorBoardCallback run-guard in after_batch and after_epoch.

When ``self.run`` is False (e.g. during ``lr_find`` or ``gather_preds``),
``_setup_writer`` is never called, so ``self.writer`` does not exist.
Both ``after_batch`` and ``after_epoch`` must bail out early in that case
to avoid an ``AttributeError``.

Because the full fastai import chain requires many heavy dependencies
(torch, numpy, scipy, ...) that may not be available in lightweight CI
environments, these tests avoid importing ``fastai.callback.tensorboard``
through the normal path.  Instead we:

1. Read the source file directly,
2. Provide a minimal ``Callback`` stub plus other names the module-level
   code expects,
3. ``exec`` the source in that prepared namespace, and
4. Exercise the resulting ``TensorBoardCallback`` class.

This keeps the tests self-contained and dependency-free.
"""

import types
import pytest
from unittest.mock import MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Build a minimal namespace that lets the tensorboard module source exec
# ---------------------------------------------------------------------------

def _load_tensorboard_module():
    """Return a dict containing all names defined by tensorboard.py."""
    src_path = Path(__file__).resolve().parent.parent / "fastai" / "callback" / "tensorboard.py"
    source = src_path.read_text()

    # Strip the ``from ..basics import *`` and other relative imports so
    # exec does not trigger the real import chain.  We replace them with
    # a curated set of names the module body actually needs.
    lines = source.splitlines(keepends=True)
    filtered = []
    for line in lines:
        # Skip lines that are relative imports or torch.utils.tensorboard
        stripped = line.strip()
        if stripped.startswith("from ..") or stripped.startswith("from ."):
            continue
        if stripped.startswith("from torch"):
            continue
        filtered.append(line)
    cleaned = "".join(filtered)

    # Provide the names the module body references
    ns: dict = {}

    # Callback base class (just needs to be a normal class)
    class _Callback:
        pass

    class _Recorder:
        order = 50

    # Minimal stubs
    ns["Callback"] = _Callback
    ns["Recorder"] = _Recorder
    ns["store_attr"] = lambda **kw: None  # no-op
    ns["rank_distrib"] = lambda: 0
    ns["hook_output"] = MagicMock()
    ns["SummaryWriter"] = MagicMock
    ns["typedispatch"] = lambda fn: fn
    ns["getcallable"] = MagicMock()
    ns["get_grid"] = MagicMock()
    ns["TensorImage"] = type("TensorImage", (), {})
    ns["TensorCategory"] = type("TensorCategory", (), {})
    ns["TensorImageBase"] = type("TensorImageBase", (), {})
    ns["TensorPoint"] = type("TensorPoint", (), {})
    ns["TensorBBox"] = type("TensorBBox", (), {})
    ns["LMLearner"] = type("LMLearner", (), {})
    ns["TextLearner"] = type("TextLearner", (), {})
    ns["__name__"] = "fastai.callback.tensorboard"
    ns["__all__"] = []
    # Python builtins are available by default in exec, but be explicit:
    ns["__builtins__"] = __builtins__
    exec(compile(cleaned, str(src_path), "exec"), ns)
    return ns


_NS = _load_tensorboard_module()
TensorBoardCallback = _NS["TensorBoardCallback"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_callback(*, run: bool, with_writer: bool):
    """Build a TensorBoardCallback with only the attributes the methods need."""
    cb = TensorBoardCallback.__new__(TensorBoardCallback)
    cb.run = run

    if with_writer:
        cb.writer = MagicMock()

    # Attributes after_batch reads when run=True
    cb.smooth_loss = 0.5
    cb.train_iter = 42
    cb.opt = MagicMock()
    cb.opt.hypers = [{"lr": 0.01}]

    # Attributes after_epoch reads when run=True
    cb.recorder = MagicMock()
    cb.recorder.metric_names = [
        "epoch", "train_loss", "valid_loss", "accuracy", "time",
    ]
    cb.recorder.log = [0, 0.3, 0.2, 0.95, "00:05"]
    cb.log_preds = False

    return cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAfterBatchGuard:
    """after_batch must bail out when run=False."""

    def test_no_error_when_run_false(self):
        cb = _make_callback(run=False, with_writer=False)
        cb.after_batch()  # must not raise

    def test_writer_not_accessed_when_run_false(self):
        cb = _make_callback(run=False, with_writer=False)
        cb.after_batch()
        assert not hasattr(cb, "writer")

    def test_writes_scalar_when_run_true(self):
        cb = _make_callback(run=True, with_writer=True)
        cb.after_batch()
        assert cb.writer.add_scalar.called


class TestAfterEpochGuard:
    """after_epoch must bail out when run=False."""

    def test_no_error_when_run_false(self):
        cb = _make_callback(run=False, with_writer=False)
        cb.after_epoch()  # must not raise

    def test_writer_not_accessed_when_run_false(self):
        cb = _make_callback(run=False, with_writer=False)
        cb.after_epoch()
        assert not hasattr(cb, "writer")

    def test_writes_scalar_when_run_true(self):
        cb = _make_callback(run=True, with_writer=True)
        cb.after_epoch()
        assert cb.writer.add_scalar.called
