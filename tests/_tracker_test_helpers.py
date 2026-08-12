"""Shared mock infrastructure for testing fastai.callback.tracker callbacks.

Both test_checkpoint_averaging.py and test_multi_metric_early_stopping.py
need to load and execute the tracker module source with mocked dependencies.
This module provides that shared setup so it is defined once.

Usage:
    from _tracker_test_helpers import tracker_module, CancelFitException, FakeRecorder
"""
import sys
import os
import types
import numpy as np


class CancelFitException(Exception):
    """Stand-in for fastai.basics.CancelFitException."""
    pass


def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _setup_mock_modules():
    """Install minimal mock modules so fastai.callback.tracker can be exec'd."""
    _make_module('torch', {'isinf': lambda x: False, 'isnan': lambda x: False})
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    _make_module('fastai.basics', {
        'np': np,
        'Callback': type('Callback', (object,), {}),
        'CancelFitException': CancelFitException,
        'store_attr': lambda *a, **kw: None,
        'float': float,
    })

    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []

    _make_module('fastai.callback.progress')
    _make_module('fastai.callback.fp16', {
        'MixedPrecision': type('MixedPrecision', (object,), {}),
    })


def _load_tracker_module():
    """Load and exec the tracker module source with import lines stripped."""
    if 'fastai.callback.tracker' in sys.modules:
        del sys.modules['fastai.callback.tracker']

    tracker_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'tracker.py')
    tracker_path = os.path.abspath(tracker_path)

    mod = types.ModuleType('fastai.callback.tracker')
    mod.__file__ = tracker_path
    mod.__package__ = 'fastai.callback'

    # Populate namespace with what star-imports would provide
    mod.np = np
    mod.Callback = sys.modules['fastai.basics'].Callback
    mod.CancelFitException = CancelFitException
    mod.store_attr = lambda *a, **kw: None
    mod.MixedPrecision = sys.modules['fastai.callback.fp16'].MixedPrecision
    mod.__builtins__ = __builtins__

    with open(tracker_path, 'r') as f:
        source = f.read()

    # Strip internal import lines that would fail without real fastai
    filtered_lines = []
    for line in source.split('\n'):
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), tracker_path, 'exec'), mod.__dict__)
    sys.modules['fastai.callback.tracker'] = mod
    return mod


# --- Module-level initialization ---
_setup_mock_modules()
tracker_module = _load_tracker_module()


# --- Shared test helpers ---

class FakeRecorder:
    """Mock recorder that simulates metric tracking."""
    def __init__(self, metric_names, values=None):
        self.metric_names = ['epoch'] + list(metric_names)
        self.values = values if values is not None else []
