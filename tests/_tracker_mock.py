"""Shared mock infrastructure for testing fastai/callback/tracker.py classes.

The tracker module depends on the full fastai import chain (torch, fastcore,
etc.). This helper mocks that chain so individual callback classes
(CheckpointAveragingCallback, MultiMetricEarlyStoppingCallback, etc.) can be
tested in isolation without heavyweight dependencies.

Usage in test files:
    from _tracker_mock import tracker_module, CancelFitException, FakeRecorder
"""
import sys
import os
import types
import numpy as np


class CancelFitException(Exception):
    """Stand-in for fastai.basics.CancelFitException."""
    pass


def _make_module(name, attrs=None):
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_tracker_module():
    """Load fastai/callback/tracker.py with mocked dependencies.

    Returns the module object with all tracker callback classes available.
    """
    # Mock torch and its submodules
    _make_module('torch', {'isinf': lambda x: False, 'isnan': lambda x: False})
    _make_module('torch.multiprocessing')
    _make_module('torch.nn')

    # Create the fastai package hierarchy
    fastai_pkg = _make_module('fastai')
    fastai_pkg.__path__ = []

    # fastai.basics
    _make_module('fastai.basics', {
        'np': np,
        'Callback': type('Callback', (object,), {}),
        'CancelFitException': CancelFitException,
        'store_attr': lambda *a, **kw: None,
        'float': float,
    })

    # fastai.callback package
    callback_pkg = _make_module('fastai.callback')
    callback_pkg.__path__ = []

    # fastai.callback.progress
    _make_module('fastai.callback.progress')

    # fastai.callback.fp16
    fp16_mod = _make_module('fastai.callback.fp16', {
        'MixedPrecision': type('MixedPrecision', (object,), {}),
    })

    # Remove any cached version
    if 'fastai.callback.tracker' in sys.modules:
        del sys.modules['fastai.callback.tracker']

    tracker_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'tracker.py')
    tracker_path = os.path.abspath(tracker_path)

    # Create the tracker module object
    tracker_mod = types.ModuleType('fastai.callback.tracker')
    tracker_mod.__file__ = tracker_path
    tracker_mod.__package__ = 'fastai.callback'

    # Populate namespace with what star-imports would provide
    tracker_mod.np = np
    tracker_mod.Callback = type('Callback', (object,), {})
    tracker_mod.CancelFitException = CancelFitException
    tracker_mod.store_attr = lambda *a, **kw: None
    tracker_mod.MixedPrecision = fp16_mod.MixedPrecision
    tracker_mod.__builtins__ = __builtins__

    # Execute the tracker source, skipping internal import lines
    with open(tracker_path, 'r') as f:
        source = f.read()

    lines = source.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('from __future__'):
            filtered_lines.append(line)
        elif line.startswith('from ..') or line.startswith('from .'):
            filtered_lines.append('pass  # skipped import')
        else:
            filtered_lines.append(line)

    exec(compile('\n'.join(filtered_lines), tracker_path, 'exec'), tracker_mod.__dict__)
    sys.modules['fastai.callback.tracker'] = tracker_mod
    return tracker_mod


class FakeRecorder:
    """Mock recorder that simulates metric tracking."""
    def __init__(self, metric_names, values=None):
        self.metric_names = ['epoch'] + list(metric_names)
        self.values = values if values is not None else []


# Load the tracker module once at import time
tracker_module = _load_tracker_module()
