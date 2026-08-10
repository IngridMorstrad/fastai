"""Shared mock infrastructure for tracker callback tests.

Both test_checkpoint_averaging.py and test_multi_metric_early_stopping.py
need to mock the entire fastai/torch import chain so tracker.py can be
imported without PyTorch. This module provides that shared setup.

Exports:
    tracker_module  - the loaded fastai.callback.tracker module
    FakeRecorder    - mock recorder that simulates metric tracking
    _CancelFitException - the mocked CancelFitException class
"""

import sys
import os
import types
import numpy as np


# ----- Mock Setup -----
# We need to mock the entire import chain so tracker.py can be imported
# without torch or full fastai dependencies.

class _CancelFitException(Exception):
    pass


# Create mock modules as proper module objects (not MagicMock)
# so Python's import system treats them as real packages.
def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Mock torch and its submodules
_make_module('torch', {'isinf': lambda x: False, 'isnan': lambda x: False})
_make_module('torch.multiprocessing')
_make_module('torch.nn')

# Create the fastai package hierarchy
fastai_pkg = _make_module('fastai')
fastai_pkg.__path__ = []  # Mark as package

# fastai.basics - provides Callback, np, CancelFitException, store_attr, etc.
basics_mod = _make_module('fastai.basics', {
    'np': np,
    'Callback': type('Callback', (object,), {}),
    'CancelFitException': _CancelFitException,
    'store_attr': lambda *a, **kw: None,
    'float': float,
})

# fastai.callback package
callback_pkg = _make_module('fastai.callback')
callback_pkg.__path__ = []  # Mark as package

# fastai.callback.progress
_make_module('fastai.callback.progress')

# fastai.callback.fp16
fp16_mod = _make_module('fastai.callback.fp16', {'MixedPrecision': type('MixedPrecision', (object,), {})})

# Now we can import the actual tracker module
# First, remove any cached version
if 'fastai.callback.tracker' in sys.modules:
    del sys.modules['fastai.callback.tracker']

# Patch the import mechanism for the tracker module.
# The tracker module does `from ..basics import *` and `from .progress import *`
# and `from .fp16 import MixedPrecision`.
# Since we set up the modules above, we need to make the star imports work.
# The simplest way: manually load and exec the tracker source with our namespace.

_tracker_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'tracker.py')
_tracker_path = os.path.abspath(_tracker_path)

# Create the tracker module
tracker_module = types.ModuleType('fastai.callback.tracker')
tracker_module.__file__ = _tracker_path
tracker_module.__package__ = 'fastai.callback'

# Populate namespace with what `from ..basics import *` would provide
tracker_module.np = np
tracker_module.Callback = basics_mod.Callback
tracker_module.CancelFitException = _CancelFitException
tracker_module.store_attr = lambda *a, **kw: None
tracker_module.MixedPrecision = fp16_mod.MixedPrecision
tracker_module.__builtins__ = __builtins__

# Execute the tracker source, skipping the import lines
with open(_tracker_path, 'r') as f:
    source = f.read()

# Remove the problematic import lines
lines = source.split('\n')
filtered_lines = []
for line in lines:
    # Skip import lines that pull from fastai internals
    if line.startswith('from __future__'):
        filtered_lines.append(line)
    elif line.startswith('from ..') or line.startswith('from .'):
        filtered_lines.append('pass  # skipped import')
    else:
        filtered_lines.append(line)

exec(compile('\n'.join(filtered_lines), _tracker_path, 'exec'), tracker_module.__dict__)
sys.modules['fastai.callback.tracker'] = tracker_module


# ----- Helpers -----

class FakeRecorder:
    """Mock recorder that simulates metric tracking."""
    def __init__(self, metric_names, values=None):
        self.metric_names = ['epoch'] + list(metric_names)
        self.values = values if values is not None else []
