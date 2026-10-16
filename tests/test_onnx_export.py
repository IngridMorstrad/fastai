"""Tests for Learner.to_onnx() method.

These tests mock the heavy fastai/torch dependencies so the ONNX export logic
can be validated without installing PyTorch.
"""
import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Mock infrastructure -- install fake modules before importing learner code
# ---------------------------------------------------------------------------

def _make_module(name, attrs=None):
    """Create a fake module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _setup_mock_modules():
    """Install minimal mock modules so fastai.learner symbols can be loaded."""
    # torch and submodules
    torch_mod = _make_module('torch', {
        'save': MagicMock(),
        'load': MagicMock(),
        'no_grad': MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        'device': str,
        'cat': MagicMock(),
        'stack': MagicMock(),
        'lerp': MagicMock(),
        'Tensor': type('Tensor', (), {}),
    })
    onnx_mod = _make_module('torch.onnx', {'export': MagicMock()})
    torch_mod.onnx = onnx_mod
    _make_module('torch.multiprocessing')
    _make_module('torch.nn', {'Module': type('Module', (), {})})

    # fastcore
    _make_module('fastcore', {})
    _make_module('fastcore.basics', {
        'patch': lambda f: f,
        'delegates': lambda *a, **kw: (lambda f: f),
    })
    _make_module('fastcore.foundation', {})


_setup_mock_modules()


# ---------------------------------------------------------------------------
# Now load just the to_onnx function by exec'ing the relevant cell
# ---------------------------------------------------------------------------

def _load_to_onnx():
    """Load the to_onnx function from learner.py source."""
    learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
    learner_path = os.path.abspath(learner_path)

    with open(learner_path, 'r') as f:
        source = f.read()

    # Find the to_onnx cell -- it starts with "# %%\n@patch\ndef to_onnx"
    marker = '# %%\n@patch\ndef to_onnx'
    idx = source.find(marker)
    if idx == -1:
        raise RuntimeError("Could not find to_onnx cell in learner.py")

    # Extract from the marker to end of file (it's the last cell)
    cell_source = source[idx:]
    # Remove the "# %%" line
    cell_lines = cell_source.split('\n')
    cell_lines = cell_lines[1:]  # skip "# %%"

    # We need some symbols available:
    # - patch, rank_distrib, get_model, join_path_file, Path
    namespace = {
        '__builtins__': __builtins__,
        'patch': lambda f: f,
        'rank_distrib': MagicMock(return_value=0),
        'get_model': lambda m: m,
        'join_path_file': None,  # will be set per-test
        'Path': Path,
        'torch': sys.modules['torch'],
        'Learner': type('Learner', (), {}),
    }

    exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
    return namespace['to_onnx']


_to_onnx_func = _load_to_onnx()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class FakeDls:
    """Mock DataLoaders that provides one_batch and n_inp."""
    def __init__(self, n_inp=1, batch=None):
        self.n_inp = n_inp
        self._batch = batch if batch is not None else [np.zeros((4, 3, 32, 32))]

    def one_batch(self):
        return self._batch


class FakeModel:
    """Mock model with eval method and state tracking."""
    def __init__(self):
        self.training = True
        self._eval_called = False

    def eval(self):
        self._eval_called = True
        self.training = False
        return self


class FakeLearner:
    """Mock Learner with path, model_dir, dls, model."""
    def __init__(self, tmp_path=None, n_inp=1, batch=None):
        self.path = Path(tmp_path or '/tmp/test_learner')
        self.model_dir = 'models'
        self.model = FakeModel()
        self.dls = FakeDls(n_inp=n_inp, batch=batch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToOnnxDynamicAxes(unittest.TestCase):
    """Test dynamic_axes construction logic."""

    def _call_to_onnx(self, learner, **kwargs):
        """Call to_onnx with mocked join_path_file and rank_distrib."""
        # Patch join_path_file to just return the fname
        def mock_join_path_file(fname, path, ext='.onnx'):
            p = path / fname.name if hasattr(fname, 'name') else path / str(fname)
            return p

        # We need to call the function with the right globals
        # Since we loaded it via exec, we re-exec with controlled globals each time
        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]  # skip "# %%"

        mock_torch_onnx_export = MagicMock()
        mock_torch = MagicMock()
        mock_torch.onnx.export = mock_torch_onnx_export

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join_path_file,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        fn = namespace['to_onnx']

        result = fn(learner, **kwargs)
        return result, mock_torch_onnx_export

    def test_default_dynamic_axes_single_input(self):
        """Default dynamic_axes=True should create batch dim for input_0 and output_0."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])
        result, mock_export = self._call_to_onnx(learner)

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        dyn_axes = call_kwargs[1]['dynamic_axes']
        self.assertEqual(dyn_axes, {
            'input_0': {0: 'batch_size'},
            'output_0': {0: 'batch_size'},
        })

    def test_default_dynamic_axes_multiple_inputs(self):
        """Multiple inputs should all get batch dim in dynamic_axes."""
        learner = FakeLearner(n_inp=2, batch=[np.zeros((4, 3, 32, 32)), np.zeros((4, 10))])
        result, mock_export = self._call_to_onnx(learner)

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        dyn_axes = call_kwargs[1]['dynamic_axes']
        self.assertEqual(dyn_axes, {
            'input_0': {0: 'batch_size'},
            'input_1': {0: 'batch_size'},
            'output_0': {0: 'batch_size'},
        })

    def test_dynamic_axes_disabled(self):
        """dynamic_axes=False should pass None for dynamic_axes."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])
        result, mock_export = self._call_to_onnx(learner, dynamic_axes=False)

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        dyn_axes = call_kwargs[1]['dynamic_axes']
        self.assertIsNone(dyn_axes)

    def test_custom_input_output_names(self):
        """Custom names should be reflected in dynamic_axes keys."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])
        result, mock_export = self._call_to_onnx(
            learner, input_names=['image'], output_names=['logits', 'probs']
        )

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args
        dyn_axes = call_kwargs[1]['dynamic_axes']
        self.assertEqual(dyn_axes, {
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'probs': {0: 'batch_size'},
        })
        # Also check input_names and output_names are passed through
        self.assertEqual(call_kwargs[1]['input_names'], ['image'])
        self.assertEqual(call_kwargs[1]['output_names'], ['logits', 'probs'])


class TestToOnnxModelEval(unittest.TestCase):
    """Test that model is set to eval mode before export."""

    def test_model_eval_called(self):
        """Model should be in eval mode when torch.onnx.export is called."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        eval_was_called_before_export = []

        def mock_export(*args, **kwargs):
            eval_was_called_before_export.append(learner.model._eval_called)

        mock_torch = MagicMock()
        mock_torch.onnx.export = mock_export

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        namespace['to_onnx'](learner)

        self.assertTrue(eval_was_called_before_export[0],
                        "model.eval() should be called before torch.onnx.export")


class TestToOnnxRankDistrib(unittest.TestCase):
    """Test rank_distrib guard prevents export in child processes."""

    def test_child_process_returns_none(self):
        """When rank_distrib() != 0, to_onnx should return None immediately."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=1),  # child process
            'get_model': lambda m: m,
            'join_path_file': lambda f, p, ext='.onnx': p / str(f),
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        result = namespace['to_onnx'](learner)

        self.assertIsNone(result)
        mock_torch.onnx.export.assert_not_called()


class TestToOnnxOpsetVersion(unittest.TestCase):
    """Test opset_version passthrough."""

    def test_opset_version_passed_to_export(self):
        """opset_version should be forwarded to torch.onnx.export."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        namespace['to_onnx'](learner, opset_version=13)

        mock_torch.onnx.export.assert_called_once()
        call_kwargs = mock_torch.onnx.export.call_args[1]
        self.assertEqual(call_kwargs['opset_version'], 13)

    def test_opset_version_default_none(self):
        """Default opset_version should be None."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        namespace['to_onnx'](learner)

        call_kwargs = mock_torch.onnx.export.call_args[1]
        self.assertIsNone(call_kwargs['opset_version'])


class TestToOnnxSimplify(unittest.TestCase):
    """Test simplify flag triggers onnxsim."""

    def test_simplify_true_calls_onnxsim(self):
        """When simplify=True, onnxsim.simplify should be called."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()
        mock_onnx = MagicMock()
        mock_onnxsim = MagicMock()
        mock_onnxsim.simplify.return_value = (MagicMock(), True)

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        # We need to intercept the import statement inside the try block
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'onnx':
                return mock_onnx
            if name == 'onnxsim':
                return mock_onnxsim
            return real_import(name, *args, **kwargs)

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)

        with unittest.mock.patch('builtins.__import__', side_effect=fake_import):
            namespace['to_onnx'](learner, simplify=True)

        mock_onnx.load.assert_called_once()
        mock_onnxsim.simplify.assert_called_once()
        mock_onnx.save.assert_called_once()

    def test_simplify_false_does_not_call_onnxsim(self):
        """When simplify=False (default), onnxsim should not be called."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        import builtins
        real_import = builtins.__import__
        import_calls = []

        def fake_import(name, *args, **kwargs):
            import_calls.append(name)
            return real_import(name, *args, **kwargs)

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        namespace['to_onnx'](learner, simplify=False)

        # onnxsim should not have been imported
        self.assertNotIn('onnxsim', import_calls)


class TestToOnnxFilePath(unittest.TestCase):
    """Test file path construction."""

    def test_default_fname(self):
        """Default fname should be 'export.onnx'."""
        learner = FakeLearner(tmp_path='/tmp/myproject', n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()
        join_calls = []

        def mock_join(fname, path, ext='.onnx'):
            join_calls.append((fname, path, ext))
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        result = namespace['to_onnx'](learner)

        # Verify join_path_file was called with correct path
        self.assertEqual(len(join_calls), 1)
        fname_arg, path_arg, ext_arg = join_calls[0]
        self.assertEqual(fname_arg, Path('export.onnx'))
        self.assertEqual(path_arg, Path('/tmp/myproject/models'))
        self.assertEqual(ext_arg, '.onnx')

    def test_custom_fname(self):
        """Custom fname should be forwarded to join_path_file."""
        learner = FakeLearner(tmp_path='/tmp/myproject', n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()
        join_calls = []

        def mock_join(fname, path, ext='.onnx'):
            join_calls.append((fname, path, ext))
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        result = namespace['to_onnx'](learner, fname='my_model.onnx')

        fname_arg, path_arg, ext_arg = join_calls[0]
        self.assertEqual(fname_arg, Path('my_model.onnx'))

    def test_returns_path(self):
        """to_onnx should return the output file path."""
        learner = FakeLearner(tmp_path='/tmp/myproject', n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        result = namespace['to_onnx'](learner)

        self.assertIsInstance(result, Path)
        self.assertEqual(result, Path('/tmp/myproject/models/export.onnx'))


class TestToOnnxKwargs(unittest.TestCase):
    """Test that extra kwargs are passed to torch.onnx.export."""

    def test_extra_kwargs_forwarded(self):
        """Extra **kwargs should be forwarded to torch.onnx.export."""
        learner = FakeLearner(n_inp=1, batch=[np.zeros((4, 3, 32, 32))])

        learner_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')
        learner_path = os.path.abspath(learner_path)

        with open(learner_path, 'r') as f:
            source = f.read()

        marker = '# %%\n@patch\ndef to_onnx'
        idx = source.find(marker)
        cell_source = source[idx:]
        cell_lines = cell_source.split('\n')[1:]

        mock_torch = MagicMock()

        def mock_join(fname, path, ext='.onnx'):
            return path / (fname.name if hasattr(fname, 'name') else str(fname))

        namespace = {
            '__builtins__': __builtins__,
            'patch': lambda f: f,
            'rank_distrib': MagicMock(return_value=0),
            'get_model': lambda m: m,
            'join_path_file': mock_join,
            'Path': Path,
            'torch': mock_torch,
            'Learner': type('Learner', (), {}),
            'getattr': getattr,
        }

        exec(compile('\n'.join(cell_lines), learner_path, 'exec'), namespace)
        namespace['to_onnx'](learner, verbose=True, training=2)

        call_kwargs = mock_torch.onnx.export.call_args[1]
        self.assertTrue(call_kwargs['verbose'])
        self.assertEqual(call_kwargs['training'], 2)


if __name__ == '__main__':
    unittest.main()
