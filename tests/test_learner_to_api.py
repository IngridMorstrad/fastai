"""Tests for `Learner.to_api` (generates a prediction web-service from a learner).

`to_api` is a `@patch` method on `Learner`, and importing `fastai.learner`
transitively imports torch (not installed in this environment). So instead of
importing the module, we extract the `to_api` function definition straight from
the generated source (`fastai/learner.py`) and exec it against a lightweight
shim. The method is pure string generation, so it runs with no torch/flask/fastapi
installed.
"""
import ast
import os
import tempfile
import unittest
from pathlib import Path


LEARNER_PY = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'learner.py')


def _load_to_api():
    """Extract and compile the `to_api` function from the generated library source.

    We read `fastai/learner.py`, locate the `to_api` function node via AST, and
    exec just that definition in a namespace with a no-op `patch` decorator and a
    placeholder `Learner` type. This avoids importing the real module (which pulls
    in torch) while exercising the exact source nbdev generates.
    """
    source = Path(LEARNER_PY).read_text()
    tree = ast.parse(source)
    func_node = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == 'to_api'),
        None,
    )
    assert func_node is not None, "to_api function not found in fastai/learner.py"

    # Drop the @patch decorator (we are defining a plain function in the shim).
    func_node.decorator_list = []
    module = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, filename=LEARNER_PY, mode='exec')

    class Learner:  # placeholder referenced by the `self:Learner` annotation
        pass

    ns = {'Learner': Learner}
    exec(code, ns)
    return ns['to_api']


class FakeLearner:
    """Minimal stand-in for `Learner` exposing only what `to_api` touches."""
    def __init__(self, path):
        self.path = Path(path)


class TestLearnerToApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.to_api = staticmethod(_load_to_api())

    def test_returns_non_empty_source_string(self):
        src = type(self).to_api(FakeLearner(tempfile.gettempdir()), fname=None)
        self.assertIsInstance(src, str)
        self.assertTrue(len(src) > 0)

    def test_flask_source_contents(self):
        src = type(self).to_api(FakeLearner(tempfile.gettempdir()), fname=None)
        self.assertIn("@app.route('/predict'", src)
        self.assertIn("load_learner(", src)
        self.assertIn(".predict(", src)
        self.assertIn("Flask", src)
        # health check endpoint
        self.assertIn("@app.route('/'", src)

    def test_flask_is_default_framework(self):
        default_src = type(self).to_api(FakeLearner(tempfile.gettempdir()), fname=None)
        flask_src = type(self).to_api(
            FakeLearner(tempfile.gettempdir()), fname=None, framework='flask')
        self.assertEqual(default_src, flask_src)

    def test_fastapi_source_contents(self):
        src = type(self).to_api(
            FakeLearner(tempfile.gettempdir()), fname=None, framework='fastapi')
        self.assertIn("FastAPI(", src)
        self.assertIn("@app.post", src)
        self.assertIn("load_learner(", src)
        self.assertIn(".predict(", src)
        # health check endpoint
        self.assertIn("@app.get('/')", src)

    def test_unsupported_framework_raises(self):
        with self.assertRaises(ValueError):
            type(self).to_api(
                FakeLearner(tempfile.gettempdir()), fname=None, framework='django')

    def test_writes_file_with_matching_contents(self):
        with tempfile.TemporaryDirectory() as d:
            learn = FakeLearner(d)
            returned = type(self).to_api(learn, fname='app.py')
            written_path = Path(d) / 'app.py'
            self.assertTrue(written_path.exists())
            self.assertEqual(written_path.read_text(), returned)

    def test_fname_none_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            learn = FakeLearner(d)
            returned = type(self).to_api(learn, fname=None)
            # Nothing should be written to the learner's path
            self.assertEqual(list(Path(d).iterdir()), [])
            self.assertIsInstance(returned, str)
            self.assertTrue(len(returned) > 0)

    def test_export_fname_and_host_port_propagate(self):
        src = type(self).to_api(
            FakeLearner(tempfile.gettempdir()), fname=None,
            host='0.0.0.0', port=8080, export_fname='model.pkl')
        self.assertIn("load_learner('model.pkl')", src)
        self.assertIn("'0.0.0.0'", src)
        self.assertIn("8080", src)

    def test_generated_flask_source_is_valid_python(self):
        src = type(self).to_api(FakeLearner(tempfile.gettempdir()), fname=None)
        # Should parse as valid Python (we are not executing it, just parsing)
        ast.parse(src)

    def test_generated_fastapi_source_is_valid_python(self):
        src = type(self).to_api(
            FakeLearner(tempfile.gettempdir()), fname=None, framework='fastapi')
        ast.parse(src)


if __name__ == '__main__':
    unittest.main()
