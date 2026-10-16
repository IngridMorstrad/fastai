"""Tests for Learner.to_api() deployment endpoint generation.

Validates that the to_api method generates correct FastAPI and Flask
prediction endpoint scripts from a trained model.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.learner import Learner


class TestToApi(unittest.TestCase):
    """Test Learner.to_api() generates correct API endpoint scripts."""

    def _make_learner(self, tmp_dir):
        """Create a minimal mock Learner with path set to tmp_dir."""
        learn = MagicMock(spec=Learner)
        learn.path = Path(tmp_dir)
        # Create a fake export.pkl so to_api doesn't try to call export()
        (learn.path / 'export.pkl').write_text('fake model')
        return learn

    def _call_to_api(self, learn, **kwargs):
        """Call the real to_api function with the mock learner."""
        # to_api is patched onto Learner via @patch from fastcore
        # We can call it as an unbound function
        return Learner.to_api(learn, **kwargs)

    def test_fastapi_generates_file(self):
        """to_api with framework='fastapi' creates the output file."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            result = self._call_to_api(learn, fname='app.py', framework='fastapi')
            self.assertTrue(result.exists())
            self.assertEqual(result.name, 'app.py')

    def test_fastapi_content_has_endpoints(self):
        """Generated FastAPI script has /predict endpoint and uvicorn."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            self._call_to_api(learn, fname='app.py', framework='fastapi')
            content = (Path(tmp) / 'app.py').read_text()
            self.assertIn('from fastapi import FastAPI', content)
            self.assertIn('@app.post("/predict")', content)
            self.assertIn('uvicorn.run', content)
            self.assertIn('load_learner', content)

    def test_flask_generates_file(self):
        """to_api with framework='flask' creates the output file."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            result = self._call_to_api(learn, fname='server.py', framework='flask')
            self.assertTrue(result.exists())
            self.assertEqual(result.name, 'server.py')

    def test_flask_content_has_endpoints(self):
        """Generated Flask script has /predict route and Flask app."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            self._call_to_api(learn, fname='server.py', framework='flask')
            content = (Path(tmp) / 'server.py').read_text()
            self.assertIn('from flask import Flask', content)
            self.assertIn('@app.route("/predict"', content)
            self.assertIn('app.run', content)
            self.assertIn('load_learner', content)

    def test_invalid_framework_raises(self):
        """to_api with unsupported framework raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            with self.assertRaises(ValueError) as ctx:
                self._call_to_api(learn, framework='django')
            self.assertIn('django', str(ctx.exception))

    def test_custom_port(self):
        """Generated script uses the specified port number."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            self._call_to_api(learn, fname='app.py', framework='fastapi', port=9090)
            content = (Path(tmp) / 'app.py').read_text()
            self.assertIn('9090', content)

    def test_exports_model_if_missing(self):
        """to_api calls self.export() if export.pkl does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            learn = self._make_learner(tmp)
            # Remove the export.pkl we created
            (learn.path / 'export.pkl').unlink()
            # Mock export to just create the file
            def fake_export(**kwargs):
                (learn.path / 'export.pkl').write_text('exported')
            learn.export = MagicMock(side_effect=fake_export)
            self._call_to_api(learn, fname='app.py', framework='fastapi')
            learn.export.assert_called_once()


if __name__ == '__main__':
    unittest.main()
