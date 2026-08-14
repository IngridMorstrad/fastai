"""Tests for fastai.check_env module.

Validates the environment checking logic with mocked dependencies.
"""
import sys
import os
import types
import importlib
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.check_env import (
    check_env, main, _check_python, _check_torch, _check_cuda,
    _check_nvidia_driver, _check_dependency, _format_results,
)


class TestCheckPython:
    """Tests for Python version checking."""

    def test_returns_ok_status(self):
        result = _check_python()
        assert result['name'] == 'Python'
        # We're running Python 3.9+ in tests
        assert result['status'] == 'OK'
        assert result['version'] is not None

    def test_version_format(self):
        result = _check_python()
        parts = result['version'].split('.')
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestCheckTorch:
    """Tests for PyTorch checking."""

    def test_torch_not_installed(self):
        with patch.dict(sys.modules, {'torch': None}):
            with patch('fastai.check_env._try_import', return_value=(None, 'No module named torch')):
                result = _check_torch()
                assert result['status'] == 'FAIL'
                assert result['version'] is None
                assert 'Not installed' in result['detail']

    def test_torch_installed(self):
        fake_torch = MagicMock()
        fake_torch.__version__ = '2.1.0'
        with patch('fastai.check_env._try_import', return_value=(fake_torch, None)):
            result = _check_torch()
            assert result['status'] == 'OK'
            assert result['version'] == '2.1.0'


class TestCheckCuda:
    """Tests for CUDA checking."""

    def test_no_torch(self):
        with patch('fastai.check_env._try_import', return_value=(None, 'No module')):
            result = _check_cuda()
            assert result['status'] == 'SKIP'

    def test_cuda_not_available(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch('fastai.check_env._try_import', return_value=(fake_torch, None)):
            result = _check_cuda()
            assert result['status'] == 'WARN'
            assert 'CPU only' in result['detail']

    def test_cuda_available(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.version.cuda = '12.1'
        fake_torch.cuda.device_count.return_value = 1
        fake_torch.cuda.get_device_name.return_value = 'NVIDIA RTX 4090'
        with patch('fastai.check_env._try_import', return_value=(fake_torch, None)):
            result = _check_cuda()
            assert result['status'] == 'OK'
            assert result['version'] == '12.1'
            assert 'RTX 4090' in result['detail']


class TestCheckNvidiaDriver:
    """Tests for NVIDIA driver checking."""

    def test_nvidia_smi_not_found(self):
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = _check_nvidia_driver()
            assert result['status'] == 'WARN'
            assert 'not found' in result['detail']

    def test_nvidia_smi_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '535.129.03\n'
        with patch('subprocess.run', return_value=mock_result):
            result = _check_nvidia_driver()
            assert result['status'] == 'OK'
            assert result['version'] == '535.129.03'

    def test_nvidia_smi_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch('subprocess.run', return_value=mock_result):
            result = _check_nvidia_driver()
            assert result['status'] == 'WARN'


class TestCheckDependency:
    """Tests for individual dependency checking."""

    def test_missing_dependency(self):
        with patch('fastai.check_env._try_import', return_value=(None, 'No module')):
            result = _check_dependency('nonexistent_pkg')
            assert result['status'] == 'FAIL'
            assert result['version'] is None

    def test_installed_dependency(self):
        fake_mod = MagicMock()
        fake_mod.__version__ = '1.2.3'
        with patch('fastai.check_env._try_import', return_value=(fake_mod, None)):
            result = _check_dependency('some_pkg')
            assert result['status'] == 'OK'
            assert result['version'] == '1.2.3'

    def test_version_too_low(self):
        fake_mod = MagicMock()
        fake_mod.__version__ = '0.1.0'
        with patch('fastai.check_env._try_import', return_value=(fake_mod, None)):
            result = _check_dependency('some_pkg', min_version='1.0.0')
            assert result['status'] == 'WARN'
            assert 'recommended' in result['detail']

    def test_version_meets_minimum(self):
        fake_mod = MagicMock()
        fake_mod.__version__ = '2.0.0'
        with patch('fastai.check_env._try_import', return_value=(fake_mod, None)):
            result = _check_dependency('some_pkg', min_version='1.0.0')
            assert result['status'] == 'OK'


class TestFormatResults:
    """Tests for output formatting."""

    def test_all_ok(self):
        results = [
            {'name': 'Python', 'version': '3.11.0', 'status': 'OK', 'detail': None},
            {'name': 'PyTorch', 'version': '2.1.0', 'status': 'OK', 'detail': None},
        ]
        output = _format_results(results)
        assert 'All checks passed' in output
        assert '✓' in output

    def test_has_failure(self):
        results = [
            {'name': 'Python', 'version': '3.11.0', 'status': 'OK', 'detail': None},
            {'name': 'torch', 'version': None, 'status': 'FAIL', 'detail': 'Not installed'},
        ]
        output = _format_results(results)
        assert 'issues detected' in output
        assert '✗' in output

    def test_has_warning(self):
        results = [
            {'name': 'CUDA', 'version': None, 'status': 'WARN', 'detail': 'Not available'},
        ]
        output = _format_results(results)
        assert '⚠' in output
        assert 'issues detected' in output

    def test_skip_status(self):
        results = [
            {'name': 'CUDA', 'version': None, 'status': 'SKIP', 'detail': 'Skipped'},
        ]
        output = _format_results(results)
        assert '–' in output


class TestCheckEnv:
    """Integration tests for check_env."""

    def test_returns_list(self):
        results = check_env()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_results_have_required_keys(self):
        results = check_env()
        for r in results:
            assert 'name' in r
            assert 'version' in r
            assert 'status' in r
            assert 'detail' in r
            assert r['status'] in ('OK', 'WARN', 'FAIL', 'SKIP')

    def test_python_always_present(self):
        results = check_env()
        names = [r['name'] for r in results]
        assert 'Python' in names


class TestMain:
    """Tests for the main CLI entry point."""

    def test_main_runs_without_crash(self, capsys):
        # In our test env, torch etc. won't be installed, so we expect FAIL exit
        try:
            main()
        except SystemExit as e:
            # Expected: exit(1) because torch is not installed
            assert e.code == 1
        captured = capsys.readouterr()
        assert 'fastai Environment Check' in captured.out

    def test_main_exits_zero_when_all_ok(self, capsys):
        fake_results = [
            {'name': 'Python', 'version': '3.11.0', 'status': 'OK', 'detail': None},
        ]
        with patch('fastai.check_env.check_env', return_value=fake_results):
            # Should not raise SystemExit
            main()
        captured = capsys.readouterr()
        assert 'All checks passed' in captured.out

    def test_main_exits_one_on_failure(self):
        fake_results = [
            {'name': 'torch', 'version': None, 'status': 'FAIL', 'detail': 'Not installed'},
        ]
        with patch('fastai.check_env.check_env', return_value=fake_results):
            try:
                main()
                assert False, "Should have raised SystemExit"
            except SystemExit as e:
                assert e.code == 1
