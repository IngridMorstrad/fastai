"""Tests for the fastai_check_env environment validator.

These tests deliberately run WITHOUT a GPU and WITHOUT torch installed. Where a
test needs torch to be present (or absent) in a particular state, it injects a
fake ``torch`` module into ``sys.modules`` so the behaviour is fully
deterministic and GPU-independent.
"""

import json
import sys
import types

import pytest

from fastai import check_env as ce
from fastai.check_env import (
    ERROR,
    INFO,
    OK,
    WARNING,
    check_cuda,
    check_dependencies,
    check_env,
    check_python,
    check_torch,
    check_version_constraints,
    format_report,
    main,
)


@pytest.fixture
def no_torch(monkeypatch):
    """Ensure ``import torch`` fails for the duration of a test."""
    monkeypatch.setitem(sys.modules, 'torch', None)  # None -> ImportError on import
    yield


def _fake_torch(cuda_available=False, compiled_cuda=None, device_names=None,
                cudnn_available=False, cudnn_version=None, version='2.1.0'):
    """Build a fake ``torch`` module with a configurable CUDA surface."""
    device_names = device_names or []
    mod = types.ModuleType('torch')
    mod.__version__ = version
    version_ns = types.SimpleNamespace(cuda=compiled_cuda)
    mod.version = version_ns

    cuda_ns = types.SimpleNamespace()
    cuda_ns.is_available = lambda: cuda_available
    cuda_ns.device_count = lambda: len(device_names)
    cuda_ns.get_device_name = lambda i: device_names[i]
    mod.cuda = cuda_ns

    cudnn_ns = types.SimpleNamespace()
    cudnn_ns.is_available = lambda: cudnn_available
    cudnn_ns.version = lambda: cudnn_version
    mod.backends = types.SimpleNamespace(cudnn=cudnn_ns)
    return mod


class TestCheckPython:
    """Python version check."""

    def test_current_python_passes(self):
        """The interpreter running the tests (>=3.9) is reported OK."""
        res = check_python()
        assert res['status'] == OK
        assert res['name'] == 'python'

    def test_old_python_flagged_error(self):
        """A min_python above the current interpreter yields a hard ERROR."""
        future = (sys.version_info.major, sys.version_info.minor + 1)
        res = check_python(min_python=future)
        assert res['status'] == ERROR

    def test_exactly_min_python_passes(self):
        """The current version is OK when it equals min_python."""
        cur = (sys.version_info.major, sys.version_info.minor)
        res = check_python(min_python=cur)
        assert res['status'] == OK


class TestCheckTorch:
    """torch presence check degrades gracefully."""

    def test_torch_absent_reports_not_installed(self, no_torch):
        """When torch cannot be imported, a WARNING is returned (no raise)."""
        res = check_torch()
        assert res['status'] == WARNING
        assert res['installed'] is False
        assert res['version'] is None

    def test_torch_present_reports_version(self, monkeypatch):
        """A present torch is reported OK with its version string."""
        monkeypatch.setitem(sys.modules, 'torch', _fake_torch(version='2.1.0'))
        res = check_torch()
        assert res['status'] == OK
        assert res['installed'] is True
        assert res['version'] == '2.1.0'


class TestCheckCuda:
    """CUDA/GPU inspection degrades gracefully in every direction."""

    def test_cuda_without_torch(self, no_torch):
        """No torch -> WARNING, never an exception."""
        res = check_cuda()
        assert res['status'] == WARNING
        assert res['available'] is False

    def test_cuda_unavailable_cpu_build(self, monkeypatch):
        """CPU-only torch build reports CUDA unavailable without raising."""
        monkeypatch.setitem(sys.modules, 'torch',
                            _fake_torch(cuda_available=False, compiled_cuda=None))
        res = check_cuda()
        assert res['status'] == WARNING
        assert res['available'] is False
        assert res['device_count'] == 0

    def test_cuda_available_reports_devices(self, monkeypatch):
        """When CUDA is available, device names and cuDNN info are reported."""
        monkeypatch.setitem(sys.modules, 'torch', _fake_torch(
            cuda_available=True, compiled_cuda='12.1',
            device_names=['Tesla T4', 'Tesla T4'],
            cudnn_available=True, cudnn_version=8902))
        res = check_cuda()
        assert res['status'] == OK
        assert res['available'] is True
        assert res['compiled_cuda'] == '12.1'
        assert res['device_count'] == 2
        assert res['devices'] == ['Tesla T4', 'Tesla T4']
        assert res['cudnn_available'] is True
        assert res['cudnn_version'] == 8902

    def test_cuda_is_available_raises_is_caught(self, monkeypatch):
        """An exception from torch.cuda.is_available is caught and reported."""
        mod = _fake_torch()
        def boom():
            raise RuntimeError('driver error')
        mod.cuda.is_available = boom
        monkeypatch.setitem(sys.modules, 'torch', mod)
        res = check_cuda()
        assert res['status'] == WARNING
        assert res['available'] is False


class TestCheckDependencies:
    """Key dependency presence reporting."""

    def test_missing_dependency_reported_not_installed(self):
        """A clearly-absent dependency is reported as WARNING, not an error."""
        results = check_dependencies(deps=['this_package_does_not_exist_xyz'])
        assert len(results) == 1
        assert results[0]['status'] == WARNING
        assert results[0]['installed'] is False

    def test_present_dependency_reported_with_version(self):
        """pytest is installed, so it is reported OK with a version."""
        results = check_dependencies(deps=['pytest'])
        assert results[0]['status'] == OK
        assert results[0]['installed'] is True
        assert results[0]['version'] is not None


class TestVersionConstraints:
    """Version-constraint validation logic."""

    def test_in_range_version_passes(self, monkeypatch):
        """An in-range installed version is OK."""
        monkeypatch.setattr(ce, '_dist_version', lambda name: '1.11.0')
        results = check_version_constraints(constraints={'torch': '>=1.10,<2.6'})
        assert results[0]['status'] == OK

    def test_out_of_range_version_flagged_error(self, monkeypatch):
        """An out-of-range installed version is a hard ERROR."""
        monkeypatch.setattr(ce, '_dist_version', lambda name: '2.9.0')
        results = check_version_constraints(constraints={'torch': '>=1.10,<2.6'})
        assert results[0]['status'] == ERROR

    def test_lower_bound_violation_flagged_error(self, monkeypatch):
        """A version below the lower bound is a hard ERROR."""
        monkeypatch.setattr(ce, '_dist_version', lambda name: '1.0.0')
        results = check_version_constraints(constraints={'fastcore': '>=1.5.29'})
        assert results[0]['status'] == ERROR

    def test_missing_dependency_is_info_not_error(self, monkeypatch):
        """A not-installed constrained dependency is INFO, never an error."""
        monkeypatch.setattr(ce, '_dist_version', lambda name: None)
        results = check_version_constraints(constraints={'torch': '>=1.10,<2.6'})
        assert results[0]['status'] == INFO

    def test_unverifiable_without_packaging_is_warning(self, monkeypatch):
        """When packaging cannot verify, the result is WARNING, never ERROR."""
        monkeypatch.setattr(ce, '_dist_version', lambda name: '1.11.0')
        monkeypatch.setattr(ce, '_specifier_contains',
                            lambda spec, ver: (None, 'packaging is not installed'))
        results = check_version_constraints(constraints={'torch': '>=1.10,<2.6'})
        assert results[0]['status'] == WARNING


class TestCheckEnv:
    """Top-level aggregation and the main() entry point."""

    def test_check_env_runs_without_torch(self, no_torch):
        """check_env() produces a full report even with torch absent."""
        report = check_env()
        assert 'checks' in report and 'summary' in report
        assert 'compatible' in report and 'platform' in report
        assert len(report['checks']) >= 3

    def test_json_output_is_valid(self, no_torch, capsys):
        """--json mode emits valid JSON containing the expected top-level keys."""
        main(['--json'])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert set(['platform', 'checks', 'summary', 'compatible']).issubset(parsed)
        assert isinstance(parsed['checks'], list)

    def test_main_returns_zero_when_compatible(self, no_torch, monkeypatch):
        """main() returns 0 when no hard incompatibility is present."""
        # Force all constraints to resolve as not-installed (INFO), so no ERROR.
        monkeypatch.setattr(ce, '_dist_version', lambda name: None)
        rc = main([])
        assert rc == 0

    def test_main_returns_nonzero_on_hard_incompatibility(self, monkeypatch, capsys):
        """main() returns non-zero when an installed dep violates its constraint."""
        monkeypatch.setattr(ce, 'check_version_constraints',
                            lambda constraints=None: [
                                {'name': 'constraint:torch', 'status': ERROR,
                                 'detail': 'torch 9.9 violates required >=1.10,<2.6'}])
        rc = main([])
        assert rc != 0

    def test_format_report_mentions_result(self, no_torch):
        """The human-readable report includes a Result line."""
        text = format_report(check_env())
        assert 'Result' in text
        assert 'fastai environment check' in text


class TestSettingsIniConsistency:
    """The in-module constants must match settings.ini (single source of truth)."""

    def _settings(self):
        import os
        root = os.path.join(os.path.dirname(__file__), '..')
        from configparser import ConfigParser
        cp = ConfigParser(delimiters=['='])
        cp.read(os.path.join(root, 'settings.ini'))
        return cp['DEFAULT']

    def test_min_python_matches_settings(self):
        """_MIN_PYTHON mirrors settings.ini min_python."""
        settings = self._settings()
        expected = tuple(int(x) for x in settings['min_python'].split('.'))
        assert ce._MIN_PYTHON == expected

    def test_torch_constraint_matches_pip_requirements(self):
        """The torch constraint mirrors settings.ini pip_requirements."""
        settings = self._settings()
        pip_req = settings['pip_requirements']
        assert 'torch' + ce._CONSTRAINED_REQUIREMENTS['torch'] in pip_req.replace(' ', '')

    def test_console_script_registered(self):
        """settings.ini registers the fastai_check_env console script."""
        settings = self._settings()
        assert 'fastai_check_env=fastai.check_env:main' in settings['console_scripts']
