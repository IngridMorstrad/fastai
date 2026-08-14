"""Validate GPU drivers, CUDA version, and dependency compatibility.

Run as:
    python -m fastai.check_env

Or, if installed as a console script:
    fastai_check_env
"""

import sys
import importlib
import subprocess
import os


__all__ = ['check_env', 'main']


def _check_python():
    """Check Python version meets minimum requirements."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = (v.major, v.minor) >= (3, 9)
    return {
        'name': 'Python',
        'version': version_str,
        'status': 'OK' if ok else 'FAIL',
        'detail': None if ok else 'fastai requires Python >= 3.9',
    }


def _try_import(module_name):
    """Try to import a module, return (module, None) or (None, error_string)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except ImportError as e:
        return None, str(e)


def _check_torch():
    """Check PyTorch installation and version."""
    mod, err = _try_import('torch')
    if mod is None:
        return {
            'name': 'PyTorch',
            'version': None,
            'status': 'FAIL',
            'detail': f'Not installed: {err}',
        }
    version = getattr(mod, '__version__', 'unknown')
    return {
        'name': 'PyTorch',
        'version': version,
        'status': 'OK',
        'detail': None,
    }


def _check_cuda():
    """Check CUDA availability and version via PyTorch."""
    mod, err = _try_import('torch')
    if mod is None:
        return {
            'name': 'CUDA',
            'version': None,
            'status': 'SKIP',
            'detail': 'PyTorch not installed; cannot check CUDA',
        }
    cuda_available = mod.cuda.is_available()
    if not cuda_available:
        return {
            'name': 'CUDA',
            'version': None,
            'status': 'WARN',
            'detail': 'CUDA not available (training will use CPU only)',
        }
    cuda_version = mod.version.cuda or 'unknown'
    device_count = mod.cuda.device_count()
    devices = []
    for i in range(device_count):
        devices.append(mod.cuda.get_device_name(i))
    detail = f"{device_count} GPU(s): {', '.join(devices)}"
    return {
        'name': 'CUDA',
        'version': cuda_version,
        'status': 'OK',
        'detail': detail,
    }


def _check_nvidia_driver():
    """Check NVIDIA driver via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip().split('\n')[0]
            return {
                'name': 'NVIDIA Driver',
                'version': driver_version,
                'status': 'OK',
                'detail': None,
            }
        else:
            return {
                'name': 'NVIDIA Driver',
                'version': None,
                'status': 'WARN',
                'detail': 'nvidia-smi failed (no NVIDIA GPU or driver not installed)',
            }
    except FileNotFoundError:
        return {
            'name': 'NVIDIA Driver',
            'version': None,
            'status': 'WARN',
            'detail': 'nvidia-smi not found in PATH',
        }
    except subprocess.TimeoutExpired:
        return {
            'name': 'NVIDIA Driver',
            'version': None,
            'status': 'WARN',
            'detail': 'nvidia-smi timed out',
        }


def _check_dependency(name, min_version=None):
    """Check a single Python dependency."""
    mod, err = _try_import(name)
    if mod is None:
        return {
            'name': name,
            'version': None,
            'status': 'FAIL',
            'detail': f'Not installed: {err}',
        }
    version = getattr(mod, '__version__', None)
    if version is None:
        version = getattr(mod, 'VERSION', 'unknown')
    status = 'OK'
    detail = None
    if min_version and version != 'unknown':
        from packaging.version import parse as parse_version
        try:
            if parse_version(str(version)) < parse_version(min_version):
                status = 'WARN'
                detail = f'Version {version} < recommended {min_version}'
        except Exception:
            pass
    return {
        'name': name,
        'version': str(version) if version else None,
        'status': status,
        'detail': detail,
    }


# Core dependencies that fastai requires
_CORE_DEPS = [
    ('fastcore', '1.5.29'),
    ('fastdownload', '0.0.5'),
    ('torchvision', '0.11'),
    ('matplotlib', None),
    ('pandas', None),
    ('numpy', None),
    ('scipy', None),
    ('scikit-learn', None),
    ('PIL', None),  # pillow
    ('spacy', None),
    ('yaml', None),  # pyyaml
    ('fastprogress', '0.2.4'),
]


def check_env():
    """Run all environment checks and return a list of result dicts.

    Each result dict has keys: name, version, status, detail.
    Status is one of: OK, WARN, FAIL, SKIP.
    """
    results = []
    results.append(_check_python())
    results.append(_check_torch())
    results.append(_check_cuda())
    results.append(_check_nvidia_driver())

    for dep_name, min_ver in _CORE_DEPS:
        results.append(_check_dependency(dep_name, min_ver))

    return results


def _format_results(results):
    """Format results for terminal output."""
    lines = []
    lines.append("fastai Environment Check")
    lines.append("=" * 50)

    max_name = max(len(r['name']) for r in results)
    has_issues = False

    for r in results:
        status_icon = {'OK': '✓', 'WARN': '⚠', 'FAIL': '✗', 'SKIP': '–'}[r['status']]
        version_str = r['version'] or 'N/A'
        line = f"  {status_icon} {r['name']:<{max_name}}  {version_str}"
        if r['detail']:
            line += f"  ({r['detail']})"
        lines.append(line)
        if r['status'] in ('FAIL', 'WARN'):
            has_issues = True

    lines.append("=" * 50)
    if has_issues:
        lines.append("Some issues detected. See details above.")
    else:
        lines.append("All checks passed!")

    return '\n'.join(lines)


def main():
    """Entry point for the fastai_check_env CLI command."""
    results = check_env()
    print(_format_results(results))

    # Exit with non-zero if any FAIL
    if any(r['status'] == 'FAIL' for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
