"""fastai_check_env - one-step validation of the fastai runtime environment.

Validates, in a single command, the three things that most often break a fastai
install: the Python interpreter version, the GPU/CUDA stack exposed through
PyTorch (drivers, compiled CUDA version, device count, cuDNN), and the versions
of fastai's key dependencies against the constraints declared in ``settings.ini``.

Every individual check is written to *report* rather than *raise*: a missing or
broken dependency (including torch itself) is surfaced as a status line instead
of crashing the tool. That is the whole point of the command - it has to run and
produce a useful diagnosis precisely in the environments that are misconfigured.

Usage (command line, registered as a console script in ``settings.ini``)::

    fastai_check_env            # human-readable report
    fastai_check_env --json     # machine-readable JSON report

Usage (programmatic)::

    from fastai.check_env import check_env, main
    report = check_env()        # structured dict of all checks
    exit_code = main([])        # 0 when compatible, non-zero on hard incompat.

The process exit code is ``0`` when no *hard* incompatibility is found and
non-zero otherwise. A hard incompatibility is defined narrowly as either the
running Python being older than ``min_python``, or an *installed* required
dependency whose version violates the range declared in ``settings.ini``. A
dependency that is simply not installed, or a version that cannot be verified
(e.g. because ``packaging`` is unavailable), is reported but never fails the run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

__all__ = [
    'OK', 'WARNING', 'ERROR', 'INFO',
    'check_python', 'check_torch', 'check_cuda', 'check_dependencies',
    'check_version_constraints', 'check_env', 'format_report', 'main',
]

# ---------------------------------------------------------------------------
# Status / severity constants
# ---------------------------------------------------------------------------
# Only ERROR causes a non-zero exit code. WARNING/INFO/OK are informational and
# never fail the run, so that a missing optional piece never blocks the user.
OK = 'ok'
WARNING = 'warning'
ERROR = 'error'
INFO = 'info'

# Dependencies whose version is validated against a settings.ini range. Each
# entry maps the distribution name to a PEP 440 specifier string. These mirror
# the `requirements`/`pip_requirements` lines in settings.ini; keeping them here
# (rather than parsing settings.ini at runtime) means the CLI works even when it
# is installed as a wheel without settings.ini on disk, and the values are
# cross-checked against settings.ini by the test suite.
_CONSTRAINED_REQUIREMENTS = {
    'torch': '>=1.10,<2.6',
    'fastcore': '>=1.5.29',
    'torchvision': '>=0.11',
    'fastprogress': '>=0.2.4',
    'fastdownload': '>=0.0.5,<2',
    'pillow': '>=9.0.0',
}

# Required dependencies reported for presence/version even when unconstrained.
# `pillow` imports as `PIL`; `pyyaml` as `yaml`; `scikit-learn` as `sklearn`.
_REQUIRED_DEPENDENCIES = [
    'fastcore', 'torch', 'torchvision', 'fastprogress', 'fastdownload',
    'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'packaging',
    'pillow', 'requests', 'pyyaml', 'spacy',
]

# Distribution name -> importable module name, where they differ.
_IMPORT_NAME = {
    'pillow': 'PIL',
    'pyyaml': 'yaml',
    'scikit-learn': 'sklearn',
}

# Minimum Python version. Mirrors `min_python` in settings.ini; verified against
# settings.ini by the tests.
_MIN_PYTHON = (3, 9)


def _result(name, status, detail, **extra):
    "Build a single check result dict with a stable schema."
    out = {'name': name, 'status': status, 'detail': detail}
    out.update(extra)
    return out


def _dist_version(dist_name):
    """Return the installed version string for *dist_name*, or ``None``.

    Tries :mod:`importlib.metadata` first (works for any installed distribution)
    and falls back to importing the module and reading ``__version__``. Never
    raises: an uninstalled or unversioned package yields ``None``.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(dist_name)
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    mod_name = _IMPORT_NAME.get(dist_name, dist_name.replace('-', '_'))
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, '__version__', None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual checks (each returns a result dict and never raises)
# ---------------------------------------------------------------------------

def check_python(min_python=_MIN_PYTHON):
    """Check the running Python version against *min_python*.

    Returns an ERROR result when the interpreter is older than the minimum
    supported version (a hard incompatibility), OK otherwise.
    """
    cur = sys.version_info[:3]
    required = '.'.join(str(p) for p in min_python)
    current = '.'.join(str(p) for p in cur)
    if cur[:len(min_python)] < tuple(min_python):
        return _result('python', ERROR,
                       f'Python {current} is older than the minimum supported {required}',
                       current=current, required=f'>={required}')
    return _result('python', OK,
                   f'Python {current} satisfies the minimum {required}',
                   current=current, required=f'>={required}')


def check_torch():
    """Report whether PyTorch is importable and its version.

    When torch is not installed (or fails to import) this returns a WARNING,
    not an error: the tool must still run. The torch *version constraint* is
    validated separately by :func:`check_version_constraints`.
    """
    try:
        import torch
    except Exception as e:
        return _result('torch', WARNING,
                       f'PyTorch is not installed or failed to import ({type(e).__name__})',
                       installed=False, version=None)
    return _result('torch', OK, f'PyTorch {torch.__version__} is installed',
                   installed=True, version=str(torch.__version__))


def check_cuda():
    """Report the GPU/CUDA stack exposed by PyTorch.

    Covers: CUDA availability, the CUDA version torch was compiled against,
    the number of visible devices and their names, and cuDNN availability and
    version. Degrades gracefully end-to-end: no torch, no CUDA build, or no GPU
    all yield informative WARNING/INFO results rather than exceptions.
    """
    try:
        import torch
    except Exception as e:
        return _result('cuda', WARNING,
                       f'Cannot inspect CUDA: PyTorch unavailable ({type(e).__name__})',
                       available=False)

    # torch.version.cuda is None for CPU-only builds.
    try:
        compiled_cuda = getattr(getattr(torch, 'version', None), 'cuda', None)
    except Exception:
        compiled_cuda = None

    try:
        available = bool(torch.cuda.is_available())
    except Exception as e:
        return _result('cuda', WARNING,
                       f'torch.cuda.is_available() raised {type(e).__name__}: {e}',
                       available=False, compiled_cuda=compiled_cuda)

    if not available:
        detail = 'CUDA is not available (no GPU detected or CPU-only build)'
        if compiled_cuda is None:
            detail += '; torch was built without CUDA'
        else:
            detail += f'; torch was built against CUDA {compiled_cuda}'
        return _result('cuda', WARNING, detail,
                       available=False, compiled_cuda=compiled_cuda, device_count=0,
                       devices=[], cudnn_available=False, cudnn_version=None)

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0
    devices = []
    for i in range(device_count):
        try:
            devices.append(torch.cuda.get_device_name(i))
        except Exception:
            devices.append(f'cuda:{i} (name unavailable)')

    try:
        cudnn_available = bool(torch.backends.cudnn.is_available())
    except Exception:
        cudnn_available = False
    try:
        cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    except Exception:
        cudnn_version = None

    detail = (f'CUDA {compiled_cuda} available with {device_count} '
              f'device(s): {", ".join(devices) if devices else "unknown"}')
    return _result('cuda', OK, detail,
                   available=True, compiled_cuda=compiled_cuda,
                   device_count=device_count, devices=devices,
                   cudnn_available=cudnn_available, cudnn_version=cudnn_version)


def check_dependencies(deps=None):
    """Report presence and version of each key fastai dependency.

    A missing dependency is reported as WARNING (``installed=False``) rather
    than failing the run. Returns a list of result dicts, one per dependency.
    """
    if deps is None:
        deps = _REQUIRED_DEPENDENCIES
    results = []
    for dep in deps:
        ver = _dist_version(dep)
        if ver is None:
            results.append(_result(f'dep:{dep}', WARNING, f'{dep} is not installed',
                                   dependency=dep, installed=False, version=None))
        else:
            results.append(_result(f'dep:{dep}', OK, f'{dep} {ver} is installed',
                                   dependency=dep, installed=True, version=str(ver)))
    return results


def _specifier_contains(spec_str, version_str):
    """Return (satisfied, reason) for *version_str* against *spec_str*.

    ``satisfied`` is ``True``/``False`` when it can be determined, or ``None``
    when it cannot be verified (packaging missing, or an unparseable version).
    Never raises.
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except Exception:
        return None, 'packaging is not installed; cannot verify version constraint'
    try:
        parsed = Version(str(version_str))
    except InvalidVersion:
        return None, f'cannot parse version {version_str!r}'
    try:
        spec = SpecifierSet(spec_str)
    except Exception as e:
        return None, f'cannot parse constraint {spec_str!r} ({type(e).__name__})'
    # prereleases=True so e.g. a dev build of an in-range version is accepted.
    return (parsed in spec) or spec.contains(parsed, prereleases=True), None


def check_version_constraints(constraints=None):
    """Validate installed dependency versions against settings.ini ranges.

    For each constrained dependency:
      * not installed -> INFO (nothing to validate, not an error);
      * cannot verify (no packaging / unparseable) -> WARNING;
      * installed and in range -> OK;
      * installed and out of range -> ERROR (a hard incompatibility).

    Returns a list of result dicts, one per constrained dependency.
    """
    if constraints is None:
        constraints = _CONSTRAINED_REQUIREMENTS
    results = []
    for dep, spec_str in constraints.items():
        ver = _dist_version(dep)
        if ver is None:
            results.append(_result(f'constraint:{dep}', INFO,
                                   f'{dep} not installed; constraint {spec_str} not evaluated',
                                   dependency=dep, constraint=spec_str, version=None))
            continue
        satisfied, reason = _specifier_contains(spec_str, ver)
        if satisfied is None:
            results.append(_result(f'constraint:{dep}', WARNING,
                                   f'{dep} {ver}: {reason}',
                                   dependency=dep, constraint=spec_str, version=str(ver)))
        elif satisfied:
            results.append(_result(f'constraint:{dep}', OK,
                                   f'{dep} {ver} satisfies {spec_str}',
                                   dependency=dep, constraint=spec_str, version=str(ver)))
        else:
            results.append(_result(f'constraint:{dep}', ERROR,
                                   f'{dep} {ver} violates required {spec_str}',
                                   dependency=dep, constraint=spec_str, version=str(ver)))
    return results


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def check_env(min_python=_MIN_PYTHON, constraints=None, deps=None):
    """Run every check and return a structured report.

    The report is a dict with a ``platform`` summary, the full ordered list of
    ``checks``, a ``compatible`` boolean (``False`` iff any check is an ERROR),
    and a ``summary`` tally of statuses. Suitable for JSON serialization.
    """
    checks = []
    checks.append(check_python(min_python))
    checks.append(check_torch())
    checks.append(check_cuda())
    checks.extend(check_dependencies(deps))
    checks.extend(check_version_constraints(constraints))

    summary = {OK: 0, WARNING: 0, ERROR: 0, INFO: 0}
    for c in checks:
        summary[c['status']] = summary.get(c['status'], 0) + 1
    compatible = summary.get(ERROR, 0) == 0

    return {
        'platform': {
            'python': platform.python_version(),
            'implementation': platform.python_implementation(),
            'system': platform.system(),
            'machine': platform.machine(),
        },
        'checks': checks,
        'summary': summary,
        'compatible': compatible,
    }


_STATUS_LABEL = {OK: 'OK   ', WARNING: 'WARN ', ERROR: 'FAIL ', INFO: 'INFO '}


def format_report(report):
    "Render *report* (from :func:`check_env`) as a human-readable string."
    plat = report['platform']
    lines = [
        'fastai environment check',
        '=' * 60,
        f"Platform : {plat['implementation']} {plat['python']} on "
        f"{plat['system']} ({plat['machine']})",
        '-' * 60,
    ]
    for c in report['checks']:
        label = _STATUS_LABEL.get(c['status'], c['status'])
        lines.append(f"[{label}] {c['name']}: {c['detail']}")
    lines.append('-' * 60)
    s = report['summary']
    lines.append(
        f"Summary  : {s.get(OK, 0)} ok, {s.get(WARNING, 0)} warning, "
        f"{s.get(ERROR, 0)} error, {s.get(INFO, 0)} info")
    lines.append(
        'Result   : ' +
        ('COMPATIBLE' if report['compatible']
         else 'INCOMPATIBLE (hard version/python conflict detected)'))
    return '\n'.join(lines)


def main(argv=None):
    """Console-script entry point for ``fastai_check_env``.

    Parses arguments, runs the full environment check, prints a human-readable
    report (or JSON with ``--json``), and returns the process exit code: ``0``
    when the environment is compatible, ``1`` when a hard incompatibility is
    present. Returning the int makes it directly usable with ``sys.exit``.
    """
    parser = argparse.ArgumentParser(
        prog='fastai_check_env',
        description='Validate the Python version, GPU/CUDA stack, and fastai '
                    'dependency compatibility in one step.')
    parser.add_argument('--json', action='store_true',
                        help='emit a machine-readable JSON report instead of text')
    args = parser.parse_args(argv)

    report = check_env()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report['compatible'] else 1


if __name__ == '__main__':
    sys.exit(main())
