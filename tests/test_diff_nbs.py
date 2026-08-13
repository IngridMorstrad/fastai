"""Tests for fastai.diff_nbs - semantic notebook diffing CLI."""
import json, os, tempfile, subprocess, sys
import pytest
from pathlib import Path
from fastai.diff_nbs import _strip_nb, _load_nb, diff_nbs, main

# Project root for subprocess PYTHONPATH
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# --- Fixtures ---

def _make_nb(cells):
    """Helper to create a minimal valid notebook dict."""
    return {
        'nbformat': 4,
        'nbformat_minor': 4,
        'metadata': {'kernelspec': {'display_name': 'python3', 'language': 'python', 'name': 'python3'}},
        'cells': cells
    }


def _write_nb(path, nb):
    """Write a notebook dict to a file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f)


# --- Tests for _strip_nb ---

class TestStripNb:
    def test_strips_outputs_and_execution_count(self):
        nb = _make_nb([{
            'cell_type': 'code',
            'source': ['print("hello")'],
            'outputs': [{'text': ['hello\n'], 'output_type': 'stream'}],
            'execution_count': 42,
            'metadata': {'scrolled': True}
        }])
        lines = _strip_nb(nb)
        content = ''.join(lines)
        assert 'print("hello")' in content
        # Outputs and execution_count should not appear
        assert '42' not in content
        assert 'scrolled' not in content

    def test_preserves_source_content(self):
        nb = _make_nb([
            {'cell_type': 'code', 'source': ['x = 1\n', 'y = 2'], 'outputs': [], 'execution_count': None, 'metadata': {}},
            {'cell_type': 'markdown', 'source': ['# Heading\n', 'paragraph'], 'metadata': {}}
        ])
        lines = _strip_nb(nb)
        content = ''.join(lines)
        assert 'x = 1' in content
        assert 'y = 2' in content
        assert '# Heading' in content
        assert 'paragraph' in content

    def test_handles_string_source(self):
        nb = _make_nb([{
            'cell_type': 'code',
            'source': 'a = 1\nb = 2',
            'outputs': [],
            'execution_count': None,
            'metadata': {}
        }])
        lines = _strip_nb(nb)
        content = ''.join(lines)
        assert 'a = 1' in content
        assert 'b = 2' in content

    def test_empty_notebook(self):
        nb = _make_nb([])
        lines = _strip_nb(nb)
        assert lines == []

    def test_cell_type_labels(self):
        nb = _make_nb([
            {'cell_type': 'code', 'source': ['pass'], 'outputs': [], 'execution_count': None, 'metadata': {}},
            {'cell_type': 'markdown', 'source': ['text'], 'metadata': {}}
        ])
        lines = _strip_nb(nb)
        assert '# %% [cell 0] (code)\n' in lines
        assert '# %% [cell 1] (markdown)\n' in lines


# --- Tests for diff_nbs ---

class TestDiffNbs:
    def test_identical_notebooks_no_diff(self):
        nb = _make_nb([{
            'cell_type': 'code',
            'source': ['x = 1'],
            'outputs': [{'text': ['1']}],
            'execution_count': 1,
            'metadata': {}
        }])
        assert diff_nbs(nb, nb) == []

    def test_different_outputs_same_source_no_diff(self):
        nb_a = _make_nb([{
            'cell_type': 'code',
            'source': ['print(1)'],
            'outputs': [{'text': ['1\n']}],
            'execution_count': 1,
            'metadata': {}
        }])
        nb_b = _make_nb([{
            'cell_type': 'code',
            'source': ['print(1)'],
            'outputs': [{'text': ['different output\n']}],
            'execution_count': 99,
            'metadata': {'collapsed': True}
        }])
        assert diff_nbs(nb_a, nb_b) == []

    def test_source_change_produces_diff(self):
        nb_a = _make_nb([{'cell_type': 'code', 'source': ['x = 1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        nb_b = _make_nb([{'cell_type': 'code', 'source': ['x = 2'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        result = diff_nbs(nb_a, nb_b)
        assert len(result) > 0
        diff_text = ''.join(result)
        assert '-x = 1' in diff_text
        assert '+x = 2' in diff_text

    def test_added_cell_produces_diff(self):
        nb_a = _make_nb([{'cell_type': 'code', 'source': ['x = 1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        nb_b = _make_nb([
            {'cell_type': 'code', 'source': ['x = 1'], 'outputs': [], 'execution_count': None, 'metadata': {}},
            {'cell_type': 'code', 'source': ['y = 2'], 'outputs': [], 'execution_count': None, 'metadata': {}}
        ])
        result = diff_nbs(nb_a, nb_b)
        assert len(result) > 0
        diff_text = ''.join(result)
        assert '+y = 2' in diff_text

    def test_context_parameter(self):
        nb_a = _make_nb([
            {'cell_type': 'code', 'source': ['a = 1\n', 'b = 2\n', 'c = 3\n', 'd = 4\n', 'e = 5'], 'outputs': [], 'execution_count': None, 'metadata': {}}
        ])
        nb_b = _make_nb([
            {'cell_type': 'code', 'source': ['a = 1\n', 'b = 2\n', 'c = CHANGED\n', 'd = 4\n', 'e = 5'], 'outputs': [], 'execution_count': None, 'metadata': {}}
        ])
        result_1 = diff_nbs(nb_a, nb_b, context=1)
        result_5 = diff_nbs(nb_a, nb_b, context=5)
        # More context means more lines
        assert len(result_5) >= len(result_1)


# --- Tests for _load_nb ---

class TestLoadNb:
    def test_loads_valid_notebook(self):
        nb = _make_nb([{'cell_type': 'code', 'source': ['x=1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(nb, f)
            f.flush()
            loaded = _load_nb(f.name)
        os.unlink(f.name)
        assert loaded == nb


# --- Tests for main() CLI ---

def _get_env():
    """Get environment with PYTHONPATH pointing to project root."""
    env = os.environ.copy()
    env['PYTHONPATH'] = _PROJECT_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    return env


class TestMainCLI:
    def test_two_identical_files_exit_0(self, tmp_path):
        nb = _make_nb([{'cell_type': 'code', 'source': ['x=1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        f1 = tmp_path / 'a.ipynb'
        f2 = tmp_path / 'b.ipynb'
        _write_nb(str(f1), nb)
        _write_nb(str(f2), nb)
        result = subprocess.run(
            [sys.executable, '-m', 'fastai.diff_nbs', str(f1), str(f2)],
            capture_output=True, text=True, cwd=str(tmp_path), env=_get_env()
        )
        assert result.returncode == 0
        assert result.stdout == ''

    def test_two_different_files_exit_1(self, tmp_path):
        nb_a = _make_nb([{'cell_type': 'code', 'source': ['x=1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        nb_b = _make_nb([{'cell_type': 'code', 'source': ['x=2'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        f1 = tmp_path / 'a.ipynb'
        f2 = tmp_path / 'b.ipynb'
        _write_nb(str(f1), nb_a)
        _write_nb(str(f2), nb_b)
        result = subprocess.run(
            [sys.executable, '-m', 'fastai.diff_nbs', str(f1), str(f2)],
            capture_output=True, text=True, cwd=str(tmp_path), env=_get_env()
        )
        assert result.returncode == 1
        assert 'x=1' in result.stdout
        assert 'x=2' in result.stdout

    def test_rev_flag_with_git(self, tmp_path):
        """Test --rev flag against a git revision."""
        # Set up a temp git repo
        subprocess.run(['git', 'init'], cwd=str(tmp_path), capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=str(tmp_path), capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=str(tmp_path), capture_output=True)

        nb_v1 = _make_nb([{'cell_type': 'code', 'source': ['original'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        nb_path = tmp_path / 'test.ipynb'
        _write_nb(str(nb_path), nb_v1)
        subprocess.run(['git', 'add', 'test.ipynb'], cwd=str(tmp_path), capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'initial'], cwd=str(tmp_path), capture_output=True)

        # Modify the notebook
        nb_v2 = _make_nb([{'cell_type': 'code', 'source': ['modified'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        _write_nb(str(nb_path), nb_v2)

        result = subprocess.run(
            [sys.executable, '-m', 'fastai.diff_nbs', 'test.ipynb', '--rev', 'HEAD'],
            capture_output=True, text=True, cwd=str(tmp_path), env=_get_env()
        )
        assert result.returncode == 1
        assert 'original' in result.stdout
        assert 'modified' in result.stdout

    def test_error_rev_with_two_paths(self, tmp_path):
        """Cannot use --rev with two positional args."""
        nb = _make_nb([{'cell_type': 'code', 'source': ['x=1'], 'outputs': [], 'execution_count': None, 'metadata': {}}])
        f1 = tmp_path / 'a.ipynb'
        f2 = tmp_path / 'b.ipynb'
        _write_nb(str(f1), nb)
        _write_nb(str(f2), nb)
        result = subprocess.run(
            [sys.executable, '-m', 'fastai.diff_nbs', str(f1), str(f2), '--rev', 'HEAD'],
            capture_output=True, text=True, cwd=str(tmp_path), env=_get_env()
        )
        assert result.returncode == 2  # argparse error

    def test_outputs_only_difference_exit_0(self, tmp_path):
        """Notebooks differing only in outputs should produce no diff."""
        nb_a = _make_nb([{'cell_type': 'code', 'source': ['print(1)'], 'outputs': [{'text': ['1\n']}], 'execution_count': 1, 'metadata': {}}])
        nb_b = _make_nb([{'cell_type': 'code', 'source': ['print(1)'], 'outputs': [{'text': ['2\n']}], 'execution_count': 99, 'metadata': {'collapsed': True}}])
        f1 = tmp_path / 'a.ipynb'
        f2 = tmp_path / 'b.ipynb'
        _write_nb(str(f1), nb_a)
        _write_nb(str(f2), nb_b)
        result = subprocess.run(
            [sys.executable, '-m', 'fastai.diff_nbs', str(f1), str(f2)],
            capture_output=True, text=True, cwd=str(tmp_path), env=_get_env()
        )
        assert result.returncode == 0
