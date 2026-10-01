"""Tests for the fastai_diff_nbs semantic notebook-diff tool."""

import json

import pytest

from fastai.diff_nbs import canonicalize_nb, diff_notebooks, fastai_diff_nbs


def _code_cell(source, execution_count=None, outputs=None, metadata=None):
    "Build a minimal code-cell dict for an in-memory notebook."
    return {
        'cell_type': 'code',
        'execution_count': execution_count,
        'metadata': metadata if metadata is not None else {},
        'outputs': outputs if outputs is not None else [],
        'source': source,
    }


def _markdown_cell(source, metadata=None):
    "Build a minimal markdown-cell dict for an in-memory notebook."
    return {
        'cell_type': 'markdown',
        'metadata': metadata if metadata is not None else {},
        'source': source,
    }


def _notebook(cells, metadata=None):
    "Build a minimal notebook dict wrapping `cells`."
    return {
        'cells': cells,
        'metadata': metadata if metadata is not None else {'kernelspec': {'name': 'python3'}},
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def _write_nb(tmp_path, name, nb_dict):
    "Write `nb_dict` to a temp .ipynb file and return its path."
    path = tmp_path / name
    path.write_text(json.dumps(nb_dict), encoding='utf-8')
    return path


class TestCanonicalizeNb:
    """Unit tests for canonicalize_nb."""

    def test_includes_code_source(self):
        """Canonical text includes code-cell source lines."""
        nb = _notebook([_code_cell(['x = 1\n', 'y = 2\n'])])
        canon = canonicalize_nb(nb)
        assert 'x = 1' in canon
        assert 'y = 2' in canon

    def test_includes_markdown_source(self):
        """Canonical text includes markdown-cell prose."""
        nb = _notebook([_markdown_cell(['# Heading\n', 'Some prose.\n'])])
        canon = canonicalize_nb(nb)
        assert '# Heading' in canon
        assert 'Some prose.' in canon

    def test_excludes_outputs(self):
        """Canonical text excludes cell output values."""
        outputs = [{'output_type': 'stream', 'name': 'stdout', 'text': ['SECRET_OUTPUT_VALUE\n']}]
        nb = _notebook([_code_cell(['print("hi")\n'], execution_count=7, outputs=outputs)])
        canon = canonicalize_nb(nb)
        assert 'print("hi")' in canon
        assert 'SECRET_OUTPUT_VALUE' not in canon

    def test_excludes_execution_count_and_metadata(self):
        """Canonical text excludes execution_count and metadata values."""
        nb = _notebook(
            [_code_cell(['a = 1\n'], execution_count=42, metadata={'collapsed': 'META_MARKER'})],
            metadata={'kernelspec': {'name': 'NB_META_MARKER'}},
        )
        canon = canonicalize_nb(nb)
        assert '42' not in canon
        assert 'META_MARKER' not in canon
        assert 'NB_META_MARKER' not in canon

    def test_preserves_cell_order(self):
        """Cells appear in document order in the canonical text."""
        nb = _notebook([_code_cell(['first\n']), _code_cell(['second\n'])])
        canon = canonicalize_nb(nb)
        assert canon.index('first') < canon.index('second')

    def test_handles_string_source(self):
        """Source stored as a single string (not a list) is handled."""
        nb = _notebook([_code_cell('line_as_string = 1\n')])
        canon = canonicalize_nb(nb)
        assert 'line_as_string = 1' in canon


class TestDiffNotebooks:
    """Unit tests for diff_notebooks."""

    def test_identical_source_differing_outputs_metadata(self, tmp_path):
        """Notebooks identical in source but differing in outputs/metadata yield no diff."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook(
            [_code_cell(['x = 1\n'], execution_count=1, outputs=[{'output_type': 'execute_result', 'data': {'text/plain': ['1']}}])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook(
            [_code_cell(['x = 1\n'], execution_count=99, outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': ['different']}], metadata={'tags': ['x']})],
            metadata={'kernelspec': {'name': 'other'}}))
        assert diff_notebooks(a, b) == ''

    def test_differing_code_source_produces_diff(self, tmp_path):
        """A changed code cell produces a diff containing the changed lines."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 2\n'])]))
        diff = diff_notebooks(a, b)
        assert diff != ''
        assert '-x = 1' in diff
        assert '+x = 2' in diff

    def test_differing_markdown_prose_produces_diff(self, tmp_path):
        """Changed markdown prose produces a diff."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_markdown_cell(['Original prose.\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_markdown_cell(['Revised prose.\n'])]))
        diff = diff_notebooks(a, b)
        assert diff != ''
        assert 'Original prose.' in diff
        assert 'Revised prose.' in diff

    def test_added_cell_reflected_in_diff(self, tmp_path):
        """Adding a cell is reflected in the diff."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 1\n']), _code_cell(['y = 2\n'])]))
        diff = diff_notebooks(a, b)
        assert diff != ''
        assert '+y = 2' in diff

    def test_removed_cell_reflected_in_diff(self, tmp_path):
        """Removing a cell is reflected in the diff."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n']), _code_cell(['y = 2\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        diff = diff_notebooks(a, b)
        assert diff != ''
        assert '-y = 2' in diff

    def test_diff_labels_paths(self, tmp_path):
        """The unified diff is labeled with the two file paths."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 2\n'])]))
        diff = diff_notebooks(a, b)
        assert str(a) in diff
        assert str(b) in diff

    def test_missing_file_raises(self, tmp_path):
        """A missing notebook path raises FileNotFoundError."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        missing = tmp_path / 'does_not_exist.ipynb'
        with pytest.raises(FileNotFoundError):
            diff_notebooks(a, missing)

    def test_unreadable_file_raises(self, tmp_path):
        """An invalid-JSON notebook raises ValueError."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        bad = tmp_path / 'bad.ipynb'
        bad.write_text('{ this is not valid json', encoding='utf-8')
        with pytest.raises(ValueError):
            diff_notebooks(a, bad)


class TestFastaiDiffNbsCli:
    """Unit tests for the fastai_diff_nbs console-script entry point."""

    def test_exit_zero_when_no_semantic_diff(self, tmp_path, monkeypatch):
        """CLI exits 0 when notebooks differ only in outputs/metadata."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'], execution_count=1)]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 1\n'], execution_count=2, outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': ['y']}])]))
        monkeypatch.setattr('sys.argv', ['fastai_diff_nbs', str(a), str(b)])
        with pytest.raises(SystemExit) as exc:
            fastai_diff_nbs()
        assert exc.value.code == 0

    def test_exit_one_when_semantic_diff(self, tmp_path, monkeypatch, capsys):
        """CLI exits 1 and prints the diff when source differs."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        b = _write_nb(tmp_path, 'b.ipynb', _notebook([_code_cell(['x = 2\n'])]))
        monkeypatch.setattr('sys.argv', ['fastai_diff_nbs', str(a), str(b)])
        with pytest.raises(SystemExit) as exc:
            fastai_diff_nbs()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert '-x = 1' in out
        assert '+x = 2' in out

    def test_exit_two_on_missing_file(self, tmp_path, monkeypatch, capsys):
        """CLI exits 2 with a readable error on a missing notebook."""
        a = _write_nb(tmp_path, 'a.ipynb', _notebook([_code_cell(['x = 1\n'])]))
        missing = tmp_path / 'missing.ipynb'
        monkeypatch.setattr('sys.argv', ['fastai_diff_nbs', str(a), str(missing)])
        with pytest.raises(SystemExit) as exc:
            fastai_diff_nbs()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert 'error:' in err
        assert 'missing.ipynb' in err
