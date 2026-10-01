"""fastai_diff_nbs - Semantic diff between two Jupyter notebook versions.

Shows a unified diff of the *semantic* content of two ``.ipynb`` files -- the
ordered source of their code and markdown cells -- while ignoring cell outputs,
execution counts, and all notebook/cell metadata. This lets reviewers focus on
the actual code and prose changes instead of noise from re-executed outputs or
shifting metadata.

This is a self-contained, stdlib-only module (json, difflib, argparse, sys). It
deliberately does NOT import torch or the heavy fastai stack, so it imports and
is testable in a minimal environment.

Usage (console script, registered in settings.ini):
    fastai_diff_nbs old.ipynb new.ipynb

Exit codes mirror ``diff`` / ``git diff --exit-code``:
    0 - notebooks are semantically identical (no diff)
    1 - a semantic diff was found
    2 - usage / file error (missing or unreadable notebook)
"""

import argparse
import difflib
import json
import sys

__all__ = ['canonicalize_nb', 'diff_notebooks', 'fastai_diff_nbs']


def _cell_source_lines(cell):
    "Return the source of a notebook `cell` as a list of lines without trailing newlines."
    src = cell.get('source', '')
    # nbformat stores source as either a list of strings or a single string.
    if isinstance(src, list):
        text = ''.join(src)
    else:
        text = src
    return text.splitlines()


def canonicalize_nb(nb_dict):
    """Render a parsed ``.ipynb`` dict to normalized semantic text.

    Includes each cell's source (code and markdown) in document order, each
    preceded by a header line marking the cell type and index. Excludes cell
    outputs, execution counts, and all notebook/cell metadata, so two notebooks
    that differ only in those respects canonicalize to the same text.

    Parameters
    ----------
    nb_dict : dict
        A parsed notebook (the object produced by ``json.load`` on an ``.ipynb``).

    Returns
    -------
    str
        The canonical semantic representation, terminated by a trailing newline.
    """
    lines = []
    cells = nb_dict.get('cells', [])
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        lines.append(f'# ===== cell {i} ({cell_type}) =====')
        lines.extend(_cell_source_lines(cell))
    return '\n'.join(lines) + '\n'


def _load_nb(path):
    "Load and parse the notebook at `path`, raising a clear error on failure."
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook not found: {path}")
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read notebook {path!r}: {e}")


def diff_notebooks(path_a, path_b):
    """Return a unified-diff string of the semantic content of two notebooks.

    The two notebooks are canonicalized (see :func:`canonicalize_nb`) and then
    compared with :func:`difflib.unified_diff`, labeled with the two paths.
    Returns an empty string when the semantic content is identical, even if the
    notebooks differ in outputs, execution counts, or metadata.

    Raises
    ------
    FileNotFoundError
        If either path does not exist.
    ValueError
        If either file is not a readable, valid notebook.
    """
    canon_a = canonicalize_nb(_load_nb(path_a))
    canon_b = canonicalize_nb(_load_nb(path_b))
    diff = difflib.unified_diff(
        canon_a.splitlines(keepends=True),
        canon_b.splitlines(keepends=True),
        fromfile=str(path_a),
        tofile=str(path_b),
    )
    return ''.join(diff)


def fastai_diff_nbs():
    """Console-script entry point: semantic-diff two notebooks given on the CLI.

    Reads two positional notebook paths from ``sys.argv``, prints the unified
    semantic diff to stdout, and exits 0 when there is no diff, 1 when there is
    one, and 2 on a file/usage error.
    """
    parser = argparse.ArgumentParser(
        prog='fastai_diff_nbs',
        description='Show a semantic diff between two Jupyter notebooks, '
                    'ignoring cell outputs, execution counts, and metadata.',
    )
    parser.add_argument('notebook_a', help='Path to the first (base) notebook')
    parser.add_argument('notebook_b', help='Path to the second (changed) notebook')
    args = parser.parse_args()

    try:
        diff = diff_notebooks(args.notebook_a, args.notebook_b)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if diff:
        sys.stdout.write(diff)
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    fastai_diff_nbs()
