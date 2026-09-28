"""Tests for the dataset versioning / lineage feature in fastai.data.external.

Covers dataset_hash determinism, DatasetSnapshot round-tripping and id
stability, the JSON lineage registry (register/list/get + failure paths),
reproduce_snapshot success and failure paths, and the untar_data
`track_lineage` parameter.

All filesystem work uses pytest's tmp_path fixture and small local fixture
files: these tests perform NO network I/O and download NO real datasets.
"""
import sys
import os
import inspect
import json
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.data.external import (
    dataset_hash, DatasetSnapshot, register_snapshot, list_snapshots,
    get_snapshot, reproduce_snapshot, fastai_lineage_path, untar_data,
)
import fastai.data.external as ext


# ============================================================
# Helpers
# ============================================================

def _make_dataset(root, files):
    """Create a small fake dataset dir under `root` from a {relpath: content} map."""
    root.mkdir(parents=True, exist_ok=True)
    for relpath, content in files.items():
        f = root / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return root


# ============================================================
# Tests for dataset_hash
# ============================================================

class TestDatasetHash:
    """Tests for the content-hashing helper."""

    def test_same_dir_twice_is_deterministic(self, tmp_path):
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'hello', 'sub/b.txt': 'world'})
        assert dataset_hash(d) == dataset_hash(d)

    def test_content_change_changes_digest(self, tmp_path):
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'hello', 'b.txt': 'world'})
        before = dataset_hash(d)
        (d / 'a.txt').write_text('HELLO')
        after = dataset_hash(d)
        assert before != after

    def test_single_file_hashing(self, tmp_path):
        f = tmp_path / 'only.txt'
        f.write_text('some contents')
        # deterministic and stable
        assert dataset_hash(f) == dataset_hash(f)
        # differs from a file with different contents
        g = tmp_path / 'other.txt'
        g.write_text('different')
        assert dataset_hash(f) != dataset_hash(g)

    def test_same_contents_different_names_differ(self, tmp_path):
        # names are part of the hash, so identical bytes under different
        # filenames must produce different digests
        d1 = _make_dataset(tmp_path / 'd1', {'foo.txt': 'same', 'bar.txt': 'data'})
        d2 = _make_dataset(tmp_path / 'd2', {'baz.txt': 'same', 'qux.txt': 'data'})
        assert dataset_hash(d1) != dataset_hash(d2)


# ============================================================
# Tests for DatasetSnapshot
# ============================================================

class TestDatasetSnapshot:
    """Tests for the snapshot record class."""

    def test_to_dict_from_dict_round_trip(self):
        snap = DatasetSnapshot(url='http://example/ds.tgz', path='/tmp/ds',
                               hash='abc123', size=42)
        rebuilt = DatasetSnapshot.from_dict(snap.to_dict())
        assert rebuilt == snap
        assert rebuilt.to_dict() == snap.to_dict()

    def test_id_deterministic_for_same_url_and_hash(self):
        a = DatasetSnapshot(url='http://example/ds.tgz', path='/tmp/a',
                            hash='deadbeef', size=1)
        b = DatasetSnapshot(url='http://example/ds.tgz', path='/tmp/b',
                            hash='deadbeef', size=999)
        # id depends only on url+hash, not path/size/created
        assert a.id == b.id

    def test_id_differs_when_url_differs(self):
        a = DatasetSnapshot(url='http://example/one.tgz', path='/tmp/a',
                            hash='deadbeef', size=1)
        b = DatasetSnapshot(url='http://example/two.tgz', path='/tmp/a',
                            hash='deadbeef', size=1)
        assert a.id != b.id

    def test_id_differs_when_hash_differs(self):
        a = DatasetSnapshot(url='http://example/ds.tgz', path='/tmp/a',
                            hash='hash1', size=1)
        b = DatasetSnapshot(url='http://example/ds.tgz', path='/tmp/a',
                            hash='hash2', size=1)
        assert a.id != b.id


# ============================================================
# Tests for the lineage registry
# ============================================================

class TestRegistry:
    """Tests for register/list/get + persistence and failure paths."""

    def test_register_writes_lineage_json(self, tmp_path):
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x', 'b.txt': 'y'})
        reg = tmp_path / 'lineage.json'
        snap = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        assert reg.exists()
        data = json.loads(reg.read_text())
        assert snap.id in data
        assert data[snap.id]['url'] == 'http://example/ds.tgz'

    def test_register_is_idempotent_for_identical_content(self, tmp_path):
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x', 'b.txt': 'y'})
        reg = tmp_path / 'lineage.json'
        first = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        second = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        assert first.id == second.id
        data = json.loads(reg.read_text())
        assert len(data) == 1

    def test_list_snapshots_returns_registered(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        d1 = _make_dataset(tmp_path / 'ds1', {'a.txt': 'one'})
        d2 = _make_dataset(tmp_path / 'ds2', {'a.txt': 'two'})
        s1 = register_snapshot('http://example/one.tgz', d1, lineage_path=reg)
        s2 = register_snapshot('http://example/two.tgz', d2, lineage_path=reg)
        snaps = list_snapshots(lineage_path=reg)
        ids = {s.id for s in snaps}
        assert ids == {s1.id, s2.id}

    def test_get_snapshot_returns_right_one(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x'})
        snap = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        got = get_snapshot(snap.id, lineage_path=reg)
        assert got == snap

    def test_get_snapshot_unknown_id_raises_keyerror(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x'})
        register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        with pytest.raises(KeyError):
            get_snapshot('does-not-exist', lineage_path=reg)

    def test_corrupt_registry_raises(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        reg.write_text('this is not valid json {{{')
        with pytest.raises(Exception):
            list_snapshots(lineage_path=reg)

    def test_reregister_different_path_warns_and_overwrites(self, tmp_path):
        # Identical content (same url + same bytes) at two different local
        # paths yields the same snapshot id. Re-registering from the second
        # path must warn about the path change and keep exactly one entry
        # whose stored path is the new one (last-writer-wins, but signalled).
        reg = tmp_path / 'lineage.json'
        files = {'a.txt': 'x', 'b.txt': 'y'}
        d1 = _make_dataset(tmp_path / 'loc1', dict(files))
        d2 = _make_dataset(tmp_path / 'loc2', dict(files))
        first = register_snapshot('http://example/ds.tgz', d1, lineage_path=reg)
        with pytest.warns(UserWarning):
            second = register_snapshot('http://example/ds.tgz', d2, lineage_path=reg)
        assert first.id == second.id
        data = json.loads(reg.read_text())
        assert len(data) == 1
        assert data[second.id]['path'] == os.fspath(d2)

    def test_reregister_same_path_does_not_warn(self, tmp_path):
        # Re-registering identical content from the SAME path is idempotent
        # and must emit no warning.
        reg = tmp_path / 'lineage.json'
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x', 'b.txt': 'y'})
        register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            register_snapshot('http://example/ds.tgz', d, lineage_path=reg)


# ============================================================
# Tests for reproduce_snapshot
# ============================================================

class TestReproduce:
    """Tests for reproduce_snapshot success and failure paths."""

    def test_reproduce_returns_path_when_unchanged(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x', 'b.txt': 'y'})
        snap = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        result = reproduce_snapshot(snap.id, lineage_path=reg)
        assert os.fspath(result) == os.fspath(d)

    def test_reproduce_raises_when_path_deleted(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        f = tmp_path / 'single.txt'
        f.write_text('contents')
        snap = register_snapshot('http://example/ds.tgz', f, lineage_path=reg)
        f.unlink()
        with pytest.raises(FileNotFoundError):
            reproduce_snapshot(snap.id, lineage_path=reg)

    def test_reproduce_raises_on_hash_mismatch(self, tmp_path):
        reg = tmp_path / 'lineage.json'
        d = _make_dataset(tmp_path / 'ds', {'a.txt': 'x', 'b.txt': 'y'})
        snap = register_snapshot('http://example/ds.tgz', d, lineage_path=reg)
        # modify a file after registering -> current hash no longer matches
        (d / 'a.txt').write_text('MODIFIED')
        with pytest.raises(ValueError):
            reproduce_snapshot(snap.id, lineage_path=reg)


# ============================================================
# Tests for untar_data track_lineage parameter
# ============================================================

class TestUntarDataLineage:
    """Tests for the opt-in track_lineage parameter on untar_data."""

    def test_track_lineage_param_defaults_to_false(self):
        sig = inspect.signature(untar_data)
        assert 'track_lineage' in sig.parameters
        assert sig.parameters['track_lineage'].default is False

    def test_track_lineage_true_registers_snapshot(self, tmp_path, monkeypatch):
        # Fake extracted data dir the "download" resolves to.
        extracted = _make_dataset(tmp_path / 'extracted', {'a.txt': 'x', 'b.txt': 'y'})
        reg = tmp_path / 'lineage.json'

        # Fake FastDownload whose .get returns the fake extracted dir, so no
        # network or real download occurs.
        class FakeFastDownload:
            def __init__(self, *args, **kwargs): pass
            def get(self, url, force=False, extract_key=None): return extracted

        monkeypatch.setattr(ext, 'FastDownload', FakeFastDownload)
        # Route the registry to a tmp file instead of ~/.fastai.
        monkeypatch.setattr(ext, 'fastai_lineage_path', lambda: reg)

        res = untar_data('http://example/ds.tgz', track_lineage=True)
        assert os.fspath(res) == os.fspath(extracted)
        assert reg.exists()
        snaps = list_snapshots(lineage_path=reg)
        assert len(snaps) == 1
        assert snaps[0].url == 'http://example/ds.tgz'

    def test_track_lineage_false_does_not_register(self, tmp_path, monkeypatch):
        extracted = _make_dataset(tmp_path / 'extracted', {'a.txt': 'x'})
        reg = tmp_path / 'lineage.json'

        class FakeFastDownload:
            def __init__(self, *args, **kwargs): pass
            def get(self, url, force=False, extract_key=None): return extracted

        monkeypatch.setattr(ext, 'FastDownload', FakeFastDownload)
        monkeypatch.setattr(ext, 'fastai_lineage_path', lambda: reg)

        res = untar_data('http://example/ds.tgz', track_lineage=False)
        assert os.fspath(res) == os.fspath(extracted)
        assert not reg.exists()
