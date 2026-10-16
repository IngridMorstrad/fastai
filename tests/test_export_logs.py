"""Tests for export_logs -- Recorder metric export to JSON and CSV."""

import csv
import json
import os

import pytest

from fastai.callback.export_logs import export_logs, _metric_columns, _avg_lr_per_epoch


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockRecorder:
    """Minimal stand-in for ``Recorder`` with configurable training history."""

    def __init__(self, *, lrs=None, losses=None, values=None, iters=None,
                 metric_names=None):
        self.lrs = lrs or []
        self.losses = losses or []
        self.values = values or []
        self.iters = iters or []
        self.metric_names = metric_names or []


class _MockLearn:
    """Minimal stand-in for ``Learner`` with a ``recorder`` attribute."""

    def __init__(self, recorder):
        self.recorder = recorder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_standard_learner():
    """Build a mock learner that simulates 3 epochs of training.

    Layout (per-epoch batches = 4):
      lrs:    [0.01]*4 + [0.008]*4 + [0.006]*4   (12 batches total)
      losses: [0.9, 0.8, 0.7, 0.6,               epoch 0
               0.5, 0.4, 0.35, 0.3,               epoch 1
               0.28, 0.25, 0.22, 0.2]              epoch 2
      iters:  [4, 8, 12]
      metric_names: ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'time']
      values (per epoch, excludes 'epoch' and 'time'):
          [ [0.65, 0.70, 0.80],
            [0.40, 0.45, 0.88],
            [0.22, 0.30, 0.92] ]
    """
    lrs = [0.01] * 4 + [0.008] * 4 + [0.006] * 4
    losses = [0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.35, 0.3,
              0.28, 0.25, 0.22, 0.2]
    iters = [4, 8, 12]
    metric_names = ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'time']
    values = [
        [0.65, 0.70, 0.80],
        [0.40, 0.45, 0.88],
        [0.22, 0.30, 0.92],
    ]
    rec = _MockRecorder(lrs=lrs, losses=losses, values=values, iters=iters,
                        metric_names=metric_names)
    return _MockLearn(rec)


# ---------------------------------------------------------------------------
# Tests -- JSON output
# ---------------------------------------------------------------------------

class TestExportLogsJSON:
    """Verify JSON (TensorBoard event-record) output."""

    def test_json_file_created(self, tmp_path):
        """export_logs creates a JSON file at the requested path."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        assert out.exists()

    def test_json_records_are_list(self, tmp_path):
        """JSON output is a list of event dicts."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    def test_json_record_schema(self, tmp_path):
        """Every record has wall_time, step, tag, and value."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        required_keys = {'wall_time', 'step', 'tag', 'value'}
        for rec in data:
            assert required_keys.issubset(rec.keys()), f"Missing keys in {rec}"

    def test_json_contains_per_batch_loss(self, tmp_path):
        """Per-batch loss records are present with tag 'loss'."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        loss_records = [r for r in data if r['tag'] == 'loss' and r['step'] < 12]
        # 12 batches total across 3 epochs
        assert len(loss_records) == 12

    def test_json_contains_per_batch_lr(self, tmp_path):
        """Per-batch lr records are present."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        # Per-batch lr records (tag='lr' and step < 12 for batch-level)
        batch_lr = [r for r in data if r['tag'] == 'lr' and r['step'] < 4]
        # At least 4 batch-level lr records for epoch 0
        assert len(batch_lr) >= 4

    def test_json_contains_per_epoch_metrics(self, tmp_path):
        """Per-epoch metric records for train_loss, valid_loss, accuracy."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        for tag in ('train_loss', 'valid_loss', 'accuracy'):
            recs = [r for r in data if r['tag'] == tag]
            assert len(recs) == 3, f"Expected 3 epoch records for {tag}, got {len(recs)}"

    def test_json_metric_values_correct(self, tmp_path):
        """Metric values match the mock recorder data."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        accuracy_recs = sorted(
            [r for r in data if r['tag'] == 'accuracy'],
            key=lambda r: r['step'],
        )
        assert [r['value'] for r in accuracy_recs] == pytest.approx([0.80, 0.88, 0.92])

    def test_json_epoch_range(self, tmp_path):
        """epoch_range limits the epochs included in the output."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json', epoch_range=(1, 3))
        data = json.loads(out.read_text())
        # Should NOT contain epoch-0 metric records
        epoch_0_metrics = [r for r in data if r['tag'] == 'train_loss' and r['step'] == 0]
        assert len(epoch_0_metrics) == 0
        # Should contain epoch 1 and 2
        train_loss_recs = [r for r in data if r['tag'] == 'train_loss']
        assert len(train_loss_recs) == 2

    def test_json_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        learn = _make_standard_learner()
        out = tmp_path / 'sub' / 'dir' / 'metrics.json'
        export_logs(learn, out, format='json')
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests -- CSV output
# ---------------------------------------------------------------------------

class TestExportLogsCSV:
    """Verify CSV output."""

    def test_csv_file_created(self, tmp_path):
        """export_logs creates a CSV file at the requested path."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        assert out.exists()

    def test_csv_headers(self, tmp_path):
        """CSV header row contains epoch, metric names, and lr."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == [
                'epoch', 'train_loss', 'valid_loss', 'accuracy', 'lr',
            ]

    def test_csv_row_count(self, tmp_path):
        """CSV has one row per epoch."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    def test_csv_values_correct(self, tmp_path):
        """CSV values match the mock recorder data."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # Epoch 0: train_loss=0.65, valid_loss=0.70, accuracy=0.80, lr=avg(0.01*4)=0.01
        row0 = rows[0]
        assert float(row0['epoch']) == 0
        assert float(row0['train_loss']) == pytest.approx(0.65)
        assert float(row0['valid_loss']) == pytest.approx(0.70)
        assert float(row0['accuracy']) == pytest.approx(0.80)
        assert float(row0['lr']) == pytest.approx(0.01)

    def test_csv_lr_is_epoch_average(self, tmp_path):
        """lr column is the mean learning rate for each epoch."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            rows = list(csv.DictReader(fh))
        assert float(rows[0]['lr']) == pytest.approx(0.01)
        assert float(rows[1]['lr']) == pytest.approx(0.008)
        assert float(rows[2]['lr']) == pytest.approx(0.006)

    def test_csv_epoch_range(self, tmp_path):
        """epoch_range limits the rows in the CSV."""
        learn = _make_standard_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv', epoch_range=(1, 2))
        with open(out) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert int(rows[0]['epoch']) == 1

    def test_csv_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        learn = _make_standard_learner()
        out = tmp_path / 'deep' / 'path' / 'metrics.csv'
        export_logs(learn, out, format='csv')
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests -- multiple custom metrics
# ---------------------------------------------------------------------------

class TestMultipleCustomMetrics:
    """Verify behaviour with more than the standard metrics."""

    def _make_multi_metric_learner(self):
        """Learner mock with accuracy and f1_score metrics."""
        lrs = [0.01] * 3
        losses = [0.5, 0.4, 0.3]
        iters = [3]
        metric_names = ['epoch', 'train_loss', 'valid_loss', 'accuracy',
                        'f1_score', 'time']
        values = [[0.4, 0.5, 0.85, 0.82]]
        rec = _MockRecorder(lrs=lrs, losses=losses, values=values,
                            iters=iters, metric_names=metric_names)
        return _MockLearn(rec)

    def test_csv_includes_all_custom_metrics(self, tmp_path):
        """CSV headers include all custom metric names."""
        learn = self._make_multi_metric_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            reader = csv.DictReader(fh)
            assert 'accuracy' in reader.fieldnames
            assert 'f1_score' in reader.fieldnames

    def test_json_includes_all_custom_metrics(self, tmp_path):
        """JSON records include all custom metric tags."""
        learn = self._make_multi_metric_learner()
        out = tmp_path / 'metrics.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        tags = {r['tag'] for r in data}
        assert 'accuracy' in tags
        assert 'f1_score' in tags

    def test_csv_custom_metric_values(self, tmp_path):
        """Custom metric values are written correctly in CSV."""
        learn = self._make_multi_metric_learner()
        out = tmp_path / 'metrics.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            rows = list(csv.DictReader(fh))
        assert float(rows[0]['f1_score']) == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Tests -- error handling and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and error paths."""

    def test_invalid_format_raises(self):
        """Unsupported format raises ValueError."""
        learn = _make_standard_learner()
        with pytest.raises(ValueError, match="format must be 'json' or 'csv'"):
            export_logs(learn, '/tmp/nope.txt', format='xml')

    def test_empty_recorder_json(self, tmp_path):
        """Empty recorder produces a valid empty JSON array."""
        rec = _MockRecorder(metric_names=['epoch', 'train_loss', 'valid_loss', 'time'])
        learn = _MockLearn(rec)
        out = tmp_path / 'empty.json'
        export_logs(learn, out, format='json')
        data = json.loads(out.read_text())
        assert data == []

    def test_empty_recorder_csv(self, tmp_path):
        """Empty recorder produces a CSV with headers only."""
        rec = _MockRecorder(metric_names=['epoch', 'train_loss', 'valid_loss', 'time'])
        learn = _MockLearn(rec)
        out = tmp_path / 'empty.csv'
        export_logs(learn, out, format='csv')
        with open(out) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 0
        # Header should still be present
        fh_check = open(out)
        header_line = fh_check.readline().strip()
        fh_check.close()
        assert 'epoch' in header_line

    def test_format_case_insensitive(self, tmp_path):
        """format='JSON' and format='Csv' are accepted."""
        learn = _make_standard_learner()
        out_json = tmp_path / 'upper.json'
        export_logs(learn, out_json, format='JSON')
        assert out_json.exists()
        out_csv = tmp_path / 'mixed.csv'
        export_logs(learn, out_csv, format='Csv')
        assert out_csv.exists()


# ---------------------------------------------------------------------------
# Tests -- monkey-patch
# ---------------------------------------------------------------------------

class TestMonkeyPatch:
    """Verify that export_logs is patched onto Learner when importable."""

    def test_learner_has_export_logs(self):
        """Learner class has the export_logs method after module import."""
        try:
            from fastai.learner import Learner
            assert hasattr(Learner, 'export_logs')
            assert Learner.export_logs is export_logs
        except ImportError:
            pytest.skip("Full fastai stack not importable")


# ---------------------------------------------------------------------------
# Tests -- helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    """Unit tests for internal helper functions."""

    def test_metric_columns_strips_epoch_and_time(self):
        """_metric_columns removes 'epoch' and 'time' from metric_names."""
        names = ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'time']
        assert _metric_columns(names) == ['train_loss', 'valid_loss', 'accuracy']

    def test_metric_columns_no_extras(self):
        """_metric_columns with only epoch returns empty."""
        assert _metric_columns(['epoch']) == []

    def test_avg_lr_per_epoch_basic(self):
        """_avg_lr_per_epoch computes correct averages."""
        lrs = [0.1, 0.1, 0.2, 0.2]
        iters = [2, 4]
        result = _avg_lr_per_epoch(lrs, iters)
        assert result == pytest.approx([0.1, 0.2])

    def test_avg_lr_per_epoch_empty(self):
        """_avg_lr_per_epoch returns [] when lrs is empty."""
        assert _avg_lr_per_epoch([], []) == []

    def test_avg_lr_per_epoch_single_epoch(self):
        """_avg_lr_per_epoch handles a single epoch."""
        result = _avg_lr_per_epoch([0.01, 0.01, 0.01], [3])
        assert result == pytest.approx([0.01])
