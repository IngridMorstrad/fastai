import json, csv, pytest
from pathlib import Path
from fastai.test_utils import *


@pytest.fixture
def trained_learner(tmp_path):
    learn = synth_learner(path=tmp_path)
    learn.fit(2)
    return learn


def test_export_logs_json(trained_learner, tmp_path):
    learn = trained_learner
    result = learn.export_logs(fname='metrics', path=tmp_path, fmt='json')
    assert isinstance(result, Path)
    assert result.name == 'metrics.json'
    assert result.exists()
    data = json.loads(result.read_text())
    assert len(data) == 2  # 2 epochs
    # Check structure
    for i, row in enumerate(data):
        assert row['epoch'] == i
        assert 'train_loss' in row
        assert 'valid_loss' in row


def test_export_logs_csv(trained_learner, tmp_path):
    learn = trained_learner
    result = learn.export_logs(fname='metrics', path=tmp_path, fmt='csv')
    assert isinstance(result, Path)
    assert result.name == 'metrics.csv'
    assert result.exists()
    with open(result) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2  # 2 epochs
    assert 'epoch' in rows[0]
    assert 'train_loss' in rows[0]
    assert 'valid_loss' in rows[0]
    assert rows[0]['epoch'] == '0'
    assert rows[1]['epoch'] == '1'


def test_export_logs_both(trained_learner, tmp_path):
    learn = trained_learner
    result = learn.export_logs(fname='metrics', path=tmp_path, fmt='both')
    assert isinstance(result, list)
    assert len(result) == 2
    json_path, csv_path = result
    assert json_path.name == 'metrics.json'
    assert csv_path.name == 'metrics.csv'
    assert json_path.exists()
    assert csv_path.exists()


def test_export_logs_values_match_recorder(trained_learner, tmp_path):
    learn = trained_learner
    rec = learn.recorder
    result = learn.export_logs(fname='metrics', path=tmp_path, fmt='json')
    data = json.loads(result.read_text())
    names = list(rec.metric_names[1:-1]) if getattr(rec, 'add_time', True) else list(rec.metric_names[1:])
    for i, row in enumerate(data):
        for j, name in enumerate(names):
            val = rec.values[i][j]
            expected = val.item() if hasattr(val, 'item') else val
            assert abs(row[name] - expected) < 1e-6, f"Mismatch at epoch {i}, metric {name}"


def test_export_logs_default_path(tmp_path):
    learn = synth_learner(path=tmp_path)
    learn.fit(1)
    result = learn.export_logs()
    assert isinstance(result, Path)
    assert result.parent == tmp_path
    assert result.name == 'logs.json'
    assert result.exists()
