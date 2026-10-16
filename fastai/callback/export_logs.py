"""export_logs -- Export Recorder training metrics to structured JSON or CSV files.

Provides an ``export_logs`` function (monkey-patched onto ``Learner``) that
serialises the metrics captured by the built-in ``Recorder`` callback into
files that are compatible with TensorBoard, Weights & Biases, and other
visualisation tools.

Supported formats
-----------------
* **json** -- TensorBoard-compatible event-style records.  Each record is a
  dict with ``wall_time`` (float epoch-seconds), ``step`` (int), ``tag``
  (metric name), and ``value`` (float).  The output file contains a JSON
  array of these records.

* **csv** -- One row per epoch.  Columns are ``epoch``, ``train_loss``,
  ``valid_loss``, any additional metric names reported by the Recorder,
  and ``lr`` (the mean learning rate for that epoch).

Usage with fastai::

    from fastai.callback.export_logs import export_logs   # registers on Learner
    learn.fit(5)
    learn.export_logs('metrics.json', format='json')
    learn.export_logs('metrics.csv',  format='csv')
"""

import csv
import json
import time as _time
from pathlib import Path

__all__ = ['export_logs']

# ---------------------------------------------------------------------------
# Fallback: when the full fastai stack is not importable (e.g. during
# isolated unit testing) we still want the module to load cleanly.
# ---------------------------------------------------------------------------
_Learner = None

try:
    from fastai.learner import Learner as _Learner
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_columns(metric_names):
    """Return the ordered list of per-epoch metric column names.

    ``Recorder.metric_names`` looks like::

        L(['epoch', 'train_loss', 'valid_loss', 'accuracy', 'time'])

    We strip the bookkeeping entries ``epoch`` and ``time`` because they are
    handled separately (``epoch`` is the row index; ``time`` is not a numeric
    metric).  The remainder -- ``train_loss``, ``valid_loss``, and any custom
    metrics -- are returned in their original order.
    """
    exclude = {'epoch', 'time'}
    return [n for n in metric_names if n not in exclude]


def _avg_lr_per_epoch(lrs, iters):
    """Compute the mean learning rate for each epoch.

    Parameters
    ----------
    lrs : list[float]
        One learning rate per training batch (length == total training batches
        across all epochs).
    iters : list[int]
        Cumulative iteration count at the *end* of each epoch, as recorded by
        ``Recorder.iters``.

    Returns
    -------
    list[float]
        One average learning rate per epoch.  If ``lrs`` is empty the list is
        empty.  If an epoch has zero training batches the average is 0.0.
    """
    if not lrs or not iters:
        return []
    avg_lrs = []
    prev = 0
    for end in iters:
        epoch_lrs = lrs[prev:end]
        avg_lrs.append(sum(epoch_lrs) / len(epoch_lrs) if epoch_lrs else 0.0)
        prev = end
    return avg_lrs


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------

def export_logs(self, path, format='json', epoch_range=None):
    """Export training metrics recorded by :class:`Recorder` to a file.

    Parameters
    ----------
    self : Learner
        The learner whose ``recorder`` holds the training history.
    path : str or Path
        Destination file path.  Parent directories are created automatically.
    format : str, optional
        ``'json'`` (default) for TensorBoard-compatible event records, or
        ``'csv'`` for a flat table with one row per epoch.
    epoch_range : tuple of (int, int) or None, optional
        If provided, export only epochs in ``range(epoch_range[0],
        epoch_range[1])``.  Zero-indexed.

    Raises
    ------
    ValueError
        If *format* is not ``'json'`` or ``'csv'``.
    """
    format = format.lower()
    if format not in ('json', 'csv'):
        raise ValueError(
            f"format must be 'json' or 'csv', got {format!r}"
        )

    recorder = self.recorder
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unpack Recorder state ------------------------------------------------
    lrs = list(getattr(recorder, 'lrs', []) or [])
    losses = list(getattr(recorder, 'losses', []) or [])
    values = list(getattr(recorder, 'values', []) or [])
    iters = list(getattr(recorder, 'iters', []) or [])
    metric_names_raw = list(getattr(recorder, 'metric_names', []) or [])

    columns = _metric_columns(metric_names_raw)
    avg_lrs = _avg_lr_per_epoch(lrs, iters)

    # Apply optional epoch range filter ------------------------------------
    n_epochs = len(values)
    if epoch_range is not None:
        start, end = epoch_range
        epoch_indices = list(range(start, min(end, n_epochs)))
    else:
        epoch_indices = list(range(n_epochs))

    if format == 'json':
        _export_json(path, epoch_indices, columns, values, avg_lrs, lrs,
                     losses, iters)
    else:
        _export_csv(path, epoch_indices, columns, values, avg_lrs)


# ---------------------------------------------------------------------------
# JSON export (TensorBoard event-record format)
# ---------------------------------------------------------------------------

def _export_json(path, epoch_indices, columns, values, avg_lrs, lrs, losses,
                 iters):
    """Write TensorBoard-compatible JSON event records.

    Record schema::

        {
            "wall_time": <float seconds since epoch>,
            "step": <int>,
            "tag": "<metric name>",
            "value": <float>
        }

    We emit:
    * Per-batch records for ``loss`` and ``lr`` (step = batch index).
    * Per-epoch records for every metric in *columns* plus ``lr`` (step =
      epoch index).
    """
    wall_time = _time.time()
    records = []

    # --- Per-batch records (loss and lr) -----------------------------------
    if epoch_indices and losses:
        # Determine batch index ranges belonging to the requested epochs
        prev_iter = 0
        for epoch_idx in range(max(epoch_indices) + 1):
            end_iter = iters[epoch_idx] if epoch_idx < len(iters) else prev_iter
            if epoch_idx in epoch_indices:
                for batch_idx in range(prev_iter, min(end_iter, len(losses))):
                    records.append({
                        'wall_time': wall_time,
                        'step': batch_idx,
                        'tag': 'loss',
                        'value': float(losses[batch_idx]),
                    })
                    if batch_idx < len(lrs):
                        records.append({
                            'wall_time': wall_time,
                            'step': batch_idx,
                            'tag': 'lr',
                            'value': float(lrs[batch_idx]),
                        })
            prev_iter = end_iter

    # --- Per-epoch metric records ------------------------------------------
    for epoch_idx in epoch_indices:
        if epoch_idx >= len(values):
            continue
        row = values[epoch_idx]
        for col_idx, col_name in enumerate(columns):
            if col_idx < len(row):
                val = row[col_idx]
                # Recorder may store None for metrics that were not computed
                if val is not None:
                    records.append({
                        'wall_time': wall_time,
                        'step': epoch_idx,
                        'tag': col_name,
                        'value': float(val),
                    })
        # Epoch-level average lr
        if epoch_idx < len(avg_lrs):
            records.append({
                'wall_time': wall_time,
                'step': epoch_idx,
                'tag': 'lr',
                'value': float(avg_lrs[epoch_idx]),
            })

    path.write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _export_csv(path, epoch_indices, columns, values, avg_lrs):
    """Write a CSV file with one row per epoch.

    Columns: ``epoch``, then each name in *columns* (typically
    ``train_loss``, ``valid_loss``, plus custom metrics), then ``lr``.
    """
    fieldnames = ['epoch'] + columns + ['lr']

    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_idx in epoch_indices:
            row = {'epoch': epoch_idx}
            vals = values[epoch_idx] if epoch_idx < len(values) else []
            for col_idx, col_name in enumerate(columns):
                row[col_name] = float(vals[col_idx]) if col_idx < len(vals) and vals[col_idx] is not None else ''
            row['lr'] = float(avg_lrs[epoch_idx]) if epoch_idx < len(avg_lrs) else ''
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Monkey-patch onto Learner
# ---------------------------------------------------------------------------
if _Learner is not None:
    _Learner.export_logs = export_logs
