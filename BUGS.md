# Known Bugs

This file tracks known bugs and issues in the fastai library. If you encounter a bug not listed here, please [open an issue](https://github.com/fastai/fastai/issues/new).

## Open

- `TensorBase.requires_grad_` uses a workaround for [pytorch#50219](https://github.com/pytorch/pytorch/issues/50219); may behave unexpectedly if the upstream fix changes semantics (`fastai/torch_core.py`)
- `SentencePieceTokenizer` does not forward special token symbols to the underlying SentencePiece model (`fastai/text/core.py`)
- `Learner.summary` does not count parameters for individual `ParameterModule` instances wrapped outside of hook-tracked layers (`fastai/callback/hook.py`)
- `TfmdDL` padding uses `L.items.index` instead of `L.index` due to an unresolved upstream bug in `L` (`fastai/text/data.py`)
- `CorpusBLEU.value` compares `self.counts` (a list) against integer `0`, so the empty-counts guard is never triggered, risking ZeroDivisionError (`fastai/metrics.py`)
- `fit_flat_cos` accepts a `start_epoch` parameter but hardcodes `start_epoch=0` in the call to `self.fit`, so the argument is silently ignored (`fastai/callback/schedule.py`)
- `log_dataset` uses `raise f'path must be a valid directory: {path}'` which raises a `TypeError` in Python 3 instead of a proper exception (`fastai/callback/wandb.py`)
- `log_model` uses `raise f'path must be a valid file: {path}'` which raises a `TypeError` in Python 3 instead of a proper exception (`fastai/callback/wandb.py`)
- `CutMix.rand_bbox` returns single-element tensors for box coordinates, so the `lam` recalculation `(1 - ((x2-x1)*(y2-y1))/float(W*H))` yields a tensor rather than a Python float, causing shape mismatches when `stack_y=False` and `torch.lerp` expects a scalar weight (`fastai/callback/mixup.py`)
- `SaveModelCallback.after_fit` always calls `self.learn.load` when `every_epoch=False` and `at_end=False`, even if no improvement was ever found and no file was saved, raising a `FileNotFoundError` on short training runs (`fastai/callback/tracker.py`)

## Fixed

- `make_vocab` uses `f'xxfake'` without interpolating the loop variable, producing duplicate padding tokens instead of unique ones (`fastai/text/data.py`) - fixed by interpolating loop variable `i` in f-string (`f'xxfake{i}'`)
- `CorpusBLEUMetric.value` compared `self.counts` (a list) against integer 0 which always evaluates False, disabling the zero-counts guard (`fastai/metrics.py`) - fixed by using `max(self.counts) == 0`
- `LMDataLoader` does not support backward language model training (`fastai/text/data.py`) - added `backwards` parameter to `LMDataLoader.__init__` that reverses text sequences when enabled

## Reporting a Bug

Please include:
1. Output of `import fastai.test_utils; fastai.test_utils.show_install(1)`
2. A minimal reproducible example
3. Full stack trace (if applicable)
4. Expected vs. actual behavior
