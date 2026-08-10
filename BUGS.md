# Known Bugs

This file tracks known bugs and issues in the fastai library. If you encounter a bug not listed here, please [open an issue](https://github.com/fastai/fastai/issues/new).

## Open

- `TensorBase.requires_grad_` uses a workaround for [pytorch#50219](https://github.com/pytorch/pytorch/issues/50219); may behave unexpectedly if the upstream fix changes semantics (`fastai/torch_core.py`)
- `SentencePieceTokenizer` does not forward special token symbols to the underlying SentencePiece model (`fastai/text/core.py`)
- `Learner.summary` does not count parameters for individual `ParameterModule` instances wrapped outside of hook-tracked layers (`fastai/callback/hook.py`)
- `TfmdDL` padding uses `L.items.index` instead of `L.index` due to an unresolved upstream bug in `L` (`fastai/text/data.py`)
- `CorpusBLEU.value` compares `self.counts` (a list) against integer `0`, so the empty-counts guard is never triggered, risking ZeroDivisionError (`fastai/metrics.py`)

- `_get_default` in `vision/augment.py` assigns `'bilinear'` interpolation mode for `TensorMask` instead of `'nearest'`, causing mask labels to be incorrectly interpolated during batch augmentation transforms (`fastai/vision/augment.py`)
- `Normalize.decodes` uses `noop` when `x` is on a non-CPU device, so if `self.mean`/`self.std` reside on CPU the denormalization raises a device-mismatch error instead of moving stats to the correct device (`fastai/data/transforms.py`)
- `ShowGraphCallback.after_epoch` passes `max(Tensor(...))` results directly into `y_bounds`, yielding single-element tensors instead of Python floats, which can break matplotlib backends that require plain numeric bounds (`fastai/callback/progress.py`)
- `GradientAccumulation` resets its counter only when `self.count >= n_acc`, so accumulated gradients from the final batches of an epoch are silently discarded if their total size does not reach `n_acc` (`fastai/callback/training.py`)
- `Optimizer.param_groups` setter assigns `pg = v_['params']` to a local variable instead of mutating the list in-place, so the parameter groups are never actually updated (`fastai/optimizer.py`)

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
