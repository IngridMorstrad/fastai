# Known Bugs

This file tracks known bugs and issues in the fastai library. If you encounter a bug not listed here, please [open an issue](https://github.com/fastai/fastai/issues/new).

## Open

- `TensorBase.requires_grad_` uses a workaround for [pytorch#50219](https://github.com/pytorch/pytorch/issues/50219); may behave unexpectedly if the upstream fix changes semantics (`fastai/torch_core.py`)
- `SentencePieceTokenizer` does not forward special token symbols to the underlying SentencePiece model (`fastai/text/core.py`)
- `Learner.summary` does not count parameters for individual `ParameterModule` instances wrapped outside of hook-tracked layers (`fastai/callback/hook.py`)
- `TfmdDL` padding uses `L.items.index` instead of `L.index` due to an unresolved upstream bug in `L` (`fastai/text/data.py`)
- `CorpusBLEU.value` compares `self.counts` (a list) against integer `0`, so the empty-counts guard is never triggered, risking ZeroDivisionError (`fastai/metrics.py`)
- `LabelSmoothingCrossEntropy.forward` computes the uniform-distribution component summed over both batch and class dimensions for `reduction='sum'`, inflating it relative to the NLL term when batch size varies (`fastai/losses.py`)
- `CutMix.before_batch` wraps `self.yb1` in a redundant single-element tuple via `tuple((self.y[shuffle],))`, causing shape mismatch when `lf` unpacks `*self.yb1` for multi-target losses (`fastai/callback/mixup.py`)
- `MixedPrecision.before_step` relies on GradScaler internally calling `self.step()` to clear `self.skipped`; if GradScaler internals change (PyTorch >= 2.3 deprecation path), the callback silently skips every optimizer step (`fastai/callback/fp16.py`)
- `Pad_Input.encodes` hardcodes `pad_idx=1` as default but `Numericalize` uses index 0 for unknown tokens (`defaultdict(int)`), so padding and unknown-token indices can collide when no explicit `pad_idx` is passed (`fastai/text/data.py`)
- `TrackerCallback.before_fit` resets `self.best` unconditionally when `reset_on_fit=True`, discarding improvement history across consecutive `.fit()` calls and causing `SaveModelCallback` to overwrite a genuinely better checkpoint with an inferior one from the first epoch of the new run (`fastai/callback/tracker.py`)

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
