# Known Bugs

This file tracks known bugs and issues in the fastai library. If you encounter a bug not listed here, please [open an issue](https://github.com/fastai/fastai/issues/new).

## Open

- `TensorBase.requires_grad_` uses a workaround for [pytorch#50219](https://github.com/pytorch/pytorch/issues/50219); may behave unexpectedly if the upstream fix changes semantics (`fastai/torch_core.py`)
- `Learner.summary` does not count parameters for individual `ParameterModule` instances wrapped outside of hook-tracked layers (`fastai/callback/hook.py`)
- `TfmdDL` padding uses `L.items.index` instead of `L.index` due to an unresolved upstream bug in `L` (`fastai/text/data.py`)
- `CorpusBLEU.value` compares `self.counts` (a list) against integer `0`, so the empty-counts guard is never triggered, risking ZeroDivisionError (`fastai/metrics.py`)

## Fixed

- `make_vocab` uses `f'xxfake'` without interpolating the loop variable, producing duplicate padding tokens instead of unique ones (`fastai/text/data.py`) - fixed by interpolating loop variable `i` in f-string (`f'xxfake{i}'`)
- `CorpusBLEUMetric.value` compared `self.counts` (a list) against integer 0 which always evaluates False, disabling the zero-counts guard (`fastai/metrics.py`) - fixed by using `max(self.counts) == 0`
- `LMDataLoader` does not support backward language model training (`fastai/text/data.py`) - added `backwards` parameter to `LMDataLoader.__init__` that reverses text sequences when enabled
- `SentencePieceTokenizer` does not forward special token symbols to the underlying SentencePiece model (`fastai/text/core.py`) - fixed: special tokens are passed via `--user_defined_symbols` in the `train()` method

## Reporting a Bug

Please include:
1. Output of `import fastai.test_utils; fastai.test_utils.show_install(1)`
2. A minimal reproducible example
3. Full stack trace (if applicable)
4. Expected vs. actual behavior
