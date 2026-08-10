# Repository Line Count

Line counts for all Python files in the repository, generated with `find . -name "*.py" -not -path "*/__pycache__/*" | sort | xargs wc -l`.

Last verified: 2026-08-08

## Summary

- **Total lines**: 24726
- **Dead code removed this round**: ~1042 lines (consolidated 3 duplicate test files into 1, removed dead `_set_grad` helper, removed unused `import types`)
- **Other fixes applied**: replaced deprecated `pkg_resources` with `packaging.version` in setup.py (prior round)

## Dead Code Removed

| File | Removed |
|------|---------|
| `tests/test_data_loader.py` | Entire file deleted (640 lines) -- unique tests merged into `tests/test_data_load.py` |
| `tests/test_dataloader.py` | Entire file deleted (700 lines) -- unique tests merged into `tests/test_data_load.py` |
| `tests/test_optimizer.py` | Dead helper function `_set_grad` (4 lines, superseded by `_get_params_with_grad`) |
| `tests/test_text_core.py` | Unused `import types` (1 line, module never referenced) |
| `fastai/imports.py` | Dead function `is_coll` (defined but never called anywhere) |
| `fastai/layers.py` | Unused import `uniform_` from `torch.nn.init` |
| `fastai/callback/tensorboard.py` | Unused imports `tensorboard` and `ModelToHalf` |
| `fastai/callback/fp16.py` | Unused import `OptState` from `torch.cuda.amp.grad_scaler` |
| `fastai/callback/schedule.py` | Unused import `SaveModelCallback` from `.tracker` |
| `fastai/callback/tracker.py` | Unused import `MixedPrecision` from `.fp16` |
| `fastai/callback/captum.py` | Unused import `tempfile` |
| `fastai/distributed.py` | Unused imports `_loaders` and `OptimWrapper` |
| `fastai/vision/augment.py` | Unused import `Bernoulli` from `torch.distributions.bernoulli` |

## Full Line Count (wc -l)

```
     24 ./dev_nbs/course/crappify.py
      6 ./fastai/basics.py
     11 ./fastai/callback/all.py
    112 ./fastai/callback/captum.py
     44 ./fastai/callback/channelslast.py
    189 ./fastai/callback/core.py
     73 ./fastai/callback/data.py
    246 ./fastai/callback/fp16.py
    283 ./fastai/callback/hook.py
      1 ./fastai/callback/__init__.py
    113 ./fastai/callback/mixup.py
     20 ./fastai/callback/preds.py
    126 ./fastai/callback/progress.py
     44 ./fastai/callback/rnn.py
    299 ./fastai/callback/schedule.py
    167 ./fastai/callback/tensorboard.py
    279 ./fastai/callback/tracker.py
     59 ./fastai/callback/training.py
    324 ./fastai/callback/wandb.py
    104 ./fastai/collab.py
      6 ./fastai/data/all.py
    247 ./fastai/data/block.py
    528 ./fastai/data/core.py
    107 ./fastai/data/download_checks.py
    138 ./fastai/data/external.py
      1 ./fastai/data/__init__.py
    215 ./fastai/data/load.py
    384 ./fastai/data/transforms.py
    224 ./fastai/distributed.py
     73 ./fastai/fp16_utils.py
     85 ./fastai/imports.py
      2 ./fastai/__init__.py
    174 ./fastai/interpret.py
    660 ./fastai/layers.py
    682 ./fastai/learner.py
    281 ./fastai/losses.py
    412 ./fastai/medical/imaging.py
      0 ./fastai/medical/__init__.py
    473 ./fastai/metrics.py
   2682 ./fastai/_modidx.py
    497 ./fastai/optimizer.py
     46 ./fastai/_pytorch_doc.py
      6 ./fastai/tabular/all.py
    404 ./fastai/tabular/core.py
     60 ./fastai/tabular/data.py
      0 ./fastai/tabular/__init__.py
     56 ./fastai/tabular/learner.py
     81 ./fastai/tabular/model.py
    165 ./fastai/test_utils.py
      6 ./fastai/text/all.py
    381 ./fastai/text/core.py
    290 ./fastai/text/data.py
      1 ./fastai/text/__init__.py
    305 ./fastai/text/learner.py
    183 ./fastai/text/models/awdlstm.py
    173 ./fastai/text/models/core.py
      1 ./fastai/text/models/__init__.py
     13 ./fastai/torch_basics.py
    903 ./fastai/torch_core.py
      9 ./fastai/torch_imports.py
      9 ./fastai/vision/all.py
   1266 ./fastai/vision/augment.py
    309 ./fastai/vision/core.py
    221 ./fastai/vision/data.py
    406 ./fastai/vision/gan.py
    231 ./fastai/vision/gradcam.py
      1 ./fastai/vision/__init__.py
    362 ./fastai/vision/learner.py
      3 ./fastai/vision/models/all.py
      3 ./fastai/vision/models/__init__.py
     11 ./fastai/vision/models/tvm.py
     98 ./fastai/vision/models/unet.py
    111 ./fastai/vision/models/xresnet.py
    105 ./fastai/vision/utils.py
    125 ./fastai/vision/widgets.py
     14 ./nbs/dltest.py
     37 ./nbs/examples/dataloader_spawn.py
     16 ./nbs/examples/distrib.py
     31 ./nbs/examples/distrib_pytorch.py
     36 ./nbs/examples/migrating_catalyst.py
     29 ./nbs/examples/migrating_fastai.py
     93 ./nbs/examples/migrating_ignite.py
     44 ./nbs/examples/migrating_lightning.py
     68 ./nbs/examples/migrating_pytorch.py
     11 ./nbs/examples/mnist_blocks.py
     11 ./nbs/examples/mnist_items.py
     87 ./nbs/examples/train_imagenette.py
     40 ./nbs/examples/train_imdbclassifier.py
     43 ./nbs/examples/train_tabular.py
     46 ./nbs/examples/train_wt2.py
     63 ./setup.py
     14 ./tests/conftest.py
      0 ./tests/__init__.py
    441 ./tests/test_checkpoint_averaging.py
    370 ./tests/test_collab.py
   1023 ./tests/test_data_load.py
    699 ./tests/test_data_transforms.py
    296 ./tests/test_gradcam.py
    477 ./tests/test_layers.py
    146 ./tests/test_lm_dataloader.py
    337 ./tests/test_losses.py
   1293 ./tests/test_metrics.py
    353 ./tests/test_multi_metric_early_stopping.py
    767 ./tests/test_optimizer.py
    574 ./tests/test_text_core.py
    528 ./tests/test_torch_core.py
  24726 total
```
