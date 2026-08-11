# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration)
- Add support for automatic dataset versioning and lineage tracking so users can reproduce experiments by referencing a specific data snapshot
- Add a `DataLoaders.profile()` method that reports per-batch wall-clock time, transform hotspots, and CPU/GPU transfer overhead to identify data-loading bottlenecks

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models
- Support automatic mixed-precision gradient scaling configuration per parameter group to allow selective full-precision training of sensitive layers (e.g. batch norm)
- [DONE] Add a checkpoint averaging callback that maintains the top-K model checkpoints by validation loss and produces a weight-averaged model at the end of training for improved generalization -- implemented as `CheckpointAveragingCallback`
- Add a `Learner.distill()` method that wraps knowledge distillation training using a frozen teacher model, configurable temperature, and soft/hard loss weighting

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`
- [DONE] Add built-in GradCAM/Grad-CAM++ visualization support to highlight class-discriminative regions in image predictions -- implemented in `fastai.vision.gradcam`

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models
- Add a `TextLearner.explain(text)` method that returns token-level attribution scores (integrated gradients or SHAP) highlighting which input tokens most influenced the prediction

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining

## Callbacks

- [DONE] Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all) -- implemented as `MultiMetricEarlyStoppingCallback`
- Add a built-in GPU memory profiling callback that logs peak GPU memory usage per training step to help users diagnose OOM issues
- Add a `WandbSweepCallback` that integrates Weights & Biases hyperparameter sweeps directly into the Learner, reporting metrics per trial and supporting early termination of underperforming runs

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model
- Add a `Learner.benchmark()` method that profiles inference latency, throughput, and memory usage across configurable batch sizes to help users choose optimal deployment settings

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
- Provide a `Learner.export_logs()` method that writes training metrics to structured JSON/CSV files compatible with TensorBoard, Weights & Biases, and other visualization tools
- Add an `nbdev_check_links` CLI command that scans all documentation notebooks for broken cross-references and dead URLs, reporting them in CI to prevent link rot
