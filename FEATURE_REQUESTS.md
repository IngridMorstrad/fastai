# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration)
- Add support for automatic dataset versioning and lineage tracking so users can reproduce experiments by referencing a specific data snapshot
- Add a `DataLoaders.summary()` method that prints dataset statistics (class distribution, missing values, sample counts per split) to help users quickly audit their data before training

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models
- Support automatic mixed-precision gradient scaling configuration per parameter group to allow selective full-precision training of sensitive layers (e.g. batch norm)
- [DONE] Add a checkpoint averaging callback that maintains the top-K model checkpoints by validation loss and produces a weight-averaged model at the end of training for improved generalization -- implemented as `CheckpointAveragingCallback`
- Add built-in support for gradient accumulation over multiple batches so users can simulate larger batch sizes on memory-constrained hardware without manual callback wiring

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`
- [DONE] Add built-in GradCAM/Grad-CAM++ visualization support to highlight class-discriminative regions in image predictions -- implemented in `fastai.vision.gradcam`

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models
- Add a `TextLearner.generate()` method with configurable decoding strategies (greedy, beam search, top-k, nucleus sampling) so users can evaluate generative models without external tooling

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining
- Add built-in support for automatic feature engineering (e.g. date-part extraction, target encoding, interaction features) configurable via a simple API in `TabularPandas`

## Callbacks

- [DONE] Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all) -- implemented as `MultiMetricEarlyStoppingCallback`
- Add a built-in GPU memory profiling callback that logs peak GPU memory usage per training step to help users diagnose OOM issues
- Add a `DebugCallback` that logs tensor shapes, dtypes, and gradient norms at each training event to help users diagnose silent shape mismatches and vanishing/exploding gradients

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model
- Add a `Learner.benchmark()` method that profiles inference latency, throughput, and memory usage across configurable batch sizes to help users choose optimal deployment settings

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
- Provide a `Learner.export_logs()` method that writes training metrics to structured JSON/CSV files compatible with TensorBoard, Weights & Biases, and other visualization tools
