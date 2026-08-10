# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration)
- Add support for automatic dataset versioning and lineage tracking so users can reproduce experiments by referencing a specific data snapshot
- Add a `DataLoaders.from_registry(name)` method that downloads and caches popular benchmark datasets (CIFAR, MNIST, ImageNet-1k subsets, GLUE) with canonical train/val/test splits in one call

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models
- Support automatic mixed-precision gradient scaling configuration per parameter group to allow selective full-precision training of sensitive layers (e.g. batch norm)
- [DONE] Add a checkpoint averaging callback that maintains the top-K model checkpoints by validation loss and produces a weight-averaged model at the end of training for improved generalization -- implemented as `CheckpointAveragingCallback`
- Add `Learner.fit_with_restarts()` implementing cosine annealing with warm restarts (SGDR), automatically adjusting cycle length and learning rate bounds across multiple restart cycles

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`
- [DONE] Add built-in GradCAM/Grad-CAM++ visualization support to highlight class-discriminative regions in image predictions -- implemented in `fastai.vision.gradcam`

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models
- Add `TextLearner.summarize(text, max_length)` for abstractive summarization with configurable length constraints and support for extractive fallback on smaller models

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining
- Add `TabularLearner.counterfactual(row, target_class)` method that generates minimal feature perturbations to flip the prediction, helping users understand decision boundaries

## Callbacks

- [DONE] Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all) -- implemented as `MultiMetricEarlyStoppingCallback`
- Add a built-in GPU memory profiling callback that logs peak GPU memory usage per training step to help users diagnose OOM issues
- Add a `StochasticWeightAveraging` callback that maintains an exponential moving average of model weights and swaps to the averaged weights at evaluation time for improved generalization

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model
- Add a `Learner.benchmark()` method that profiles inference latency, throughput, and memory usage across configurable batch sizes to help users choose optimal deployment settings

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
- Provide a `Learner.export_logs()` method that writes training metrics to structured JSON/CSV files compatible with TensorBoard, Weights & Biases, and other visualization tools
