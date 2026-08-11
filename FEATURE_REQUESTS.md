# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration)
- Add support for automatic dataset versioning and lineage tracking so users can reproduce experiments by referencing a specific data snapshot
- Add `DataLoaders.auto_augment(policy='imagenet')` that searches for an optimal augmentation policy (AutoAugment/RandAugment) using a proxy task on a subset of training data

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models
- Support automatic mixed-precision gradient scaling configuration per parameter group to allow selective full-precision training of sensitive layers (e.g. batch norm)
- [DONE] Add a checkpoint averaging callback that maintains the top-K model checkpoints by validation loss and produces a weight-averaged model at the end of training for improved generalization -- implemented as `CheckpointAveragingCallback`
- Add `Learner.federated_fit()` that coordinates training across multiple data silos using federated averaging, allowing users to train on distributed private datasets without centralizing raw data

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`
- [DONE] Add built-in GradCAM/Grad-CAM++ visualization support to highlight class-discriminative regions in image predictions -- implemented in `fastai.vision.gradcam`

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining
- Add `TabularLearner.feature_importance(method='permutation')` that computes per-feature importance scores via permutation or SHAP values and returns a ranked DataFrame with optional visualization

## Callbacks

- [DONE] Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all) -- implemented as `MultiMetricEarlyStoppingCallback`
- Add a built-in GPU memory profiling callback that logs peak GPU memory usage per training step to help users diagnose OOM issues
- Add a `ReproducibilityCallback` that snapshots the full training configuration (random seeds, library versions, git hash, hyperparameters) at `before_fit` and writes it to a JSON manifest alongside the saved model

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model
- Add a `Learner.benchmark()` method that profiles inference latency, throughput, and memory usage across configurable batch sizes to help users choose optimal deployment settings

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
- Provide a `Learner.export_logs()` method that writes training metrics to structured JSON/CSV files compatible with TensorBoard, Weights & Biases, and other visualization tools
- Add `Learner.model_card()` that generates a Markdown model card documenting architecture, training data summary, performance metrics, intended use, and known limitations following the standard model card template
