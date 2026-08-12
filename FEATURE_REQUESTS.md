# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- [DONE] Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration) -- `DataLoader` natively detects `IterableDataset` instances and supports non-indexed streaming iteration
- Add support for automatic dataset versioning and lineage tracking so users can reproduce experiments by referencing a specific data snapshot
- Add multi-GPU `DataLoaders.distributed()` method that automatically shards data across GPUs with balanced partition sizes and deterministic ordering for reproducibility
- Add a `DataLoaders.from_registry(name)` method that downloads and caches popular benchmark datasets (CIFAR, MNIST, ImageNet-1k subsets, GLUE) with canonical train/val/test splits in one call
- Add a DataLoader option for automatic class-balanced sampling that reweights batches to handle heavily imbalanced classification datasets without manual oversampling

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models
- Support automatic mixed-precision gradient scaling configuration per parameter group to allow selective full-precision training of sensitive layers (e.g. batch norm)
- [DONE] Add a checkpoint averaging callback that maintains the top-K model checkpoints by validation loss and produces a weight-averaged model at the end of training for improved generalization -- implemented as `CheckpointAveragingCallback`
- Add a `Learner.prune(method='structured')` method that applies structured weight pruning with an integrated fine-tuning schedule to recover accuracy after sparsification
- [DONE] Add `Learner.fit_with_restarts()` implementing cosine annealing with warm restarts (SGDR), automatically adjusting cycle length and learning rate bounds across multiple restart cycles -- already implemented as `Learner.fit_sgdr()` in `fastai/callback/schedule.py`
- Add a built-in gradient accumulation wrapper that transparently simulates larger batch sizes across multiple forward passes for memory-constrained GPUs

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`
- [DONE] Add built-in GradCAM/Grad-CAM++ visualization support to highlight class-discriminative regions in image predictions -- implemented in `fastai.vision.gradcam`
- Add a `vision_learner.ssl_pretrain()` method that supports self-supervised pretraining strategies (SimCLR, BYOL, DINO) on unlabeled image datasets before fine-tuning
- Add a built-in test-time augmentation (TTA) pipeline that supports configurable augmentation policies and ensembling strategies beyond simple averaging

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models
- Add `TextLearner.summarize(text, max_length)` for abstractive summarization with configurable length constraints and support for extractive fallback on smaller models
- Add a tokenizer-agnostic vocabulary inspector that reports token frequency, coverage, and out-of-vocabulary rate on a given corpus to aid text preprocessing decisions

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining
- Add `TabularLearner.counterfactual(row, target_class)` method that generates minimal feature perturbations to flip the prediction, helping users understand decision boundaries

## Callbacks

- [DONE] Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all) -- implemented as `MultiMetricEarlyStoppingCallback`
- Add a built-in GPU memory profiling callback that logs peak GPU memory usage per training step to help users diagnose OOM issues
- Add a `GradientNoiseCallback` that injects decayed Gaussian noise into gradients during training to help escape sharp minima and improve generalization on small datasets
- Add a `StochasticWeightAveraging` callback that maintains an exponential moving average of model weights and swaps to the averaged weights at evaluation time for improved generalization
- Add a learning rate warmup callback that linearly or exponentially ramps the learning rate over a configurable number of initial steps to stabilize early training

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model
- Add a `Learner.benchmark()` method that profiles inference latency, throughput, and memory usage across configurable batch sizes to help users choose optimal deployment settings

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
- Provide a `Learner.export_logs()` method that writes training metrics to structured JSON/CSV files compatible with TensorBoard, Weights & Biases, and other visualization tools
- Add a `fastai_diff_nbs` CLI command that shows semantic diffs between notebook versions, ignoring cell outputs and metadata so reviewers can focus on actual code and prose changes
