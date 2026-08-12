# How to contribute to fastai

First, thanks a lot for wanting to help! Make sure you have read the [doc on code style](
https://docs.fast.ai/dev/style.html) first. (Note that we don't follow PEP8, but instead follow a coding style designed specifically for numerical and interactive programming.) For help running and building the code, see the [developers guide](https://docs.fast.ai/dev/develop.html).

## Note for new contributors from Jeremy

It can be tempting to jump into a new project by questioning the stylistic decisions that have been made, such as naming, formatting, and so forth. This can be especially so for python programmers contributing to this project, which is unusual in following a number of conventions that are common in other programming communities, but not in Python. However, please don’t do this, for (amongst others) the following reasons:

- Contributing to [Parkinson’s law of triviality](https://www.wikiwand.com/en/Law_of_triviality) has negative consequences for a project. Let’s focus on deep learning!
- It’s exhausting to repeat the same discussion over and over again, especially when it’s been well documented already. When you have a question about the project, please check the pages in the docs website linked here.
- You’re likely to get a warmer welcome from the community if you start out by contributing something that’s been requested on the forum, since you’ll be solving someone’s current problem.
- If you start out by just telling us your point of view, rather than studying the background behind the decisions that have been made, you’re unlikely to be contributing anything new or useful.
- I’ve been writing code for nearly 40 years now, across dozens of languages, and other folks involved have quite a bit of experience too - the approaches used are based on significant experience and research. Whilst there’s always room for improvement, it’s much more likely you’ll be making a positive contribution if you spend a few weeks studying and working within the current framework before suggesting wholesale changes.

## How to get started

Here are some ways that you can learn a lot about the library, whilst also contributing to the community:

- Pick a class, function, or method and write tests for it. For instance, here are the tests for [fastai.core](https://github.com/fastai/fastai1/blob/master/tests/test_core.py). Adding tests for anything without good test coverage is a great way to really understand that part of the library deeply, and have in-depth conversations with the dev team about the reasoning behind decisions in the code.
- Document something that is currently undocumented. You can find them by looking for the “new methods” section in any doc notebook. Here’s a [search](https://github.com/fastai/fastai/search?q=%22new+methods%22&unscoped_q=%22new+methods%22) that lists them
- Add an example of use to the docs for something that doesn’t currently have an example of use. We’d like everything soon in the docs to include an actual piece of working code demonstrating it. Currently, we’ve largely only provided working examples for stuff higher up the abstraction ladder.

## Project Architecture

The fastai library is built with [nbdev](https://nbdev.fast.ai/), meaning the source of truth lives in Jupyter notebooks under `/nbs`. The generated Python modules in `/fastai` are exported from those notebooks and should not be hand-edited.

### Module layout

| Directory/File | Purpose |
|---|---|
| `nbs/` | Source notebooks (numbered by topic; `nbdev_export` generates `.py` files from them) |
| `fastai/torch_core.py` | Tensor subclass utilities, type dispatch, and PyTorch interop foundations |
| `fastai/layers.py` | Reusable neural-network building blocks (activations, normalization, pooling) |
| `fastai/data/` | DataLoaders pipeline: `core.py` (transforms, datasets), `load.py` (DataLoader), `block.py` (DataBlock API) |
| `fastai/learner.py` | `Learner` class that ties together model, data, loss, optimizer, and callbacks |
| `fastai/optimizer.py` | Optimizers and learning-rate scheduling |
| `fastai/callback/` | Training-loop callbacks: scheduling, mixed precision, hooks, progress, and more |
| `fastai/vision/` | Computer-vision: augmentations, models (xresnet, UNet), and GAN support |
| `fastai/text/` | NLP: tokenization (`core.py`), numericalization and DataLoaders (`data.py`), and AWD-LSTM models |
| `fastai/tabular/` | Tabular learning: pandas integration, categorical/continuous processing, and TabularModel |
| `fastai/medical/` | Domain-specific medical imaging utilities |
| `fastai/metrics.py` | Training metrics (accuracy, Bleu, perplexity, etc.) |
| `fastai/interpret.py` | Interpretation tools (confusion matrices, top losses) |

### Layered abstraction

fastai follows a layered design: low-level PyTorch utilities (`torch_core`) are composed into data pipelines (`data/`), which feed into `Learner`, which is extended by callbacks (`callback/`). Domain modules (`vision/`, `text/`, `tabular/`) build on all layers and expose high-level `DataLoaders` factories and pre-built architectures.

### Notebook-to-module mapping

Notebook filenames encode module paths. For example, `31_text.data.ipynb` generates `fastai/text/data.py`. The leading number determines build order; the dotted name after it maps directly to the package path.

## Did you find a bug?

* Nobody is perfect, especially not us. But first, please double-check the bug doesn't come from something on your side. The [forum](http://forums.fast.ai/) is a tremendous source for help, and we'd advise to use it as a first step. Be sure to include as much code as you can so that other people can easily help you.
* Then, ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/fastai/fastai/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/fastai/fastai/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages as well as the result of the line `import fastai.test_utils; fastai.test_utils.show_install(1)`.

#### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure that your PR includes tests that fail without your patch, and pass with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Before submitting, please be sure you abide by our [coding style](https://docs.fast.ai/dev/style.html) and [the guide on abbreviations](https://docs.fast.ai/dev/abbr.html) and clean-up your code accordingly.

## Do you intend to add a new feature or change an existing one?

* You can suggest your change on the [fastai forum](http://forums.fast.ai/) to see if others are interested or want to help. [This topic](http://forums.fast.ai/t/fastai-v1-adding-features/23041/8) lists the features that will be added to fastai in the foreseeable future. Be sure to read it too!
* Before implementing a non-trivial new feature, first create a notebook version of your new feature, like those in [dev_nb](https://github.com/fastai/fastai_docs/tree/master/dev_nb). It should show step-by-step what your code is doing, and why, with the result of each step. Try to simplify the code as much as possible. When you're happy with it, let us know on the forum (include a link to gist with your notebook.)
* Once your approach has been discussed and confirmed on the forum, you are welcome to push a PR, including a complete description of the new feature and an example of how it's used. Be sure to document your code and read the [doc on code style](https://docs.fast.ai/dev/style.html) and [the one on abbreviations](https://docs.fast.ai/dev/abbr.html).
* Ensure that your PR includes tests that exercise not only your feature, but also any other code that might be impacted. Currently we have poor test coverage of existing features, so often you'll need to add tests of existing code. Your help here is much appreciated!

## How to add a new module or subpackage

fastai uses [nbdev](https://nbdev.fast.ai/), so new modules start life as notebooks. Here's the workflow:

1. **Create a notebook** in `nbs/` following the naming convention `NN_module.name.ipynb` (e.g., `45_vision.filters.ipynb`). The numeric prefix controls the ordering in docs; pick a number that places your module logically among its neighbors.

2. **Use `#|export` at the top of cells** that contain code you want in the final library. Only exported cells end up in the generated `.py` file.

3. **Run `nbdev_export`** to generate (or regenerate) the corresponding `.py` file under `fastai/`. Never hand-edit generated `.py` files directly.

4. **If you're adding a new subpackage**, make sure the subpackage directory is listed in `settings.ini` under `lib_path` and that the directory has an `__init__.py`.

5. **Update `__init__.py` or `all.py`** for the subpackage so that the new module is importable via the package's public API (e.g., add the module to `_all_` or import it in `all.py`).

6. **Add tests** in `tests/` following existing patterns (e.g., `test_modulename.py`). Look at nearby test files for conventions around imports and fixtures.

7. **Do not edit `_modidx.py`** - it is auto-generated by nbdev. If it becomes stale, running `nbdev_export` will refresh it.

## How to submit notebook PRs?

Please run [`nbdev_install_hooks`](https://nbdev.fast.ai/api/clean.html#nbdev_install_hooks) in your terminal after cloning the repository. This sets up git hooks, which clean up the notebooks to remove the extraneous stuff stored in the notebooks (e.g. which cells you ran) which causes unnecessary merge conflicts.

If you made a change to the notebooks in one of the exported cells, you can export it to the library with [`nbdev_export`](https://nbdev.fast.ai/api/doclinks.html#nbdev_export).
If you made a change to the library, you can export it back to the notebooks with [`nbdev_update`](https://nbdev.fast.ai/api/sync.html#nbdev_update).

Furthermore, you can run tests in parallel by launching [`nbdev_test`](https://nbdev.fast.ai/api/test.html#nbdev_test).

If you'd like to learn the nbdev commands available and more about the project, please visit [`the docs`](https://nbdev.fast.ai/getting_started.html#how-to-use-nbdev).


## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.

* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.

* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.

* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.

* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.


### Code PRs

* If your PR is a bug fix, please also include a test that demonstrates the problem, or modifies an existing test that wasn't catching that problem already. Of course, it's not a requirement, so proceed anyway if you can't figure out how to write a test, but do try. Without having a test your fix could be lost down the road. By supplying a test, you're ensuring that your projects won't break in the future.

* Same applies for PRs that implement new features - without having a test case validating this new feature, it'd be very easy for that new feature to break in the future. A test case ensures that the feature will not break.


## Do you have questions about the source code?

* Please ask it on the [fastai forum](http://forums.fast.ai/) (after searching someone didn't ask the same one before with a quick search). We'd rather have the maximum of discussions there so that the largest number can benefit from it.

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the `/nbs` directory.
* To switch the `docs` submodule to ssh, `cd docs && git remote set-url origin git@github.com:fastai/fastai-docs.git`

## Backward Compatibility Guidelines

fastai is used in production systems, courses, and published research. Changes that silently break existing user code erode trust and create support burden. Follow these guidelines to keep the library stable while still evolving it.

### Rules for public API changes

1. **Never remove or rename a public function/class without a deprecation cycle.** Add a wrapper that calls the new implementation, emits a `DeprecationWarning` with the replacement name, and keep it for at least one minor release.
2. **Never change default parameter values** in a way that alters existing behavior. If the new default is better for new users, introduce a new parameter or create a separate method.
3. **Never change return types.** If a function returned a `list` and you want to return a `tensor`, add a new method or a `return_type` kwarg.
4. **Additive changes are safe.** New parameters with defaults that preserve old behavior, new methods, and new modules do not require a deprecation period.

### How to check for breakage

- Search the notebooks in `/nbs` for usage of the symbol you are changing. These notebooks serve as integration tests and documentation; if they break, users will too.
- Run `nbdev_test` on any notebook that exercises the changed API.
- Grep the test suite (`tests/`) for direct calls to the modified function.
- If your change modifies a callback's event signature, check that existing callbacks in `fastai/callback/` still work with the new signature.

### Deprecation pattern

```python
def old_name(*args, **kwargs):
    warnings.warn("`old_name` is deprecated; use `new_name` instead.", DeprecationWarning, stacklevel=2)
    return new_name(*args, **kwargs)
```

Add a brief note in the PR description indicating which symbol is deprecated and what replaces it so maintainers can track removal timelines.

## Migrating Code Contributions from fastai v1 to v2

If you are porting a feature, fix, or example from fastai v1, keep the following differences in mind:

### Key API changes

| v1 pattern | v2 equivalent | Notes |
|---|---|---|
| `DataBunch.create(...)` | `DataLoaders(train_dl, valid_dl)` | DataLoaders is a thin wrapper; create the underlying DataLoader objects with `TfmdDL` |
| `Learner(data, model, ...)` | `Learner(dls, model, ...)` | The first argument is now a `DataLoaders` instance, not a `DataBunch` |
| `fit_one_cycle(cyc_len, max_lr)` | `fit_one_cycle(n_epoch, lr_max)` | Parameter names changed; `lr_max` replaces `max_lr` |
| `ItemList.from_folder(...)` | `DataBlock(...).dataloaders(path)` | The mid-level `DataBlock` API replaces the `ItemList` pipeline |
| `callback_fns=[...]` | `cbs=[...]` | Callbacks are passed as instances, not classes/factories |

### Transform pipeline differences

- v1 transforms are plain functions applied via `ItemList.add_tfms()`; v2 transforms are `Transform` subclasses with `encodes`/`decodes` methods that enable reversible pipelines.
- Type dispatch means you define transforms once and they apply differently depending on whether the input is a `TensorImage`, `TensorText`, or `Category`. When porting, ensure your transform has proper type annotations.
- v2 transforms can carry state through the `setup()` method (e.g., vocabulary, normalization statistics). Prefer this over global variables or class-level caches from v1.

### Testing ported code

1. Confirm the v1 behavior you are porting still makes sense in v2 - some features were intentionally removed or redesigned.
2. Write your test against the v2 API from the start rather than adapting a v1 test. The test infrastructure uses `pytest` with fixtures defined in `conftest.py`.
3. If the ported feature touches a notebook, remember that `nbdev_export` must be run to regenerate the `.py` files - edits to the generated files will be overwritten.

### Common pitfalls when porting

- **Importing from moved modules**: Many v1 utilities moved from `fastai.core` to `fastcore.foundation` or `fastcore.basics`. Check `fastcore` first if an import fails.
- **`Callback` ordering**: v2 uses an explicit `order` attribute (lower runs first). If your v1 callback assumed a specific execution order relative to other callbacks, set `order` explicitly.
- **`L` vs plain lists**: v2 uses the `L` container pervasively. It supports indexing with lists/masks and has different iteration semantics than a plain Python list. Wrap outputs in `L()` when the downstream API expects it.
## Code Quality and Testing

### Identifying Dead Code

Dead code is code that is defined but never actually used at runtime. Common examples in Python include:

- **Unused imports**: modules or names imported at the top of a file but never referenced in the body.
  ```python
  # Bad: numpy is imported but never used
  import numpy as np
  import torch

  def my_func(x):
      return torch.relu(x)
  ```
  ```python
  # Good: only import what you use
  import torch

  def my_func(x):
      return torch.relu(x)
  ```

- **Unreferenced helper functions**: functions defined (often during iterative development) that no caller ever invokes.
  ```python
  # Bad: _old_helper is defined but never called anywhere
  def _old_helper(x):
      return x * 2

  def compute(x):
      return x + 1
  ```

To find unused imports, search for `import` lines whose imported names do not appear elsewhere in the file. Tools like `flake8` (with the F401 rule) or `pylint` can automate this detection.

### Running Tests

Run the full test suite from the repository root:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_optimizer.py
```

To run a specific test class or method:

```bash
pytest tests/test_optimizer.py::TestSGD::test_basic_step
```

### Test File Conventions

- Test files in `tests/` are **not** notebook-managed. Unlike files in `fastai/` (which are auto-generated from notebooks via `nbdev_export`), test files can be edited directly without breaking nbdev sync.
- The `tests/conftest.py` file already adds the repository root to `sys.path`, so pytest can import `fastai` modules without needing a package install.
- Tests mock heavy dependencies (PyTorch training loops, GPU ops) where possible, so many tests run without a full training environment.
## Understanding Callback Ordering and Interactions

Callbacks are fastai's primary extension mechanism. Getting their `order` attribute and inter-callback dependencies right is essential for writing correct callbacks and diagnosing bugs.

### Execution order

Every callback has an `order` class attribute (default 0). Lower values run first. Key built-in orderings:

| Callback | order | Why |
|----------|-------|-----|
| `TrainEvalCallback` | 10 | Sets train/eval mode before anything else |
| `Recorder` | 50 | Records metrics after the batch is done |
| `ProgressCallback` | 60 | Displays after recording |
| `TrackerCallback` | 60 | Reads recorded metrics |
| `SaveModelCallback` | 61 | Acts on tracker results |
| `EarlyStoppingCallback` | 63 | Acts after save decisions |
| `MixedPrecision` | 10 | Wraps forward pass in autocast |

When writing a new callback, pick an `order` relative to the callbacks you depend on. If your callback reads `self.smooth_loss`, it must run **after** `Recorder` (order > 50).

### Attribute resolution via GetAttr

Callbacks inherit from `GetAttr` with `_default='learn'`. This means:
- `self.model` resolves to `self.learn.model`
- `self.opt` resolves to `self.learn.opt`
- `self.recorder` finds the `Recorder` callback instance (learner searches callbacks by lowercase class name)

When one callback needs state from another, use this delegation (e.g., `self.recorder.values`). Do **not** store cross-callback references in `__init__`.

### Common pitfalls

1. **Missing `self.run` guards** - If your callback sets `self.run = False` in `before_fit` (e.g., during `lr_find`), every other event method must check `if not self.run: return` before accessing state created in `before_fit`.

2. **Accessing uninitialized state** - If `before_fit` creates `self.writer` or `self.hps`, ensure `after_batch`/`after_epoch` only access them after confirming initialization succeeded.

3. **Order conflicts with `MixedPrecision`** - The `MixedPrecision` callback enters an autocast context in `before_batch` and exits in `after_loss`. Any callback that modifies the loss between these events must account for the precision state.

4. **`store_attr` with explicit names** - When calling `store_attr('a, b, c')` with an explicit list, parameters not in the list are silently dropped. Always verify that every parameter your methods reference is included.

### Testing callbacks

Write tests that exercise your callback in isolation using `ShortEpochCallback` to limit training:

```python
learn = synth_learner(cbs=[MyCallback(), ShortEpochCallback()])
learn.fit(1)
# assert on state your callback should have set
```

## PR Checklist

Before marking your pull request as ready for review, verify the following:

- [ ] **Branch is up to date** with `master` (rebase or merge latest changes)
- [ ] **`nbdev_export`** has been run if you changed any notebook cells tagged with `#|export`
- [ ] **`nbdev_update`** has been run if you changed library `.py` files directly
- [ ] **`nbdev_test`** passes locally (use `--do_print` to see which notebooks run)
- [ ] **No generated file edits** - never hand-edit files in `fastai/` that are auto-generated from notebooks; edit the notebook instead
- [ ] **Coding style** follows the [fastai style guide](https://docs.fast.ai/dev/style.html) and [abbreviation conventions](https://docs.fast.ai/dev/abbr.html)
- [ ] **Tests included** - add at least one test that fails without your change and passes with it
- [ ] **Docs updated** - if your change affects public API, update or add a docstring and an example in the relevant notebook
- [ ] **Single concern** - the PR addresses one bug fix or one feature, not a mix of unrelated changes
- [ ] **Clean history** - squash fixup commits; each commit in the PR should represent a logical unit of work

## Release and Versioning

fastai uses [SemVer](https://semver.org/) versioning. Here is how releases relate to contributions:

- **Version bumps** are handled by maintainers, not contributors. Do not modify `settings.ini` version fields in your PR.
- **Release cadence**: there is no fixed schedule. Releases happen when enough significant changes accumulate or a critical fix lands.
- **Which branch to target**: always base your PR on `master`. This is the only integration branch.
- **When your change ships**: after your PR is merged to `master`, it will be included in the next PyPI/conda release. There is no separate staging branch.
- **Breaking changes**: if your PR changes public API behavior, call it out clearly in the PR description. Breaking changes require a major version bump and additional review from maintainers.
- **Changelog**: fastai does not maintain a manual CHANGELOG file. Release notes are generated from merged PR titles and descriptions, so write clear PR titles that summarize the user-visible effect.
## Dead Code Removal Guide

Removing unused code reduces maintenance burden and makes the codebase easier to navigate. Here is how to safely identify and remove dead code in this project:

### What qualifies as dead code

- **Unused imports**: modules or names imported but never referenced in the file
- **Unreachable functions**: functions defined but never called from any module
- **Commented-out code**: old code left behind in comments with no explanatory reason

### How to find dead code

1. **Python AST analysis** - Parse a file's abstract syntax tree and compare imported names against all `Name` nodes used in the file body. An import that only appears in the import statement itself is dead.
2. **Grep across the repo** - For a function or class defined in one file, search the entire repository (`grep -rn "function_name" --include="*.py"`) to confirm it is called somewhere.
3. **Static analysis tools** - Tools like `vulture` or `pyflakes` automate unused-name detection.

### Important constraints

- **Do not edit autogenerated files** directly. Files starting with `# AUTOGENERATED! DO NOT EDIT!` are exported from notebooks via `nbdev_export`. To remove dead code from these files, edit the corresponding notebook in `nbs/` and re-export.
- **Non-autogenerated files** (e.g., `fastai/fp16_utils.py`, `setup.py`, `tests/*.py`) can be edited directly.
- **Always run the test suite** after removing imports to confirm nothing breaks. Some imports may have side effects (e.g., registering a module) even if the imported name is never referenced explicitly.
- **Update `REPO_LINE_SIZE.md`** with the new `wc -l` output after your changes, so the line count stays current.

## Testing Best Practices

Writing good tests for fastai requires understanding how the library uses notebooks, GPU resources, and dynamic dispatch. Follow these guidelines to write tests that are reliable, fast, and useful.

### Where to put tests

- **Notebook-driven tests**: For features defined in notebooks (cells tagged with `#|export`), add test cells in the same notebook directly below the implementation. These run via `nbdev_test`.
- **Standalone test files**: For integration tests or tests that require complex fixtures, add them under the `tests/` directory following the naming convention `test_<module>.py`.

### Structuring a test

1. **Arrange**: Create minimal synthetic data rather than downloading datasets. Use `synth_learner()` or small random tensors to avoid network dependencies and keep tests fast.
2. **Act**: Call the function or train for 1-2 epochs only. Never train to convergence in a test.
3. **Assert**: Check concrete values (tensor shapes, metric ranges, file existence) rather than just "no exception was raised."

```python
def test_lr_find_returns_suggestion():
    learn = synth_learner()
    lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
    assert 1e-7 < lr_min < 1.0, f"lr_min out of range: {lr_min}"
    assert 1e-7 < lr_steep < 1.0, f"lr_steep out of range: {lr_steep}"
```

### Handling GPU and slow tests

- Mark GPU-requiring tests with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")` so they are skipped gracefully in CPU-only CI environments.
- For tests that take more than a few seconds (large model loading, multi-epoch training), add a `@pytest.mark.slow` marker and document why the duration is necessary.
- Always set a small `bs` (batch size) and use tiny image sizes (e.g. 32x32) to minimize runtime.

### Testing callbacks

Callbacks interact with the training loop at specific events. Test them in isolation where possible:

```python
def test_custom_callback_fires():
    called = []
    class _TestCb(Callback):
        def after_batch(self): called.append('after_batch')
    learn = synth_learner(cbs=[_TestCb()])
    learn.fit(1)
    assert 'after_batch' in called
```

### Common pitfalls

- **Non-determinism**: Set `torch.manual_seed()` and `random.seed()` at the top of tests that check numeric values. Floating-point comparisons should use `torch.allclose()` with an appropriate tolerance.
- **Global state leakage**: Each test should create its own `Learner` and data. Never rely on objects created in a previous test.
- **File cleanup**: If your test writes files (model exports, logs), use `tmp_path` (pytest) or Python's `tempfile` module and clean up in a `finally` block or fixture teardown.
- **Mocking external services**: If a feature calls an external API (e.g. Weights & Biases logging), mock the network call rather than requiring credentials in CI.

## GPU/CUDA Troubleshooting for Contributors

Running fastai locally requires a working PyTorch installation, but CUDA version mismatches are one of the most common stumbling blocks for new contributors. This section covers how to diagnose and work around GPU-related issues during development.

### Verifying your CUDA setup

```python
import torch
print(torch.__version__)           # e.g. 2.1.0+cu121
print(torch.cuda.is_available())   # True if GPU is usable
print(torch.version.cuda)          # CUDA toolkit version bundled with PyTorch
```

If `torch.cuda.is_available()` returns `False`, the most common causes are:

1. **Driver/toolkit mismatch**: Your NVIDIA driver is older than what the installed PyTorch CUDA build requires. Run `nvidia-smi` to see your driver version and compare against the [PyTorch compatibility matrix](https://pytorch.org/get-started/locally/).
2. **CPU-only PyTorch installed**: If you installed PyTorch from a channel without specifying the CUDA variant, you may have a CPU-only build. Reinstall with the correct `--index-url` for your CUDA version.
3. **No GPU present**: On machines without a discrete NVIDIA GPU (e.g., CI runners, Mac laptops), CUDA is unavailable by design.

### Running tests without a GPU

Most fastai tests can run on CPU. The test suite skips GPU-requiring tests automatically when no CUDA device is detected. To explicitly force CPU-only execution (useful for debugging non-GPU issues without interference):

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/
```

Setting `CUDA_VISIBLE_DEVICES` to an empty string hides all GPUs from PyTorch, ensuring your tests exercise the CPU code paths.

### Common CUDA errors and fixes

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `CUDA out of memory` | Batch size too large for GPU VRAM | Reduce `bs` in your test or use smaller image sizes (32x32) |
| `CUDA error: device-side assert triggered` | Index out of bounds in a kernel (often wrong number of classes) | Check that `n_out` matches your label range; run with `CUDA_LAUNCH_BLOCKING=1` for a clearer stack trace |
| `RuntimeError: CUDA error: no kernel image is available` | PyTorch built for a different GPU architecture | Reinstall PyTorch matching your GPU compute capability |
| `libcudnn.so: cannot open shared object file` | cuDNN not installed or not on `LD_LIBRARY_PATH` | Install cuDNN matching your CUDA version, or use a conda environment that bundles it |

### Debugging CUDA errors

CUDA errors are asynchronous by default, meaning the Python stack trace points to a later operation rather than the one that actually failed. To get an accurate trace:

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pytest tests/test_vision.py -x
```

This forces synchronous execution so the error is raised at the exact offending line. Note that this significantly slows down execution, so only use it for debugging.

### Writing GPU-aware tests

When contributing tests that require a GPU, guard them so they are skipped gracefully in CPU-only environments:

```python
import pytest, torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_mixed_precision_training():
    # test body that requires CUDA
    ...
```

This keeps the test suite green for contributors who develop on CPU-only machines or in CI environments without GPU access.
## Troubleshooting Common Issues

When working with the fastai codebase, you may encounter confusing errors. This section documents common issues and their solutions.

### Import errors after editing notebooks

**Symptom**: `ImportError` or `AttributeError` when importing a module you just changed.

**Cause**: The `.py` files under `fastai/` are auto-generated from notebooks. If you edited the notebook but forgot to export, the `.py` file is stale.

**Fix**: Run `nbdev_export` from the repo root. Never hand-edit files that begin with `# AUTOGENERATED! DO NOT EDIT!`.

### `store_attr` silently drops parameters

**Symptom**: Accessing `self.some_param` raises `AttributeError` even though the parameter is in `__init__`.

**Cause**: When `store_attr` is called with an explicit string list (e.g., `store_attr('a, b, c')`), only the names in that list are stored. If you add a new parameter to `__init__` but forget to add it to the `store_attr` string, it is silently dropped.

**Fix**: Add the new parameter name to the `store_attr` string, or switch to calling `store_attr()` with no arguments (which stores all `__init__` params except `self`).

### Callbacks not firing during training

**Symptom**: Your callback's `after_epoch` or `after_batch` method is never called.

**Cause**: The callback system checks `self.run` before invoking event methods. If another callback (or `before_fit` in your own callback) sets `self.run = False`, all subsequent events are skipped. Additionally, `run_valid = False` (the default for some mixins) suppresses callbacks during the validation phase.

**Fix**: Check whether your callback inherits from a class that sets `run_valid = False`. Also verify that `before_fit` does not accidentally set `self.run = False` for your use case (e.g., during `lr_find` or `gather_preds`, many tracker callbacks disable themselves).

### `CancelBatchException` vs `CancelTrainException` confusion

**Symptom**: Training stops unexpectedly, or a single batch skip aborts the entire epoch.

**Cause**: fastai uses a hierarchy of cancellation exceptions (`CancelBatchException`, `CancelTrainException`, `CancelEpochException`, `CancelFitException`). Raising the wrong one has different effects:
- `CancelBatchException` skips the current batch but continues training.
- `CancelTrainException` ends the training phase of the current epoch and moves to validation.
- `CancelFitException` ends all training immediately.

**Fix**: Use the most specific exception for your intent. If you only want to skip the optimizer step for one batch (e.g., gradient overflow), use `CancelBatchException`.

### `DataLoader` hangs on macOS or Windows with `num_workers > 0`

**Symptom**: Training hangs or crashes immediately when using multiprocessing workers.

**Cause**: fastai's `DataLoader` automatically sets `num_workers = 0` on macOS and Windows in notebook environments, but this guard may not activate in all contexts (e.g., plain scripts).

**Fix**: Explicitly pass `num_workers=0` when creating your `DataLoaders` on macOS/Windows, or use the `defaults.cpus` setting to control parallelism globally.

### Mixed precision training fails with "RuntimeError: expected scalar type Half"

**Symptom**: After calling `learn.to_fp16()`, certain operations fail with dtype mismatch errors.

**Cause**: The `MixedPrecision` callback wraps forward passes in `torch.cuda.amp.autocast`, but operations outside the forward pass (custom loss functions, metric calculations, callback computations) run in full precision. If your custom code casts tensors manually or stores intermediate results from inside autocast, dtype mismatches occur.

**Fix**: Do not manually cast tensors to `float16` inside your loss function or callbacks. Let `autocast` handle precision automatically. If you need a tensor in full precision inside an autocast region, use `tensor.float()` explicitly for that computation.

### Optimizer state not loaded after `learn.load()`

**Symptom**: After loading a saved model, the learning rate schedule or momentum appears reset.

**Cause**: By default, `learn.save()` and `learn.load()` do not include optimizer state. The `with_opt` parameter defaults to `False`.

**Fix**: Use `learn.save('model', with_opt=True)` and `learn.load('model', with_opt=True)` to persist and restore optimizer state alongside model weights.
