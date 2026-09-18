# Dead Code Findings

## File: fastai/layers.py

### Item: `inplace_relu`
- **Type**: function (partial)
- **Lines**: L454
- **Evidence**: Only appears in `__all__` (L13) and its definition (L454). Not imported or referenced by any other module, no tests.
  ```
  grep results: only fastai/layers.py (definition + __all__)
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 1

### Item: `Debugger`
- **Type**: function (creates nn.Module via `@module()`)
- **Lines**: L95-L99
- **Evidence**: Only appears in `__all__` (L14) and its definition (L95-L99). Not imported or referenced by any other module, no tests. Calls `set_trace()` which is a debug utility not meant for production.
  ```
  grep results: only fastai/layers.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 5

### Item: `SimpleCNN`
- **Type**: class
- **Lines**: L437-L446
- **Evidence**: Only appears in `__all__` (L19), its definition (L437-L446), and `_modidx.py`. No tests, not imported or referenced by any other module.
  ```
  grep results: only fastai/layers.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 10

### Item: `SeparableBlock`
- **Type**: function
- **Lines**: L505-L506
- **Evidence**: Only appears in `__all__` (L20), its definition (L505-L506), and `_modidx.py`. No tests, not imported or referenced by any other module.
  ```
  grep results: only fastai/layers.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 2

### Item: `TimeDistributed`
- **Type**: class
- **Lines**: L514-L548
- **Evidence**: Only appears in `__all__` (L20), its definition (L514-L548), and `_modidx.py`. Also references internal helper `_stack_tups` (L509-L511). No tests, not imported or referenced by any other module.
  ```
  grep results: only fastai/layers.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes (also removes `_stack_tups` which is only used by `TimeDistributed`)
- **Estimated lines saved**: 38

### Item: `_stack_tups`
- **Type**: function (private helper)
- **Lines**: L509-L511
- **Evidence**: Only used by `TimeDistributed.low_mem_forward` (L538). If `TimeDistributed` is removed, this becomes dead.
  ```
  grep results: only fastai/layers.py (definition + usage in TimeDistributed)
  ```
- **Safe to remove**: yes (with TimeDistributed)
- **Estimated lines saved**: (counted with TimeDistributed above)

---

## File: fastai/torch_core.py

### Item: `_fa_rebuild_tensor` and `_fa_rebuild_qtensor`
- **Type**: function
- **Lines**: L216-L217
- **Evidence**: Defined but never called anywhere in the codebase. These were legacy pickle reconstruction helpers superseded by `_rebuild_from_type` (L341) which is what `TensorBase.__reduce_ex__` actually uses.
  ```
  grep results: only fastai/torch_core.py (definition), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 2

### Item: `show_image_batch`
- **Type**: function
- **Lines**: L804-L809
- **Evidence**: Only appears in `__all__` (L21), its definition (L804-L809), and `_modidx.py`. Not called by any other module. No tests.
  ```
  grep results: only fastai/torch_core.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 6

### Item: Duplicate `init_default` (torch_core.py vs layers.py)
- **Type**: duplicated function
- **Lines**: L819-L823 (torch_core.py) vs L214-L217 (layers.py)
- **Evidence**: `init_default` is defined in both `torch_core.py` and `layers.py`. Since `layers.py` does `from .torch_core import *`, and then redefines `init_default`, the layers.py version shadows the torch_core.py version for any consumer that imports from layers (or from modules that do `from .layers import *`). The torch_core.py version is only used internally by `cond_init` (L829) which is in the same file. The two implementations differ slightly:
  - **layers.py** (L214-L217): uses `nested_callable(m, 'bias.fill_')(0.)` for bias init
  - **torch_core.py** (L819-L823): uses `m.bias.data.fill_(0.)` for bias init, with explicit `hasattr` check for bias

  The torch_core.py version is used by `cond_init` -> `apply_init` -> `apply_leaf`, which are model initialization utilities. The layers.py version is used by `vision/gan.py` and test_layers.py. Both are exported in their respective `__all__`.
- **Safe to remove**: The torch_core.py version could be consolidated -- `cond_init` could import from layers instead. But this requires careful testing.
- **Estimated lines saved**: 5 (if consolidated)

### Item: Redundant `show` methods in `TitledInt`, `TitledFloat`, `TitledStr`, `TitledTuple`
- **Type**: duplicated logic
- **Lines**: L549-L572 (the 4 subclasses of ShowTitle)
- **Evidence**: `ShowTitle` (L542-L547) defines `_show_args = {'label': 'text'}` and `def show(self, ctx=None, **kwargs)`. All four subclasses (`TitledInt`, `TitledFloat`, `TitledStr`, `TitledTuple`) repeat both `_show_args` and `show` identically. The inherited versions from `ShowTitle` would behave the same. Each redundant copy is 3 lines (the `_show_args` assignment + the `show` method). 4 copies = 12 redundant lines.
  ```python
  # All four subclasses have this identical block:
  _show_args = {'label': 'text'}
  def show(self, ctx=None, **kwargs):
      "Show self"
      return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))
  ```
- **Safe to remove**: yes (remove redundant `_show_args` and `show` from each subclass; they inherit from `ShowTitle`)
- **Estimated lines saved**: 12

---

## File: fastai/vision/augment.py

### Item: `OldRandomCrop`
- **Type**: class
- **Lines**: L227-L233
- **Evidence**: Only appears in `__all__` (L12), its definition (L227-L233), and `_modidx.py`. Not imported or referenced by any other module. No tests. Superseded by `RandomCrop` (L198).
  ```
  grep results: only fastai/vision/augment.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 7

### Item: `DeterministicFlip`
- **Type**: class
- **Lines**: L643-L653
- **Evidence**: Only appears in `__all__` (L14), its definition (L643-L653), and `_modidx.py`. Not imported or referenced by any other module. No tests.
  ```
  grep results: only fastai/vision/augment.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 11

### Item: `DeterministicDihedral`
- **Type**: class
- **Lines**: L703-L712
- **Evidence**: Only appears in `__all__` (L15), its definition (L703-L712), and `_modidx.py`. Not imported or referenced by any other module. No tests.
  ```
  grep results: only fastai/vision/augment.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 10

### Item: `DeterministicDraw`
- **Type**: class
- **Lines**: L635-L639
- **Evidence**: Only used by `DeterministicFlip` (L652) and `DeterministicDihedral` (L711). If both are removed, this becomes dead. Not imported from outside, no tests.
  ```
  grep results: only fastai/vision/augment.py (definition + usage in DeterministicFlip/Dihedral), fastai/_modidx.py
  ```
- **Safe to remove**: yes (with DeterministicFlip and DeterministicDihedral)
- **Estimated lines saved**: 5

---

## File: fastai/interpret.py

### Item: `SegmentationInterpretation`
- **Type**: class (empty stub)
- **Lines**: L172-L174
- **Evidence**: Empty class (`pass` body) that only inherits from `Interpretation` without adding anything. Only appears in `__all__` (L14), its definition (L172-L174), and `_modidx.py`. Not imported or referenced by any other module. No tests.
  ```
  grep results: only fastai/interpret.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 3

---

## File: fastai/metrics.py

### Item: `optim_metric`
- **Type**: function
- **Lines**: L91-L103
- **Evidence**: Only appears in `__all__` (L12), its definition (L91-L103), and `_modidx.py`. Not imported or referenced by any other module. No tests. Uses `scipy.optimize.minimize_scalar` but is never called.
  ```
  grep results: only fastai/metrics.py (definition + __all__), fastai/_modidx.py
  ```
- **Safe to remove**: yes
- **Estimated lines saved**: 13

---

## Summary

| File | Item | Type | Lines Saved |
|------|------|------|-------------|
| layers.py | `inplace_relu` | dead function | 1 |
| layers.py | `Debugger` | dead function | 5 |
| layers.py | `SimpleCNN` | dead class | 10 |
| layers.py | `SeparableBlock` | dead function | 2 |
| layers.py | `TimeDistributed` + `_stack_tups` | dead class + helper | 38 |
| torch_core.py | `_fa_rebuild_tensor` + `_fa_rebuild_qtensor` | dead functions | 2 |
| torch_core.py | `show_image_batch` | dead function | 6 |
| torch_core.py | duplicate `init_default` | duplicated function | 5 |
| torch_core.py | redundant `show`/`_show_args` in Titled* | duplicated logic | 12 |
| vision/augment.py | `OldRandomCrop` | dead class | 7 |
| vision/augment.py | `DeterministicDraw` | dead class | 5 |
| vision/augment.py | `DeterministicFlip` | dead class | 11 |
| vision/augment.py | `DeterministicDihedral` | dead class | 10 |
| interpret.py | `SegmentationInterpretation` | dead stub class | 3 |
| metrics.py | `optim_metric` | dead function | 13 |
| **Total** | | | **~130** |

Note: Line counts above are for the function/class bodies only. Removing these items also requires updating `__all__` lists and `_modidx.py` entries.

### Items NOT included (covered by existing PRs)
- Dead JIT autograd utils (`script_use_ctx`, `script_save_ctx`, `script_fwd`, `script_bwd`, `grad_module`) -- PR #309
- Dead JIT custom autograd activation functions (`SwishJit`, `MishJit`, `_SwishJitAutoFn`, `MishJitAutoFn`, related jit fwd/bwd functions) -- PR #307
- Dead example scripts (`distrib.py`, `mnist_blocks.py`, `mnist_items.py`) -- PR #305
- Dead imports from `imports.py` -- PR #304
- Dead `migrating_fastai.py` and `fp16_utils.py` refactor -- PR #302
- Dead workflow files and test consolidation -- PRs #292, #291, #289, #287, #316, #298
