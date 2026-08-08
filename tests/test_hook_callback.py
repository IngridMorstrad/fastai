"""Tests for fastai.callback.hook module.

Covers Hook, Hooks, hook_output, hook_outputs, has_params, total_params,
dummy_eval, model_sizes, and num_features_model utility functions.
"""
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastai.callback.hook import (
    Hook, Hooks, hook_output, hook_outputs,
    has_params, total_params, dummy_eval, model_sizes, num_features_model,
)


# ============================================================
# Helper fixtures
# ============================================================

@pytest.fixture
def linear_layer():
    """A simple linear layer for testing."""
    return nn.Linear(4, 2)


@pytest.fixture
def conv_model():
    """A simple conv model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
    )


@pytest.fixture
def simple_sequential():
    """A sequential model with multiple layers."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


# ============================================================
# Tests for Hook
# ============================================================

class TestHook:
    """Tests for the Hook class."""

    def test_creation_forward(self, linear_layer):
        """Hook can be created as a forward hook."""
        hook = Hook(linear_layer, lambda m, i, o: o, is_forward=True)
        assert hook.removed is False
        assert hook.stored is None
        hook.remove()

    def test_creation_backward(self, linear_layer):
        """Hook can be created as a backward hook."""
        hook = Hook(linear_layer, lambda m, i, o: o, is_forward=False)
        assert hook.removed is False
        hook.remove()

    def test_hook_fn_stores_output(self, linear_layer):
        """Hook function stores the result of hook_func."""
        hook = Hook(linear_layer, lambda m, i, o: o)
        x = torch.randn(2, 4)
        out = linear_layer(x)
        assert hook.stored is not None
        assert hook.stored.shape == (2, 2)
        hook.remove()

    def test_hook_fn_stores_custom_result(self, linear_layer):
        """Hook function can store custom computed values."""
        hook = Hook(linear_layer, lambda m, i, o: o.mean())
        x = torch.randn(2, 4)
        linear_layer(x)
        assert hook.stored is not None
        assert hook.stored.ndim == 0  # scalar
        hook.remove()

    def test_detach_true(self, linear_layer):
        """With detach=True (default), stored tensor has no grad_fn."""
        hook = Hook(linear_layer, lambda m, i, o: o, detach=True)
        x = torch.randn(2, 4, requires_grad=True)
        linear_layer(x)
        assert hook.stored.grad_fn is None
        hook.remove()

    def test_detach_false(self, linear_layer):
        """With detach=False, stored tensor retains grad_fn."""
        hook = Hook(linear_layer, lambda m, i, o: o, detach=False)
        x = torch.randn(2, 4, requires_grad=True)
        linear_layer(x)
        assert hook.stored.grad_fn is not None
        hook.remove()

    def test_cpu_flag(self):
        """With cpu=True, stored tensor is on CPU."""
        layer = nn.Linear(4, 2)
        hook = Hook(layer, lambda m, i, o: o, cpu=True)
        x = torch.randn(2, 4)
        layer(x)
        assert hook.stored.device.type == 'cpu'
        hook.remove()

    def test_remove(self, linear_layer):
        """After remove(), the hook no longer captures."""
        hook = Hook(linear_layer, lambda m, i, o: o)
        x = torch.randn(2, 4)
        linear_layer(x)
        assert hook.stored is not None
        hook.remove()
        assert hook.removed is True
        # Reset stored and forward again
        hook.stored = None
        linear_layer(x)
        assert hook.stored is None  # hook is removed, not called

    def test_remove_idempotent(self, linear_layer):
        """Calling remove() multiple times does not raise."""
        hook = Hook(linear_layer, lambda m, i, o: o)
        hook.remove()
        hook.remove()  # Should not raise
        assert hook.removed is True

    def test_context_manager(self, linear_layer):
        """Hook works as a context manager and removes on exit."""
        with Hook(linear_layer, lambda m, i, o: o) as hook:
            x = torch.randn(2, 4)
            linear_layer(x)
            assert hook.stored is not None
        assert hook.removed is True

    def test_context_manager_no_capture_after_exit(self, linear_layer):
        """After context manager exits, hook does not capture."""
        with Hook(linear_layer, lambda m, i, o: o) as hook:
            pass
        hook.stored = None
        x = torch.randn(2, 4)
        linear_layer(x)
        assert hook.stored is None

    def test_hook_receives_module_input_output(self, linear_layer):
        """Hook function receives the correct module, input, and output."""
        received = {}

        def capture(m, i, o):
            received['module'] = m
            received['output'] = o
            return o

        hook = Hook(linear_layer, capture)
        x = torch.randn(2, 4)
        out = linear_layer(x)
        assert received['module'] is linear_layer
        assert torch.allclose(received['output'], out.detach())
        hook.remove()

    def test_multiple_forward_passes(self, linear_layer):
        """Hook updates stored on each forward pass."""
        hook = Hook(linear_layer, lambda m, i, o: o)
        x1 = torch.randn(2, 4)
        linear_layer(x1)
        stored1 = hook.stored.clone()

        x2 = torch.randn(2, 4)
        linear_layer(x2)
        stored2 = hook.stored.clone()

        # Different inputs should give different stored values
        assert not torch.equal(stored1, stored2)
        hook.remove()


# ============================================================
# Tests for hook_output
# ============================================================

class TestHookOutput:
    """Tests for hook_output convenience function."""

    def test_captures_output(self, linear_layer):
        """hook_output captures the layer output tensor."""
        hook = hook_output(linear_layer)
        x = torch.randn(3, 4)
        out = linear_layer(x)
        assert hook.stored is not None
        assert hook.stored.shape == out.shape
        hook.remove()

    def test_detach_default(self, linear_layer):
        """By default, hook_output detaches stored tensor."""
        hook = hook_output(linear_layer)
        x = torch.randn(3, 4, requires_grad=True)
        linear_layer(x)
        assert hook.stored.grad_fn is None
        hook.remove()

    def test_detach_false(self, linear_layer):
        """With detach=False, stored tensor retains grad_fn."""
        hook = hook_output(linear_layer, detach=False)
        x = torch.randn(3, 4, requires_grad=True)
        linear_layer(x)
        assert hook.stored.grad_fn is not None
        hook.remove()

    def test_cpu_flag(self, linear_layer):
        """With cpu=True, stored tensor is on CPU."""
        hook = hook_output(linear_layer, cpu=True)
        x = torch.randn(3, 4)
        linear_layer(x)
        assert hook.stored.device.type == 'cpu'
        hook.remove()

    def test_context_manager(self, linear_layer):
        """hook_output can be used as a context manager."""
        with hook_output(linear_layer) as hook:
            x = torch.randn(3, 4)
            linear_layer(x)
            assert hook.stored is not None
        assert hook.removed is True

    def test_grad_mode(self, linear_layer):
        """With grad=True, hook is registered as backward hook."""
        hook = hook_output(linear_layer, grad=True)
        # The hook should be created (backward hooks only fire during backward pass)
        assert hook.removed is False
        hook.remove()


# ============================================================
# Tests for Hooks
# ============================================================

class TestHooks:
    """Tests for the Hooks class."""

    def test_creation_multiple_modules(self, simple_sequential):
        """Hooks can be created on multiple modules."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        assert len(hooks) == 3
        hooks.remove()

    def test_len(self, simple_sequential):
        """__len__ returns number of hooks."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        assert len(hooks) == len(modules)
        hooks.remove()

    def test_getitem(self, simple_sequential):
        """__getitem__ returns individual Hook."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        assert isinstance(hooks[0], Hook)
        assert isinstance(hooks[1], Hook)
        assert isinstance(hooks[2], Hook)
        hooks.remove()

    def test_iter(self, simple_sequential):
        """__iter__ allows iterating over hooks."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        hook_list = list(hooks)
        assert len(hook_list) == 3
        assert all(isinstance(h, Hook) for h in hook_list)
        hooks.remove()

    def test_stored_property(self, simple_sequential):
        """stored property returns L of stored values after forward pass."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        x = torch.randn(2, 4)
        simple_sequential(x)
        stored = hooks.stored
        assert len(stored) == 3
        # First layer (Linear 4->8) output shape
        assert stored[0].shape == (2, 8)
        # Second layer (ReLU) same shape
        assert stored[1].shape == (2, 8)
        # Third layer (Linear 8->2) output shape
        assert stored[2].shape == (2, 2)
        hooks.remove()

    def test_stored_before_forward(self, simple_sequential):
        """stored property returns L of None values before any forward pass."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        stored = hooks.stored
        assert len(stored) == 3
        assert all(s is None for s in stored)
        hooks.remove()

    def test_remove(self, simple_sequential):
        """remove() removes all hooks."""
        modules = list(simple_sequential.children())
        hooks = Hooks(modules, lambda m, i, o: o)
        hooks.remove()
        assert all(h.removed for h in hooks)

    def test_context_manager(self, simple_sequential):
        """Hooks works as a context manager."""
        modules = list(simple_sequential.children())
        with Hooks(modules, lambda m, i, o: o) as hooks:
            x = torch.randn(2, 4)
            simple_sequential(x)
            assert len(hooks.stored) == 3
        assert all(h.removed for h in hooks)

    def test_context_manager_no_capture_after_exit(self, simple_sequential):
        """After exiting context manager, hooks do not capture."""
        modules = list(simple_sequential.children())
        with Hooks(modules, lambda m, i, o: o) as hooks:
            pass
        # Reset stored
        for h in hooks:
            h.stored = None
        x = torch.randn(2, 4)
        simple_sequential(x)
        # Hooks removed, should not have captured
        assert all(h.stored is None for h in hooks)

    def test_empty_modules(self):
        """Hooks with empty module list creates no hooks."""
        hooks = Hooks([], lambda m, i, o: o)
        assert len(hooks) == 0
        hooks.remove()


# ============================================================
# Tests for hook_outputs
# ============================================================

class TestHookOutputs:
    """Tests for hook_outputs convenience function."""

    def test_captures_all_outputs(self, simple_sequential):
        """hook_outputs captures output from all modules."""
        modules = list(simple_sequential.children())
        with hook_outputs(modules) as hooks:
            x = torch.randn(2, 4)
            simple_sequential(x)
            assert len(hooks.stored) == 3
            assert all(s is not None for s in hooks.stored)

    def test_stored_shapes(self, simple_sequential):
        """Stored shapes match forward pass dimensions."""
        modules = list(simple_sequential.children())
        with hook_outputs(modules) as hooks:
            x = torch.randn(2, 4)
            simple_sequential(x)
            assert hooks.stored[0].shape == (2, 8)
            assert hooks.stored[1].shape == (2, 8)
            assert hooks.stored[2].shape == (2, 2)

    def test_detach_default(self, simple_sequential):
        """By default, detaches stored tensors."""
        modules = list(simple_sequential.children())
        with hook_outputs(modules) as hooks:
            x = torch.randn(2, 4, requires_grad=True)
            simple_sequential(x)
            for s in hooks.stored:
                assert s.grad_fn is None

    def test_cpu_flag(self, simple_sequential):
        """With cpu=True, stored tensors are on CPU."""
        modules = list(simple_sequential.children())
        with hook_outputs(modules, cpu=True) as hooks:
            x = torch.randn(2, 4)
            simple_sequential(x)
            for s in hooks.stored:
                assert s.device.type == 'cpu'

    def test_grad_mode(self, simple_sequential):
        """With grad=True, hooks are backward hooks."""
        modules = list(simple_sequential.children())
        hooks = hook_outputs(modules, grad=True)
        assert len(hooks) == 3
        hooks.remove()


# ============================================================
# Tests for has_params
# ============================================================

class TestHasParams:
    """Tests for has_params utility."""

    def test_linear_has_params(self):
        """Linear layer has parameters."""
        assert has_params(nn.Linear(4, 2)) is True

    def test_conv2d_has_params(self):
        """Conv2d layer has parameters."""
        assert has_params(nn.Conv2d(3, 8, 3)) is True

    def test_relu_no_params(self):
        """ReLU has no parameters."""
        assert has_params(nn.ReLU()) is False

    def test_dropout_no_params(self):
        """Dropout has no parameters."""
        assert has_params(nn.Dropout()) is False

    def test_batchnorm_has_params(self):
        """BatchNorm has parameters."""
        assert has_params(nn.BatchNorm2d(8)) is True

    def test_sequential_with_params(self):
        """Sequential with parameterized layers has params."""
        model = nn.Sequential(nn.Linear(4, 2))
        assert has_params(model) is True

    def test_empty_sequential_no_params(self):
        """Empty Sequential has no parameters."""
        model = nn.Sequential()
        assert has_params(model) is False


# ============================================================
# Tests for total_params
# ============================================================

class TestTotalParams:
    """Tests for total_params utility."""

    def test_linear_param_count(self):
        """Linear(4, 2) has 4*2 + 2 = 10 params, trainable."""
        layer = nn.Linear(4, 2)
        params, trainable = total_params(layer)
        assert params == 10
        assert trainable is True

    def test_linear_no_bias(self):
        """Linear(4, 2, bias=False) has 4*2 = 8 params."""
        layer = nn.Linear(4, 2, bias=False)
        params, trainable = total_params(layer)
        assert params == 8
        assert trainable is True

    def test_frozen_params(self):
        """Frozen parameters show trainable=False."""
        layer = nn.Linear(4, 2)
        for p in layer.parameters():
            p.requires_grad = False
        params, trainable = total_params(layer)
        assert params == 10
        assert trainable is False

    def test_conv2d_param_count(self):
        """Conv2d(3, 8, 3) has 3*8*3*3 + 8 = 224 params."""
        layer = nn.Conv2d(3, 8, 3)
        params, trainable = total_params(layer)
        assert params == 224
        assert trainable is True

    def test_relu_no_params(self):
        """ReLU has 0 params, trainable=False."""
        layer = nn.ReLU()
        params, trainable = total_params(layer)
        assert params == 0
        assert trainable is False

    def test_sequential_total(self):
        """Sequential reports total across children."""
        model = nn.Sequential(
            nn.Linear(4, 8),  # 4*8 + 8 = 40
            nn.Linear(8, 2),  # 8*2 + 2 = 18
        )
        params, trainable = total_params(model)
        assert params == 58
        assert trainable is True

    def test_mixed_frozen_unfrozen(self):
        """Model with first param frozen returns trainable based on first param."""
        model = nn.Sequential(
            nn.Linear(4, 2),
            nn.Linear(2, 1),
        )
        # Freeze only the first layer
        for p in model[0].parameters():
            p.requires_grad = False
        params, trainable = total_params(model)
        # total_params checks first param's requires_grad
        assert params == 13  # (4*2+2) + (2*1+1) = 10 + 3
        assert trainable is False  # first param is frozen


# ============================================================
# Tests for dummy_eval
# ============================================================

class TestDummyEval:
    """Tests for dummy_eval utility."""

    def test_basic_conv_model(self, conv_model):
        """Evaluates a conv model on default size."""
        result = dummy_eval(conv_model)
        assert isinstance(result, torch.Tensor)
        # Input: (1, 3, 64, 64), with padding=1, same spatial dims
        assert result.shape == (1, 16, 64, 64)

    def test_custom_size(self, conv_model):
        """Evaluates on custom input size."""
        result = dummy_eval(conv_model, size=(32, 32))
        assert result.shape == (1, 16, 32, 32)

    def test_no_grad(self, conv_model):
        """Output should not require grad."""
        result = dummy_eval(conv_model)
        assert result.requires_grad is False

    def test_single_conv(self):
        """Works with a single Conv2d layer."""
        layer = nn.Conv2d(3, 16, 3, padding=1)
        result = dummy_eval(layer, size=(8, 8))
        assert result.shape == (1, 16, 8, 8)

    def test_model_with_pooling(self):
        """Works with a model that includes pooling."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        result = dummy_eval(model, size=(16, 16))
        assert result.shape == (1, 8, 8, 8)


# ============================================================
# Tests for model_sizes
# ============================================================

class TestModelSizes:
    """Tests for model_sizes utility."""

    def test_returns_list_of_shapes(self, conv_model):
        """Returns a list of shapes for each layer."""
        sizes = model_sizes(conv_model, size=(32, 32))
        assert isinstance(sizes, list)
        assert len(sizes) == 4  # Conv, ReLU, Conv, ReLU

    def test_shapes_match_forward(self, conv_model):
        """Shapes match expected dimensions."""
        sizes = model_sizes(conv_model, size=(32, 32))
        # Conv2d(3,8,3,p=1): (1,8,32,32)
        assert sizes[0] == torch.Size([1, 8, 32, 32])
        # ReLU: same
        assert sizes[1] == torch.Size([1, 8, 32, 32])
        # Conv2d(8,16,3,p=1): (1,16,32,32)
        assert sizes[2] == torch.Size([1, 16, 32, 32])
        # ReLU: same
        assert sizes[3] == torch.Size([1, 16, 32, 32])

    def test_default_size(self, conv_model):
        """Default size is (64,64)."""
        sizes = model_sizes(conv_model)
        assert sizes[-1] == torch.Size([1, 16, 64, 64])

    def test_model_with_pooling(self):
        """Captures size changes from pooling layers."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.MaxPool2d(2),
        )
        sizes = model_sizes(model, size=(16, 16))
        assert sizes[0] == torch.Size([1, 8, 16, 16])
        assert sizes[1] == torch.Size([1, 8, 8, 8])

    def test_different_spatial_sizes(self):
        """Works with non-square input sizes."""
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
        )
        sizes = model_sizes(model, size=(32, 64))
        assert sizes[0] == torch.Size([1, 4, 32, 64])


# ============================================================
# Tests for num_features_model
# ============================================================

class TestNumFeaturesModel:
    """Tests for num_features_model utility."""

    def test_basic_conv_model(self, conv_model):
        """Returns number of output channels for conv model."""
        nf = num_features_model(conv_model)
        assert nf == 16

    def test_single_conv(self):
        """Returns output channels for a single Conv2d."""
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1))
        nf = num_features_model(model)
        assert nf == 32

    def test_model_with_stride(self):
        """Works with strided convolutions."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 64, 3, stride=2, padding=1),
        )
        nf = num_features_model(model)
        assert nf == 64

    def test_model_needing_larger_input(self):
        """Handles models that need larger input by doubling size."""
        # A model with many downsampling layers needs larger input
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
        )
        nf = num_features_model(model)
        assert nf == 32
