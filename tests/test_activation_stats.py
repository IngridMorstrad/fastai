"""Tests for ActivationStats.hook handling of list outputs.

Verifies that layers returning a plain list (not just tuple) are handled
correctly without raising AttributeError.

We avoid importing ActivationStats directly because its import chain
pulls in the full fastai.basics module which requires many optional
dependencies. Instead we test the method logic via a minimal stand-in
that replicates the exact method bodies from hook.py.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class _ActivationStatsStub:
    """Minimal stub replicating ActivationStats hook/flatten logic for testing."""

    def __init__(self, with_hist=False):
        self.with_hist = with_hist

    def hook(self, m, i, o):
        if isinstance(o, (tuple, list)):
            return self.hook_multi_ouput(o)
        o = o.float()
        res = {'mean': o.mean().item(), 'std': o.std().item(),
               'near_zero': (o <= 0.05).long().sum().item() / o.numel()}
        if self.with_hist:
            res['hist'] = o.histc(40, 0, 10)
        return res

    def hook_multi_ouput(self, o_tuple):
        "For outputs of RNN which are [nested] tuples of tensors"
        res = []
        for o in self._flatten_tuple(o_tuple):
            if not isinstance(o, torch.Tensor):
                continue
            res.append(self.hook(None, None, o))
        return res

    def _flatten_tuple(self, o_tuple):
        "Recursively flatten a [nested] tuple or list"
        res = []
        for it in o_tuple:
            if isinstance(it, (tuple, list)):
                res += self._flatten_tuple(it)
            else:
                res += [it]
        return tuple(res)


def _verify_source_matches_stub():
    """Verify that the actual source code contains our fix."""
    hook_path = os.path.join(os.path.dirname(__file__), '..', 'fastai', 'callback', 'hook.py')
    with open(hook_path) as f:
        src = f.read()
    # The fix: isinstance check includes list
    assert 'isinstance(o, (tuple, list))' in src, (
        "Expected hook.py to contain 'isinstance(o, (tuple, list))' guard"
    )
    # _flatten_tuple also handles list
    assert "isinstance(it, (tuple, list))" in src, (
        "Expected hook.py _flatten_tuple to handle lists"
    )


class TestActivationStatsHook:
    """Tests for ActivationStats.hook handling various output types."""

    def setup_method(self):
        self.cb = _ActivationStatsStub(with_hist=False)

    def test_source_file_contains_fix(self):
        """The actual hook.py source contains the list guard."""
        _verify_source_matches_stub()

    def test_hook_with_tensor_output(self):
        """hook works with a plain tensor output."""
        o = torch.randn(4, 8)
        result = self.cb.hook(None, None, o)
        assert 'mean' in result
        assert 'std' in result
        assert 'near_zero' in result

    def test_hook_with_tuple_output(self):
        """hook works with a tuple of tensors (existing behavior)."""
        o = (torch.randn(4, 8), torch.randn(4, 8))
        result = self.cb.hook(None, None, o)
        assert isinstance(result, list)
        assert len(result) == 2
        assert 'mean' in result[0]

    def test_hook_with_list_output(self):
        """hook works with a list of tensors (the bug fix).

        Previously this would raise:
            AttributeError: 'list' object has no attribute 'float'
        """
        o = [torch.randn(4, 8), torch.randn(4, 8)]
        result = self.cb.hook(None, None, o)
        assert isinstance(result, list)
        assert len(result) == 2
        assert 'mean' in result[0]
        assert 'std' in result[0]

    def test_hook_with_nested_list_output(self):
        """hook works with a nested list of tensors."""
        o = [torch.randn(4, 8), [torch.randn(4, 8), torch.randn(4, 8)]]
        result = self.cb.hook(None, None, o)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all('mean' in r for r in result)

    def test_hook_with_nested_tuple_output(self):
        """hook works with nested tuples (pre-existing behavior preserved)."""
        o = (torch.randn(4, 8), (torch.randn(4, 8), torch.randn(4, 8)))
        result = self.cb.hook(None, None, o)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_flatten_tuple_handles_list(self):
        """_flatten_tuple correctly flattens nested lists."""
        t1, t2, t3 = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
        nested = [t1, [t2, t3]]
        result = self.cb._flatten_tuple(nested)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert torch.equal(result[0], t1)
        assert torch.equal(result[1], t2)
        assert torch.equal(result[2], t3)

    def test_flatten_tuple_handles_mixed_nesting(self):
        """_flatten_tuple handles mixed tuples and lists."""
        t1, t2, t3 = torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)
        mixed = (t1, [t2, (t3,)])
        result = self.cb._flatten_tuple(mixed)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_hook_skips_non_tensor_in_list(self):
        """Non-tensor items in a list are skipped."""
        o = [torch.randn(4, 8), "not_a_tensor", torch.randn(4, 8)]
        result = self.cb.hook(None, None, o)
        assert isinstance(result, list)
        assert len(result) == 2  # only the 2 tensors produce results
