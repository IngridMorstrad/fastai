"""Tests for Pad_Chunk in fastai.text.data.

Validates that `Pad_Chunk.__init__` correctly stores the `decode` parameter
via `store_attr` and that `decodes` does not raise `AttributeError`.
"""
import sys
import os
import pytest

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from fastai.text.data import Pad_Chunk, TensorText


class TestPadChunkStoreAttr:
    """Verify that Pad_Chunk stores all constructor parameters."""

    def test_decode_defaults_to_true(self):
        pc = Pad_Chunk()
        assert hasattr(pc, 'decode'), "Pad_Chunk must store 'decode' attribute"
        assert pc.decode is True

    def test_decode_false_stored(self):
        pc = Pad_Chunk(decode=False)
        assert pc.decode is False

    def test_pad_idx_stored(self):
        pc = Pad_Chunk(pad_idx=0)
        assert pc.pad_idx == 0

    def test_pad_first_stored(self):
        pc = Pad_Chunk(pad_first=False)
        assert pc.pad_first is False

    def test_seq_len_stored(self):
        pc = Pad_Chunk(seq_len=128)
        assert pc.seq_len == 128


class TestPadChunkDecodes:
    """Verify that decodes works without raising AttributeError."""

    def test_decodes_with_decode_true_filters_padding(self):
        pc = Pad_Chunk(pad_idx=1, decode=True)
        # Tensor with some pad tokens (1) and real tokens (2, 3, 4)
        x = TensorText(torch.tensor([1, 1, 2, 3, 4, 1]))
        result = pc.decodes(x)
        expected = torch.tensor([2, 3, 4])
        assert torch.equal(result, expected)

    def test_decodes_with_decode_false_returns_unchanged(self):
        pc = Pad_Chunk(pad_idx=1, decode=False)
        x = TensorText(torch.tensor([1, 1, 2, 3, 4, 1]))
        result = pc.decodes(x)
        assert torch.equal(result, x)

    def test_decodes_no_attribute_error(self):
        """Regression: store_attr must include 'decode' to avoid AttributeError."""
        pc = Pad_Chunk(pad_idx=0, pad_first=False, seq_len=64, decode=True)
        x = TensorText(torch.tensor([0, 5, 6, 0]))
        # Must not raise AttributeError
        result = pc.decodes(x)
        expected = torch.tensor([5, 6])
        assert torch.equal(result, expected)
