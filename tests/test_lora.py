"""Tests for the LoRA / QLoRA parameter-efficient fine-tuning API."""

import pytest
import torch
import torch.nn as nn

from fastai.text.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    mark_only_lora_as_trainable,
)


class TestLoRALinear:
    """Unit tests for LoRALinear and the high-level LoRA API."""

    def test_invalid_rank_raises(self):
        """Non-positive rank raises ValueError."""
        base = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="rank must be a positive integer"):
            LoRALinear(base, rank=0)
        with pytest.raises(ValueError, match="rank must be a positive integer"):
            LoRALinear(base, rank=-3)

    def test_invalid_alpha_raises(self):
        """Negative alpha raises ValueError."""
        base = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            LoRALinear(base, alpha=-1)

    def test_invalid_dropout_raises(self):
        """Dropout outside [0, 1) raises ValueError."""
        base = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRALinear(base, dropout=1.0)
        with pytest.raises(ValueError, match="dropout must be in"):
            LoRALinear(base, dropout=-0.1)

    def test_non_linear_base_raises(self):
        """A non-Linear base layer raises ValueError."""
        with pytest.raises(ValueError, match="base_linear must be an nn.Linear"):
            LoRALinear(nn.ReLU())

    def test_init_output_equals_base(self):
        """At initialization (B is zero), output equals the base layer's output."""
        torch.manual_seed(0)
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4, alpha=8)
        x = torch.randn(5, 16)
        assert torch.allclose(lora(x), base(x), atol=1e-6)

    def test_nonzero_adapter_changes_output(self):
        """With nonzero A and B, output differs from the base layer."""
        torch.manual_seed(0)
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4, alpha=8)
        base_out = base(torch.ones(1, 16))  # compute before mutating A/B
        with torch.no_grad():
            lora.lora_A.copy_(torch.randn_like(lora.lora_A))
            lora.lora_B.copy_(torch.randn_like(lora.lora_B))
        x = torch.ones(1, 16)
        assert not torch.allclose(lora(x), base_out, atol=1e-4)

    def test_requires_grad_flags(self):
        """Base params are frozen; LoRA A/B are trainable."""
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4)
        assert lora.base.weight.requires_grad is False
        assert lora.base.bias.requires_grad is False
        assert lora.lora_A.requires_grad is True
        assert lora.lora_B.requires_grad is True

    def test_scaling_formula(self):
        """scaling == alpha / rank."""
        base = nn.Linear(8, 4)
        lora = LoRALinear(base, rank=8, alpha=16)
        assert lora.scaling == 16 / 8

    def test_apply_lora_replaces_linears(self):
        """apply_lora wraps all nn.Linear modules in a Sequential."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        apply_lora(model, rank=4, alpha=8)
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)
        assert isinstance(model[1], nn.ReLU)

    def test_apply_lora_marks_only_lora_trainable(self):
        """After apply_lora + mark_only_lora_as_trainable, only adapters are trainable."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        apply_lora(model, rank=4, alpha=8)
        mark_only_lora_as_trainable(model)
        for name, p in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                assert p.requires_grad is True, f"{name} should be trainable"
            else:
                assert p.requires_grad is False, f"{name} should be frozen"

    def test_target_modules_filtering(self):
        """target_modules only wraps submodules whose attribute name matches."""

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(16, 16)
                self.mlp = nn.Linear(16, 16)

        model = Block()
        apply_lora(model, rank=4, target_modules=["attn"])
        assert isinstance(model.attn, LoRALinear)
        assert isinstance(model.mlp, nn.Linear)
        assert not isinstance(model.mlp, LoRALinear)

    def test_lora_parameters_yields_only_adapters(self):
        """lora_parameters yields exactly the adapter params."""
        model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
        apply_lora(model, rank=4)
        params = list(lora_parameters(model))
        # 2 wrapped linears x (lora_A + lora_B) = 4 params
        assert len(params) == 4
        for p in params:
            assert p.requires_grad is True

    def test_merge_matches_unmerged(self):
        """merge() then forward matches the unmerged forward within tolerance."""
        torch.manual_seed(1)
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4, alpha=8, dropout=0.0)
        with torch.no_grad():
            lora.lora_A.copy_(torch.randn_like(lora.lora_A))
            lora.lora_B.copy_(torch.randn_like(lora.lora_B))
        lora.eval()
        x = torch.randn(5, 16)
        unmerged = lora(x)
        lora.merge()
        merged = lora(x)
        assert lora.merged is True
        assert torch.allclose(unmerged, merged, atol=1e-5)

    def test_double_merge_guarded(self):
        """Merging twice without unmerge raises ValueError."""
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4)
        lora.merge()
        with pytest.raises(ValueError, match="already merged"):
            lora.merge()

    def test_unmerge_restores(self):
        """unmerge() restores the original base weight; unmerge without merge raises."""
        torch.manual_seed(2)
        base = nn.Linear(16, 8)
        lora = LoRALinear(base, rank=4, alpha=8)
        with torch.no_grad():
            lora.lora_A.copy_(torch.randn_like(lora.lora_A))
            lora.lora_B.copy_(torch.randn_like(lora.lora_B))
        original_weight = lora.base.weight.data.clone()
        lora.merge()
        lora.unmerge()
        assert lora.merged is False
        assert torch.allclose(lora.base.weight.data, original_weight, atol=1e-5)
        with pytest.raises(ValueError, match="not merged"):
            lora.unmerge()

    def test_quantize_base_dtype_and_forward(self):
        """quantize_base stores base in float16, adapters stay float32, forward matches base."""
        torch.manual_seed(3)
        base = nn.Linear(16, 8)
        base_fp32 = nn.Linear(16, 8)
        base_fp32.load_state_dict(base.state_dict())
        lora = LoRALinear(base, rank=4, alpha=8, quantize_base=True)
        assert lora.base.weight.dtype == torch.float16
        assert lora.base.bias.dtype == torch.float16
        assert lora.lora_A.dtype == torch.float32
        assert lora.lora_B.dtype == torch.float32
        x = torch.randn(5, 16)
        # At init (B is zero) the output should match the fp32 base within fp16 tolerance.
        assert torch.allclose(lora(x), base_fp32(x), atol=1e-2)

    def test_param_count_is_small(self):
        """Trainable (LoRA) params are far fewer than total params - the point of PEFT."""
        model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        apply_lora(model, rank=8, alpha=16)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total
        # LoRA params should be well under 10% of the full model here.
        assert trainable < 0.1 * total
