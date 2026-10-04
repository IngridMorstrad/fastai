"""LoRA / QLoRA - A high-level API for parameter-efficient fine-tuning (PEFT) of LLMs.

Based on the paper "LoRA: Low-Rank Adaptation of Large Language Models"
(Hu et al., 2021, https://arxiv.org/abs/2106.09685). Instead of fine-tuning all
weights of a large model, LoRA freezes the pretrained weights and injects a pair of
trainable low-rank matrices (A, B) into selected `nn.Linear` layers. The update is:

    y = base(x) + scaling * dropout(x @ A^T) @ B^T,   scaling = alpha / rank

A is initialized with a Kaiming-normal distribution and B is initialized to zeros,
so the adapter is a no-op at initialization (training starts from the pretrained
behavior). Only A and B are trained, which makes the number of trainable parameters
orders of magnitude smaller than the full model.

This module also provides a lightweight, dependency-free "QLoRA" approximation: when
`quantize_base=True`, the frozen base weight is stored in a lower-precision dtype
(default `torch.float16`) while the LoRA adapters stay in float32. This mimics the
memory-saving spirit of QLoRA without requiring bitsandbytes; it is a dtype-based
approximation (NOT true 4-bit NF4 quantization) and is documented as such.

Usage:
    import torch.nn as nn
    from fastai.text.lora import apply_lora, mark_only_lora_as_trainable, lora_parameters

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    apply_lora(model, rank=8, alpha=16, dropout=0.05, target_modules=None)
    mark_only_lora_as_trainable(model)
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LoRALinear', 'apply_lora', 'lora_parameters', 'mark_only_lora_as_trainable']


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank adapter.

    The base layer's weight and bias are frozen (`requires_grad=False`); only the
    low-rank matrices A (rank, in_features) and B (out_features, rank) are trainable.
    The forward computes `base(x) + scaling * dropout(x @ A^T) @ B^T` where
    `scaling = alpha / rank`. B is initialized to zeros so the adapter is a no-op at
    initialization and the wrapped layer reproduces the base layer's output exactly.

    Parameters
    ----------
    base_linear : nn.Linear
        The pretrained linear layer to adapt. Its parameters are frozen in place.
    rank : int, default=8
        Rank of the low-rank update. Must be > 0.
    alpha : float, default=16
        LoRA scaling numerator. The effective scaling is `alpha / rank`. Must be >= 0.
    dropout : float, default=0.0
        Dropout probability applied to the input of the low-rank path. Must be in [0, 1).
    quantize_base : bool, default=False
        If True, store the frozen base weight (and bias) in `quantize_dtype` to save
        memory, while keeping the LoRA adapters in float32. This is a dtype-based
        approximation of QLoRA, NOT true 4-bit NF4 quantization (no bitsandbytes).
    quantize_dtype : torch.dtype, default=torch.float16
        Lower-precision dtype used for the base weight when `quantize_base=True`.
    """

    def __init__(self, base_linear, rank=8, alpha=16, dropout=0.0,
                 quantize_base=False, quantize_dtype=torch.float16):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise ValueError(f"base_linear must be an nn.Linear, got {type(base_linear).__name__}")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.quantize_base = quantize_base
        self.quantize_dtype = quantize_dtype
        self.merged = False

        # Freeze the base layer: only the LoRA adapters are trainable.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Optional QLoRA-style memory saving: store the frozen base weight in a
        # lower-precision dtype. Adapters remain in float32 for stable training.
        if self.quantize_base:
            self.base.weight.data = self.base.weight.data.to(self.quantize_dtype)
            if self.base.bias is not None:
                self.base.bias.data = self.base.bias.data.to(self.quantize_dtype)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # A: (rank, in_features), B: (out_features, rank). B starts at zero so the
        # adapter contributes nothing at initialization.
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        # Base path. When the base is quantized, run it in float32 for a correct,
        # numerically comparable result, then the delta is added in float32.
        if self.quantize_base:
            weight = self.base.weight.to(x.dtype)
            bias = self.base.bias.to(x.dtype) if self.base.bias is not None else None
            base_out = F.linear(x, weight, bias)
        else:
            base_out = self.base(x)

        if self.merged:
            # The low-rank delta has already been folded into the base weight.
            return base_out

        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def _delta_weight(self):
        "The low-rank weight delta `scaling * (B @ A)` with shape (out_features, in_features)."
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge(self):
        "Fold the low-rank delta into the base weight for efficient inference."
        if self.merged:
            raise ValueError("LoRALinear is already merged; call unmerge() before merging again.")
        delta = self._delta_weight().to(self.base.weight.dtype)
        self.base.weight.data += delta
        self.merged = True
        return self

    def unmerge(self):
        "Undo a previous merge(), restoring the separate base weight and adapter."
        if not self.merged:
            raise ValueError("LoRALinear is not merged; nothing to unmerge.")
        delta = self._delta_weight().to(self.base.weight.dtype)
        self.base.weight.data -= delta
        self.merged = False
        return self

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, quantize_base={self.quantize_base}")


def _is_lora_param_name(name):
    "True if a parameter name belongs to a LoRA adapter (lora_A / lora_B)."
    return 'lora_A' in name or 'lora_B' in name


def apply_lora(model, rank=8, alpha=16, dropout=0.0, target_modules=None,
               quantize_base=False, quantize_dtype=torch.float16):
    """Replace matching `nn.Linear` submodules of `model` with `LoRALinear` wrappers in place.

    Walks the module tree and swaps each eligible `nn.Linear` for a `LoRALinear` that
    wraps it. After injection, every parameter except the LoRA adapters is frozen
    (see `mark_only_lora_as_trainable`). The model is modified in place and returned.

    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    rank, alpha, dropout, quantize_base, quantize_dtype
        Forwarded to `LoRALinear` (see its docstring).
    target_modules : list[str] | set[str] | None, default=None
        If provided, only `nn.Linear` submodules whose attribute name contains one of
        these substrings are wrapped. If None, all `nn.Linear` submodules are wrapped.

    Returns
    -------
    nn.Module
        The same model instance, modified in place.
    """
    if target_modules is not None:
        target_modules = list(target_modules)

    # Collect replacements first to avoid mutating the tree while iterating it.
    replacements = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if target_modules is not None and not any(t in child_name for t in target_modules):
                continue
            replacements.append((module, child_name, child))

    for parent, child_name, child in replacements:
        wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout,
                             quantize_base=quantize_base, quantize_dtype=quantize_dtype)
        setattr(parent, child_name, wrapped)

    mark_only_lora_as_trainable(model)
    return model


def lora_parameters(model):
    "Yield only the trainable LoRA adapter parameters (lora_A / lora_B) of `model`."
    for name, param in model.named_parameters():
        if _is_lora_param_name(name):
            yield param


def mark_only_lora_as_trainable(model):
    """Freeze all non-LoRA parameters and unfreeze LoRA adapters.

    Sets `requires_grad=False` on every parameter whose name is not a LoRA adapter,
    and `requires_grad=True` on `lora_A` / `lora_B`. Returns the model.
    """
    for name, param in model.named_parameters():
        param.requires_grad = _is_lora_param_name(name)
    return model
