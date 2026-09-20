"""Opt-in, eager inference SDPA head padding; no model or backend mutation."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["inference_sdpa"]


def _eligible(query, key, value, attn_mask, dropout_p, enable_gqa):
    # Deliberately narrow: existing native behavior handles every other case.
    if torch.is_grad_enabled() or dropout_p != 0.0 or enable_gqa or attn_mask is not None:
        return False
    if torch.compiler.is_compiling():
        return False
    tensors = (query, key, value)
    if any(t.device.type != "cuda" or t.dtype not in (torch.float16, torch.bfloat16)
           or t.requires_grad or t.layout != torch.strided or getattr(t, "is_nested", False) or t.ndim != 4
           for t in tensors):
        return False
    if any(t.device != query.device or t.dtype != query.dtype
           or t.stride(-1) != 1 or min(t.shape) <= 0 for t in tensors):
        return False
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        return False
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        return False
    return query.shape[-1] % 8 != 0 or value.shape[-1] % 8 != 0


def _enabled_fused_backends():
    backend = torch.backends.cuda
    return [check for enabled, check in (
        (backend.flash_sdp_enabled, backend.can_use_flash_attention),
        (backend.mem_efficient_sdp_enabled, backend.can_use_efficient_attention),
        (backend.cudnn_sdp_enabled, backend.can_use_cudnn_attention),
    ) if enabled()]


def _has_fused_backend(query, key, value, is_causal, checks):
    params = torch.backends.cuda.SDPAParams(query, key, value, None, 0.0, is_causal, False)
    return any(check(params, debug=False) for check in checks)


def _pad_qkv(query: Tensor, key: Tensor, value: Tensor, scale: float | None):
    """Pure zero extension; retain the original query scale and value width."""
    width = ((max(query.shape[-1], value.shape[-1]) + 7) // 8) * 8
    padded = tuple(F.pad(t, (0, width - t.shape[-1])) for t in (query, key, value))
    return (*padded, query.shape[-1] ** -0.5 if scale is None else scale, value.shape[-1])


def inference_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    enabled: bool = False,
) -> Tensor:
    """Call SDPA unchanged unless inference head padding enables a fused path.

    Opt in at the consumer attention call with enabled=True, inside no_grad or
    inference_mode. The candidate keeps Q/K logits' original scale and removes
    only zero-extended value channels. Fused kernels can differ numerically.
    Masks, dropout, GQA, autograd, compilation and unsupported inputs delegate
    to native SDPA. Existing fused eligibility also retains the native call.
    Eligibility is a hint: PyTorch still selects the backend; nothing is forced.
    """
    native_options = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if enabled and _eligible(query, key, value, attn_mask, dropout_p, enable_gqa):
        try:
            checks = _enabled_fused_backends()
            already_supported = not checks or _has_fused_backend(query, key, value, is_causal, checks)
        except (AttributeError, TypeError, RuntimeError):
            # Capability API missing/unsupported: preserve the original call.
            already_supported = True
        if not already_supported:
            q, k, v, original_scale, value_width = _pad_qkv(query, key, value, scale)
            try:
                candidate_supported = _has_fused_backend(q, k, v, is_causal, checks)
            except (AttributeError, TypeError, RuntimeError):
                candidate_supported = False
            if candidate_supported:
                # Allocation and actual SDPA execution errors are not swallowed.
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal,
                    scale=original_scale, enable_gqa=False,
                )[..., :value_width]
    return F.scaled_dot_product_attention(query, key, value, **native_options)
