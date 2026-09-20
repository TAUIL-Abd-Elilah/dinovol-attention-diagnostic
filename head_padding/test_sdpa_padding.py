"""CPU-only semantic and routing tests; no CUDA allocation or kernel execution."""
import unittest
from dataclasses import dataclass, replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

import sdpa_padding as sut


@dataclass
class Metadata:
    shape: tuple = (1, 2, 7, 54)
    device: torch.device = torch.device("cuda:0")
    dtype: torch.dtype = torch.bfloat16
    requires_grad: bool = False
    layout: torch.layout = torch.strided
    last_stride: int = 1
    is_nested: bool = False

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dim):
        assert dim == -1
        return self.last_stride


def inputs(value_width=54):
    generator = torch.Generator(device="cpu").manual_seed(20260920)
    return tuple(torch.randn(shape, generator=generator, dtype=torch.float64)
                 for shape in ((2, 3, 7, 54), (2, 3, 9, 54), (2, 3, 9, value_width)))


def reference(q, k, v, scale=None, mask=None, causal=False):
    logits = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5 if scale is None else scale)
    if causal:
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        logits = logits.masked_fill(~keep, float("-inf"))
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
    return logits.softmax(-1) @ v


class PaddingMathTests(unittest.TestCase):
    def test_matches_independent_attention_and_preserves_inputs(self):
        # Includes unequal value width and explicit zero scale (uniform attention).
        for width in (37, 54, 61):
            for scale in (None, 0.0, 0.31):
                with self.subTest(width=width, scale=scale):
                    q, k, v = inputs(width)
                    originals = [x.clone() for x in (q, k, v)]
                    qp, kp, vp, original_scale, value_width = sut._pad_qkv(q, k, v, scale)
                    actual = F.scaled_dot_product_attention(qp, kp, vp, scale=original_scale)[..., :value_width]
                    torch.testing.assert_close(actual, reference(q, k, v, scale), rtol=1e-12, atol=1e-12)
                    self.assertEqual(actual.shape, (2, 3, 7, width))
                    self.assertEqual(actual.dtype, q.dtype)
                    for old, current in zip(originals, (q, k, v)):
                        self.assertTrue(torch.equal(old, current))

    def test_preserves_causal_boolean_and_additive_masks(self):
        q, k, v = inputs()
        keep = torch.ones(7, 9, dtype=torch.bool)
        keep[:, 2::3] = False
        additive = torch.linspace(-3, 1, 63, dtype=torch.float64).reshape(7, 9)
        for mask, causal in ((None, True), (keep, False), (additive, False)):
            with self.subTest(causal=causal, mask_dtype=None if mask is None else mask.dtype):
                qp, kp, vp, scale, width = sut._pad_qkv(q, k, v, None)
                actual = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask,
                                                        is_causal=causal, scale=scale)[..., :width]
                torch.testing.assert_close(actual, reference(q, k, v, mask=mask, causal=causal),
                                           rtol=1e-12, atol=1e-12)

    def test_default_scale_regression_is_detectable(self):
        q, k, v = inputs()
        qp, kp, vp, _, width = sut._pad_qkv(q, k, v, None)
        wrong_scale = F.scaled_dot_product_attention(qp, kp, vp)[..., :width]
        self.assertGreater((wrong_scale - reference(q, k, v)).abs().max().item(), 1e-3)


class DispatchTests(unittest.TestCase):
    def assert_native(self, q, k, v, **options):
        sentinel = object()
        with patch.object(sut.F, "scaled_dot_product_attention", return_value=sentinel) as native, \
             patch.object(sut, "_pad_qkv", side_effect=AssertionError("must not allocate padding")), \
             patch.object(sut, "_enabled_fused_backends", side_effect=AssertionError("must not query CUDA")):
            result = sut.inference_sdpa(q, k, v, **options)
        self.assertIs(result, sentinel)
        native.assert_called_once()
        for received, original in zip(native.call_args.args, (q, k, v)):
            self.assertIs(received, original)
        expected = dict(attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)
        expected.update({k: v for k, v in options.items() if k != "enabled"})
        self.assertEqual(native.call_args.kwargs, expected)

    def test_disabled_is_native_even_for_unknown_objects(self):
        self.assert_native(object(), object(), object())

    def test_unsupported_cases_keep_original_call(self):
        base = Metadata()
        cases = [
            (replace(base, device=torch.device("cpu")), base, base, {}),
            (replace(base, dtype=torch.float32), base, base, {}),
            (replace(base, requires_grad=True), base, base, {}),
            (replace(base, is_nested=True), base, base, {}),
            (replace(base, last_stride=2), base, base, {}),
            (replace(base, shape=(1, 7, 54)), base, base, {}),
            (base, replace(base, device=torch.device("cuda:1")), base, {}),
            (base, replace(base, shape=(1, 3, 7, 54)), base, {}),
            (base, replace(base, shape=(1, 2, 7, 53)), base, {}),
            (base, base, replace(base, shape=(1, 2, 8, 54)), {}),
            (base, base, base, {"dropout_p": 0.1}),
            (base, base, base, {"enable_gqa": True}),
            (base, base, base, {"attn_mask": object(), "scale": 0.0, "is_causal": True}),
            (*[replace(base, shape=(1, 2, 7, 56))] * 3, {}),
        ]
        with torch.inference_mode():
            for q, k, v, options in cases:
                with self.subTest(options=options, q=q, k=k, v=v):
                    self.assert_native(q, k, v, enabled=True, **options)

    def test_grad_enabled_is_native(self):
        with torch.enable_grad():
            self.assert_native(Metadata(), Metadata(), Metadata(), enabled=True)

    def test_compilation_is_native(self):
        with torch.inference_mode(), patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assert_native(Metadata(), Metadata(), Metadata(), enabled=True)

    def test_no_enabled_backend_and_existing_fused_support_avoid_padding(self):
        for checks, supported in (([], False), ([object()], True)):
            with self.subTest(checks=checks), torch.inference_mode(), \
                 patch.object(sut, "_enabled_fused_backends", return_value=checks), \
                 patch.object(sut, "_has_fused_backend", return_value=supported), \
                 patch.object(sut, "_pad_qkv", side_effect=AssertionError("must not pad")), \
                 patch.object(sut.F, "scaled_dot_product_attention", return_value="native"):
                self.assertEqual(sut.inference_sdpa(Metadata(), Metadata(), Metadata(), enabled=True), "native")

    def test_candidate_dispatch_preserves_scale_and_value_width(self):
        # Substitute capability checks only; real CPU SDPA executes the candidate.
        q, k, v = inputs(37)
        with torch.inference_mode(), patch.object(sut, "_eligible", return_value=True), \
             patch.object(sut, "_enabled_fused_backends", return_value=[object()]), \
             patch.object(sut, "_has_fused_backend", side_effect=[False, True]):
            result = sut.inference_sdpa(q, k, v, enabled=True)
        torch.testing.assert_close(result, reference(q, k, v), rtol=1e-12, atol=1e-12)

    def test_unsupported_candidate_or_query_error_uses_original_tensors(self):
        q, k, v = inputs()
        for answers in ([False, False], [False, RuntimeError("unsupported query")], [TypeError("old API")]):
            with self.subTest(answers=answers), torch.inference_mode(), \
                 patch.object(sut, "_eligible", return_value=True), \
                 patch.object(sut, "_enabled_fused_backends", return_value=[object()]), \
                 patch.object(sut, "_has_fused_backend", side_effect=answers), \
                 patch.object(sut.F, "scaled_dot_product_attention", return_value="native") as native:
                self.assertEqual(sut.inference_sdpa(q, k, v, enabled=True, scale=0.0), "native")
                self.assertIs(native.call_args.args[0], q)
                self.assertIs(native.call_args.args[1], k)
                self.assertIs(native.call_args.args[2], v)
                self.assertEqual(native.call_args.kwargs["scale"], 0.0)

    def test_cpu_autograd_remains_native(self):
        q, k, v = [t.requires_grad_() for t in inputs()]
        actual = sut.inference_sdpa(q, k, v, enabled=True)
        native = F.scaled_dot_product_attention(q, k, v)
        actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
        native_grad = torch.autograd.grad(native.square().sum(), (q, k, v))
        self.assertTrue(torch.equal(actual, native))
        for actual_part, native_part in zip(actual_grad, native_grad):
            self.assertTrue(torch.equal(actual_part, native_part))

    def test_actual_sdpa_error_is_not_silently_retried(self):
        q, k, v = inputs()
        with torch.inference_mode(), patch.object(sut, "_eligible", return_value=True), \
             patch.object(sut, "_enabled_fused_backends", return_value=[object()]), \
             patch.object(sut, "_has_fused_backend", side_effect=[False, True]), \
             patch.object(sut.F, "scaled_dot_product_attention", side_effect=RuntimeError("kernel failed")) as native:
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                sut.inference_sdpa(q, k, v, enabled=True)
            native.assert_called_once()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
