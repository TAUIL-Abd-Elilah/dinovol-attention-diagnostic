# Opt-in inference SDPA head padding

Standalone candidate only. No Villa source, model, projection, input precision, or backend setting is modified. CUDA performance and model-output validation belong to the separate real-CT benchmark; these files contain only the helper and CPU tests.

At the existing attention call, use an explicit import and opt-in:

    from sdpa_padding import inference_sdpa

    # Inside the consumer's existing inference_mode/no_grad context:
    output = inference_sdpa(q, k, v, dropout_p=0.0,
                            is_causal=False, enabled=True)

The default enabled=False delegates to PyTorch SDPA. No module-level monkeypatch is needed. Preserve every existing attention option when integrating. A consumer that has already scaled Q must retain its original scale argument; this helper does not infer preprocessing.

For the target head width 54, the candidate appends two zero channels to Q/K/V, passes the original scale 54**-0.5, and slices the attention result back to the original V width. Explicit scale values, including zero, are preserved. Q/K/V projection weights, tokens, and tensor dtypes stay unchanged. Unequal Q and V widths use the next shared multiple of eight and restore the original V width.

The implementation delegates unchanged for disabled opt-in, CPU, unsupported dtype/device/layout/shape, noncontiguous last dimension, enabled autograd or requires-grad input, dropout, attention masks, GQA, compilation, and already aligned widths. It also retains the original call if that call already qualifies for any enabled fused backend, avoiding unnecessary padding on platforms with native support. If no enabled fused backend accepts the padded tensors, it uses the original call. Eligibility queries are routing hints; they do not prove which kernel PyTorch selects. The external profiler must establish that. Capability API errors fall back, while allocation and actual SDPA execution errors propagate normally.

Zero extension preserves the attention expression mathematically only when the original scale is retained. Switching the selected kernel can change accumulation and softmax rounding, particularly for fp16/bfloat16. This is not a bitwise-equivalence promise. The returned shape and dtype match SDPA; its stride/contiguity need not match the original backend's output. Use the consumer's existing output transpose/reshape handling.

Head padding is established prior art: the [FlashAttention interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L798-L847) computes the original softmax scale, pads nonmultiples of eight, and slices the result. This helper is an application of that technique through PyTorch SDPA, not a novel attention algorithm. See also [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for scale and kernel numerical behavior. Prior-art source inspected on 20 September 2026; the linked main branch can change.

## CPU verification

    python head_padding/test_sdpa_padding.py

Tests use an independent float64 softmax(QK-transpose * scale)V reference for default, explicit, and zero scale; unequal value widths; causal and masked expressions; and unchanged input tensors. A regression control confirms that accidentally using the padded default scale changes the answer. Masked tests validate the mathematical core only: production masked calls deliberately remain native. Routing tests use metadata fakes or substitute capability checks, never CUDA tensors. They check fallback argument identity, native CPU gradients, preexisting fused eligibility, unsupported capability queries, and propagation of an actual SDPA error.

Passing CPU tests establishes the algebra and conservative routing contract. It does not establish GPU performance, backend selection, full-model tolerance, or improved reading.
