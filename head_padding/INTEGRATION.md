# Explicit Dinovol integration example

example_villa_integration.patch targets Villa f07d33be6a00d12ace7d6a9465efe17c78ed7b47. It has not been applied to the checkout. It adds the authored helper as pretrained_backbones/sdpa_padding.py and threads one boolean, sdpa_head_padding=False, through the existing model factory and Eva constructors. The patch retains the upstream source's MIT licensing; Villa's LICENSE credits Vesuvius Challenge (2024). Retain that notice when distributing derived code.

The actual Dinovol call is in pretrained_backbones/dinovol_2_eva.py, EvaAttention.forward. The similarly named transformers/eva.py is a different model path. The factory in dinovol_2_builder.py only forwards keys in its defaults; the example adds the new default to both v1 and v2 dictionaries. Routing is:

    model_config["sdpa_head_padding"]
        -> build_dinovol_2_backbone
        -> Eva (or EvaWithChunking, through **kwargs)
        -> EvaBlock
        -> EvaAttention

Example construction after applying the patch:

    model_config = dict(checkpoint_config["model"])
    model_config["sdpa_head_padding"] = True  # actual boolean, opt in explicitly
    backbone = build_dinovol_2_backbone(model_config)
    backbone.load_pretrained_weights(existing_backbone_state)
    # Continue the existing eval/device/dtype/freeze setup unchanged.

No checkpoint tensor or serialized checkpoint needs modification. Missing/false flag uses the original SDPA branch verbatim; training also uses that branch even when the flag is true. Only opt-in eval calls enter the helper, whose inference/dtype/platform guards remain in force. The flag is an ordinary attribute and creates no parameters or state-dict keys. Existing transpose/reshape handles the sliced output's possible noncontiguity.

The guided-ink convenience loader load_frozen_dino_backbone currently takes its model mapping from the checkpoint and does not expose a runtime override. This example makes the option reachable through the model factory; it does not invent an existing CLI option. If integration into that convenience API is desired, add a separate default-false keyword and copy/override its model mapping immediately before the factory call. Do not rewrite the checkpoint or globally replace torch.nn.functional.

## Verification and remaining limits

[Patch receipt](evidence/INTEGRATION_PATCH_RECEIPT.json) records git apply --check and AST validation. [Factory receipt](evidence/INTEGRATION_CPU_RECEIPT.json) records four tiny CPU factory smoke cases (v1/v2, ordinary/chunked). Edited source was compiled in memory with unchanged local dependencies: flag routing, identical initialized state dictionaries, exact native/default/CPU-opt-in attention outputs, and training bypass all passed. The checkout stayed clean and CUDA was not initialized. These checks do not replace the external real-CT/CUDA comparison.

Helper review fixed one portable guard: nested tensors can report strided layout while refusing shape access, so they now delegate before shape inspection. [Guard receipt](evidence/CPU_GUARD_FOLLOWUP.json) records the targeted follow-up after the original 12-test run; the original [CPU receipt](evidence/CPU_TEST_RECEIPT.json) retains the earlier version's hashes. Current frozen helper SHA256 is b87a8ea8e04d22eec144399188d51245e8657ae81ba9081b669791f82f99826d.

No additional mathematical defect was found. Remaining limits are substantive: capability checks are only hints; switching kernels changes rounding; output strides may differ; TorchScript/export behavior has not been validated; the candidate deliberately bypasses compiled execution. The factory smoke does not establish CUDA speed or full-model quality tolerance. Keep the default false until those measured results support use.

To regenerate/check the example without modifying Villa:

    python head_padding/make_integration_example.py --villa /path/to/pinned/villa
    python head_padding/check_integration_example.py --villa /path/to/pinned/villa

Run these from the diagnostic repository root. They write fresh local receipts beside the scripts; the published historical receipts remain under evidence/.
