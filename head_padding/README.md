# Opt-in Dinovol attention padding

An explicit inference option avoids the official ps8 teacher's head-width-54 math-SDPA fallback on the tested **Windows / RTX 3090 / Torch 2.13.0+cu126** setup. It pads Q/K/V from 54 to 56 channels, retains the original attention scale, and removes the extra output channels. PyTorch then executes its existing memory-efficient attention kernel.

**The complete frozen-DINO similarity calculation was 4.71–5.01× faster on three real CT cubes, with 76.4% lower peak allocated GPU memory. Outputs are not bitwise equal:** 0.467–0.985% of voxels changed their similarity >0.5 eligibility. This option is disabled by default; production pseudo-label equivalence and reading accuracy have not been established.

## Complete-similarity measurements

| PHerc0139 cube | Native median / p90 | Helper median / p90 | Median speedup | Eligibility flips |
|---|---:|---:|---:|---:|
| A, original diagnostic | 3.837 / 3.850 s | 0.765 / 0.771 s | 5.014× | 165,329 / 16,777,216 (0.9854%) |
| B, Z4352 | 3.725 / 3.729 s | 0.788 / 0.792 s | 4.726× | 78,355 / 16,777,216 (0.4670%) |
| C, Z4608 | 3.712 / 3.728 s | 0.788 / 0.791 s | 4.708× | 114,145 / 16,777,216 (0.6804%) |

Every profiled native arm used **192 math** attention calls; every helper arm used **192 efficient** attention calls. No FlashAttention or cuDNN attention executed. Peak allocated GPU memory was 3,065,467,904 → 722,444,800 bytes (2.855 → 0.673 GiB) on each cube. Reserved memory, individual timings and operator counts are in the evidence.

These are five synchronized calls after two warmups in fresh processes, with a separate profiler pass. The fixed order alternated helper/native order across cubes. The measured call includes all 24 blocks, eight 128³ windows, Gaussian blending, interpolation and final minmax normalization. Padding and dispatch checks are included. Startup, CPU normalization and transfers, U-Net prediction, dataset sampling and student training are excluded. This is not whole-pipeline wall time.

All inputs are disjoint 256³ uint8 cubes of PHerc0139, scan `20250728140407-9.362um-1.2m-113keV-masked.zarr`, at 9.362 µm. B and C share one face, in a different region from A. They are a local replication on one scroll, not three independent scrolls. Input coordinates were selected before inference results; all additional cached pixels were checked against the official S3 chunks. See [input selection](ADDITIONAL_INPUT_PLAN.md).

The same official Paris4-trained step-352500 ps8 teacher and 864-dimensional reference were used throughout. Model weights, bf16 precision, stride128, window128, minibatch1 and preprocessing were held fixed. Minibatch1 is an explicit memory cap, different from the historical minibatch16 recipe. Full identities and environment are recorded in each run report.

## Numerical consequences

| Cube | Final-map maximum absolute difference | Mean absolute difference | Newly eligible | Newly ineligible | Native eligible |
|---|---:|---:|---:|---:|---:|
| A | 0.0140121 | 0.00217922 | 164,673 | 656 | 9,126,848 |
| B | 0.00859299 | 0.00099688 | 11,927 | 66,428 | 8,344,327 |
| C | 0.01004294 | 0.00145637 | 111,102 | 3,043 | 6,727,566 |

“Eligible” here means **similarity >0.5**, not an ink prediction. Actual v1/v2 pseudo-labels threshold the product of similarity and phase-specific frozen U-Net probabilities; v2 additionally requires raw intensity >50. The exact archived 60k/63k teachers could not be located publicly or locally. Later checkpoints are not interchangeable. No production-label equivalence or downstream ink-quality claim follows from these numbers.

The mechanism control on A used the same zero-padding but forced math attention: every saved token, blended grid, pre-minmax map and final map was **exactly equal**, with zero eligibility flips and no speed gain. This isolates the kernel change as the numerical/performance difference in this fixture. The opt-in helper reproduced all 11 saved arrays of the original padded experiment byte-for-byte. Final minmax normalization can magnify small changes to the input range, so token cosine alone is insufficient validation.

## Use and reproduce

[sdpa_padding.py](sdpa_padding.py) provides `inference_sdpa(..., enabled=True)`; its default is `False`. It preserves native calls for unsupported inputs, autograd, dropout, masks, GQA, compilation, and when an enabled fused kernel already accepts the original dimensions. Capability checks are only hints; actual execution is profiled. See [helper contract](HELPER.md).

For explicit model integration, use the [example patch](example_villa_integration.patch) and [integration instructions](INTEGRATION.md), targeting Villa `f07d33be6a00d12ace7d6a9465efe17c78ed7b47`. The new model-factory flag defaults false and training retains the native branch. The patch passed apply/AST checks and four CPU factory smoke cases; the real-CT benchmark tests the helper through an isolated process interception. The current guided convenience loader does not expose this new runtime flag. No checkpoint must be rewritten and no global monkeypatch belongs in deployment.

Use the existing working Villa environment and pinned checkout described in the [parent README](../README.md); no packages are installed by these scripts. From the repository root:

```text
python fetch_inputs.py --output inputs --include-checkpoint
python head_padding/fetch_additional_inputs.py --output inputs/additional
python head_padding/test_sdpa_padding.py
```

Omit the first command's checkpoint flag to reuse the verified 864 MB teacher. The additional fetcher downloads exactly 32 MiB of CT plus small metadata and verifies fixed hashes. Weights, CT, large outputs and profiler traces are not bundled.

Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1` and `AGENTS_AGENT_MODE=1`. Run one GPU process at a time:

```text
python head_padding/profile_helper.py --baseline-dir . --helper-dir head_padding --vesuvius-src /path/to/pinned/villa/vesuvius/src --checkpoint inputs/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt --reference inputs/avg_ref_embedding.npy --volume inputs/pherc0139_ct.npy --mode helper --output outputs/A_helper
python head_padding/profile_helper.py --baseline-dir . --helper-dir head_padding --vesuvius-src /path/to/pinned/villa/vesuvius/src --checkpoint inputs/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt --reference inputs/avg_ref_embedding.npy --volume inputs/pherc0139_ct.npy --mode unchanged --output outputs/A_native
python head_padding/compare_runs.py --baseline outputs/A_native --candidate outputs/A_helper --output outputs/comparison_A.json
```

For B/C, replace `--volume` with `inputs/additional/pherc0139_z4352.npy` / `pherc0139_z4608.npy` and supply its corresponding `.json` through `--manifest`. Use native→helper for B and helper→native for C, retaining every result. Output directories must be new. Each report captures exact inputs, helper/source hashes, timings and backend evidence. Captured token/map transfers occur only in the separate profile pass, not timed calls.

## Evidence and scope

- [Compact numerical summary](evidence/SUMMARY.json) and [verified additional input receipts](evidence/fixtures/fetch_receipt.json).
- [A comparison](evidence/comparison_replica_A.json), [B comparison](evidence/comparison_replica_B.json), [C comparison](evidence/comparison_replica_C.json), including all stage errors, threshold margins and per-token cosines.
- [Original raw-padding and forced-math control](evidence/comparison_full_01.json), [helper identity](evidence/HELPER_IDENTITY.json).
- [Initial full-comparison plan](FULL_COMPARISON_PLAN.md), [replication plan](HELPER_REPLICATION_PLAN.md), [CPU tests](evidence/CPU_TEST_RECEIPT.json), [nested-input guard follow-up](evidence/CPU_GUARD_FOLLOWUP.json), [factory integration check](evidence/INTEGRATION_CPU_RECEIPT.json).

### Separate precision follow-up

After seeing the bf16 differences, a [separately declared experiment](PRECISION_FOLLOWUP_PLAN.md) cast Q/K/V to FP32 only inside efficient attention and cast the result back to bf16. On **cube A only**, it reduced median time from 3.733 to 2.164 s (1.725×) and eligibility flips to 71,709 (0.4274%). Mean final-map error fell to 0.000959, but maximum error **increased** to 0.01516. It passed its development screen but is not uniformly more accurate, bitwise equal, or validated on the other cubes. It is a research alternative, not an option in the published helper.

The [full comparison](evidence/comparison_precision_A.json), [gate receipt](evidence/comparison_precision_A_gate.json) and [first-call probe](evidence/probe_fp32.json) retain this mixed result. The probe's equality fraction is an approximate float32 reduction; the full-map flip counts use exact CPU integer counts. All per-call casting is included in timing. Efficient attention was explicitly forced in this diagnostic only.

To reproduce the full follow-up on A, invoke `profile_precision.py` with the same baseline/checkpoint/reference/volume/source/output arguments as `profile_helper.py`, omit `--helper-dir`, and use `--mode pad56_fp32`; then run `--mode unchanged` into a new directory. Compare the paired directories with `compare_runs.py`. The order, screen and scope are in the linked plan. This follow-up does not change the three-cube bf16 helper result or establish production-label equivalence.

Public JSON copies redact workspace prefixes only and retain their unredacted original hashes. The shipped helper and profiling drivers match their measured hashes, as does the comparator used for A/B/C. The initial `comparison_full_01.json` records an earlier comparator revision, before support for additional input manifests; that historical comparator snapshot is not bundled. The current comparator reproduces the same stage/eligibility analysis and validates both input schemas. Timings are from this machine, not a universal performance guarantee. The initial raw-padding run measured 3.739 → 0.787 s (4.75×), separately from the later paired table above.

Head padding with preserved scale and output slicing is established [FlashAttention prior art](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py). Villa already uses SDPA. The contribution is this Dinovol fallback diagnosis, guarded application, reproducible full-consumer benchmark and disclosed numerical consequences. A public-code audit found no corresponding Dinovol padding in the inspected current Villa sources; unpublished work cannot be excluded. This does not claim a new attention algorithm, upstream adoption, new readable area or recovered text.

By TAUIL Abd Elilah with AI assistance. Credit to ScrollPrize/Vesuvius Challenge for Villa, models and public CT, and PyTorch for the kernels. Authored helper/harness code is covered by the repository MIT license; preserve [Villa's MIT notice](LICENSE_VILLA) with the integration patch. Original assets retain their terms.
