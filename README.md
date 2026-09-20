# Dinovol attention compatibility diagnostic

**20 September update:** this project now includes a measured, explicitly opt-in [head-padding implementation](head_padding/README.md). Across three real PHerc0139 cubes on Windows/RTX 3090, complete frozen-DINO similarity was **4.71–5.01× faster**, with **76.4% lower peak allocated GPU memory**. Outputs differ: **0.467–0.985%** of voxels changed their similarity >0.5 eligibility decision. Production labels and reading accuracy remain untested. [Plain-text submission update](SUBMISSION.txt).

The original diagnostic and its reproducible baseline remain below. Its historical timings are separate from the new paired comparison.

The official ps8 Dinovol teacher uses the **math SDPA backend** on the tested Windows / RTX 3090 environment, even though all fused-backend enable flags are true. This report records why, with a reproducible real-CT input and the unchanged Villa model.

The original stage was a diagnostic contribution with no optimization measured at that time. Neither stage establishes a bottleneck on Linux/H100, whole-training performance, ink accuracy, or recovered text.

## Findings

The first real attention call has Q/K/V shape `[1, 16, 4101, 54]`, dtype bf16. PyTorch's eligibility diagnostics report:

- Flash Attention was not compiled into this installed Torch build.
- Memory-efficient attention rejects head dimension 54 because it requires a multiple of 8.
- cuDNN attention also rejects the head dimension.
- All three backend enable flags were already true.

The complete similarity trace contains **192 math-SDPA calls**: 24 transformer blocks × 8 windows. The [recorded operator counts](results/operator_counts.json) include the trace hash and extraction rule. Simply checking whether the code calls `scaled_dot_product_attention`, or whether its enable flags are true, does not identify the backend that actually executes. No backend was forced for this experiment.

| Measurement | Recorded result |
|---|---|
| Data | One PHerc0139 256³ raw-CT cube, 9.362 µm |
| Workload | Frozen DINO similarity only; excludes the U-Net and training |
| Model | Official step-352500 ps8 teacher; 215,859,168 parameters |
| Settings | bf16, inference mode, stride 128, window 128³, minibatch 1 |
| Timing | Two warmups, five synchronized passes |
| Median / p90 | **3.2967 / 3.2992 seconds** per 256³ cube |
| Peak allocated / reserved CUDA memory | **2.855 / 2.924 GiB** |
| Repeatability | Five timed outputs and the profiled output exactly equal |

Startup, normalization, output-to-CPU copies and saving are excluded from the timing. Minibatch 1 is an explicit memory cap, different from the historical recipe's minibatch 16. One cube and one machine do not establish general throughput. Similarity output in [0,1] is not evidence that ink is present.

## Reproduce

The measured environment was Windows 11, RTX 3090, Python 3.14.6, Torch 2.13.0+cu126, timm 1.0.22, NumPy 2.4.6, einops 0.8.2, Zarr 3.2.1 and tifffile 2026.7.14. Full recorded versions are in [results/baseline.json](results/baseline.json). Backend availability can differ with the platform and build; report those differences rather than expecting this outcome everywhere.

Use a working [Villa Vesuvius environment](https://github.com/ScrollPrize/villa/tree/f07d33be6a00d12ace7d6a9465efe17c78ed7b47/vesuvius), with the source checkout pinned to `f07d33be6a00d12ace7d6a9465efe17c78ed7b47`. The fetcher additionally needs `requests` and NumPy. These scripts do not install packages or alter Villa. Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1` and `AGENTS_AGENT_MODE=1` in your shell.

```text
python fetch_inputs.py --output inputs --include-checkpoint
```

This downloads 12 uncompressed CT chunks (24 MiB), keeps the fixed 16 MiB pixel crop, fetches the 3.5 KiB reference embedding, and optionally downloads the 863,606,099-byte slim teacher. Omit `--include-checkpoint` to use an existing verified teacher. SHA-256 checks enforce all input identities. Weights and CT are fetched from their original publishers and are not included in this repository.

First verify the inputs and strict loading without CUDA:

```text
python profile_existing.py --vesuvius-src /path/to/villa/vesuvius/src --checkpoint inputs/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt --reference inputs/avg_ref_embedding.npy --volume inputs/pherc0139_ct.npy --output outputs/cpu --check-only
```

To measure, use the same command with a new output directory and omit `--check-only`. It calls the existing production `_dino_similarity` method, performs two warmups and five timed passes, then writes a profiler trace and output checks. The CUDA allocator is capped at 85%; this does not cap all other GPU consumers. Use an otherwise idle GPU.

Extract the backend call counts with `python summarize_trace.py outputs/run/trace.json --output outputs/run/operator_counts.json`, replacing `outputs/run` with your measured output directory. The profile refuses modified source under the supplied `vesuvius/src` tree, and checks the fixed reference embedding hash as well as the CT and checkpoint hashes.

For the first-call eligibility diagnostic:

```text
python diagnose_attention.py --vesuvius-src /path/to/villa/vesuvius/src --checkpoint inputs/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt --reference inputs/avg_ref_embedding.npy --volume inputs/pherc0139_ct.npy --output outputs/diagnosis 2> diagnosis-native.log
```

This intercepts the **first actual model attention call**, records its tensors and backend eligibility, and stops before attention executes. Native C++ warnings appear on stderr, so retain `diagnosis-native.log` as well as the JSON. It is not a synthetic tensor microbenchmark.

The public scripts use explicit input paths and a fixed `.npy` CT crop. The original run assembled the same pixels from eight cached TIFFs. The exported records retain numerical fields and original report hashes while omitting local filesystem paths and unrelated training configuration. Public packaging was checked against the exact pixel identity and strict checkpoint load; the recorded timing is from the original run.

## Evidence and provenance

- [Recorded baseline](results/baseline.json), [first-call diagnosis](results/attention.json), [native warnings](results/warnings.txt), [kernel durations](results/kernel_breakdown.json), [portable-package validation](results/packaging_validation.json).
- [Official Dinovol teacher](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500/tree/6a8cccbafef191a966da815e22ff5c6eae075aae), SHA-256 `e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59`.
- [Reference embedding source](https://huggingface.co/scrollprize/ink_3d_dino_guided/tree/73a79525466037432191284dfa237baf830c49ec), SHA-256 `61bdf93bc5e3fd956eebdbed52618985d27264b8a9e5cb043087d0f234507a81`.
- CT source and bbox are encoded in `fetch_inputs.py`. Pixel-array SHA-256: `fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5`.
- The kernel summary sums actual CUDA kernel events. Do not add the profiler's nested operator ranges and their kernels together.

Villa already implements SDPA, fused QKV and AMP. This report claims neither invention nor implementation of those techniques. [PR 775](https://github.com/ScrollPrize/villa/pull/775) is additional prior art for SDPA adoption elsewhere in Villa. Tierval had volunteered for Dinovol performance work; coordination was requested before starting an optimization. This report does not claim an exclusive work area.

Credit: ScrollPrize / Vesuvius Challenge for Villa, pretrained models and open CT data; PyTorch for the existing backend diagnostics. Report and reproduction harness by TAUIL Abd Elilah, with AI assistance. Original dependencies and downloaded assets retain their own licenses and terms.
