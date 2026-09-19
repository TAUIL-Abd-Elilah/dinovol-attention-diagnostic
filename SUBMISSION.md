# September 2026 Progress Prize submission draft

Contributor: TAUIL Abd Elilah (individual; Discord: Abd Elilah / iluss.)

Contribution: https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic

## What did you contribute?

A reproducible compatibility diagnosis for the official ps8 Dinovol teacher on Windows and an RTX 3090. The code already uses PyTorch SDPA, but a real-CT profile shows that all 192 attention calls use its math backend in the measured environment. The first real model call explains why: this Torch build lacks Flash Attention, and the other fused backends reject the 54-wide heads. All backend enable flags were already true.

The repository includes a fixed public CT input downloader with pixel-hash verification, strict checkpoint loading, profiling and first-call diagnostics, recorded results, and reproduction instructions. It uses PHerc0139 CT with the published Paris4-trained teacher; the test measures execution and does not validate cross-scroll ink accuracy.

## How does this help the project?

It gives contributors a concrete way to distinguish actual backend execution from configuration flags before spending time optimizing Dinovol. It also supplies a measured consumer-GPU baseline for this specific guided-label inference path. This can help reproduce or rule out the same compatibility issue on another installation.

## Evidence and limits

On one fixed 256³ cube, five synchronized runs after two warmups gave a median of 3.2967 seconds and peak allocated CUDA memory of 2.855 GiB. Repeated and profiled outputs were exactly equal. The first-call diagnostic records actual Q/K/V shapes and PyTorch's backend rejection reasons.

This is a diagnostic contribution, not a speedup, a new attention algorithm, an ink-detection improvement, or a recovered-surface result. It does not establish the bottleneck on Linux/H100 or during training. No award amount is claimed. Existing SDPA work and the original model/data authors are credited in the README.

Prepared with AI assistance. Not submitted by the agent; review and paste into the prize form if appropriate.
