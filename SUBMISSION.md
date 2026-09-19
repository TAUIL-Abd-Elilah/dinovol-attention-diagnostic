# What is your contribution?

I published a [reproducible Dinovol attention diagnostic](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic) that identifies why the official ps8 teacher uses PyTorch's math attention backend on my Windows / RTX 3090 setup, despite all fused-backend flags being enabled.

## (1) Which scroll data did you work on for this submission?

I used a fixed 256³ raw-CT crop from **PHerc0139**, at **9.362 µm** resolution, from the public `20250728140407-9.362um-1.2m-113keV-masked.zarr` volume. The model was the published **Paris4-trained ps8 Dinovol teacher, step 352500**, with its reference embedding. This is an execution benchmark on PHerc0139; it does not establish cross-scroll ink-detection accuracy.

## (2) How does it substantially increase the probability of yourself or someone else reading those scrolls or others?

The contribution supports work on making DINO-guided ink experiments more practical on consumer GPUs. It identifies a concrete compatibility issue and provides a measured baseline, allowing contributors to check the backend that actually executes before choosing an optimization.

The reading benefit is currently indirect: I have not yet demonstrated a speedup, improved ink detection or recovered text. A substantial increase in reading probability therefore remains to be established by follow-up optimization and validation.

## (3) What does it enable that was not possible before?

It packages this specific failure case into a reproducible workflow: download and hash-check the exact public inputs, strictly load the checkpoint, profile the production similarity path, and inspect the first real attention call's backend eligibility. Contributors can reproduce or rule out the same issue on their own installation and compare later changes against the recorded baseline.

The underlying PyTorch profiling tools and Villa SDPA implementation already existed. The new contribution is the verified case, input preparation and reproduction harness; it does not introduce a new reading capability or attention algorithm.

## (4) What evidence have you provided for this?

- [Trace-derived operator counts](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/results/operator_counts.json): **192 math-SDPA calls**, covering 24 blocks across eight windows, with the trace hash and extraction rule.
- [Actual model-input diagnostics](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/results/attention.json) and [native warnings](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/results/warnings.txt): bf16 Q/K/V shape `[1,16,4101,54]`; Flash Attention unavailable in this build; memory-efficient and cuDNN attention reject the 54-wide heads.
- [Recorded measurements](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/results/baseline.json): two warmups and five synchronized passes; **3.2967 seconds median** per 256³ cube and **2.855 GiB peak allocated CUDA memory**. Repeated and profiled outputs were exactly equal.
- [Reproduction instructions](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/README.md), public input downloader, pinned model/source identities, and [portable-package validation](https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/5409507e8eaaadb92e728f1ef2da47392e3b4404/results/packaging_validation.json), including exact pixel identity and strict loading of all 463 checkpoint state keys.

These results cover one input and one environment. Loading, normalization and output-copy time are excluded from the timing; training and Linux/H100 performance were not measured. Original model, data and SDPA authors are credited in the README.

Contributor: TAUIL Abd Elilah (individual; Discord: Abd Elilah / iluss.). Prepared with AI assistance. This Markdown is ready for review and has not been submitted to the prize form.
