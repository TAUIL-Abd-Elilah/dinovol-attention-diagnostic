Discord display name:
Abd Elilah

URL of your open-source / publicly available contribution:
https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/tree/main/head_padding
https://github.com/TAUIL-Abd-Elilah/dinovol-attention-diagnostic/blob/main/head_padding/evidence/SUMMARY.json

What is your contribution?

I turned my earlier Dinovol attention diagnostic into a tested, opt-in optimization: 4.71–5.01 times faster frozen-DINO similarity calculation, with 76.4% lower peak allocated GPU memory on my Windows / RTX 3090 setup. This is a September update to my existing Dinovol entry.

(1) Which scroll data did you work on?

Three fixed, non-overlapping 256 x 256 x 256 raw CT cubes from PHerc0139, using the public 9.362-micrometre volume 20250728140407-9.362um-1.2m-113keV-masked.zarr. The two additional cubes were selected by coordinates before inference and verified pixel-for-pixel against the official source. I used the official Paris4-trained Dinovol v2 ps8 teacher, step 352500, and official reference embedding. All input, model and source hashes are published.

(2) How does this increase the probability of reading these or other scrolls?

DINO similarity is a component of the DINO-guided ink workflow. On the three tested cubes, its median runtime fell from 3.71–3.84 seconds to 0.765–0.788 seconds, and peak allocated GPU memory fell from 2.855 GiB to 0.673 GiB. Reducing this cost lets contributors run more similarity experiments with existing consumer hardware. The demonstrated benefit is faster, less memory-intensive computation; improved ink accuracy or recovered text has not yet been demonstrated.

(3) What does it enable that was not possible before?

The unmodified teacher fell back to math attention despite fused backends being enabled. The new helper enables PyTorch's existing efficient-attention kernel on this tested setup by padding 54-channel heads to 56, preserving the original attention scale and restoring the original output width. It is disabled by default, retains native behavior outside supported cases, and includes an explicit Villa integration example. The contribution is a tested Dinovol application of established head-padding techniques, not a new attention algorithm.

(4) What evidence have you provided?

The public repository includes runnable code, reproducible commands, protocols fixed before the corresponding experiments, verified input hashes, CPU tests, integration checks and three complete frozen-DINO similarity comparisons. Each arm used two warmups and five synchronized timed runs; separate profiler traces confirm 192 math-attention calls replaced by 192 efficient-attention calls. A forced-math padding control on cube A produced exactly equal outputs, isolating the backend change. Timing includes padding and dispatch but excludes startup, CPU normalization/transfers, U-Net prediction and training.

The numerical tradeoff is measured: 0.467–0.985% of voxels change their similarity >0.5 eligibility decision. These decisions are not the final ink pseudo-labels. Production-label equivalence and reading accuracy remain untested, and the results cover one scroll and one hardware/software environment. The helper therefore remains experimental and opt-in.

Developed and checked with AI assistance; code, measurements and limitations are public.
