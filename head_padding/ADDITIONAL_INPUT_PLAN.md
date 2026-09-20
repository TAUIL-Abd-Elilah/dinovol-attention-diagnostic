# Frozen additional real CT inputs

Preselection date: 2026-09-20. This plan was written before opening any head-padding model results or inspecting CT image quality. These inputs test numerical replication, not ink-reading quality.

## Deterministic selection

Use the existing raw PHerc0139 grid `data/PHerc0139-4x5x5`, whose manifest declares L0 box `[4352,4864,3072,3712,2560,3200]`. Anchor at its minimum Z/Y/X and take the first two consecutive, non-overlapping 256-voxel blocks along Z, retaining the minimum Y and X. No intensity, similarity, label, or performance result contributes to selection. This cache-based rule was chosen before image or inference-result inspection.

| Input | Half-open box Z0,Z1,Y0,Y1,X0,X1 | Source chunk indices Z × Y × X |
|---|---|---|
| `pherc0139_z4352` | `[4352,4608,3072,3328,2560,2816]` | `{34,35} × {24,25} × {20,21}` |
| `pherc0139_z4608` | `[4608,4864,3072,3328,2560,2816]` | `{36,37} × {24,25} × {20,21}` |

The two cubes share a Z face and have zero voxel overlap. Both are disjoint from the existing diagnostic cube `[3840,4096,3712,3968,1344,1600]`. Each output is `(256,256,256)` uint8 in ZYX order, at 9.362 micrometres per voxel.

## Source and cache provenance

All cubes use the same official source as the original diagnostic:

`s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr`

L0 HTTP base: `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0`.

Declared Zarr v2 layout: shape `[20974,6621,6621]`, chunks `[128,128,128]`, dtype `|u1`, C order, uncompressed, no filters, dimension separator `/`.

Expected cache is `data/PHerc0139-4x5x5/cubes_RAW/zZZZZZ_yYYYYY_xXXXXX.tif`, with each coordinate the start voxel of one 128-cubed raw chunk. Eight TIFFs per selected input cover each cube exactly. The cache manifest identifies this exact source and full grid box. Cache reuse must validate source, grid membership, individual TIFF shape/dtype and record each file SHA-256. These are raw TIFFs; do not use the nearby Gaussian-smoothed `ct_sigma1.npy` or prediction files. TIFF pixel identity has not been independently revalidated against S3, so cached origin remains a documented provenance limitation.

Other existing caches do not replace this selection: `locked_pherc0139_raw/context.npy` covers only `[3776,4160,3648,4032,1280,1664]`; its 64-voxel margins cannot produce two full cubes adjacent to the original. The additional z4864 grid is only 128 voxels deep.

## Fixed preparation and transfer bound

Prepare with `fetch_additional_inputs.py`, validating strict source layout with one bounded `.zarray` request, and prefer the above local TIFFs. If a selected TIFF is absent, fetch that exact 128-cubed S3 raw chunk with a strict 2 MiB streaming bound; require exactly 2,097,152 bytes. Never replace a selected cube based on data or outcomes.

Maximum uncached payload is 16 chunks = 32 MiB, plus at most 64 KiB metadata, below the 48 MiB task ceiling. Expected new payload with complete cache: metadata only. No model/reference downloads and no GPU calls.

Write `inputs_additional/pherc0139_z4352.npy` and `inputs_additional/pherc0139_z4608.npy` plus corresponding `.json` manifests. Required manifest fields are `source_uri`, `bbox_zyx`, `array_sha256` (SHA-256 of C-order pixels), `shape`, and `dtype`; also record axis order, scale, file hash, metadata hash and per-chunk provenance/hash. Array hashes are populated by fixed extraction, not used to accept or reject input quality.

Use both cubes for the same already frozen full-inference comparison and report every result. No cube substitution, tuning, success-based early stopping, or interpretation of these unlabelled inputs as reading ground truth.

## Preparation receipt appended after selection

The fixed selection above was preserved. `fetch_additional_inputs.py --cache-root data/PHerc0139-4x5x5 --verify-source` assembled the cached raw TIFFs and compared all 16 chunks, pixel for pixel, with the official S3 source. Every comparison passed. Transfer was exactly 33,554,432 raw bytes (32 MiB) plus 239 metadata bytes. Thus the initially stated unverified-cache provenance limitation was resolved for these two cubes; no image-quality inspection or model-output inspection was performed.

Fixed C-order pixel SHA-256 values:

- `pherc0139_z4352`: `819b93a9fa7d3ab30735caea4f8d558f6e83439e9957942b886f1118a0a369f4`.
- `pherc0139_z4608`: `8755eceeffffe19ebc89fb1c001aa29a4b7f3c31e5b577714a31297f34268b60`.

Arrays and individual provenance manifests are in `inputs_additional/`; `inputs_additional/fetch_receipt.json` records the transfer and verification outcome. The fetcher embeds the fixed hashes and can independently reproduce both cubes without a local cache using only NumPy and requests (tifffile is required only for optional TIFF reuse).
