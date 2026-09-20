"""Reproduce two preselected PHerc0139 cubes; no weights, inference or installs.

The bounded request/layout checks follow the original diagnostic fetch_inputs.py.
Without --cache-root, download exactly 16 raw 2 MiB chunks (plus metadata).
With a cache, --verify-source compares every cached pixel to its official chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import requests

SOURCE_URI = ('s3://vesuvius-challenge-open-data/PHerc0139/volumes/'
              '20250728140407-9.362um-1.2m-113keV-masked.zarr')
CT_BASE = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
           'PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0')
LAYOUT = {'shape': [20974, 6621, 6621], 'chunks': [128, 128, 128],
          'dtype': '|u1', 'order': 'C', 'compressor': None, 'filters': None,
          'dimension_separator': '/', 'zarr_format': 2}
INPUTS = (
    ('pherc0139_z4352', [4352, 4608, 3072, 3328, 2560, 2816],
     '819b93a9fa7d3ab30735caea4f8d558f6e83439e9957942b886f1118a0a369f4'),
    ('pherc0139_z4608', [4608, 4864, 3072, 3328, 2560, 2816],
     '8755eceeffffe19ebc89fb1c001aa29a4b7f3c31e5b577714a31297f34268b60'),
)
CHUNK_BYTES = 128 ** 3
MAX_CHUNKS = 16


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def small_get(url, cap):
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        data = bytearray()
        for block in response.iter_content(2 ** 20):
            if len(data) + len(block) > cap:
                raise ValueError('Response exceeded declared size limit')
            data.extend(block)
    return bytes(data)


def validate_cache(root):
    if root is None:
        return None
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('sources', {}).get('raw') != SOURCE_URI or
            manifest.get('chunk_size') != 128 or
            manifest.get('bbox_l0_zyx') != [4352, 4864, 3072, 3712, 2560, 3200]):
        raise ValueError('Cache source/layout does not match the frozen selection')
    return file_sha(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, help='Optional original raw grid with manifest.json and cubes_RAW/')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'inputs_additional')
    parser.add_argument('--verify-source', action='store_true', help='Verify each reused cached TIFF against official S3 pixels')
    args = parser.parse_args()
    cache_manifest_sha = validate_cache(args.cache_root)
    metadata_raw = small_get(CT_BASE + '/.zarray', 65536)
    metadata = json.loads(metadata_raw)
    if any(metadata.get(k) != v for k, v in LAYOUT.items()):
        raise ValueError('Public Zarr metadata differs from the fixed layout')
    args.output.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    receipts = []

    def remote_chunk(index):
        if index not in downloaded:
            if len(downloaded) >= MAX_CHUNKS:
                raise ValueError('32 MiB raw-chunk budget exhausted')
            raw = small_get(CT_BASE + '/' + '/'.join(map(str, index)), CHUNK_BYTES)
            if len(raw) != CHUNK_BYTES:
                raise ValueError(f'Unexpected raw chunk byte count: {index}')
            downloaded[index] = raw
        return np.frombuffer(downloaded[index], dtype=np.uint8).reshape((128,) * 3)

    for name, bbox, expected_sha in INPUTS:
        origin = (bbox[0], bbox[2], bbox[4])
        array = np.empty((256,) * 3, dtype=np.uint8)
        provenance = []
        for offset in itertools.product((0, 128), repeat=3):
            start = tuple(a + b for a, b in zip(origin, offset))
            index = tuple(a // 128 for a in start)
            filename = f'z{start[0]:05d}_y{start[1]:05d}_x{start[2]:05d}.tif'
            cached = args.cache_root / 'cubes_RAW' / filename if args.cache_root else None
            entry = {'index_zyx': list(index), 'url': CT_BASE + '/' + '/'.join(map(str, index))}
            if cached is not None and cached.is_file():
                import tifffile  # Optional: S3-only reproduction does not require it.
                cube = tifffile.imread(cached)
                if cube.shape != (128,) * 3 or cube.dtype != np.uint8:
                    raise ValueError(f'Cache TIFF shape/dtype differs: {filename}')
                entry.update({'cache_file': 'cubes_RAW/' + filename,
                              'cache_file_sha256': file_sha(cached), 'origin': 'cached_tiff',
                              'source_verified': False})
                if args.verify_source:
                    if not np.array_equal(cube, remote_chunk(index)):
                        raise ValueError(f'Cached TIFF differs from official source: {filename}')
                    entry['source_verified'] = True
            else:
                cube = remote_chunk(index)
                entry.update({'origin': 'official_s3', 'source_verified': True})
            entry['pixel_sha256'] = sha(cube.tobytes(order='C'))
            target = tuple(slice(a, a + 128) for a in offset)
            array[target] = cube
            provenance.append(entry)
        array_sha = sha(array.tobytes(order='C'))
        if array_sha != expected_sha:
            raise ValueError(f'Frozen array identity failed: {name}')
        array_path = args.output / (name + '.npy')
        if array_path.exists():
            existing = np.load(array_path, allow_pickle=False)
            if (existing.shape != (256,) * 3 or existing.dtype != np.uint8 or
                    sha(existing.tobytes(order='C')) != expected_sha):
                raise ValueError(f'Existing array differs: {array_path}')
        else:
            with array_path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
        receipt = {
            'schema_version': 'dinovol-additional-ct-v1', 'source_uri': SOURCE_URI,
            'level': 0, 'axis_order': 'zyx', 'voxel_size_um': 9.362,
            'bbox_zyx': bbox, 'shape': [256] * 3, 'dtype': 'uint8',
            'array_sha256': array_sha, 'array_file': array_path.name,
            'array_file_sha256': file_sha(array_path),
            'zarr_metadata_sha256': sha(metadata_raw), 'zarr_layout': LAYOUT,
            'cache_manifest_sha256': cache_manifest_sha,
            'all_chunks_verified_against_source': all(p['source_verified'] for p in provenance),
            'chunks': provenance,
        }
        (args.output / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        receipts.append({'name': name, 'array_sha256': array_sha,
                         'all_chunks_verified_against_source': receipt['all_chunks_verified_against_source']})
    summary = {'inputs': receipts, 'downloaded_raw_chunks': len(downloaded),
               'downloaded_raw_bytes': sum(map(len, downloaded.values())),
               'downloaded_metadata_bytes': len(metadata_raw)}
    (args.output / 'fetch_receipt.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
