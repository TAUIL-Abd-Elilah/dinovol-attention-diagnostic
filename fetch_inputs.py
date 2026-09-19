"""Fetch the fixed public CT crop/reference; optionally fetch the 864 MB teacher.

No credentials, package installation, source edits, or GPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import requests

CT_BASE = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
           'PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0')
CT_SHA = 'fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5'
REF_URL = ('https://huggingface.co/scrollprize/ink_3d_dino_guided/resolve/'
           '73a79525466037432191284dfa237baf830c49ec/avg_ref_embedding.npy')
REF_SHA = '61bdf93bc5e3fd956eebdbed52618985d27264b8a9e5cb043087d0f234507a81'
TEACHER_NAME = 'dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt'
TEACHER_URL = ('https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500/'
               'resolve/6a8cccbafef191a966da815e22ff5c6eae075aae/' + TEACHER_NAME)
TEACHER_SHA = 'e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59'


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def small_get(url, cap):
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        data = bytearray()
        for block in response.iter_content(2**20):
            if len(data) + len(block) > cap:
                raise ValueError('Response exceeded declared size limit')
            data.extend(block)
    return bytes(data)


def download_verified(url, path, size, sha):
    if path.exists():
        if path.stat().st_size != size or file_sha(path) != sha:
            raise ValueError(f'Existing input differs: {path.name}')
        return
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise ValueError('Partial input exceeds expected size')
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or
                               response.headers.get('Content-Range') != f'bytes {offset}-{size-1}/{size}'):
                    raise ValueError('Server did not honor the resume range')
                with part.open('ab' if offset else 'xb') as stream:
                    for block in response.iter_content(8 * 2**20):
                        offset += len(block)
                        if offset > size:
                            raise ValueError('Input exceeds expected size')
                        stream.write(block)
        except requests.RequestException:
            print(f'Transfer interrupted; retaining partial bytes (attempt {attempt+1}/3)', flush=True)
    if part.stat().st_size != size or file_sha(part) != sha:
        raise ValueError('Input incomplete or SHA-256 mismatch; partial retained')
    part.replace(path)


def fetch_ct(path):
    if path.exists():
        array = np.load(path, allow_pickle=False)
    else:
        meta = json.loads(small_get(CT_BASE + '/.zarray', 65536))
        expected = {'shape': [20974, 6621, 6621], 'chunks': [128, 128, 128],
                    'dtype': '|u1', 'order': 'C', 'compressor': None,
                    'filters': None, 'dimension_separator': '/', 'zarr_format': 2}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError('Public Zarr metadata differs from the recorded layout')
        origin = np.array([3840, 3712, 1344])
        end = origin + 256
        array = np.empty((256, 256, 256), dtype=np.uint8)
        chunks = [range(int(a // 128), int((b - 1) // 128) + 1) for a, b in zip(origin, end)]
        for index in itertools.product(*chunks):
            raw = small_get(CT_BASE + '/' + '/'.join(map(str, index)), 128**3)
            if len(raw) != 128**3:
                raise ValueError(f'Unexpected chunk length: {index}')
            cube = np.frombuffer(raw, dtype=np.uint8).reshape(128, 128, 128)
            start = np.array(index) * 128
            lo, hi = np.maximum(start, origin), np.minimum(start + 128, end)
            src = tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
            dst = tuple(slice(int(a), int(b)) for a, b in zip(lo - origin, hi - origin))
            array[dst] = cube[src]
    if array.shape != (256, 256, 256) or array.dtype != np.uint8 or hashlib.sha256(array.tobytes()).hexdigest() != CT_SHA:
        raise ValueError('Fixed CT pixel identity failed')
    if not path.exists():
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
    return {'source': CT_BASE, 'shape': [256, 256, 256], 'dtype': 'uint8',
            'bbox_zyx': [3840, 4096, 3712, 3968, 1344, 1600], 'array_sha256': CT_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('inputs'))
    parser.add_argument('--include-checkpoint', action='store_true', help='Also fetch the 863,606,099-byte teacher')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {'ct': fetch_ct(args.output / 'pherc0139_ct.npy')}
    download_verified(REF_URL, args.output / 'avg_ref_embedding.npy', 3584, REF_SHA)
    receipt['reference'] = {'url': REF_URL, 'sha256': REF_SHA}
    if args.include_checkpoint:
        download_verified(TEACHER_URL, args.output / TEACHER_NAME, 863606099, TEACHER_SHA)
        receipt['checkpoint'] = {'url': TEACHER_URL, 'sha256': TEACHER_SHA}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
