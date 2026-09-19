"""Verify and load the fixed public PHerc0139 real-CT benchmark crop."""

from pathlib import Path
import hashlib

import numpy as np


EXPECTED_PIXEL_SHA256 = "fb5d3659016e1bb760c608336e3ae1f356414464362021ec4e96ce392de94ad5"
SOURCE_URI = "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
BOX_Z0_Z1_Y0_Y1_X0_X1 = [3840, 4096, 3712, 3968, 1344, 1600]


def load_volume(volume_path):
    """Return the exact 256³ uint8 ZYX pixel array and JSON-safe provenance."""
    volume_path = Path(volume_path)
    if volume_path.suffix.lower() != ".npy":
        raise ValueError("The fixed CT volume must be a .npy file")
    volume = np.load(volume_path, allow_pickle=False)
    if volume.shape != (256, 256, 256) or volume.dtype != np.uint8:
        raise ValueError(f"Expected 256³ uint8 CT, got {volume.shape}, {volume.dtype}")
    pixel_sha256 = hashlib.sha256(volume.tobytes(order="C")).hexdigest()
    if pixel_sha256 != EXPECTED_PIXEL_SHA256:
        raise ValueError(f"CT pixel SHA-256 mismatch: {pixel_sha256}")
    metadata = {
        "volume_path": str(volume_path.resolve()),
        "volume_file_sha256": hashlib.sha256(volume_path.read_bytes()).hexdigest(),
        "source": {"uri": SOURCE_URI, "level": 0, "shape": [20974, 6621, 6621],
                   "chunks": [128, 128, 128], "dtype": "uint8", "voxel_size_um": 9.362},
        "grid_box_l0_zyx": BOX_Z0_Z1_Y0_Y1_X0_X1,
        "box_order": "z0,z1,y0,y1,x0,x1", "array_axis_order": "ZYX",
        "shape": list(volume.shape), "dtype": str(volume.dtype),
        "array_sha256": pixel_sha256,
        "nonzero_fraction": float(np.mean(volume != 0)),
        "min": int(volume.min()), "max": int(volume.max()),
        "percentiles_1_50_99": np.percentile(volume, [1, 50, 99]).tolist(),
    }
    return volume, metadata
