"""Profile a bounded runtime experiment; the pinned production checkout is unmodified."""
import argparse
import json
import sys
import time
import traceback
import hashlib
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir', type=Path, required=True)
known, _ = parser.parse_known_args()
sys.path.insert(0, str(known.baseline_dir.resolve()))
import profile_existing as baseline
baseline.add_input_arguments(parser)
parser.add_argument('--mode', choices=['unchanged', 'helper'], required=True)
parser.add_argument('--helper-dir', type=Path, required=True)
parser.add_argument('--manifest', type=Path, help='Explicit hash-verified additional CT fixture provenance')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = {'status': 'STARTED', 'mode': args.mode, 'production_source_modified': False,
          'quality_capture': 'Last eight windows and eighth interpolation only, during separate profiler pass; excluded from timing',
          'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    sys.path.insert(0, str(args.vesuvius_src.resolve()))
    from vesuvius.models.build.pretrained_backbones.dinovol_2_eva import Eva
    original_sdpa, original_forward, original_interpolate = F.scaled_dot_product_attention, Eva.forward_features, F.interpolate
    sys.path.insert(0, str(args.helper_dir.resolve()))
    import sdpa_padding
    # Isolated benchmark only: retain the native callable to prevent recursive
    # dispatch while intercepting production calls. The product helper is unchanged.
    sdpa_padding.F = SimpleNamespace(pad=F.pad, scaled_dot_product_attention=original_sdpa)
    report['helper'] = {'sha256': baseline.sha256_file(Path(sdpa_padding.__file__)),
                        'path': str(Path(sdpa_padding.__file__).resolve()),
                        'integration': 'isolated benchmark F proxy preserving saved native callable'}
    if args.manifest:
        import load_real_ct
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        def load_additional(path):
            array = np.load(path, allow_pickle=False)
            if array.shape != (256, 256, 256) or array.dtype != np.uint8:
                raise ValueError('Additional fixture must be 256-cubed uint8')
            digest = hashlib.sha256(array.tobytes(order='C')).hexdigest()
            if digest != manifest['array_sha256']:
                raise ValueError('Additional fixture pixel hash mismatch')
            return array, dict(manifest, volume_path=str(Path(path).resolve()),
                volume_file_sha256=baseline.sha256_file(Path(path)), array_sha256=digest,
                nonzero_fraction=float(np.mean(array != 0)),
                percentiles_1_50_99=np.percentile(array, [1, 50, 99]).tolist())
        load_real_ct.load_volume = load_additional
        report['fixture_loader_override'] = {'manifest': manifest,
            'manifest_sha256': baseline.sha256_file(args.manifest),
            'implementation': 'load_additional in this recorded runner; original loader bypassed explicitly'}
    counts = {'sdpa': 0, 'windows': 0, 'map_interpolations': 0, 'captured_windows': 0}

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        counts['sdpa'] += 1
        assert list(q.shape) == [1, 16, 4101, 54] and q.shape == k.shape == v.shape
        assert q.dtype == torch.bfloat16 and torch.is_inference_mode_enabled()
        assert attn_mask is None and dropout_p == 0 and not is_causal and not enable_gqa
        if args.mode == 'unchanged':
            return original_sdpa(q, k, v, scale=scale)
        return sdpa_padding.inference_sdpa(q, k, v, scale=scale, enabled=True)

    def forward(self, *positional, **keywords):
        result = original_forward(self, *positional, **keywords)
        index = counts['windows']
        counts['windows'] += 1
        if index >= 56:
            assert index < 64, 'Baseline protocol changed; refuse misleading capture/timing'
            tokens = result['x_norm_patchtokens'].detach().float().cpu().numpy()
            np.save(args.output / f'window_{index-56}_tokens.npy', tokens)
            counts['captured_windows'] += 1
        return result

    def interpolate(tensor, *positional, **keywords):
        out = original_interpolate(tensor, *positional, **keywords)
        if tuple(tensor.shape) == (1, 1, 32, 32, 32) and tuple(out.shape) == (1, 1, 256, 256, 256):
            index = counts['map_interpolations']
            counts['map_interpolations'] += 1
            if index == 7:
                grid = tensor.detach().float().cpu().numpy()
                pre_norm = out.detach().float().cpu().numpy()
                np.save(args.output / 'blended_grid.npy', grid)
                np.save(args.output / 'interpolated_before_minmax.npy', pre_norm)
                report['pre_minmax'] = {'min': float(pre_norm.min()), 'max': float(pre_norm.max()),
                                         'range': float(pre_norm.max()-pre_norm.min())}
        return out

    run_args = argparse.Namespace(checkpoint=args.checkpoint, vesuvius_src=args.vesuvius_src,
        reference=args.reference, volume=args.volume, output=args.output, check_only=False)
    with patch.object(F, 'scaled_dot_product_attention', sdpa), patch.object(Eva, 'forward_features', forward), patch.object(F, 'interpolate', interpolate):
        baseline.run(run_args, report)
    assert counts == {'sdpa': 1536, 'windows': 64, 'map_interpolations': 8, 'captured_windows': 8}, counts
    report['counts'] = counts
    report['status'] = 'FULL_EXPERIMENT_COMPLETE'
except Exception:
    report['status'] = 'FAILED'
    report['traceback'] = traceback.format_exc()
    raise
finally:
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    report['experiment_sha256'] = baseline.sha256_file(Path(__file__))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'mode', 'timing_summary', 'gpu_memory_timed', 'counts', 'pre_minmax']}, indent=2))
