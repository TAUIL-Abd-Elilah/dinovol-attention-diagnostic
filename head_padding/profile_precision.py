"""Profile a bounded runtime experiment; the pinned production checkout is unmodified."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir', type=Path, required=True)
known, _ = parser.parse_known_args()
sys.path.insert(0, str(known.baseline_dir.resolve()))
import profile_existing as baseline
baseline.add_input_arguments(parser)
parser.add_argument('--mode', choices=['unchanged', 'pad56_fp32'], required=True)
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
    counts = {'sdpa': 0, 'windows': 0, 'map_interpolations': 0, 'captured_windows': 0}

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        counts['sdpa'] += 1
        assert list(q.shape) == [1, 16, 4101, 54] and q.shape == k.shape == v.shape
        assert q.dtype == torch.bfloat16 and torch.is_inference_mode_enabled()
        assert attn_mask is None and dropout_p == 0 and not is_causal and not enable_gqa
        if args.mode == 'unchanged':
            return original_sdpa(q, k, v, scale=scale)
        q, k, v = (F.pad(t.float(), (0, 2)) for t in (q, k, v))
        preserved = scale if scale is not None else 54 ** -0.5
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = original_sdpa(q, k, v, scale=preserved)
        return out[..., :54].to(dtype=torch.bfloat16)

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
