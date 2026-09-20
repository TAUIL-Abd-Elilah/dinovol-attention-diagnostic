"""Bounded padding feasibility probe on the first actual ps8 teacher SDPA call.

No production files are changed. Stops before finishing the first transformer
block. Padding is established prior art; this tests the measured Windows case.
"""
import argparse
import json
import math
import statistics
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
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = {'status': 'STARTED', 'scope': 'One real first-block attention call only; no full-forward speedup claim',
          'variants_predeclared': ['unchanged', 'pad56', 'pad64', 'pad56_math_control'],
          'production_source_modified': False}


class ProbeComplete(Exception):
    pass


try:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    original = F.scaled_dot_product_attention

    def intercept(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        assert list(q.shape) == [1, 16, 4101, 54]
        assert q.shape == k.shape == v.shape and q.dtype == torch.bfloat16
        assert attn_mask is None and dropout_p == 0 and not is_causal and not enable_gqa
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        original_scale = scale if scale is not None else q.shape[-1] ** -0.5
        report['input'] = {'shape': list(q.shape), 'dtype': str(q.dtype), 'q_stride': list(q.stride()),
                           'scale_argument': scale, 'preserved_scale': original_scale}
        report['eligibility'] = {}
        for width in (54, 56, 64):
            tensors = [F.pad(x, (0, width-54)) if width != 54 else x for x in (q, k, v)]
            params = torch.backends.cuda.SDPAParams(*tensors, None, 0.0, False, False)
            report['eligibility'][str(width)] = {
                'flash': torch.backends.cuda.can_use_flash_attention(params),
                'efficient': torch.backends.cuda.can_use_efficient_attention(params),
                'cudnn': torch.backends.cuda.can_use_cudnn_attention(params)}
        del tensors, params

        def invoke(name):
            if name == 'unchanged':
                return original(q, k, v, scale=scale)
            width = 64 if name == 'pad64' else 56
            padded = [F.pad(x, (0, width-54)) for x in (q, k, v)]
            if name == 'pad56_math_control':
                with sdpa_kernel(SDPBackend.MATH):
                    out = original(*padded, scale=original_scale)
            else:
                out = original(*padded, scale=original_scale)
            return out[..., :54]

        reference = invoke('unchanged').clone()
        report['variants'] = {}
        for name in report['variants_predeclared']:
            try:
                for _ in range(2):
                    out = invoke(name)
                torch.cuda.synchronize()
                durations = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(5):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    out = invoke(name)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter()-started)*1000)
                delta = out.float() - reference.float()
                metrics = {
                    'max_abs': float(delta.abs().max()), 'mean_abs': float(delta.abs().mean()),
                    'relative_l2': float(delta.norm()/reference.float().norm()),
                    'cosine': float(F.cosine_similarity(out.float().flatten(), reference.float().flatten(), dim=0)),
                    'exact': bool(torch.equal(out, reference)), 'finite': bool(torch.isfinite(out).all())}
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    invoke(name)
                    torch.cuda.synchronize()
                operators = {event.key: event.count for event in prof.key_averages()
                             if 'scaled_dot_product' in event.key}
                report['variants'][name] = {'status': 'PASS', 'milliseconds': durations,
                    'median_ms': statistics.median(durations), 'output_vs_unchanged': metrics,
                    'actual_sdpa_operators': operators,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            except Exception as exc:
                report['variants'][name] = {'status': 'FAILED', 'error': repr(exc)}
        raise ProbeComplete()

    run_args = argparse.Namespace(checkpoint=args.checkpoint, vesuvius_src=args.vesuvius_src,
        reference=args.reference, volume=args.volume, output=args.output, check_only=False)
    provenance = {}
    with patch.object(F, 'scaled_dot_product_attention', intercept):
        try:
            baseline.run(run_args, provenance)
        except ProbeComplete:
            report['status'] = 'PROBE_COMPLETE'
        else:
            raise RuntimeError('Expected interception did not stop the model')
    report['provenance'] = provenance
except Exception:
    report['status'] = 'FAILED'
    report['traceback'] = traceback.format_exc()
    raise
finally:
    report['script_sha256'] = baseline.sha256_file(Path(__file__))
    (args.output/'probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'provenance'}, indent=2))
