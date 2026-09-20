"""Follow-up first-call precision probe motivated by observed full-map differences.

Reconstructs the pinned teacher with the existing strict real-CT baseline loader.
Captures the first actual Q/K/V and stops before completing that attention block.
This is a new diagnostic experiment, not a revision of the original protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
from unittest.mock import patch


class ProbeComplete(Exception):
    pass


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--baseline-dir", type=Path, required=True)
    known, _ = bootstrap.parse_known_args()
    sys.path.insert(0, str(known.baseline_dir.resolve()))
    import profile_existing as baseline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    baseline.add_input_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    variants = ["native_bf16_math", "pad56_bf16", "pad56_fp32_efficient_cast_bf16"]
    report = {
        "status": "STARTED",
        "scope": "First actual attention call only; no full-model speed or quality conclusion",
        "experiment_motivation": {
            "stage": "New follow-up after full bf16 fused-vs-math comparison",
            "parent_reported_full_map_max_abs": 0.0140,
            "parent_reported_full_map_mean_abs": 0.00218,
            "parent_reported_eligibility_flip_fraction": 0.00985,
            "parent_reported_math_padding_control": "exact",
            "question": "Can FP32 efficient SDPA reduce error to native bf16 math while retaining useful speed?",
        },
        "variants_predeclared": variants,
        "production_source_modified": False,
        "helper_source_modified": False,
        "protocol": {
            "warmups": 2, "measured_repeats": 5, "profiler_passes_per_variant": 1,
            "padded_width": 56, "original_width": 54,
            "timing": "Synchronized wall latency including per-call casts, padding, attention, output slicing and output cast",
            "fixed_order": variants,
            "fp32_variant": "Original bf16 Q/K/V cast to float32 only inside attention, then output cast back to bf16",
            "backend_forcing": "MATH for reference; automatic for existing bf16 padding; EFFICIENT_ATTENTION only for new FP32 variant",
        },
    }
    provenance = {}
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        native_sdpa = F.scaled_dot_product_attention

        def intercept(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                      *, scale=None, enable_gqa=False):
            if list(q.shape) != [1, 16, 4101, 54] or q.shape != k.shape or q.shape != v.shape:
                raise ValueError("Unexpected actual first-call QKV shape")
            if any(t.dtype != torch.bfloat16 or t.device.type != "cuda" for t in (q, k, v)):
                raise ValueError("Expected actual CUDA bf16 QKV")
            if attn_mask is not None or dropout_p != 0 or is_causal or enable_gqa:
                raise ValueError("Unexpected first-call attention options")
            if not torch.is_inference_mode_enabled() or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
                raise ValueError("Expected inference mode without autograd/autocast")
            preserved_scale = q.shape[-1] ** -0.5 if scale is None else scale
            report["input"] = {
                "shape": list(q.shape), "dtype": str(q.dtype), "device": str(q.device),
                "strides": {name: list(t.stride()) for name, t in zip(("q", "k", "v"), (q, k, v))},
                "scale_argument": scale, "preserved_scale": preserved_scale,
                "qkv_sha256": {name: hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                               for name, t in zip(("q", "k", "v"), (q, k, v))},
            }
            report["precision_settings_unchanged"] = {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "bf16_reduced_precision_matmul_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "fp16_bf16_reduction_math_sdp_allowed": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
            report["backend_flags_before_probe"] = {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            }
            report["eligibility_hints"] = {}
            for dtype in (torch.bfloat16, torch.float32):
                padded = [F.pad(t.to(dtype=dtype), (0, 2)) for t in (q, k, v)]
                params = torch.backends.cuda.SDPAParams(*padded, None, 0.0, False, False)
                report["eligibility_hints"][str(dtype)] = {
                    "flash": torch.backends.cuda.can_use_flash_attention(params),
                    "efficient": torch.backends.cuda.can_use_efficient_attention(params),
                    "cudnn": torch.backends.cuda.can_use_cudnn_attention(params),
                }
                del padded, params

            def invoke(name):
                if name == "native_bf16_math":
                    with sdpa_kernel(SDPBackend.MATH):
                        return native_sdpa(q, k, v, scale=scale)
                if name == "pad56_bf16":
                    padded = [F.pad(t, (0, 2)) for t in (q, k, v)]
                    return native_sdpa(*padded, scale=preserved_scale)[..., :54]
                if name == "pad56_fp32_efficient_cast_bf16":
                    padded = [F.pad(t.float(), (0, 2)) for t in (q, k, v)]
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        out = native_sdpa(*padded, scale=preserved_scale)
                    return out[..., :54].to(dtype=q.dtype)
                raise ValueError(name)

            reference = invoke("native_bf16_math").clone()
            torch.cuda.synchronize()
            reference_fp32 = reference.float()
            report["reference_output_sha256"] = hashlib.sha256(
                reference.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            report["variants"] = {}
            for name in variants:
                try:
                    out = None
                    for _ in range(2):
                        out = invoke(name)
                    torch.cuda.synchronize()
                    del out
                    live_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    durations = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        begun = time.perf_counter()
                        out = invoke(name)
                        torch.cuda.synchronize()
                        durations.append((time.perf_counter() - begun) * 1000)
                        del out
                    timed_peak = torch.cuda.max_memory_allocated()
                    # Separate quality evaluation is outside every timed interval.
                    out = invoke(name)
                    delta = out.float() - reference_fp32
                    metrics = {
                        "max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
                        "relative_l2": float(delta.norm() / reference_fp32.norm().clamp_min(1e-30)),
                        "cosine": float(F.cosine_similarity(out.float().flatten(), reference_fp32.flatten(), dim=0)),
                        "exact_fraction": float((out == reference).float().mean()),
                        "exact": bool(torch.equal(out, reference)), "finite": bool(torch.isfinite(out).all()),
                        "output_dtype": str(out.dtype), "output_shape": list(out.shape),
                    }
                    del delta, out
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        profiled_out = invoke(name)
                        torch.cuda.synchronize()
                    del profiled_out
                    trace = args.output / (name + "_trace.json")
                    prof.export_chrome_trace(str(trace))
                    events = prof.key_averages()
                    operators = {event.key: event.count for event in events if "scaled_dot_product" in event.key}
                    if name == "native_bf16_math" and not any("attention_math" in key for key in operators):
                        raise RuntimeError("Profiler did not confirm requested math reference backend")
                    if name == "pad56_fp32_efficient_cast_bf16" and not any("efficient_attention" in key for key in operators):
                        raise RuntimeError("Profiler did not confirm requested FP32 efficient backend")
                    report["variants"][name] = {
                        "status": "PASS", "milliseconds": durations, "median_ms": statistics.median(durations),
                        "min_ms": min(durations), "max_ms": max(durations),
                        "output_vs_native_bf16_math": metrics, "actual_sdpa_operators": operators,
                        "timed_live_allocated_before_bytes": live_before, "timed_peak_allocated_bytes": timed_peak,
                        "timed_peak_increment_bytes": timed_peak - live_before,
                        "trace": trace.name,
                        "top20_self_cuda_operators": events.table(sort_by="self_device_time_total", row_limit=20),
                    }
                except Exception as error:
                    report["variants"][name] = {"status": "FAILED", "error": repr(error),
                                                "traceback": traceback.format_exc()}
            raise ProbeComplete()

        run_args = argparse.Namespace(checkpoint=args.checkpoint, vesuvius_src=args.vesuvius_src,
                                      reference=args.reference, volume=args.volume, output=args.output, check_only=False)
        with patch.object(F, "scaled_dot_product_attention", intercept):
            try:
                baseline.run(run_args, provenance)
            except ProbeComplete:
                report["status"] = "PROBE_COMPLETE" if all(v["status"] == "PASS" for v in report["variants"].values()) else "PROBE_COMPLETE_WITH_FAILED_VARIANT"
            else:
                raise RuntimeError("The expected first-call interception did not stop execution")
    except Exception:
        report["status"] = "FAILED"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["provenance"] = provenance
        report["script_sha256"] = baseline.sha256_file(Path(__file__))
        (args.output / "probe_fp32.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "variants": report.get("variants", {})}, indent=2))


if __name__ == "__main__":
    main()
