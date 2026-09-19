"""Capture the first real SDPA inputs and stop before executing attention.

Reuses profile_existing's verified real-CT loading, normalization, strict
checkpoint loading and production consumer setup. No attention backend is forced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from unittest.mock import patch

import profile_existing as baseline


class StopBeforeAttention(Exception):
    """Expected successful stop after inspecting the first attention inputs."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline.add_input_arguments(parser)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; an existing directory is refused.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTED", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "argv": sys.argv, "attention_executed": False, "sdpa_interceptions": 0,
              "backend_forcing": False, "diagnostic_warnings": []}
    warning_lines = []

    def tensor_info(tensor):
        if tensor is None:
            return None
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "device": str(tensor.device),
                "contiguous": tensor.is_contiguous(), "requires_grad": tensor.requires_grad,
                "storage_offset": tensor.storage_offset()}

    try:
        import torch
        import torch.nn.functional as functional

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        report["cpu_threads"] = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}
        report["diagnostic_source"] = {"path": str(Path(__file__).resolve()),
                                       "sha256": baseline.sha256_file(Path(__file__))}

        def inspect_first(query, key, value, attn_mask=None, dropout_p=0.0,
                          is_causal=False, *, scale=None, enable_gqa=False):
            report["sdpa_interceptions"] += 1
            if report["sdpa_interceptions"] != 1:
                raise RuntimeError("More than one SDPA interception")
            report["first_sdpa"] = {
                "query": tensor_info(query), "key": tensor_info(key), "value": tensor_info(value),
                "attn_mask": tensor_info(attn_mask), "dropout_p": dropout_p,
                "is_causal": is_causal, "scale": scale, "enable_gqa": enable_gqa,
                "inference_mode": torch.is_inference_mode_enabled(), "grad_enabled": torch.is_grad_enabled(),
            }
            backend = torch.backends.cuda
            report["backend_flags"] = {
                "cuda_built": backend.is_built(),
                "flash_compiled_available": backend.is_flash_attention_available(),
                "flash_enabled": backend.flash_sdp_enabled(),
                "efficient_enabled": backend.mem_efficient_sdp_enabled(),
                "cudnn_enabled": backend.cudnn_sdp_enabled(),
                "math_enabled": backend.math_sdp_enabled(),
                "math_fp16_bf16_reduction_allowed": backend.fp16_bf16_reduction_math_sdp_allowed(),
                "cudnn_available": torch.backends.cudnn.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "device_capability": list(torch.cuda.get_device_capability(query.device)),
            }
            params = backend.SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
            report["backend_eligibility"] = {}
            for name, checker in (
                ("flash", backend.can_use_flash_attention),
                ("efficient", backend.can_use_efficient_attention),
                ("cudnn", backend.can_use_cudnn_attention),
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        eligible = bool(checker(params, debug=True))
                        report["backend_eligibility"][name] = {"can_use": eligible}
                    except Exception as exc:
                        report["backend_eligibility"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                    entries = [{"backend": name, "category": item.category.__name__,
                                "message": str(item.message), "filename": item.filename,
                                "lineno": item.lineno} for item in caught]
                report["diagnostic_warnings"].extend(entries)
                warning_lines.extend(f"[{name}] {entry['category']}: {entry['message']}" for entry in entries)
            torch.cuda.synchronize(query.device)
            report["gpu_memory_at_stop"] = {
                "allocated_bytes": torch.cuda.memory_allocated(query.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(query.device),
                "reserved_bytes": torch.cuda.memory_reserved(query.device),
            }
            raise StopBeforeAttention()

        run_args = argparse.Namespace(
            checkpoint=args.checkpoint, vesuvius_src=args.vesuvius_src,
            volume=args.volume, reference=args.reference,
            output=args.output, check_only=False,
        )
        with patch.object(functional, "scaled_dot_product_attention", new=inspect_first):
            try:
                baseline.run(run_args, report)
            except StopBeforeAttention:
                report["status"] = "DIAGNOSED_FIRST_SDPA_ABORTED_BEFORE_ATTENTION"
            else:
                raise RuntimeError("Production call did not reach the intercepted SDPA")
        report["diagnostic_scope"] = "One first-block QKV projection; SDPA not executed; no warmup or full forward completed"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output / "warnings.txt").write_text("\n".join(warning_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "first_sdpa": report["first_sdpa"],
                      "backend_flags": report["backend_flags"],
                      "backend_eligibility": report["backend_eligibility"],
                      "warnings": report["diagnostic_warnings"]}, indent=2))


if __name__ == "__main__":
    main()
