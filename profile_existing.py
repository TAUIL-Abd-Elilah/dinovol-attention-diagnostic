"""Bounded real-CT profiling of the unchanged guided-ink DINO consumer.

This driver neither trains nor optimizes a model. It calls the production
DinoGuidedLabelGenerator._dino_similarity method with its minimal field owner.
The historical recipe's stride is 128; minibatch 1 is an explicit memory cap,
not the historical minibatch 16. Run only when the GPU is available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace


EXPECTED_CHECKPOINT_SHA256 = "e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59"
EXPECTED_REFERENCE_SHA256 = "61bdf93bc5e3fd956eebdbed52618985d27264b8a9e5cb043087d0f234507a81"
EXPECTED_SOURCE_HEAD = "f07d33be6a00d12ace7d6a9465efe17c78ed7b47"
CUDA_ALLOCATOR_FRACTION = 0.85


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def add_input_arguments(parser):
    parser.add_argument("--vesuvius-src", type=Path, required=True,
                        help="vesuvius/src directory in the pinned Villa checkout.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official slim ps8 teacher checkpoint; its SHA-256 is fixed.")
    parser.add_argument("--reference", type=Path, required=True,
                        help="864-dimensional reference embedding .npy file.")
    parser.add_argument("--volume", type=Path, required=True,
                        help="Exact PHerc0139 256-cubed uint8 .npy volume; pixel SHA-256 is fixed.")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; an existing directory is refused.")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify inputs and strict CPU loading, without calling any CUDA API.")
    return parser.parse_args()


def strict_load(checkpoint_path, expected_digest, build_backbone, torch):
    digest = sha256_file(checkpoint_path)
    if digest != expected_digest:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise ValueError("Checkpoint must contain config mapping")
    config = payload["config"]
    if not isinstance(config.get("model"), Mapping) or not isinstance(payload.get("teacher"), Mapping):
        raise ValueError("Checkpoint must contain config.model and teacher mappings")
    model_config = dict(config["model"])
    model = build_backbone(model_config)
    expected = model.state_dict()
    teacher = payload["teacher"]
    if set(teacher) == set(expected):
        state, adapter = dict(teacher), "teacher keys already exactly match backbone"
    elif teacher and all(str(key).startswith("backbone.") for key in teacher):
        state = {str(key).removeprefix("backbone."): value for key, value in teacher.items()}
        adapter = "strip exactly one backbone. prefix from every teacher key"
    else:
        raise ValueError("Teacher keys are neither an exact backbone nor a uniformly prefixed backbone")
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatch = {key: {"expected": list(expected[key].shape),
                      "actual": list(state[key].shape) if isinstance(state[key], torch.Tensor)
                      else {"type": type(state[key]).__name__}}
                for key in expected.keys() & state.keys()
                if not isinstance(state[key], torch.Tensor) or expected[key].shape != state[key].shape}
    if missing or unexpected or mismatch:
        raise ValueError(json.dumps({"missing": missing, "unexpected": unexpected, "shape_mismatch": mismatch}))
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": str(checkpoint_path.resolve()), "sha256": digest, "bytes": checkpoint_path.stat().st_size,
        "step": payload.get("step"), "config": config, "config_sha256": canonical_hash(config),
        "model_config": model_config, "adapter": adapter, "strict_key_count": len(expected),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def memory_host():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {"rss_bytes": info.rss, "peak_working_set_bytes": getattr(info, "peak_wset", None)}
    except ImportError:
        return {"note": "psutil unavailable; host memory not measured"}


def run(args, report):
    report["source_clean_verified"] = False
    git = subprocess.run(["git", "-C", str(args.vesuvius_src), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    report["source_checkout_head"] = git.stdout.strip() if git.returncode == 0 else None
    if report["source_checkout_head"] != EXPECTED_SOURCE_HEAD:
        raise RuntimeError(f"Expected source checkout {EXPECTED_SOURCE_HEAD}, got {report['source_checkout_head']}")
    status = subprocess.run(
        ["git", "-C", str(args.vesuvius_src), "status", "--porcelain", "--untracked-files=all", "--", "."],
        capture_output=True, text=True, check=False)
    if status.returncode != 0:
        raise RuntimeError(f"Could not verify source cleanliness: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError(f"Source directory has tracked or untracked changes:\n{status.stdout}")
    report["source_clean_verified"] = True
    sys.path.insert(0, str(args.vesuvius_src.resolve()))
    import numpy as np
    import load_real_ct
    import torch
    import vesuvius.ink_detection.training.dynamic_labels as dynamic
    import vesuvius.ink_detection.data.normalization as normalization
    import vesuvius.ink_detection.config as ink_config
    import vesuvius.models.build.pretrained_backbones.dinovol_2_builder as builder
    import vesuvius.models.build.pretrained_backbones.dinovol_2_eva as eva
    import vesuvius.models.build.pretrained_backbones.rope as rope

    modules = (dynamic, normalization, ink_config, builder, eva, rope)
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(args.vesuvius_src.resolve()):
            raise RuntimeError(f"Source shadowing detected: {module.__file__}")
    report["source"] = {module.__name__: {
        "path": str(Path(module.__file__).resolve()), "sha256": sha256_file(Path(module.__file__))
    } for module in modules}
    report["source"]["driver"] = {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))}
    report["source"]["ct_loader"] = {"path": str(Path(load_real_ct.__file__).resolve()),
                                       "sha256": sha256_file(Path(load_real_ct.__file__))}
    report["consumer_method_sha256"] = hashlib.sha256(
        inspect.getsource(dynamic.DinoGuidedLabelGenerator._dino_similarity).encode()).hexdigest()
    report["versions"] = {"python": sys.version, "platform": platform.platform(), "torch_cuda": torch.version.cuda}
    for package in ("torch", "timm", "numpy", "tifffile", "einops", "zarr"):
        report["versions"][package] = importlib.metadata.version(package)
    report["protocol"] = {
        "scope": "frozen DINO similarity only; excludes U-Net, label threshold, student training and dataset sampling",
        "chunk_size": 256, "window_size": 128, "stride": 128, "patch_size": 8,
        "embedding_dim": 864, "image_batch": 1, "dino_minibatch": 1, "precision": "bfloat16",
        "precision_note": "Backbone and input explicitly bf16 as in the consumer; no extra autocast or compilation",
        "historical_recipe_note": "stride128 preserved; minibatch1 is a declared memory cap versus recipe minibatch16",
        "normalization": "whole256cube percentile_minmax p1/p99; no geometric or photometric augmentation",
        "blend_sigma": 4.0, "warmups": 2, "measured_repeats": 5, "profile_passes": 1,
        "source_model_unchanged": True, "output_transfers_excluded_from_measured_similarity": True,
        "inference_mode": True,
    }
    report["cuda_allocator"] = {"configured_fraction": CUDA_ALLOCATOR_FRACTION, "applied": False}
    device = None
    if not args.check_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU performance fallback")
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(CUDA_ALLOCATOR_FRACTION, device)
        report["cuda_allocator"]["applied"] = True
        report["cuda_allocator"]["applied_before_checkpoint_load"] = True
    start = time.perf_counter()
    raw, report["data"] = load_real_ct.load_volume(args.volume)
    report["input_read_hash_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    normalized = normalization.normalize_image(
        raw.astype(np.float32), ink_config.NormalizationConfig.from_value("percentile_minmax"))
    report["cpu_normalization_ms"] = (time.perf_counter() - start) * 1000
    if not np.isfinite(normalized).all() or normalized.min() < 0 or normalized.max() > 1:
        raise ValueError("Normalized CT fails finiteness/range invariant")
    report["data"]["normalized_sha256"] = hashlib.sha256(normalized.tobytes()).hexdigest()
    start = time.perf_counter()
    backbone, report["checkpoint"] = strict_load(
        args.checkpoint, EXPECTED_CHECKPOINT_SHA256, builder.build_dinovol_2_backbone, torch)
    report["checkpoint_hash_and_cpu_load_ms"] = (time.perf_counter() - start) * 1000
    reference_digest = sha256_file(args.reference)
    report["reference"] = {"path": str(args.reference.resolve()), "sha256": reference_digest}
    report["reference_verified"] = reference_digest == EXPECTED_REFERENCE_SHA256
    if not report["reference_verified"]:
        raise ValueError(f"Reference SHA-256 mismatch: {reference_digest}")
    ref = dynamic.load_reference_embedding(args.reference, embedding_dim=864, device="cpu", dtype=torch.bfloat16)
    report["reference"].update({"shape": list(ref.shape), "norm": float(ref.norm()), "dtype": str(ref.dtype)})
    grid = dynamic.DinoGridSpec()
    starts = dynamic.dino_window_starts(chunk_size=256, window_size=128, stride=128)
    if starts != [0, 128] or tuple(backbone.patch_size) != (8, 8, 8):
        raise ValueError("Input/window/checkpoint does not match ps8 historical grid")
    report["host_memory_after_load"] = memory_host()
    if args.check_only:
        report["status"] = "CPU_INPUT_AND_STRICT_CHECKPOINT_CHECK_PASSED_NO_CUDA_CALLS"
        return

    report["gpu"] = {"name": torch.cuda.get_device_name(device),
                     "total_memory": torch.cuda.get_device_properties(device).total_memory,
                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                     "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    backbone = backbone.to(device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    report["backbone_device_transfer_ms"] = (time.perf_counter() - start) * 1000
    cpu_input = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    image = cpu_input.to(device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    report["input_device_transfer_ms"] = (time.perf_counter() - start) * 1000
    reference_on_device = ref.to(device=device, dtype=torch.float32)
    reference_on_device = reference_on_device / reference_on_device.norm().clamp_min(1e-12)
    report["reference"]["constructor_device_renormalization"] = True
    report["reference"]["device_norm"] = float(reference_on_device.norm())
    owner = SimpleNamespace(device=device, dtype=torch.bfloat16, grid=grid, starts=starts,
                            dino_minibatch=1, dino=backbone,
                            reference_embedding=reference_on_device,
                            weight=dynamic.gaussian_window_3d(16, 4.0).to(device))

    @torch.inference_mode()
    def invoke():
        return dynamic.DinoGuidedLabelGenerator._dino_similarity(owner, image)

    def timed_call():
        torch.cuda.synchronize(device)
        begun = time.perf_counter()
        result = invoke()
        torch.cuda.synchronize(device)
        return result, (time.perf_counter() - begun) * 1000

    def checked_output(result):
        array = result.detach().float().cpu().numpy()
        if array.shape != (1, 1, 256, 256, 256) or not np.isfinite(array).all():
            raise ValueError(f"Invalid similarity shape/finiteness: {array.shape}")
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Similarity output outside [0,1]")
        return array

    report["warmup_ms"] = []
    for _ in range(2):
        result, duration = timed_call()
        checked_output(result)
        report["warmup_ms"].append(duration)
        del result
    torch.cuda.reset_peak_memory_stats(device)
    report["measured_ms"], report["repeat_errors"] = [], []
    reference_output = None
    for index in range(5):
        result, duration = timed_call()
        array = checked_output(result)
        report["measured_ms"].append(duration)
        if reference_output is None:
            reference_output = array.copy()
            output_path = args.output / "similarity_first.npy"
            np.save(output_path, reference_output)
            report["output"] = {"path": str(output_path.resolve()), "sha256": sha256_file(output_path),
                                "shape": list(array.shape), "dtype": str(array.dtype),
                                "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}
        delta = np.abs(array - reference_output)
        report["repeat_errors"].append({"repeat": index, "max_abs": float(delta.max()),
                                        "mean_abs": float(delta.mean()), "exact_equal": bool(np.array_equal(array, reference_output))})
        print(json.dumps({"pass": index + 1, "milliseconds": duration}), flush=True)
        del result, array, delta
    report["timing_summary"] = {"median_ms": float(np.median(report["measured_ms"])),
                                "p90_ms": float(np.percentile(report["measured_ms"], 90)),
                                "windows_per_pass": 8, "tokens_per_pass": 8 * 4096}
    report["gpu_memory_timed"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

    class TracedBackbone:
        def forward_features(self, *values, **kwargs):
            with torch.profiler.record_function("production_backbone.forward_features"):
                output = backbone.forward_features(*values, **kwargs)
            tokens = output["x_norm_patchtokens"]
            if tuple(tokens.shape) != (1, 4096, 864):
                raise ValueError(f"Unexpected patch tokens: {tuple(tokens.shape)}")
            return output

    owner.dino = TracedBackbone()  # Identical model call, only a profiler scope.
    torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True, with_stack=False) as profiler:
        with torch.profiler.record_function("production_dino_similarity"):
            result = invoke()
        torch.cuda.synchronize(device)
    profile_array = checked_output(result)
    delta = np.abs(profile_array - reference_output)
    report["profile_output_error"] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    events = profiler.key_averages()
    table = events.table(sort_by="self_device_time_total", row_limit=20)
    (args.output / "top20_self_cuda.txt").write_text(table, encoding="utf-8")
    operators = [{"name": event.key, "count": event.count,
                  "self_cuda_us": float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))),
                  "self_cpu_us": float(event.self_cpu_time_total)} for event in events]
    report["top20_self_cuda_operators"] = sorted(operators, key=lambda item: item["self_cuda_us"], reverse=True)[:20]
    report["gpu_memory_profile"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    report["host_memory_final"] = memory_host()
    report["status"] = "BASELINE_PROFILE_COMPLETE_NO_OPTIMIZATION"


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTED", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "argv": sys.argv}
    try:
        run(args, report)
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
