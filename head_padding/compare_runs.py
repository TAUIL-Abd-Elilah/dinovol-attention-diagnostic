"""CPU-only comparison of frozen Dinovol padding runs; never imports Torch.

Inputs are mmap'ed; reductions use at most CHUNK float64 entries at once.
Exact error quantiles partition a temporary disk-backed float64 vector in place.
Token cosines are computed independently for every 864-dimensional patch token.
"""
import argparse
import collections
import hashlib
import json
import math
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

CHUNK = 262144
TOKENS = [f"window_{i}_tokens.npy" for i in range(8)]
STAGES = {"tokens": TOKENS, "blended_grid": ["blended_grid.npy"],
          "interpolated_before_minmax": ["interpolated_before_minmax.npy"],
          "similarity_first": ["similarity_first.npy"]}
OPERATORS = ("aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention_math",
             "aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
             "aten::_scaled_dot_product_cudnn_attention")
EXPECTED_COUNTS = {"sdpa": 1536, "windows": 64, "map_interpolations": 8, "captured_windows": 8}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_events(path):
    """Stream Chrome trace event objects without retaining the complete trace."""
    decoder, buffer = json.JSONDecoder(), ""
    with path.open(encoding="utf-8") as stream:
        while '"traceEvents"' not in buffer:
            block = stream.read(65536)
            if not block:
                raise ValueError("Missing traceEvents")
            buffer += block
        buffer = buffer[buffer.index('"traceEvents"') + len('"traceEvents"'):]
        while "[" not in buffer:
            buffer += stream.read(65536)
        buffer = buffer[buffer.index("[") + 1:]
        while True:
            buffer = buffer.lstrip(" \n\t\r,")
            if buffer.startswith("]"):
                return
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(65536)
                if not block:
                    raise ValueError("Truncated profiler trace")
                buffer += block
                if len(buffer) > 16 * 1024**2:
                    raise ValueError("Unexpected trace event larger than bounded parser buffer")
                continue
            yield event
            buffer = buffer[end:]


def backend_counts(path):
    # Same exact-name/ph/category rule as the existing summarize_trace.py.
    counts = collections.Counter(e.get("name") for e in trace_events(path)
                                 if e.get("ph") == "X" and e.get("cat") == "cpu_op")
    selected = {name: counts[name] for name in OPERATORS}
    if selected[OPERATORS[0]] != 192 or sum(selected[name] for name in OPERATORS[1:]) != 192:
        raise ValueError(f"Unexpected actual attention counts in {path}: {selected}")
    return {"trace_sha256": sha(path), "count_rule": "Complete-duration (ph=X) CPU operator events, exact names",
            "operator_counts": selected}


def close_array(array):
    array._mmap.close()


def validate_run(directory):
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "FULL_EXPERIMENT_COMPLETE" or report["counts"] != EXPECTED_COUNTS:
        raise ValueError(f"Incomplete or changed protocol in {directory}")
    if report["production_source_modified"] or not report["source_clean_verified"]:
        raise ValueError("Production source was modified")
    hashes = {}
    for name in sum(STAGES.values(), []):
        path = directory / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (1, 4096, 864) if name in TOKENS else ((1, 1, 32, 32, 32) if name == "blended_grid.npy" else (1, 1, 256, 256, 256))
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"Unexpected array contract in {path}: {array.shape}, {array.dtype}")
        close_array(array)
        hashes[name] = sha(path)
    if hashes["similarity_first.npy"] != report["output"]["sha256"]:
        raise ValueError("Saved final output hash differs from run report")
    return report, {"directory": str(directory.resolve()), "report_sha256": sha(directory / "report.json"),
                    "array_sha256": hashes, "backend": backend_counts(directory / "trace.json"),
                    "mode": report["mode"], "timing_summary": report["timing_summary"],
                    "measured_ms": report["measured_ms"], "gpu_memory_timed": report["gpu_memory_timed"],
                    "counts": report["counts"], "pre_minmax": report["pre_minmax"],
                    "experiment_sha256": report["experiment_sha256"], "helper": report.get("helper"),
                    "fixture_loader_override": report.get("fixture_loader_override"),
                    "profile_output_error": report["profile_output_error"]}


def shared_contract(a, b):
    fields = ["source_checkout_head", "consumer_method_sha256", "versions", "protocol", "cuda_allocator",
              "experiment_sha256", "gpu"]
    for field in fields:
        if a[field] != b[field]:
            raise ValueError(f"Run contract differs: {field}")
    for field in ("checkpoint", "reference"):
        if a[field]["sha256"] != b[field]["sha256"]:
            raise ValueError(f"Input hash differs: {field}")
    for field in ("volume_file_sha256", "array_sha256", "normalized_sha256"):
        if a["data"][field] != b["data"][field]:
            raise ValueError(f"Data contract differs: {field}")
    # The original loader uses grid_box_l0_zyx/source; the additional frozen
    # manifest uses bbox_zyx/source_uri. Match all provenance within each pair
    # instead of silently dropping unknown fields or requiring one schema.
    for record in (a["data"], b["data"]):
        if not any(key in record for key in ("grid_box_l0_zyx", "bbox_zyx")):
            raise ValueError("Missing physical sample bounding box")
        if not any(key in record for key in ("source", "source_uri")):
            raise ValueError("Missing physical sample source")
    if {k: v for k, v in a["data"].items() if k != "volume_path"} != {
            k: v for k, v in b["data"].items() if k != "volume_path"}:
        raise ValueError("Data provenance differs between paired runs")
    for field in ("helper", "fixture_loader_override"):
        if a.get(field) != b.get(field):
            raise ValueError(f"Run contract differs: {field}")
    if {k: v["sha256"] for k, v in a["source"].items()} != {k: v["sha256"] for k, v in b["source"].items()}:
        raise ValueError("Source hashes differ")


def exact_percentiles_inplace(values, percentiles):
    indices = [(len(values) - 1) * p / 100 for p in percentiles]
    needed = sorted({i for index in indices for i in (math.floor(index), math.ceil(index))})
    values.partition(needed)
    return [float(values[math.floor(index)] * (1 - (index % 1)) + values[math.ceil(index)] * (index % 1))
            for index in indices]


def stage_metrics(base_dir, cand_dir, names):
    sizes = []
    for name in names:
        array = np.load(base_dir / name, mmap_mode="r", allow_pickle=False)
        sizes.append(array.size)
        close_array(array)
    total = sum(sizes)
    absolute_sum = squared_sum = reference_squared_sum = 0.0
    max_abs, different, offset = 0.0, 0, 0
    ranges = {"baseline_min": math.inf, "baseline_max": -math.inf,
              "candidate_min": math.inf, "candidate_max": -math.inf}
    with tempfile.TemporaryDirectory(prefix="dinovol-comparison-") as scratch:
        errors = np.memmap(Path(scratch) / "errors.dat", dtype=np.float64, mode="w+", shape=(total,))
        for name, size in zip(names, sizes):
            a = np.load(base_dir / name, mmap_mode="r", allow_pickle=False)
            b = np.load(cand_dir / name, mmap_mode="r", allow_pickle=False)
            af, bf = a.reshape(-1), b.reshape(-1)
            for start in range(0, size, CHUNK):
                av = np.asarray(af[start:start + CHUNK], dtype=np.float64)
                bv = np.asarray(bf[start:start + CHUNK], dtype=np.float64)
                if not np.isfinite(av).all() or not np.isfinite(bv).all():
                    raise ValueError(f"Nonfinite values in {name}")
                delta = bv - av
                diff = np.abs(delta)
                errors[offset + start:offset + start + len(diff)] = diff
                max_abs = max(max_abs, float(diff.max()))
                absolute_sum += float(diff.sum(dtype=np.float64))
                squared_sum += float(np.dot(delta, delta))
                reference_squared_sum += float(np.dot(av, av))
                different += int(np.count_nonzero(delta))
                ranges["baseline_min"] = min(ranges["baseline_min"], float(av.min()))
                ranges["baseline_max"] = max(ranges["baseline_max"], float(av.max()))
                ranges["candidate_min"] = min(ranges["candidate_min"], float(bv.min()))
                ranges["candidate_max"] = max(ranges["candidate_max"], float(bv.max()))
            offset += size
            del af, bf
            close_array(a); close_array(b)
        p99, p999 = exact_percentiles_inplace(errors, [99, 99.9])
        close_array(errors)
    return {"files": names, "element_count": total, "finite": True, "different_elements": different,
            "exact_equal": different == 0, "max_abs": max_abs, "mean_abs": absolute_sum / total,
            "rmse": math.sqrt(squared_sum / total), "p99_abs": p99, "p99_9_abs": p999,
            "relative_l2": math.sqrt(squared_sum / reference_squared_sum) if reference_squared_sum else None,
            **ranges}


def token_cosines(base_dir, cand_dir):
    windows, all_cosines = [], []
    for name in TOKENS:
        a = np.load(base_dir / name, mmap_mode="r", allow_pickle=False)
        b = np.load(cand_dir / name, mmap_mode="r", allow_pickle=False)
        cosines = []
        for start in range(0, 4096, 128):
            av, bv = np.asarray(a[0, start:start + 128], dtype=np.float64), np.asarray(b[0, start:start + 128], dtype=np.float64)
            denom = np.sqrt(np.einsum("ij,ij->i", av, av) * np.einsum("ij,ij->i", bv, bv))
            if np.any(denom == 0):
                raise ValueError("Undefined cosine for zero-norm token")
            cosines.extend((np.einsum("ij,ij->i", av, bv) / denom).tolist())
        windows.append({"file": name, "per_token_cosine": cosines})
        all_cosines.extend(cosines)
        close_array(a); close_array(b)
    values = np.asarray(all_cosines)
    return {"definition": "dot(baseline,candidate)/(L2baseline*L2candidate) across embedding dimension, separately for each patch token; float64 reductions",
            "token_count": len(all_cosines), "min": float(values.min()), "mean": float(values.mean()),
            "median": float(np.median(values)), "p1": float(np.percentile(values, 1)),
            "p0_1": float(np.percentile(values, 0.1)), "windows": windows}


def margin_summary(values):
    if not len(values):
        return None
    return {"min_signed": float(values.min()), "max_signed": float(values.max()),
            "mean_signed": float(values.mean()), "median_abs": float(np.median(np.abs(values))),
            "max_abs": float(np.abs(values).max()), "p99_abs": float(np.percentile(np.abs(values), 99))}


def eligibility(base_dir, cand_dir):
    a = np.load(base_dir / "similarity_first.npy", mmap_mode="r", allow_pickle=False)
    b = np.load(cand_dir / "similarity_first.npy", mmap_mode="r", allow_pickle=False)
    af, bf = a.reshape(-1), b.reshape(-1)
    counts = collections.Counter()
    # At most two scalar margins for each voxel, allocated on disk. Only
    # changed entries are populated; unchanged voxels never enter summaries.
    with tempfile.TemporaryDirectory(prefix="dinovol-margins-") as scratch:
        margins = np.memmap(Path(scratch) / "margins.dat", dtype=np.float64, mode="w+", shape=(4, a.size))
        for start in range(0, a.size, CHUNK):
            av, bv = af[start:start + CHUNK], bf[start:start + CHUNK]
            ap, bp = av > 0.5, bv > 0.5
            counts["baseline_positive"] += int(np.count_nonzero(ap))
            counts["candidate_positive"] += int(np.count_nonzero(bp))
            counts["baseline_exactly_threshold"] += int(np.count_nonzero(av == 0.5))
            counts["candidate_exactly_threshold"] += int(np.count_nonzero(bv == 0.5))
            for direction, mask, row in (("ineligible_to_eligible", ~ap & bp, 0), ("eligible_to_ineligible", ap & ~bp, 2)):
                n = int(np.count_nonzero(mask)); offset = counts[direction]
                margins[row, offset:offset + n] = av[mask].astype(np.float64) - 0.5
                margins[row + 1, offset:offset + n] = bv[mask].astype(np.float64) - 0.5
                counts[direction] += n
        flipped = counts["ineligible_to_eligible"] + counts["eligible_to_ineligible"]
        margin_stats = {}
        for direction, row in (("ineligible_to_eligible", 0), ("eligible_to_ineligible", 2)):
            n = counts[direction]
            # Usually small; hard bound is the 256^3 voxel count. Summaries are
            # evaluated one direction/array at a time, not all maps in RAM.
            margin_stats[direction] = {"baseline_minus_threshold": margin_summary(margins[row, :n]),
                                       "candidate_minus_threshold": margin_summary(margins[row + 1, :n])}
        close_array(margins)
    result = {"definition": "strict similarity_first > 0.5; eligibility only, not production pseudo-label",
              "threshold": 0.5, "voxel_count": int(a.size), **dict(counts), "flipped": flipped,
              "flipped_fraction_all_voxels": flipped / a.size,
              "baseline_positive_fraction": counts["baseline_positive"] / a.size,
              "candidate_positive_fraction": counts["candidate_positive"] / a.size,
              "flips_relative_to_baseline_positive_count": flipped / counts["baseline_positive"] if counts["baseline_positive"] else None,
              "flip_margins": margin_stats}
    del af, bf
    close_array(a); close_array(b)
    return result


def compare(base_dir, cand_dir, base_report, base_provenance):
    report, provenance = validate_run(cand_dir)
    shared_contract(base_report, report)
    stages = {key: stage_metrics(base_dir, cand_dir, names) for key, names in STAGES.items()}
    token_details = {name: stage_metrics(base_dir, cand_dir, [name]) for name in TOKENS}
    median_ratio = report["timing_summary"]["median_ms"] / base_report["timing_summary"]["median_ms"]
    allocated_ratio = report["gpu_memory_timed"]["peak_allocated_bytes"] / base_report["gpu_memory_timed"]["peak_allocated_bytes"]
    return {"candidate": provenance, "same_input_model_and_source_contract": True,
            "stages": stages, "per_window_token_errors": token_details,
            "token_cosines": token_cosines(base_dir, cand_dir), "eligibility": eligibility(base_dir, cand_dir),
            "all_stages_exact_equal": all(s["exact_equal"] for s in stages.values()),
            "timing": {"median_speedup": 1 / median_ratio, "median_reduction_fraction": 1 - median_ratio,
                       "p90_ratio": report["timing_summary"]["p90_ms"] / base_report["timing_summary"]["p90_ms"],
                       "peak_allocated_ratio": allocated_ratio,
                       "peak_reserved_ratio": report["gpu_memory_timed"]["peak_reserved_bytes"] / base_report["gpu_memory_timed"]["peak_reserved_bytes"],
                       "frozen_performance_screen_pass": median_ratio <= 0.8 and allocated_ratio <= 1,
                       "scope": "Five timed complete-similarity calls after two warmups; separate fresh processes in fixed order; excludes model startup, CPU transfers, U-Net and training"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).parent / "full_unchanged_01")
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    report, provenance = validate_run(args.baseline)
    comparisons = [compare(args.baseline, directory, report, provenance) for directory in args.candidate]
    output = {"status": "COMPARISON_COMPLETE", "script_sha256": sha(Path(__file__)),
              "plan_sha256": sha(Path(__file__).parent / "FULL_COMPARISON_PLAN.md"),
              "additional_plans_sha256": {name: sha(Path(__file__).parent / name) for name in
                  ("HELPER_REPLICATION_PLAN.md", "ADDITIONAL_INPUT_PLAN.md") if (Path(__file__).parent / name).exists()},
              "python": platform.python_version(), "numpy": np.__version__, "cpu_only": True,
              "method": "mmap inputs; chunked float64 reductions; exact linearly interpolated percentiles using disk-backed in-place partition; fixed threshold0.5",
              "input_provenance": {"checkpoint_sha256": report["checkpoint"]["sha256"],
                  "reference_sha256": report["reference"]["sha256"], "data": report["data"],
                  "source_checkout_head": report["source_checkout_head"], "protocol": report["protocol"]},
              "baseline": provenance, "comparisons": comparisons,
              "production_label_equivalence": "UNTESTED: exact phase-specific60k/63k frozen U-Net teachers unavailable; similarity eligibility is not sigmoid(U-Net)*similarity >0.5, and v2 additionally requires raw>50",
              "elapsed_seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for item in comparisons:
        print(json.dumps({"mode": item["candidate"]["mode"], "timing": item["timing"],
                          "final_error": item["stages"]["similarity_first"],
                          "eligibility": item["eligibility"], "all_stages_exact_equal": item["all_stages_exact_equal"]}))


if __name__ == "__main__":
    main()
