#!/usr/bin/env python3
"""Audit raw measurements and write the depth-sweep report; standard library only."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def key(row):
    return int(row["batch"]), row["mode"], int(row["depth"]), row["telemetry"], int(row["round"])


def close(actual, expected, label):
    assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), (label, actual, expected)


def median(rows, field):
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def analyze(root, expected_trials, kind):
    rows = read_csv(root / "raw.csv")
    expected_configs = {(mode, depth, telemetry)
                        for mode, depth, telemetry in [("serial_pageable", 1, "sampling"),
                                                      ("serial_pinned", 1, "sampling"),
                                                      ("serialized_8", 8, "sampling")]}
    expected_configs |= {("pipeline", depth, telemetry)
                         for depth in (1, 2, 4, 8) for telemetry in ("sampling", "off")}
    expected_keys = {(batch, mode, depth, telemetry, trial)
                     for batch in (1, 16, 128) for mode, depth, telemetry in expected_configs
                     for trial in range(expected_trials)}
    assert len(rows) == len(expected_keys), "Missing or extra trials"
    assert {key(row) for row in rows} == expected_keys, "Missing or duplicate configuration"
    for batch in (1, 16, 128):
        for trial in range(expected_trials):
            orders = [int(row["order"]) for row in rows
                      if int(row["batch"]) == batch and int(row["round"]) == trial]
            assert sorted(orders) == list(range(1, 12)), "Invalid execution order"

    by_key = {key(row): row for row in rows}
    power_groups = defaultdict(list)
    for sample in read_csv(root / "power.csv"):
        k = key(sample)
        assert k in by_key, "Unexpected power sample"
        row = by_key[k]
        inside = float(row["t0_s"]) <= float(sample["steady_s"]) <= float(row["t1_s"])
        assert int(sample["in_window"]) == int(inside), "Incorrect window flag"
        assert math.isfinite(float(sample["power_w"])) and float(sample["power_w"]) > 0
        if inside:
            power_groups[k].append(sample)

    latency_groups = defaultdict(list)
    for sample in read_csv(root / "latency.csv"):
        k = key(sample)
        assert k in by_key, "Unexpected latency sample"
        request = int(sample["request_id"])
        assert request % 31 == 0 and 0 <= request < int(by_key[k]["completed_requests"])
        assert int(sample["input_bank"]) == request % 9, "Wrong input bank"
        assert int(sample["slot"]) == request % int(sample["depth"]), "Wrong buffer slot"
        assert math.isfinite(float(sample["host_observed_ms"])) and float(sample["host_observed_ms"]) >= 0
        latency_groups[k].append(sample)

    valid_power = 0
    for row in rows:
        k = key(row)
        n, depth = int(row["completed_requests"]), int(row["depth"])
        seconds = float(row["seconds"])
        assert n >= depth and seconds > 0
        close(float(row["t1_s"]) - float(row["t0_s"]), seconds, "elapsed time")
        close(n / seconds, float(row["requests_s"]), "throughput")
        close(int(row["batch"]) * n / seconds, float(row["inputs_s"]), "vector throughput")
        close(seconds * 1000 / n, float(row["amortized_ms"]), "amortized time")
        assert int(row["mismatches"]) == 0
        assert int(row["stale_control_passed"]) == int(row["injected_control_passed"]) == 1
        assert int(row["checked_requests"]) == 256 + depth
        batch = int(row["batch"])
        xb, yb, wb = batch * 4096, batch * 512 * 4, 512 * 4096
        close(int(row["device_tensor_bytes"]), wb + depth * (xb + yb), "device buffers")
        close(int(row["pinned_host_bytes"]), 0 if row["mode"] == "serial_pageable"
              else 9 * xb + depth * yb, "pinned buffers")
        samples = latency_groups[k]
        assert len(samples) == int(row["latency_samples"]) == (n + 30) // 31
        assert sorted(int(sample["request_id"]) for sample in samples) == list(range(0, n, 31))
        values = sorted(float(sample["host_observed_ms"]) for sample in samples)
        close(values[(len(values) - 1) // 2], float(row["host_p50_ms"]), "latency p50")
        close(values[int(0.95 * (len(values) - 1))], float(row["host_p95_ms"]), "latency p95")
        samples = sorted(power_groups[k], key=lambda sample: float(sample["steady_s"]))
        assert len(samples) == int(row["power_samples"])
        integral, span, max_gap = 0.0, 0.0, 0.0
        for previous, current in zip(samples, samples[1:]):
            dt = float(current["steady_s"]) - float(previous["steady_s"])
            assert dt > 0
            span += dt
            integral += dt * (float(previous["power_w"]) + float(current["power_w"])) / 2
            max_gap = max(max_gap, dt)
        coverage = span / seconds
        close(coverage, float(row["power_coverage"]), "power coverage")
        close(max_gap, float(row["power_max_gap_s"]), "power gap")
        valid = len(samples) >= 20 and coverage >= 0.9 and max_gap <= 0.5
        if row["telemetry"] == "off":
            assert not samples, "Disabled telemetry still queried power"
        if valid:
            valid_power += 1
            close(integral / span, float(row["gpu_mean_w"]), "mean power")
            close(integral / span * seconds, float(row["gpu_est_j"]), "energy")
            close(integral / span * seconds * 1000 / n,
                  float(row["gpu_est_mj_per_request"]), "energy per request")
        else:
            assert all(math.isnan(float(row[field])) for field in
                       ("gpu_mean_w", "gpu_est_j", "gpu_est_mj_per_request"))

    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)[:4]].append(row)
    summary = []
    for k in sorted(grouped):
        group = grouped[k]
        summary.append(dict(batch=k[0], mode=k[1], depth=k[2], telemetry=k[3], trials=len(group),
                            requests_s=median(group, "requests_s"), inputs_s=median(group, "inputs_s"),
                            p50_ms=median(group, "host_p50_ms"), p95_ms=median(group, "host_p95_ms"),
                            energy_mj=median(group, "gpu_est_mj_per_request"),
                            min_requests_s=min(float(r["requests_s"]) for r in group),
                            max_requests_s=max(float(r["requests_s"]) for r in group),
                            power_trials=sum(math.isfinite(float(r["gpu_mean_w"])) for r in group),
                            device_bytes=int(group[0]["device_tensor_bytes"]),
                            pinned_bytes=int(group[0]["pinned_host_bytes"])))

    get = lambda batch, mode, depth, telemetry: next(
        cell for cell in summary if (cell["batch"], cell["mode"], cell["depth"], cell["telemetry"])
        == (batch, mode, depth, telemetry))
    findings = []
    for batch in (1, 16, 128):
        serial = max((get(batch, mode, 1, "sampling") for mode in ("serial_pageable", "serial_pinned")),
                     key=lambda cell: cell["requests_s"])
        pipeline = [get(batch, "pipeline", depth, "sampling") for depth in (1, 2, 4, 8)]
        best = max(pipeline, key=lambda cell: cell["requests_s"])
        depth2 = get(batch, "pipeline", 2, "sampling")
        findings.append(dict(batch=batch, best_depth=best["depth"], baseline=serial["mode"],
                             throughput_ratio=best["requests_s"] / serial["requests_s"],
                             best_vs_depth2=best["requests_s"] / depth2["requests_s"],
                             p95_ratio=best["p95_ms"] / serial["p95_ms"],
                             energy_ratio=None if best["energy_mj"] is None or serial["energy_mj"] is None
                             else best["energy_mj"] / serial["energy_mj"]))
    overhead = []
    for batch in (1, 16, 128):
        for depth in (1, 2, 4, 8):
            pairs = [100 * (float(by_key[batch, "pipeline", depth, "sampling", trial]["requests_s"])
                            / float(by_key[batch, "pipeline", depth, "off", trial]["requests_s"]) - 1)
                     for trial in range(expected_trials)]
            overhead.append(dict(batch=batch, depth=depth, median_sampling_change_pct=statistics.median(pairs),
                                 min_pct=min(pairs), max_pct=max(pairs), paired_changes_pct=pairs))

    audit = dict(trials=len(rows), completed_requests=sum(int(r["completed_requests"]) for r in rows),
                 checked_requests=sum(int(r["checked_requests"]) for r in rows),
                 valid_power_trials=valid_power,
                 sampling_trials=sum(r["telemetry"] == "sampling" for r in rows),
                 latency_samples=sum(int(r["latency_samples"]) for r in rows),
                 raw_audit_passed=True, kind=kind)
    for name in ("pipeline_depth.cu", "dp4a_benchmark.cu", "run_pipeline_depth.sh", "analyze_pipeline_depth.py"):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in (root / "SHA256SUMS").read_text(), "Source snapshot hash mismatch"
    with (root / "summary.json").open("w", encoding="utf-8") as destination:
        json.dump(dict(audit=audit, cells=summary, findings=findings, telemetry=overhead),
                  destination, indent=2, allow_nan=False)
        destination.write("\n")

    report = ["# Pipeline Depth and Power-Query Overhead", "",
              "Status: complete; raw-data audit and all correctness controls passed.", "",
              f"Run type: {kind}. Trials: {len(rows)}. Checked requests: {audit['checked_requests']:,}.", "",
              "## Method", "",
              "RTX 2060 / WSL, cuBLAS integer GEMM, 4096 inputs and 512 outputs per vector; "
              "int8 [-8,7] inputs/weights and int32 outputs. Seed 12345. Nine distinct input banks "
              "ensure each slot sees changing inputs even at depth 8. Weights stay GPU-resident.", "",
              "Depths 1/2/4/8 use independent nonblocking streams, cuBLAS handles, device input/output "
              "buffers and pinned host outputs. Both serial baselines and depth-8 forced-serialization "
              "control use the same GEMM. Allocation, reference generation and input preparation are excluded.", "",
              "Five trials in the full run, shuffled method-group order with seed 20261010. "
              "Sampling/off pairs are adjacent with alternating within-pair order. Each full cell has "
              "0.5 s warm-up and at least 3 s timing, with final drain inside timing. Closed-loop saturation; "
              "no externally arriving request queue or full-model inference.", "",
              "Host-observed latency runs from submission until completion is observed at buffer reuse. "
              "A result may already have finished before it is observed. It is not GPU execution time, "
              "amortized throughput, or open-loop service latency. Samples every 31 requests cover all "
              "slots and banks. Displayed quantiles are medians of trial-level quantiles.", "",
              "Sampling calls NVML about every 100 ms. Off uses the same sleeping worker without NVML calls: "
              "the paired comparison isolates query overhead, not all monitoring/instrumentation overhead. "
              "GPU board energy is a gross estimate, includes other GPU activity, and excludes CPU and wall "
              "electricity. Trapezoidal time-weighted mean is extrapolated across the timing window; "
              "require >=20 samples, >=90% coverage, and <=0.5 s maximum gap. No idle subtraction.", "",
              "128 fully checked sequence requests before and after timing plus final buffers per cell. "
              "Timed requests are not all checked. Wrong-bank stale output and one-bit host-output corruption "
              "must be rejected. Submitted/completed counts must match. No direct profiling proof of overlap.", "",
              "Buffer bytes below exclude library workspaces, CUDA context/allocator overhead and CPU reference "
              "arrays; they are not total process memory or peak VRAM.", "",
              "## Results", "",
              "| Batch | Mode | Depth | Telemetry | Requests/s | p50 ms | p95 ms | GPU mJ/request | Device MiB | Pinned MiB | Valid power |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in summary:
        report.append(f"| {cell['batch']} | {cell['mode']} | {cell['depth']} | {cell['telemetry']} | "
                      f"{fmt(cell['requests_s'], 1)} | {fmt(cell['p50_ms'])} | {fmt(cell['p95_ms'])} | "
                      f"{fmt(cell['energy_mj'])} | {cell['device_bytes'] / 2**20:.3f} | "
                      f"{cell['pinned_bytes'] / 2**20:.3f} | {cell['power_trials']}/{cell['trials']} |")
    report += ["", "## Depth Selection", "",
               "Descriptive best median among tested depths, not a statistically proven optimum or universal setting.", ""]
    if kind == "smoke":
        report.append("Smoke run only: short timings validate execution and auditing, not performance conclusions.")
    else:
        for finding in findings:
            report.append(f"- Batch {finding['batch']}: best measured throughput at depth {finding['best_depth']}; "
                          f"{finding['throughput_ratio']:.3f}x versus {finding['baseline']}, "
                          f"{finding['best_vs_depth2']:.3f}x versus depth 2. "
                          f"Host p95 ratio {finding['p95_ratio']:.3f}x. "
                          f"GPU energy ratio {fmt(finding['energy_ratio'])}x.")
    report += ["", "## Power-Query Control", "",
               "Paired percent change in throughput with sampling versus off; negative means sampling slower. "
               "Ranges show run variability and may include scheduling, temperature and other-GPU-work effects.", "",
               "| Batch | Depth | Median change % | Minimum % | Maximum % |",
               "|---|---:|---:|---:|---:|"]
    for item in overhead:
        report.append(f"| {item['batch']} | {item['depth']} | {item['median_sampling_change_pct']:.2f} | "
                      f"{item['min_pct']:.2f} | {item['max_pct']:.2f} |")
    report += ["", "## Raw Audit", "", f"```json\n{json.dumps(audit, indent=2)}\n```", "",
               "The independent Python audit reconstructs throughput, sampled latency quantiles, "
               "power-window flags, energy integration, buffer counts and the expected trial grid "
               "from CSVs, and verifies source snapshot hashes.", "",
               "## Reproduction", "",
               "From this result folder in WSL (creates a new result folder, preserving this run):", "",
               "```bash\nbash run_pipeline_depth.sh full\n```", "",
               "Short correctness check: `bash run_pipeline_depth.sh smoke`.", "",
               "The wrapper compiles the frozen source, records environment/compiler details and checksums, "
               "runs the benchmark, and audits the raw data. Requires CUDA 13.2 at /usr/local/cuda-13.2, "
               "WSL GPU access and Python 3. See environment.txt, benchmark_metadata.txt, build.txt and run.txt.", "",
               "## References", "",
               "- CUDA streams and pinned transfers: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-best-practices-guide/index.html",
               "- NVML power query: https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html", ""]
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Report: {root / 'REPORT.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--kind", choices=("full", "smoke"), required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.expected_trials, args.kind)
