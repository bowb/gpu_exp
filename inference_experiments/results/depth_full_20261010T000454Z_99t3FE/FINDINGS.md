# Findings: How Deep Should the GPU Pipeline Be?

This benchmark was compiled and run locally through PowerShell into
Ubuntu-24.04-Test on the RTX 2060. It tests scheduling of integer matrix
multiplication, not a full AI model or a new mathematical algorithm.

## Verification

- 165 timed trials completed; five repetitions per configuration.
- 3,715,516 timed requests completed, including the final drain.
- 42,840 request-level correctness checks: zero mismatches.
- Wrong-bank stale outputs and deliberately flipped output bits were rejected.
- All 105 power-sampling trials met the coverage and sampling-gap requirements.
- Independent CSV audit reconstructed throughput, sampled latency quantiles,
  power integration, buffer sizes, trial coverage, and source hashes.
- Twelve additional auditor tests passed: intact data accepted; deliberately
  corrupted copies rejected. Tests used the short smoke run as their fixture.
- Timed outputs were not all checked. Power estimates cover GPU board energy
  only, including other GPU work; CPU and whole-PC electricity were not measured.

## Main Results

Medians of five trials with power sampling enabled, relative to the fastest
serial baseline for each batch. Each request contains the stated number of vectors.

| Batch | Best Measured Depth | Serial Requests/s | Pipeline Requests/s | Throughput Gain | GPU Energy/Request Change | Serial p95 ms | Pipeline p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 6,483 | 11,565 | +78.4% | -37.5% | 0.231 | 0.734 |
| 16 | 4 | 6,223 | 10,574 | +69.9% | +8.8% | 0.231 | 0.669 |
| 128 | 8 | 3,984 | 10,152 | +154.8% | -52.4% | 0.351 | 1.533 |

Best measured depth is descriptive, not a statistically proven optimum.
Trial variability and overlapping ranges matter, especially for small differences.

## The Useful Trade-Off at Batch 128

| Depth | Requests/s | Host p95 ms | GPU mJ/Request | Device Tensor MiB | Pinned Host MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 3,961 | 0.303 | 14.648 | 2.75 | 4.75 |
| 2 | 7,221 | 0.399 | 8.999 | 3.50 | 5.00 |
| 4 | 9,657 | 0.725 | 7.223 | 5.00 | 5.50 |
| 8 | 10,152 | 1.533 | 6.988 | 8.00 | 6.50 |

Depth 8 gained only 5.1% in median throughput over depth 4, while host-observed
p95 latency grew 2.11x. Its energy estimate was only 3.3% lower, with overlapping
trial variability. Depth 4 is a useful throughput/latency compromise in this run;
depth 2 keeps observed latency lower while retaining a substantial speed gain.
Memory figures count explicit tensor and pinned buffers, not total VRAM or RAM:
cuBLAS workspaces, context, allocator overhead and CPU reference arrays are excluded.

Latency is submission-to-completion-observation at buffer reuse under closed-loop
saturation. Already-finished results can wait to be observed. This is not exact
GPU completion time or open-loop server latency.

## Controls and Limits

- At batch 128, eight streams forced to wait after every request achieved about
  4,000 requests/s, versus 10,152 when allowed to pipeline. Creating streams alone
  did not produce the gain. Hardware copy/kernel overlap was not directly profiled.
- The gains persisted with power queries disabled. At batch 128, query-off median
  rates were 4,044 / 6,920 / 9,498 / 11,776 requests/s at depths 1 / 2 / 4 / 8.
  Energy cannot be inferred for query-off trials from separately sampled trials.
- Paired monitoring comparisons were noisy and sometimes favored sampling. At
  batch 128/depth 8, sampling changed paired throughput by a median -4.1%, with
  a range of -19.3% to +10.4%. This is not a reliable universal overhead estimate.
- Batch 16 still used more GPU energy per request than its fastest serial baseline.
  Faster throughput is not automatically lower energy use.
- Compared with the prior experiment, this run used nine input banks, 3-second
  timing windows, 0.5-second warm-ups and randomized method-group order. Compare
  methods within this run rather than treating historical timing differences as
  a controlled test of depth alone.

## Files and Reproduction

REPORT.md contains the complete method and tables. raw.csv, power.csv and
latency.csv contain the measurements; summary.json contains audited statistics.
Source snapshots, SHA256SUMS, environment.txt, benchmark_metadata.txt, build.txt,
run.txt and STATUS.txt document the completed run.

Rerun from WSL, preserving previous results:

```bash
bash /mnt/e/Codex/inference_experiments/cuda/run_pipeline_depth.sh full
```

The frozen runner can also be executed from this result folder. Audit unit tests:

```bash
python3 /mnt/e/Codex/inference_experiments/cuda/test_pipeline_depth_audit.py /mnt/e/Codex/inference_experiments/results/depth_smoke_20261010T000407Z_fswoWw
```
