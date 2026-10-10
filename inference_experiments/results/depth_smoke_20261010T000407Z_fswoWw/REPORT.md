# Pipeline Depth and Power-Query Overhead

Status: complete; raw-data audit and all correctness controls passed.

Run type: smoke. Trials: 33. Checked requests: 8,568.

## Method

RTX 2060 / WSL, cuBLAS integer GEMM, 4096 inputs and 512 outputs per vector; int8 [-8,7] inputs/weights and int32 outputs. Seed 12345. Nine distinct input banks ensure each slot sees changing inputs even at depth 8. Weights stay GPU-resident.

Depths 1/2/4/8 use independent nonblocking streams, cuBLAS handles, device input/output buffers and pinned host outputs. Both serial baselines and depth-8 forced-serialization control use the same GEMM. Allocation, reference generation and input preparation are excluded.

Five trials in the full run, shuffled method-group order with seed 20261010. Sampling/off pairs are adjacent with alternating within-pair order. Each full cell has 0.5 s warm-up and at least 3 s timing, with final drain inside timing. Closed-loop saturation; no externally arriving request queue or full-model inference.

Host-observed latency runs from submission until completion is observed at buffer reuse. A result may already have finished before it is observed. It is not GPU execution time, amortized throughput, or open-loop service latency. Samples every 31 requests cover all slots and banks. Displayed quantiles are medians of trial-level quantiles.

Sampling calls NVML about every 100 ms. Off uses the same sleeping worker without NVML calls: the paired comparison isolates query overhead, not all monitoring/instrumentation overhead. GPU board energy is a gross estimate, includes other GPU activity, and excludes CPU and wall electricity. Trapezoidal time-weighted mean is extrapolated across the timing window; require >=20 samples, >=90% coverage, and <=0.5 s maximum gap. No idle subtraction.

128 fully checked sequence requests before and after timing plus final buffers per cell. Timed requests are not all checked. Wrong-bank stale output and one-bit host-output corruption must be rejected. Submitted/completed counts must match. No direct profiling proof of overlap.

Buffer bytes below exclude library workspaces, CUDA context/allocator overhead and CPU reference arrays; they are not total process memory or peak VRAM.

## Results

| Batch | Mode | Depth | Telemetry | Requests/s | p50 ms | p95 ms | GPU mJ/request | Device MiB | Pinned MiB | Valid power |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | pipeline | 1 | off | 5919.0 | 0.151 | 0.173 | unavailable | 2.006 | 0.037 | 0/1 |
| 1 | pipeline | 1 | sampling | 5175.1 | 0.167 | 0.285 | unavailable | 2.006 | 0.037 | 0/1 |
| 1 | pipeline | 2 | off | 10435.5 | 0.173 | 0.347 | unavailable | 2.012 | 0.039 | 0/1 |
| 1 | pipeline | 2 | sampling | 10503.5 | 0.177 | 0.228 | unavailable | 2.012 | 0.039 | 0/1 |
| 1 | pipeline | 4 | off | 6913.5 | 0.588 | 0.767 | unavailable | 2.023 | 0.043 | 0/1 |
| 1 | pipeline | 4 | sampling | 12745.0 | 0.313 | 0.367 | unavailable | 2.023 | 0.043 | 0/1 |
| 1 | pipeline | 8 | off | 10963.8 | 0.600 | 1.693 | unavailable | 2.047 | 0.051 | 0/1 |
| 1 | pipeline | 8 | sampling | 11161.3 | 0.703 | 0.836 | unavailable | 2.047 | 0.051 | 0/1 |
| 1 | serial_pageable | 1 | sampling | 5541.1 | 0.142 | 0.200 | unavailable | 2.006 | 0.000 | 0/1 |
| 1 | serial_pinned | 1 | sampling | 4937.3 | 0.193 | 0.248 | unavailable | 2.006 | 0.037 | 0/1 |
| 1 | serialized_8 | 8 | sampling | 6046.5 | 0.146 | 0.225 | unavailable | 2.047 | 0.051 | 0/1 |
| 16 | pipeline | 1 | off | 3249.9 | 0.284 | 0.422 | unavailable | 2.094 | 0.594 | 0/1 |
| 16 | pipeline | 1 | sampling | 4903.4 | 0.197 | 0.236 | unavailable | 2.094 | 0.594 | 0/1 |
| 16 | pipeline | 2 | off | 8582.7 | 0.198 | 0.418 | unavailable | 2.188 | 0.625 | 0/1 |
| 16 | pipeline | 2 | sampling | 5957.6 | 0.230 | 0.738 | unavailable | 2.188 | 0.625 | 0/1 |
| 16 | pipeline | 4 | off | 10109.3 | 0.369 | 0.622 | unavailable | 2.375 | 0.688 | 0/1 |
| 16 | pipeline | 4 | sampling | 9607.5 | 0.351 | 0.792 | unavailable | 2.375 | 0.688 | 0/1 |
| 16 | pipeline | 8 | off | 8566.9 | 0.669 | 1.578 | unavailable | 2.750 | 0.812 | 0/1 |
| 16 | pipeline | 8 | sampling | 10014.7 | 0.600 | 1.547 | unavailable | 2.750 | 0.812 | 0/1 |
| 16 | serial_pageable | 1 | sampling | 5461.7 | 0.173 | 0.219 | unavailable | 2.094 | 0.000 | 0/1 |
| 16 | serial_pinned | 1 | sampling | 3930.9 | 0.223 | 0.411 | unavailable | 2.094 | 0.594 | 0/1 |
| 16 | serialized_8 | 8 | sampling | 3950.2 | 0.196 | 0.365 | unavailable | 2.750 | 0.812 | 0/1 |
| 128 | pipeline | 1 | off | 2832.2 | 0.326 | 0.454 | unavailable | 2.750 | 4.750 | 0/1 |
| 128 | pipeline | 1 | sampling | 3154.7 | 0.294 | 0.434 | unavailable | 2.750 | 4.750 | 0/1 |
| 128 | pipeline | 2 | off | 6867.1 | 0.263 | 0.435 | unavailable | 3.500 | 5.000 | 0/1 |
| 128 | pipeline | 2 | sampling | 5235.5 | 0.287 | 0.732 | unavailable | 3.500 | 5.000 | 0/1 |
| 128 | pipeline | 4 | off | 7854.0 | 0.475 | 0.662 | unavailable | 5.000 | 5.500 | 0/1 |
| 128 | pipeline | 4 | sampling | 6304.9 | 0.612 | 0.788 | unavailable | 5.000 | 5.500 | 0/1 |
| 128 | pipeline | 8 | off | 11063.7 | 0.693 | 0.811 | unavailable | 8.000 | 6.500 | 0/1 |
| 128 | pipeline | 8 | sampling | 10136.2 | 0.689 | 1.447 | unavailable | 8.000 | 6.500 | 0/1 |
| 128 | serial_pageable | 1 | sampling | 1971.4 | 0.481 | 0.521 | unavailable | 2.750 | 0.000 | 0/1 |
| 128 | serial_pinned | 1 | sampling | 3412.7 | 0.311 | 0.370 | unavailable | 2.750 | 4.750 | 0/1 |
| 128 | serialized_8 | 8 | sampling | 3187.8 | 0.300 | 0.368 | unavailable | 8.000 | 6.500 | 0/1 |

## Depth Selection

Descriptive best median among tested depths, not a statistically proven optimum or universal setting.

Smoke run only: short timings validate execution and auditing, not performance conclusions.

## Power-Query Control

Paired percent change in throughput with sampling versus off; negative means sampling slower. Ranges show run variability and may include scheduling, temperature and other-GPU-work effects.

| Batch | Depth | Median change % | Minimum % | Maximum % |
|---|---:|---:|---:|---:|
| 1 | 1 | -12.57 | -12.57 | -12.57 |
| 1 | 2 | 0.65 | 0.65 | 0.65 |
| 1 | 4 | 84.35 | 84.35 | 84.35 |
| 1 | 8 | 1.80 | 1.80 | 1.80 |
| 16 | 1 | 50.88 | 50.88 | 50.88 |
| 16 | 2 | -30.59 | -30.59 | -30.59 |
| 16 | 4 | -4.96 | -4.96 | -4.96 |
| 16 | 8 | 16.90 | 16.90 | 16.90 |
| 128 | 1 | 11.39 | 11.39 | 11.39 |
| 128 | 2 | -23.76 | -23.76 | -23.76 |
| 128 | 4 | -19.72 | -19.72 | -19.72 |
| 128 | 8 | -8.38 | -8.38 | -8.38 |

## Raw Audit

```json
{
  "trials": 33,
  "completed_requests": 34098,
  "checked_requests": 8568,
  "valid_power_trials": 0,
  "sampling_trials": 21,
  "latency_samples": 1116,
  "raw_audit_passed": true,
  "kind": "smoke"
}
```

The independent Python audit reconstructs throughput, sampled latency quantiles, power-window flags, energy integration, buffer counts and the expected trial grid from CSVs, and verifies source snapshot hashes.

## Reproduction

From this result folder in WSL (creates a new result folder, preserving this run):

```bash
bash run_pipeline_depth.sh full
```

Short correctness check: `bash run_pipeline_depth.sh smoke`.

The wrapper compiles the frozen source, records environment/compiler details and checksums, runs the benchmark, and audits the raw data. Requires CUDA 13.2 at /usr/local/cuda-13.2, WSL GPU access and Python 3. See environment.txt, benchmark_metadata.txt, build.txt and run.txt.

## References

- CUDA streams and pinned transfers: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-best-practices-guide/index.html
- NVML power query: https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html
