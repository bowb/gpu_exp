# Pipeline Depth and Power-Query Overhead

Status: complete; raw-data audit and all correctness controls passed.

Run type: full. Trials: 165. Checked requests: 42,840.

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
| 1 | pipeline | 1 | off | 4574.2 | 0.199 | 0.273 | unavailable | 2.006 | 0.037 | 0/5 |
| 1 | pipeline | 1 | sampling | 4770.9 | 0.188 | 0.264 | 3.877 | 2.006 | 0.037 | 5/5 |
| 1 | pipeline | 2 | off | 8526.7 | 0.194 | 0.367 | unavailable | 2.012 | 0.039 | 0/5 |
| 1 | pipeline | 2 | sampling | 8853.3 | 0.207 | 0.356 | 2.486 | 2.012 | 0.039 | 5/5 |
| 1 | pipeline | 4 | off | 12224.0 | 0.272 | 0.676 | unavailable | 2.023 | 0.043 | 0/5 |
| 1 | pipeline | 4 | sampling | 11565.0 | 0.265 | 0.734 | 1.868 | 2.023 | 0.043 | 5/5 |
| 1 | pipeline | 8 | off | 10336.5 | 0.583 | 1.435 | unavailable | 2.047 | 0.051 | 0/5 |
| 1 | pipeline | 8 | sampling | 10500.7 | 0.629 | 1.382 | 2.070 | 2.047 | 0.051 | 5/5 |
| 1 | serial_pageable | 1 | sampling | 6483.1 | 0.134 | 0.231 | 2.987 | 2.006 | 0.000 | 5/5 |
| 1 | serial_pinned | 1 | sampling | 4625.9 | 0.188 | 0.302 | 3.978 | 2.006 | 0.037 | 5/5 |
| 1 | serialized_8 | 8 | sampling | 4815.3 | 0.195 | 0.283 | 3.913 | 2.047 | 0.051 | 5/5 |
| 16 | pipeline | 1 | off | 5215.7 | 0.175 | 0.274 | unavailable | 2.094 | 0.594 | 0/5 |
| 16 | pipeline | 1 | sampling | 4348.0 | 0.213 | 0.273 | 6.038 | 2.094 | 0.594 | 5/5 |
| 16 | pipeline | 2 | off | 8765.3 | 0.229 | 0.358 | unavailable | 2.188 | 0.625 | 0/5 |
| 16 | pipeline | 2 | sampling | 9187.4 | 0.224 | 0.343 | 7.296 | 2.188 | 0.625 | 5/5 |
| 16 | pipeline | 4 | off | 10175.0 | 0.303 | 0.715 | unavailable | 2.375 | 0.688 | 0/5 |
| 16 | pipeline | 4 | sampling | 10574.3 | 0.310 | 0.669 | 7.046 | 2.375 | 0.688 | 5/5 |
| 16 | pipeline | 8 | off | 9985.4 | 0.689 | 1.407 | unavailable | 2.750 | 0.812 | 0/5 |
| 16 | pipeline | 8 | sampling | 10490.5 | 0.644 | 1.272 | 7.257 | 2.750 | 0.812 | 5/5 |
| 16 | serial_pageable | 1 | sampling | 6222.8 | 0.141 | 0.231 | 6.475 | 2.094 | 0.000 | 5/5 |
| 16 | serial_pinned | 1 | sampling | 5011.9 | 0.182 | 0.298 | 7.655 | 2.094 | 0.594 | 5/5 |
| 16 | serialized_8 | 8 | sampling | 4434.5 | 0.209 | 0.295 | 5.904 | 2.750 | 0.812 | 5/5 |
| 128 | pipeline | 1 | off | 4043.7 | 0.245 | 0.303 | unavailable | 2.750 | 4.750 | 0/5 |
| 128 | pipeline | 1 | sampling | 3961.4 | 0.244 | 0.303 | 14.648 | 2.750 | 4.750 | 5/5 |
| 128 | pipeline | 2 | off | 6919.8 | 0.265 | 0.428 | unavailable | 3.500 | 5.000 | 0/5 |
| 128 | pipeline | 2 | sampling | 7220.6 | 0.264 | 0.399 | 8.999 | 3.500 | 5.000 | 5/5 |
| 128 | pipeline | 4 | off | 9497.9 | 0.353 | 0.745 | unavailable | 5.000 | 5.500 | 0/5 |
| 128 | pipeline | 4 | sampling | 9657.4 | 0.386 | 0.725 | 7.223 | 5.000 | 5.500 | 5/5 |
| 128 | pipeline | 8 | off | 11776.3 | 0.580 | 1.228 | unavailable | 8.000 | 6.500 | 0/5 |
| 128 | pipeline | 8 | sampling | 10151.5 | 0.651 | 1.533 | 6.988 | 8.000 | 6.500 | 5/5 |
| 128 | serial_pageable | 1 | sampling | 3416.8 | 0.286 | 0.362 | 16.401 | 2.750 | 0.000 | 5/5 |
| 128 | serial_pinned | 1 | sampling | 3984.0 | 0.241 | 0.351 | 14.678 | 2.750 | 4.750 | 5/5 |
| 128 | serialized_8 | 8 | sampling | 4000.3 | 0.240 | 0.302 | 14.621 | 8.000 | 6.500 | 5/5 |

## Depth Selection

Descriptive best median among tested depths, not a statistically proven optimum or universal setting.

- Batch 1: best measured throughput at depth 4; 1.784x versus serial_pageable, 1.306x versus depth 2. Host p95 ratio 3.179x. GPU energy ratio 0.625x.
- Batch 16: best measured throughput at depth 4; 1.699x versus serial_pageable, 1.151x versus depth 2. Host p95 ratio 2.897x. GPU energy ratio 1.088x.
- Batch 128: best measured throughput at depth 8; 2.548x versus serial_pinned, 1.406x versus depth 2. Host p95 ratio 4.362x. GPU energy ratio 0.476x.

## Power-Query Control

Paired percent change in throughput with sampling versus off; negative means sampling slower. Ranges show run variability and may include scheduling, temperature and other-GPU-work effects.

| Batch | Depth | Median change % | Minimum % | Maximum % |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.84 | -1.51 | 10.40 |
| 1 | 2 | 1.67 | -6.26 | 23.25 |
| 1 | 4 | -0.82 | -15.82 | 19.22 |
| 1 | 8 | 3.33 | -13.78 | 23.26 |
| 16 | 1 | -4.53 | -16.64 | 10.18 |
| 16 | 2 | 1.54 | -12.24 | 21.10 |
| 16 | 4 | 22.60 | -14.80 | 27.24 |
| 16 | 8 | 17.15 | -38.58 | 31.07 |
| 128 | 1 | 0.84 | -2.49 | 2.42 |
| 128 | 2 | 0.41 | -3.61 | 10.72 |
| 128 | 4 | -2.66 | -6.08 | 1.87 |
| 128 | 8 | -4.09 | -19.28 | 10.42 |

## Raw Audit

```json
{
  "trials": 165,
  "completed_requests": 3715516,
  "checked_requests": 42840,
  "valid_power_trials": 105,
  "sampling_trials": 105,
  "latency_samples": 119938,
  "raw_audit_passed": true,
  "kind": "full"
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
