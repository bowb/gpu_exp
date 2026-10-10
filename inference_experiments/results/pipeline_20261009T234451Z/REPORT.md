# Two-Stream Pipeline Experiment

GPU: NVIDIA GeForce RTX 2060; cuBLAS: 130401; CUDA compiler: 13.2.86; copy engines: 2; concurrent kernels: 1.

Seed 12345; int8 [-8,7], int32 outputs, four distinct precomputed input banks, 4096 inputs and 512 outputs per vector. Weights stay GPU-resident. Allocation, CPU reference and input preparation excluded. Five trials with rotated method order, 1-second warm-up, >=5 seconds timed including final drain.

Throughput counts completed requests; each request contains Batch input vectors. Latency is host-observed submission-to-completion, sampled every 31st request under closed-loop saturation, not amortized throughput or open-loop service latency. Full correctness checks before/after timing and on final buffers; timed requests are not all checked. Stale-output and injected-error controls must reject errors. No profiling proof of kernel/copy overlap.

GPU board energy is estimated from ~100ms power samples, not CPU or wall energy; other GPU activity contributes. nan means unavailable/poor coverage. raw.csv records exact buffer allocation bytes excluding library/context/allocator overhead. DISABLE_POWER=1 disables queries for a telemetry-overhead control.

Median of five trials (latency quantiles are per-trial sampled statistics):

| Batch | Mode | Completed requests/s | Host p50 ms | Host p95 ms | GPU mJ/request | Throughput vs best serial | Valid power trials |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | serial_pageable | 5812.98 | 0.142968 | 0.236924 | 3.17846 | 1x | 5/5 |
| 1 | serial_pinned | 4741.6 | 0.191419 | 0.293549 | 4.08272 | 0.815692x | 5/5 |
| 1 | async_1stream | 4507.31 | 0.199944 | 0.273622 | 4.0128 | 0.775388x | 5/5 |
| 1 | async_2streams | 8187.64 | 0.220793 | 0.378388 | 2.60855 | 1.40851x | 5/5 |
| 1 | async_2streams_serialized | 4303.93 | 0.199834 | 0.280535 | 4.24105 | 0.7404x | 5/5 |
| 16 | serial_pageable | 5878.54 | 0.150411 | 0.238927 | 6.20214 | 1x | 5/5 |
| 16 | serial_pinned | 4305.46 | 0.216896 | 0.333184 | 6.17287 | 0.732403x | 5/5 |
| 16 | async_1stream | 4139.51 | 0.210795 | 0.301604 | 5.78291 | 0.704173x | 5/5 |
| 16 | async_2streams | 8628.66 | 0.22991 | 0.329476 | 6.81068 | 1.46782x | 5/5 |
| 16 | async_2streams_serialized | 4782.49 | 0.210994 | 0.272179 | 6.2891 | 0.81355x | 5/5 |
| 128 | serial_pageable | 3147.28 | 0.291146 | 0.42771 | 17.1224 | 0.819829x | 5/5 |
| 128 | serial_pinned | 3838.95 | 0.249006 | 0.362158 | 13.7712 | 1x | 5/5 |
| 128 | async_1stream | 3987.25 | 0.238977 | 0.296225 | 13.7015 | 1.03863x | 5/5 |
| 128 | async_2streams | 6999.9 | 0.263443 | 0.41677 | 8.90726 | 1.82339x | 5/5 |
| 128 | async_2streams_serialized | 3859.16 | 0.248946 | 0.350376 | 14.6316 | 1.00526x | 5/5 |

## Reproduction

From this result directory:

```bash
/usr/local/cuda-13.2/bin/nvcc -O3 -std=c++17 -arch=sm_75 pipeline_energy.cu -lcublas -ldl -Xcompiler -pthread -o pipeline_energy
out="$PWD/rerun_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$out"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PIPELINE_RESULTS="$out" ./pipeline_energy
```

All validation passes and both negative correctness controls passed. Each cell validated 128 sequence requests plus final buffers. Outputs: raw.csv, power.csv, latency.csv; source snapshots included.
