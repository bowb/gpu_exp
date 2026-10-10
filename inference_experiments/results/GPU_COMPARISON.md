# GPU Comparison

GPU: NVIDIA GeForce RTX 2060 cuBLAS version: 130401

Same int8 inputs, int32 outputs; seed 12345. All errors must be zero.

Seven samples per row; median per-operation time. Compute: 50 operations/sample. Transfers: 8 requests/sample.
Pageable host memory; allocations, initialization and CPU reference are outside timing. Energy is not measured.

| Batch | Scenario | DP4A ms | cuBLAS ms | DP4A speedup vs cuBLAS |
|---|---|---:|---:|---:|
| 1 | compute_only | 0.0427514 | 0.0453018 | 1.05966x |
| 1 | weights_resident | 0.194296 | 0.177039 | 0.91118x |
| 1 | upload_all | 0.444268 | 0.429319 | 0.966351x |
| 4 | compute_only | 0.0910547 | 0.0276013 | 0.303129x |
| 4 | weights_resident | 0.180569 | 0.111114 | 0.615355x |
| 4 | upload_all | 0.464945 | 0.384778 | 0.827576x |
| 16 | compute_only | 0.239446 | 0.0401056 | 0.167493x |
| 16 | weights_resident | 0.408348 | 0.150309 | 0.36809x |
| 16 | upload_all | 0.656746 | 0.424031 | 0.645654x |
| 128 | compute_only | 1.40179 | 0.0443597 | 0.0316451x |
| 128 | weights_resident | 1.6521 | 0.270652 | 0.163823x |
| 128 | upload_all | 1.90969 | 0.499234 | 0.261422x |

All correctness checks passed, including an injected-error control. cuBLAS uses its default algorithm heuristic; no exhaustive tuning.
Resident: upload inputs + compute + download output. Upload-all also uploads weights. All requests await completion.
