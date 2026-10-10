# Transfer and Energy Comparison

GPU: NVIDIA GeForce RTX 2060 cuBLAS version: 130401

Same int8 inputs, int32 outputs; seed 12345. All errors must be zero.

Three samples per row; median per-operation time. At least 5 seconds/sample after 1 second warm-up; all timings are host wall averages, including dispatch and synchronization.
Pageable versus CUDA-allocated pinned memory; allocations, initialization and CPU reference are outside timing. GPU energy is estimated from sampled board power in the raw CSV, not whole-PC energy. nan means unavailable or insufficient coverage. Other GPU applications contribute.

| Batch | Scenario | Pageable ms | Pinned ms | Pinned speedup |
|---|---|---:|---:|---:|
| 1 | compute_no_transfers | 0.0188568 | 0.0198723 | 0.948896x |
| 1 | weights_resident | 0.163687 | 0.241243 | 0.678515x |
| 1 | upload_all | 0.411757 | 0.381849 | 1.07833x |
| 16 | compute_no_transfers | 0.0335554 | 0.034076 | 0.984721x |
| 16 | weights_resident | 0.148663 | 0.204041 | 0.728597x |
| 16 | upload_all | 0.432678 | 0.384468 | 1.1254x |
| 128 | compute_no_transfers | 0.0501376 | 0.0510445 | 0.982233x |
| 128 | weights_resident | 0.279444 | 0.247799 | 1.12771x |
| 128 | upload_all | 0.534218 | 0.462014 | 1.15628x |

All correctness checks passed, including an injected-error control. cuBLAS uses its default algorithm heuristic; no exhaustive tuning.
Resident: upload inputs + compute + download output. Upload-all also uploads weights. All requests await completion.
