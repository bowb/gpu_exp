# Inference Experiment Report

Run timestamp: `2026-10-09T15:03:53-0600`

## Environment

- Python: `C:\Users\bowb\AppData\Local\Programs\Python\Python39\python.exe`
- Python version: `3.9.1 (tags/v3.9.1:1e5d33e, Dec  7 2020, 17:08:21) [MSC v.1927 64 bit (AMD64)]`
- NumPy: `1.26.4`
- Platform: `Windows-10-10.0.22621-SP0`
- Processor env: `{'PROCESSOR_IDENTIFIER': 'AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD', 'NUMBER_OF_PROCESSORS': '16'}`
- RAM visible to Python: `15.92 GiB`
- `nvidia-smi`: `C:\WINDOWS\system32\nvidia-smi.EXE`
- `nvcc`: `None`

```text
Fri Oct  9 15:03:53 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.95                 Driver Version: 595.95         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2060      WDDM  |   00000000:23:00.0  On |                  N/A |
| 44%   36C    P8             14W /  170W |     984MiB /   6144MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
```

## Summary Table

| Experiment | Case | Status | Reference ms | Candidate ms | Speedup | Max abs error | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| event_driven_incremental_matvec | changed_fraction=0.000 | ran | 2.719 | 0.002 | 1431.32x | 0 | 16 recurrent updates, 0/4096 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.005 | ran | 3.048 | 5.453 | 0.56x | 8.34465e-07 | 16 recurrent updates, 20/4096 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.050 | ran | 2.995 | 42.366 | 0.07x | 1.93715e-06 | 16 recurrent updates, 205/4096 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.500 | ran | 3.042 | 375.787 | 0.01x | 2.86102e-06 | 16 recurrent updates, 2048/4096 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| low_rank_plus_sparse_update | low_rank_low_sparse | ran | 0.142 | 0.152 | 0.93x | 1.43051e-06 | rank=8, sparse_nnz=2097, factored_estimate=4305876 bytes, materialized_estimate=4198400 bytes. |
| low_rank_plus_sparse_update | high_rank_high_sparse | ran | 0.139 | 0.617 | 0.23x | 1.69873e-06 | rank=192, sparse_nnz=52428, factored_estimate=6819824 bytes, materialized_estimate=4198400 bytes. |
| binary_bitwise_dot | batch=128,out=512,n=4096 | ran | 444.938 | 209.872 | 2.12x | 0 | Exact binary {-1,+1} dot product using XNOR/popcount. |
| ternary_bitwise_dot | batch=128,out=512,n=4096,zero_prob=0.50 | ran | 442.044 | 754.889 | 0.59x | 0 | Exact ternary {-1,0,+1} dot product using positive/negative bit masks. |
| fixed_point_lut_sigmoid | q8_in_domain_elements=2000000 | ran | 8.592 | 13.071 | 0.66x | 0.000488281 | Q8 fixed-point input with 4097-entry sigmoid LUT over [-8,8]. |
| fixed_point_lut_sigmoid | q4_coarse_negative_control_elements=2000000 | ran | 8.681 | 13.273 | 0.65x | 0.00781161 | Coarser Q4 LUT is a negative control for accuracy. |
| fixed_point_lut_sigmoid | q8_wide_clipped_negative_control_elements=2000000 | ran | 8.446 | 12.327 | 0.69x | 0.000488013 | Wide input range clips at [-8,8]; this checks saturation error. |
| fft_convolution | valid_conv_batch=8,n=16384,kernel=15 | ran | 2.055 | 5.672 | 0.36x | 1.2555e-06 | Small-kernel negative control; direct convolution should usually win. |
| fft_convolution | valid_conv_batch=8,n=16384,kernel=1024 | ran | 7.023 | 5.109 | 1.37x | 1.86891e-05 | Long-kernel candidate; FFT can amortize the larger convolution. |
| cuda_dp4a_int8 | RTX_2060_dp4a_kernel | skipped | - | - | - | - | Skipped locally because nvcc was not found. The RTX 2060 supports DP4A, but compiling the included CUDA kernel requires the NVIDIA CUDA Toolkit. |

## Readout

- Event-driven incremental computation is expected to win only when few input channels change. The high-change row is included as a negative control.
- Low-rank plus sparse updates test whether an adapted model can avoid materializing a full updated matrix. The high-rank/high-sparse row is the stress case.
- Binary/ternary bitwise rows are exact correctness tests. In NumPy they may lose to dense matrix multiplication because the prototype creates large temporary arrays instead of using fused CPU SIMD or GPU kernels.
- Fixed-point/LUT sigmoid trades transcendental math for table indexing. Q4 and wide-clipped rows show accuracy limits.
- FFT convolution should mainly help with long kernels. The short-kernel row is a negative control.
- CUDA DP4A did not run unless a CUDA compiler is present. A source benchmark is included under `cuda/`.

## Energy

`nvidia-smi` power snapshots are stored in `raw_results.json` for each timed section when available. CPU package energy was not measured because no reliable local energy counter was exposed to this Python harness. GPU energy is not estimated for CPU-only tests.

## Reproduce

```powershell
cd E:\Codex\inference_experiments
python .\experiments.py --out .\results --repeats 5 --seed 12345
```

Raw data: `raw_results.json`; compact data: `summary.csv`.
