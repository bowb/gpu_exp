# Inference Experiment Report

Run timestamp: `2026-10-09T14:54:22-0600`

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
Fri Oct  9 14:54:22 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.95                 Driver Version: 595.95         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
```

## Summary Table

| Experiment | Case | Status | Reference ms | Candidate ms | Speedup | Max abs error | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| event_driven_incremental_matvec | changed_fraction=0.000 | ran | 1.107 | 0.007 | 150.56x | 0 | 8 recurrent updates, 0/2048 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.005 | ran | 1.534 | 0.398 | 3.86x | 4.76837e-07 | 8 recurrent updates, 10/2048 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.050 | ran | 1.103 | 4.505 | 0.24x | 9.53674e-07 | 8 recurrent updates, 102/2048 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| event_driven_incremental_matvec | changed_fraction=0.500 | ran | 1.132 | 25.174 | 0.04x | 1.90735e-06 | 8 recurrent updates, 1024/2048 changed inputs per update. High-change case is a negative control for incremental recomputation. |
| low_rank_plus_sparse_update | low_rank_low_sparse | ran | 0.188 | 0.197 | 0.96x | 1.54972e-06 | rank=8, sparse_nnz=1179, factored_estimate=2435100 bytes, materialized_estimate=2362368 bytes. |
| low_rank_plus_sparse_update | high_rank_high_sparse | ran | 0.170 | 0.483 | 0.35x | 1.65403e-06 | rank=96, sparse_nnz=29491, factored_estimate=3542012 bytes, materialized_estimate=2362368 bytes. |
| binary_bitwise_dot | batch=64,out=256,n=2048 | ran | 55.902 | 27.431 | 2.04x | 0 | Exact binary {-1,+1} dot product using XNOR/popcount. |
| ternary_bitwise_dot | batch=64,out=256,n=2048,zero_prob=0.50 | ran | 56.529 | 92.912 | 0.61x | 0 | Exact ternary {-1,0,+1} dot product using positive/negative bit masks. |
| fixed_point_lut_sigmoid | q8_in_domain_elements=500000 | ran | 2.484 | 3.525 | 0.70x | 0.000488102 | Q8 fixed-point input with 4097-entry sigmoid LUT over [-8,8]. |
| fixed_point_lut_sigmoid | q4_coarse_negative_control_elements=500000 | ran | 2.649 | 3.467 | 0.76x | 0.00780994 | Coarser Q4 LUT is a negative control for accuracy. |
| fixed_point_lut_sigmoid | q8_wide_clipped_negative_control_elements=500000 | ran | 2.454 | 2.859 | 0.86x | 0.000487983 | Wide input range clips at [-8,8]; this checks saturation error. |
| fft_convolution | valid_conv_batch=4,n=8192,kernel=15 | ran | 0.578 | 1.796 | 0.32x | 8.31804e-07 | Small-kernel case is a negative control; FFT should mainly help for long kernels. |
| fft_convolution | valid_conv_batch=4,n=8192,kernel=512 | ran | 1.110 | 2.114 | 0.52x | 1.1648e-05 | Small-kernel case is a negative control; FFT should mainly help for long kernels. |
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
