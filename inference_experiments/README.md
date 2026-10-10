# Inference Math Experiments

Reproducible experiments for testing unconventional inference approaches on this PC.

The harness runs CPU/NumPy tests immediately when the required packages are present, and it records skipped CUDA tests honestly when the CUDA toolkit or Python GPU libraries are missing.

## What This Tests

- Event-driven incremental matvec updates
- Low-rank plus sparse weight updates
- Binary and ternary bitwise dot products
- Fixed-point/LUT sigmoid approximations
- FFT-based convolution
- CUDA DP4A int8 dot products, provided as a compilable CUDA test when `nvcc` is installed

## Reproducible Windows Commands

From PowerShell:

```powershell
cd E:\Codex\inference_experiments
python .\experiments.py --out .\results --repeats 5 --seed 12345
```

For a quicker smoke test:

```powershell
cd E:\Codex\inference_experiments
python .\experiments.py --out .\results_smoke --repeats 2 --quick
```

## Optional CUDA DP4A Test

This PC has an RTX 2060, which supports DP4A, but the CUDA toolkit must be installed to compile the included kernel.

When `nvcc` is available:

```powershell
cd E:\Codex\inference_experiments
.\cuda\build_and_run_dp4a.ps1
```

The CUDA source is in `cuda\dp4a_benchmark.cu`.

## Outputs

Each run writes:

- `results\raw_results.json`
- `results\summary.csv`
- `results\REPORT.md`

The report includes environment detection, correctness checks, negative controls, timing, memory estimates, and notes on energy telemetry. CPU energy is not estimated without a reliable local power sensor. GPU power telemetry is recorded when `nvidia-smi` is available, but GPU energy is only meaningful for tests that actually run GPU kernels.
