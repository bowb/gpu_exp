$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "dp4a_benchmark.cu"
$Exe = Join-Path $Root "dp4a_benchmark.exe"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Error "nvcc was not found. Install the NVIDIA CUDA Toolkit, then re-run this script."
}

nvcc -O3 -arch=sm_75 $Source -o $Exe
& $Exe
