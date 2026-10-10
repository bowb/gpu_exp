#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kind="${1:-full}"
case "$kind" in
    full) seconds=3; warmup=0.5; trials=5 ;;
    smoke) seconds=0.15; warmup=0.05; trials=1 ;;
    *) printf 'Usage: bash run_pipeline_depth.sh [full|smoke]\n' >&2; exit 3 ;;
esac
mkdir -p "$here/../results"
out="$(mktemp -d "$here/../results/depth_${kind}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'Results: %s\n' "$out"
cp -- "$here/pipeline_depth.cu" "$here/dp4a_benchmark.cu" \
    "$here/analyze_pipeline_depth.py" "$here/run_pipeline_depth.sh" "$out/"
printf 'RUNNING\n' > "$out/STATUS.txt"
trap 'printf "FAILED\n" > "$out/STATUS.txt"' ERR
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
{
    date -u --iso-8601=seconds
    uname -a
    lscpu
    free -b
    /usr/local/cuda-13.2/bin/nvcc --version
    g++ --version
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.draw,temperature.gpu,utilization.gpu --format=csv
    printf 'kind=%s seconds=%s warmup=%s trials=%s\n' "$kind" "$seconds" "$warmup" "$trials"
    printf 'batches=1,16,128 CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
} > "$out/environment.txt" 2>&1
(
    cd "$out"
    sha256sum pipeline_depth.cu dp4a_benchmark.cu analyze_pipeline_depth.py run_pipeline_depth.sh > SHA256SUMS
)
/usr/local/cuda-13.2/bin/nvcc -O3 -std=c++17 -arch=sm_75 \
    "$out/pipeline_depth.cu" -lcublas -ldl -Xcompiler -pthread -o "$out/pipeline_depth" \
    > "$out/build.txt" 2>&1
"$out/pipeline_depth" --output "$out" --seconds "$seconds" --warmup "$warmup" \
    --trials "$trials" --batches 1,16,128 2>&1 | tee "$out/run.txt"
python3 "$out/analyze_pipeline_depth.py" "$out" --expected-trials "$trials" --kind "$kind"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,power.draw,temperature.gpu,utilization.gpu --format=csv > "$out/gpu_after.txt"
printf 'COMPLETE\n' > "$out/STATUS.txt"
trap - ERR
printf 'Finished. Report: %s/REPORT.md\n' "$out"
