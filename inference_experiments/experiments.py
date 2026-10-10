#!/usr/bin/env python3
"""
Reproducible CPU/GPU-toolchain-aware experiments for unconventional inference math.

The script only marks an experiment as "ran" when it executes locally. CUDA DP4A is
reported as skipped unless nvcc is present and the separate CUDA benchmark is built.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover - environment dependent
    psutil = None


POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_command(args: List[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}


def query_gpu_power() -> Optional[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return None
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,power.limit,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not result["ok"] or not result["stdout"]:
        return {"raw": result}
    parts = [part.strip() for part in result["stdout"].splitlines()[0].split(",")]
    keys = [
        "power_draw_w",
        "power_limit_w",
        "gpu_util_percent",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
    ]
    parsed: Dict[str, Any] = {}
    for key, value in zip(keys, parts):
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "ran_at": now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_env": {
            "PROCESSOR_IDENTIFIER": os.environ.get("PROCESSOR_IDENTIFIER"),
            "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS"),
        },
        "numpy_version": np.__version__,
        "psutil_available": psutil is not None,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "nvcc_path": shutil.which("nvcc"),
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        env["memory"] = {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
        }
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)

    if shutil.which("nvidia-smi"):
        env["nvidia_smi"] = run_command(["nvidia-smi"], timeout=10)
        env["gpu_power_snapshot"] = query_gpu_power()
    else:
        env["nvidia_smi"] = {"ok": False, "stdout": "", "stderr": "nvidia-smi not found"}

    env["nvcc_version"] = (
        run_command(["nvcc", "--version"], timeout=10)
        if shutil.which("nvcc")
        else {"ok": False, "stdout": "", "stderr": "nvcc not found"}
    )
    return env


def rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def measure(fn: Callable[[], Any], repeats: int, warmups: int = 1) -> Dict[str, Any]:
    for _ in range(warmups):
        fn()

    before_rss = rss_bytes()
    gpu_before = query_gpu_power()
    times_ms: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    gpu_after = query_gpu_power()
    after_rss = rss_bytes()

    return {
        "times_ms": times_ms,
        "median_ms": median(times_ms),
        "mean_ms": mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "std_ms": pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "rss_before_bytes": before_rss,
        "rss_after_bytes": after_rss,
        "rss_delta_bytes": None
        if before_rss is None or after_rss is None
        else int(after_rss - before_rss),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "last_result": last,
    }


def result_row(
    experiment: str,
    case: str,
    reference_name: str,
    candidate_name: str,
    reference_measure: Optional[Dict[str, Any]],
    candidate_measure: Optional[Dict[str, Any]],
    max_abs_error: Optional[float],
    status: str = "ran",
    notes: str = "",
    control_type: str = "candidate",
    memory_estimate_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    ref_ms = reference_measure.get("median_ms") if reference_measure else None
    cand_ms = candidate_measure.get("median_ms") if candidate_measure else None
    speedup = None
    if ref_ms is not None and cand_ms is not None and cand_ms > 0:
        speedup = ref_ms / cand_ms
    row = {
        "experiment": experiment,
        "case": case,
        "control_type": control_type,
        "status": status,
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_median_ms": ref_ms,
        "candidate_median_ms": cand_ms,
        "speedup_ref_over_candidate": speedup,
        "max_abs_error": max_abs_error,
        "memory_estimate_bytes": memory_estimate_bytes,
        "notes": notes,
    }
    for prefix, measure_obj in [("reference", reference_measure), ("candidate", candidate_measure)]:
        if measure_obj:
            row[f"{prefix}_mean_ms"] = measure_obj["mean_ms"]
            row[f"{prefix}_min_ms"] = measure_obj["min_ms"]
            row[f"{prefix}_max_ms"] = measure_obj["max_ms"]
            row[f"{prefix}_std_ms"] = measure_obj["std_ms"]
            row[f"{prefix}_rss_delta_bytes"] = measure_obj["rss_delta_bytes"]
            row[f"{prefix}_gpu_before"] = measure_obj["gpu_before"]
            row[f"{prefix}_gpu_after"] = measure_obj["gpu_after"]
    return row


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def experiment_event_driven(rng: np.random.Generator, repeats: int, quick: bool) -> List[Dict[str, Any]]:
    n_in = 2048 if quick else 4096
    n_out = 512 if quick else 1024
    steps = 8 if quick else 16
    W = (rng.standard_normal((n_out, n_in)).astype(np.float32) / math.sqrt(n_in))
    x0 = rng.standard_normal(n_in).astype(np.float32)
    y0 = W @ x0
    rows: List[Dict[str, Any]] = []
    fractions = [0.0, 0.005, 0.05, 0.50]

    for frac in fractions:
        k = int(round(n_in * frac))
        changes: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(steps):
            if k == 0:
                idx = np.empty(0, dtype=np.int64)
                delta = np.empty(0, dtype=np.float32)
            else:
                idx = rng.choice(n_in, size=k, replace=False)
                delta = (rng.standard_normal(k).astype(np.float32) * 0.01)
            changes.append((idx, delta))

        def dense_apply() -> np.ndarray:
            x = x0.copy()
            out = y0
            for idx, delta in changes:
                if idx.size:
                    x[idx] += delta
                out = W @ x
            return out

        def incremental_apply() -> np.ndarray:
            out = y0.copy()
            for idx, delta in changes:
                if idx.size:
                    out = out + (W[:, idx] @ delta)
            return out

        dense_out = dense_apply()
        inc_out = incremental_apply()
        err = max_abs(dense_out, inc_out)
        ref = measure(dense_apply, repeats=repeats)
        cand = measure(incremental_apply, repeats=repeats)
        control = "negative_control" if frac >= 0.50 else "candidate"
        notes = (
            f"{steps} recurrent updates, {k}/{n_in} changed inputs per update. "
            "High-change case is a negative control for incremental recomputation."
        )
        rows.append(
            result_row(
                "event_driven_incremental_matvec",
                f"changed_fraction={frac:.3f}",
                "dense_recompute_W_at_x",
                "incremental_y_plus_W_delta",
                ref,
                cand,
                err,
                notes=notes,
                control_type=control,
                memory_estimate_bytes=int(W.nbytes + x0.nbytes + y0.nbytes),
            )
        )
    return rows


def make_sparse_update(
    rng: np.random.Generator, m: int, n: int, sparsity: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nnz = max(1, int(m * n * sparsity))
    rows = rng.integers(0, m, size=nnz, dtype=np.int64)
    cols = rng.integers(0, n, size=nnz, dtype=np.int64)
    vals = (rng.standard_normal(nnz).astype(np.float32) * 0.05)
    return rows, cols, vals


def experiment_low_rank_sparse(rng: np.random.Generator, repeats: int, quick: bool) -> List[Dict[str, Any]]:
    m = 768 if quick else 1024
    n = 768 if quick else 1024
    cases = [(8, 0.002, "low_rank_low_sparse"), (192 if not quick else 96, 0.05, "high_rank_high_sparse")]
    W = (rng.standard_normal((m, n)).astype(np.float32) / math.sqrt(n))
    x = rng.standard_normal(n).astype(np.float32)
    rows_out: List[Dict[str, Any]] = []

    for rank, sparsity, case_name in cases:
        U = (rng.standard_normal((m, rank)).astype(np.float32) / math.sqrt(rank))
        V = (rng.standard_normal((n, rank)).astype(np.float32) / math.sqrt(n))
        s_rows, s_cols, s_vals = make_sparse_update(rng, m, n, sparsity)

        W_update = W + (U @ V.T)
        np.add.at(W_update, (s_rows, s_cols), s_vals)

        def dense_apply() -> np.ndarray:
            return W_update @ x

        def factored_apply() -> np.ndarray:
            out = W @ x
            out = out + U @ (V.T @ x)
            sparse_out = np.zeros(m, dtype=np.float32)
            np.add.at(sparse_out, s_rows, s_vals * x[s_cols])
            return out + sparse_out

        dense_out = dense_apply()
        factored_out = factored_apply()
        err = max_abs(dense_out, factored_out)
        ref = measure(dense_apply, repeats=repeats)
        cand_repeats = max(2, min(repeats, 3)) if sparsity >= 0.05 else repeats
        cand = measure(factored_apply, repeats=cand_repeats)
        memory_estimate = int(
            W.nbytes
            + U.nbytes
            + V.nbytes
            + s_rows.nbytes
            + s_cols.nbytes
            + s_vals.nbytes
            + x.nbytes
        )
        materialized_memory = int(W_update.nbytes + x.nbytes)
        notes = (
            f"rank={rank}, sparse_nnz={len(s_vals)}, factored_estimate={memory_estimate} bytes, "
            f"materialized_estimate={materialized_memory} bytes."
        )
        rows_out.append(
            result_row(
                "low_rank_plus_sparse_update",
                case_name,
                "materialized_dense_updated_weight",
                "base_plus_U_Vt_plus_sparse_triplets",
                ref,
                cand,
                err,
                notes=notes,
                control_type="negative_control" if "high_rank" in case_name else "candidate",
                memory_estimate_bytes=memory_estimate,
            )
        )
    return rows_out


def packbits_bool(a: np.ndarray) -> np.ndarray:
    return np.packbits(a.astype(np.uint8), axis=1)


def popcount_bytes(a: np.ndarray) -> np.ndarray:
    return POPCOUNT8[a].sum(axis=2, dtype=np.int32)


def experiment_bitwise(rng: np.random.Generator, repeats: int, quick: bool) -> List[Dict[str, Any]]:
    batch = 64 if quick else 128
    n = 2048 if quick else 4096
    out_features = 256 if quick else 512
    x_bool = rng.integers(0, 2, size=(batch, n), dtype=np.uint8).astype(bool)
    w_bool = rng.integers(0, 2, size=(out_features, n), dtype=np.uint8).astype(bool)
    x_dense = np.where(x_bool, 1, -1).astype(np.int32)
    w_dense = np.where(w_bool, 1, -1).astype(np.int32)
    xp = packbits_bool(x_bool)
    wp = packbits_bool(w_bool)

    def dense_binary() -> np.ndarray:
        return x_dense @ w_dense.T

    def bitwise_binary() -> np.ndarray:
        xnor = np.bitwise_not(np.bitwise_xor(xp[:, None, :], wp[None, :, :]))
        matches = popcount_bytes(xnor)
        return (2 * matches - n).astype(np.int32)

    dense_out = dense_binary()
    bitwise_out = bitwise_binary()
    rows = [
        result_row(
            "binary_bitwise_dot",
            f"batch={batch},out={out_features},n={n}",
            "dense_int32_dot",
            "packed_xnor_popcount",
            measure(dense_binary, repeats=repeats),
            measure(bitwise_binary, repeats=max(2, min(repeats, 3))),
            max_abs(dense_out, bitwise_out),
            notes="Exact binary {-1,+1} dot product using XNOR/popcount.",
            memory_estimate_bytes=int(xp.nbytes + wp.nbytes),
        )
    ]

    w_ter = rng.choice(np.array([-1, 0, 1], dtype=np.int32), size=(out_features, n), p=[0.25, 0.50, 0.25])
    w_pos = packbits_bool(w_ter == 1)
    w_neg = packbits_bool(w_ter == -1)
    x_pos = xp
    x_neg = packbits_bool(~x_bool)

    def dense_ternary() -> np.ndarray:
        return x_dense @ w_ter.T

    def bitwise_ternary() -> np.ndarray:
        good = popcount_bytes(np.bitwise_and(x_pos[:, None, :], w_pos[None, :, :]))
        good += popcount_bytes(np.bitwise_and(x_neg[:, None, :], w_neg[None, :, :]))
        bad = popcount_bytes(np.bitwise_and(x_pos[:, None, :], w_neg[None, :, :]))
        bad += popcount_bytes(np.bitwise_and(x_neg[:, None, :], w_pos[None, :, :]))
        return (good - bad).astype(np.int32)

    dense_ter_out = dense_ternary()
    bitwise_ter_out = bitwise_ternary()
    rows.append(
        result_row(
            "ternary_bitwise_dot",
            f"batch={batch},out={out_features},n={n},zero_prob=0.50",
            "dense_int32_dot",
            "packed_mask_popcount",
            measure(dense_ternary, repeats=repeats),
            measure(bitwise_ternary, repeats=max(2, min(repeats, 3))),
            max_abs(dense_ter_out, bitwise_ter_out),
            notes="Exact ternary {-1,0,+1} dot product using positive/negative bit masks.",
            memory_estimate_bytes=int(xp.nbytes + x_neg.nbytes + w_pos.nbytes + w_neg.nbytes),
        )
    )
    return rows


def sigmoid_float(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def experiment_lut(rng: np.random.Generator, repeats: int, quick: bool) -> List[Dict[str, Any]]:
    size = 500_000 if quick else 2_000_000
    x = rng.uniform(-8.0, 8.0, size=size).astype(np.float32)
    x_wide = rng.uniform(-20.0, 20.0, size=size).astype(np.float32)

    q8_min = -2048
    q8_max = 2048
    table_q8 = sigmoid_float((np.arange(q8_min, q8_max + 1, dtype=np.float32) / 256.0)).astype(np.float32)
    x_q8 = np.clip(np.rint(x * 256.0), q8_min, q8_max).astype(np.int16)
    x_wide_q8 = np.clip(np.rint(x_wide * 256.0), q8_min, q8_max).astype(np.int16)

    q4_min = -128
    q4_max = 128
    table_q4 = sigmoid_float((np.arange(q4_min, q4_max + 1, dtype=np.float32) / 16.0)).astype(np.float32)
    x_q4 = np.clip(np.rint(x * 16.0), q4_min, q4_max).astype(np.int16)

    def ref_sigmoid() -> np.ndarray:
        return sigmoid_float(x)

    def q8_lut() -> np.ndarray:
        return table_q8[(x_q8.astype(np.int32) - q8_min)]

    def q4_lut() -> np.ndarray:
        return table_q4[(x_q4.astype(np.int32) - q4_min)]

    def ref_sigmoid_wide() -> np.ndarray:
        return sigmoid_float(x_wide)

    def q8_lut_wide() -> np.ndarray:
        return table_q8[(x_wide_q8.astype(np.int32) - q8_min)]

    ref_out = ref_sigmoid()
    q8_out = q8_lut()
    q4_out = q4_lut()
    wide_ref = ref_sigmoid_wide()
    wide_q8 = q8_lut_wide()

    rows = [
        result_row(
            "fixed_point_lut_sigmoid",
            f"q8_in_domain_elements={size}",
            "float32_exp_sigmoid",
            "q8_fixed_point_table_lookup",
            measure(ref_sigmoid, repeats=repeats),
            measure(q8_lut, repeats=repeats),
            max_abs(ref_out, q8_out),
            notes="Q8 fixed-point input with 4097-entry sigmoid LUT over [-8,8].",
            memory_estimate_bytes=int(table_q8.nbytes + x_q8.nbytes),
        ),
        result_row(
            "fixed_point_lut_sigmoid",
            f"q4_coarse_negative_control_elements={size}",
            "float32_exp_sigmoid",
            "q4_fixed_point_table_lookup",
            measure(ref_sigmoid, repeats=max(2, min(repeats, 3))),
            measure(q4_lut, repeats=max(2, min(repeats, 3))),
            max_abs(ref_out, q4_out),
            notes="Coarser Q4 LUT is a negative control for accuracy.",
            control_type="negative_control",
            memory_estimate_bytes=int(table_q4.nbytes + x_q4.nbytes),
        ),
        result_row(
            "fixed_point_lut_sigmoid",
            f"q8_wide_clipped_negative_control_elements={size}",
            "float32_exp_sigmoid_wide_range",
            "q8_fixed_point_table_lookup_clipped",
            measure(ref_sigmoid_wide, repeats=max(2, min(repeats, 3))),
            measure(q8_lut_wide, repeats=max(2, min(repeats, 3))),
            max_abs(wide_ref, wide_q8),
            notes="Wide input range clips at [-8,8]; this checks saturation error.",
            control_type="negative_control",
            memory_estimate_bytes=int(table_q8.nbytes + x_wide_q8.nbytes),
        ),
    ]
    return rows


def next_pow_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def experiment_fft_convolution(rng: np.random.Generator, repeats: int, quick: bool) -> List[Dict[str, Any]]:
    batch = 4 if quick else 8
    n = 8192 if quick else 16384
    kernels = [15, 1024 if not quick else 512]
    rows: List[Dict[str, Any]] = []
    x = rng.standard_normal((batch, n)).astype(np.float32)

    for k in kernels:
        h = rng.standard_normal(k).astype(np.float32)
        fft_len = next_pow_two(n + k - 1)
        H = np.fft.rfft(h, fft_len)

        def direct_valid() -> np.ndarray:
            return np.stack([np.convolve(x[i], h, mode="valid") for i in range(batch)])

        def fft_valid() -> np.ndarray:
            X = np.fft.rfft(x, fft_len, axis=1)
            y = np.fft.irfft(X * H[None, :], fft_len, axis=1)
            return y[:, k - 1 : n]

        direct_out = direct_valid()
        fft_out = fft_valid()
        control = "negative_control" if k <= 15 else "candidate"
        fft_note = (
            "Small-kernel negative control; direct convolution should usually win."
            if k <= 15
            else "Long-kernel candidate; FFT can amortize the larger convolution."
        )
        rows.append(
            result_row(
                "fft_convolution",
                f"valid_conv_batch={batch},n={n},kernel={k}",
                "direct_np_convolve",
                "rfft_precomputed_kernel",
                measure(direct_valid, repeats=max(2, min(repeats, 3)) if k > 256 else repeats),
                measure(fft_valid, repeats=max(2, min(repeats, 3))),
                max_abs(direct_out, fft_out),
                notes=fft_note,
                control_type=control,
                memory_estimate_bytes=int(x.nbytes + h.nbytes + H.nbytes),
            )
        )
    return rows


def skipped_cuda_row(env: Dict[str, Any]) -> Dict[str, Any]:
    if env.get("nvcc_path"):
        note = "nvcc was detected; run cuda/build_and_run_dp4a.ps1 to compile and execute the CUDA DP4A benchmark."
    else:
        note = (
            "Skipped locally because nvcc was not found. The RTX 2060 supports DP4A, "
            "but compiling the included CUDA kernel requires the NVIDIA CUDA Toolkit."
        )
    return result_row(
        "cuda_dp4a_int8",
        "RTX_2060_dp4a_kernel",
        "cpu_int8_reference",
        "cuda___dp4a_kernel",
        None,
        None,
        None,
        status="skipped",
        notes=note,
        memory_estimate_bytes=None,
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys and not key.endswith("_gpu_before") and not key.endswith("_gpu_after"):
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def fmt_speedup(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}x"


def fmt_error(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def generate_report(path: Path, env: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    gpu_summary = "nvidia-smi not available"
    nvidia = env.get("nvidia_smi", {})
    if nvidia.get("ok") and nvidia.get("stdout"):
        first_lines = nvidia["stdout"].splitlines()[:14]
        gpu_summary = "\n".join(first_lines)

    lines: List[str] = []
    lines.append("# Inference Experiment Report")
    lines.append("")
    lines.append(f"Run timestamp: `{env.get('ran_at')}`")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: `{env.get('python_executable')}`")
    lines.append(f"- Python version: `{str(env.get('python')).splitlines()[0]}`")
    lines.append(f"- NumPy: `{env.get('numpy_version')}`")
    lines.append(f"- Platform: `{env.get('platform')}`")
    lines.append(f"- Processor env: `{env.get('cpu_env')}`")
    if env.get("memory"):
        total_gib = env["memory"]["total_bytes"] / (1024 ** 3)
        lines.append(f"- RAM visible to Python: `{total_gib:.2f} GiB`")
    lines.append(f"- `nvidia-smi`: `{env.get('nvidia_smi_path')}`")
    lines.append(f"- `nvcc`: `{env.get('nvcc_path')}`")
    lines.append("")
    lines.append("```text")
    lines.append(gpu_summary)
    lines.append("```")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Experiment | Case | Status | Reference ms | Candidate ms | Speedup | Max abs error | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {experiment} | {case} | {status} | {ref} | {cand} | {speedup} | {err} | {notes} |".format(
                experiment=row["experiment"],
                case=row["case"],
                status=row["status"],
                ref=fmt_ms(row.get("reference_median_ms")),
                cand=fmt_ms(row.get("candidate_median_ms")),
                speedup=fmt_speedup(row.get("speedup_ref_over_candidate")),
                err=fmt_error(row.get("max_abs_error")),
                notes=row.get("notes", "").replace("|", "/"),
            )
        )
    lines.append("")
    lines.append("## Readout")
    lines.append("")
    lines.append("- Event-driven incremental computation is expected to win only when few input channels change. The high-change row is included as a negative control.")
    lines.append("- Low-rank plus sparse updates test whether an adapted model can avoid materializing a full updated matrix. The high-rank/high-sparse row is the stress case.")
    lines.append("- Binary/ternary bitwise rows are exact correctness tests. In NumPy they may lose to dense matrix multiplication because the prototype creates large temporary arrays instead of using fused CPU SIMD or GPU kernels.")
    lines.append("- Fixed-point/LUT sigmoid trades transcendental math for table indexing. Q4 and wide-clipped rows show accuracy limits.")
    lines.append("- FFT convolution should mainly help with long kernels. The short-kernel row is a negative control.")
    lines.append("- CUDA DP4A did not run unless a CUDA compiler is present. A source benchmark is included under `cuda/`.")
    lines.append("")
    lines.append("## Energy")
    lines.append("")
    lines.append("`nvidia-smi` power snapshots are stored in `raw_results.json` for each timed section when available. CPU package energy was not measured because no reliable local energy counter was exposed to this Python harness. GPU energy is not estimated for CPU-only tests.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("cd E:\\Codex\\inference_experiments")
    lines.append("python .\\experiments.py --out .\\results --repeats 5 --seed 12345")
    lines.append("```")
    lines.append("")
    lines.append("Raw data: `raw_results.json`; compact data: `summary.csv`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(out_dir: Path, repeats: int, seed: int, quick: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = collect_environment()
    rows: List[Dict[str, Any]] = []
    experiments = [
        experiment_event_driven,
        experiment_low_rank_sparse,
        experiment_bitwise,
        experiment_lut,
        experiment_fft_convolution,
    ]

    for fn in experiments:
        try:
            rows.extend(fn(rng, repeats, quick))
        except Exception as exc:
            rows.append(
                result_row(
                    fn.__name__,
                    "experiment_failed",
                    "",
                    "",
                    None,
                    None,
                    None,
                    status="failed",
                    notes=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )

    rows.append(skipped_cuda_row(env))
    payload = {"environment": env, "results": rows}
    (out_dir / "raw_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    generate_report(out_dir / "REPORT.md", env, rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats per benchmark")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Use smaller shapes for a smoke test")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    payload = run_all(args.out, args.repeats, args.seed, args.quick)
    ran = sum(1 for row in payload["results"] if row["status"] == "ran")
    skipped = sum(1 for row in payload["results"] if row["status"] == "skipped")
    failed = sum(1 for row in payload["results"] if row["status"] == "failed")
    print(f"Wrote results to {args.out}")
    print(f"rows: ran={ran} skipped={skipped} failed={failed}")
    print(f"report: {args.out / 'REPORT.md'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
