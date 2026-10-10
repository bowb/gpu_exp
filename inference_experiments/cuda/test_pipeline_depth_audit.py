#!/usr/bin/env python3
"""Exercise the independent audit against intact and deliberately corrupted smoke data."""
import argparse
import contextlib
import csv
import importlib.util
import io
import shutil
import tempfile
import unittest
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("smoke_results", type=Path)
args = parser.parse_args()
source = args.smoke_results.resolve()
spec = importlib.util.spec_from_file_location("depth_audit", source / "analyze_pipeline_depth.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="audit_fixture_", dir=source.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("raw.csv", "power.csv", "latency.csv", "pipeline_depth.cu", "dp4a_benchmark.cu",
                     "analyze_pipeline_depth.py", "run_pipeline_depth.sh", "SHA256SUMS"):
            shutil.copyfile(source / name, self.root / name)

    def mutate(self, name, change):
        path = self.root / name
        rows = audit.read_csv(path)
        fields = list(rows[0])
        change(rows)
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def run_audit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            audit.analyze(self.root, 1, "smoke")

    def rejects(self):
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_intact_run_passes(self):
        self.run_audit()
        self.assertTrue((self.root / "summary.json").is_file())

    def test_missing_trial(self):
        self.mutate("raw.csv", lambda rows: rows.pop())
        self.rejects()

    def test_duplicate_trial(self):
        self.mutate("raw.csv", lambda rows: rows.__setitem__(-1, rows[0].copy()))
        self.rejects()

    def test_mismatch_flag(self):
        self.mutate("raw.csv", lambda rows: rows[0].update(mismatches="1"))
        self.rejects()

    def test_failed_error_control(self):
        self.mutate("raw.csv", lambda rows: rows[0].update(injected_control_passed="0"))
        self.rejects()

    def test_incorrect_completed_count(self):
        self.mutate("raw.csv", lambda rows: rows[0].update(
            completed_requests=str(int(rows[0]["completed_requests"]) + 1)))
        self.rejects()

    def test_missing_latency_sample(self):
        self.mutate("latency.csv", lambda rows: rows.pop(0))
        self.rejects()

    def test_incorrect_input_bank(self):
        self.mutate("latency.csv", lambda rows: rows[0].update(input_bank="8"))
        self.rejects()

    def test_incorrect_buffer_slot(self):
        self.mutate("latency.csv", lambda rows: rows[0].update(slot="99"))
        self.rejects()

    def test_incorrect_power_window(self):
        self.mutate("power.csv", lambda rows: rows[0].update(
            in_window=str(1 - int(rows[0]["in_window"]))))
        self.rejects()

    def test_unsupported_energy_claim(self):
        self.mutate("raw.csv", lambda rows: rows[0].update(gpu_est_j="0"))
        self.rejects()

    def test_changed_source_snapshot(self):
        with (self.root / "pipeline_depth.cu").open("a", encoding="utf-8") as destination:
            destination.write("\n// changed fixture\n")
        self.rejects()


unittest.main(argv=[__file__], verbosity=2)
