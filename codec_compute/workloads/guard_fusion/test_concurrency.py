"""Local protocol tests with fake foreground work; these do not exercise NVDEC."""

import ast
import builtins
from contextlib import redirect_stdout
import io
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from . import concurrency as runner
from ..cpu_comparison.benchmark import Accounting


class FakeClock:
    def __init__(self, now=0):
        self.now = now

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeCorpus:
    def __init__(self, delay=0, fail=False):
        self.calls, self.delay, self.fail = [], delay, fail
        self.expected = runner.chunk(1000)

    def cpu(self, key, index):
        if self.fail and index == 0:
            raise AssertionError("test output mismatch")
        assert runner.chunk(1000) == self.expected
        self.calls.append((key, index))
        time.sleep(self.delay)


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux process affinity is required")
class ConcurrencyTests(unittest.TestCase):
    def test__given_late_completion__when_releasing_absolute_jobs__then_queueing_counts_and_releases_do_not_drift(self):
        clock = FakeClock(5)
        called = []

        def operation(index):
            called.append(index)
            clock.sleep((1.3, 0.3, 1.7)[index])

        jobs = runner.releases(operation, 3, 5, 1, 0.75, clock.clock, clock.sleep)
        self.assertEqual(called, [0, 1, 2])
        self.assertEqual([job["release_offset_s"] for job in jobs], [0, 1, 2])
        self.assertEqual([job["deadline_miss"] for job in jobs], [True, False, True])
        self.assertAlmostEqual(jobs[1]["release_lateness_ms"], 300)
        self.assertAlmostEqual(jobs[1]["release_to_completion_ms"], 600)
        self.assertAlmostEqual(jobs[2]["start_offset_s"], 2)

    def test__given_overload__when_all_releases_are_due__then_every_job_runs_and_lateness_accumulates(self):
        clock = FakeClock()
        calls = []

        def operation(index):
            calls.append(index)
            clock.sleep(1.5)

        jobs = runner.releases(operation, 4, 0, 1, 0.75, clock.clock, clock.sleep)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual([job["release_lateness_ms"] for job in jobs], [0, 500, 1000, 1500])
        self.assertEqual(jobs[-1]["finish_offset_s"], 6)
        self.assertTrue(all(job["deadline_miss"] for job in jobs))

    def test__given_rotated_repeats__when_ordering_modes__then_each_mode_occupies_every_position(self):
        orders = [runner.rotation(repeat) for repeat in range(4)]
        for position in range(4):
            self.assertEqual({order[position] for order in orders}, set(runner.MODES))
        self.assertEqual(runner.rotation(4), runner.MODES)

    def test__given_affinity_budget__when_work_raises__then_original_thread_masks_are_restored(self):
        original = runner.thread_masks(os.getpid())
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with runner.one_cpu_budget() as logical_cpu:
                self.assertEqual(logical_cpu, min(original[os.getpid()]))
                self.assertEqual(runner.audit_affinity(os.getpid(), logical_cpu)["allowed_logical_cpus"], [logical_cpu])
                raise RuntimeError("test failure")
        self.assertEqual(runner.thread_masks(os.getpid()), original)

    def background(self):
        return {"iterations": 1000, "expected_hex": runner.chunk(1000).hex()}

    def test__given_real_background_worker_and_fake_foreground__when_observing_fixed_windows__then_progress_and_checks_span_the_window(self):
        with runner.one_cpu_budget() as logical_cpu:
            for mode in ("background-only", "cpu"):
                corpus = FakeCorpus()
                row = runner.trial(corpus, mode, logical_cpu=logical_cpu, background_spec=self.background(),
                                   accounting_factory=Accounting, duration=0.12, period=0.04, deadline=0.03)
                with self.subTest(mode=mode):
                    expected = 0 if mode == "background-only" else 3
                    self.assertEqual(row["scheduled_jobs"], expected)
                    self.assertEqual(row["completed_jobs"], expected)
                    self.assertEqual(len(corpus.calls), 0 if mode == "background-only" else 4)
                    progress = row["background"]
                    self.assertGreater(progress["completed_chunks"], 0)
                    self.assertIn(progress["checked_chunks"]-progress["completed_chunks"], (0, 1))
                    self.assertAlmostEqual(progress["chunks_per_second"], progress["completed_chunks"]/0.12, places=3)
                    self.assertGreater(row["background_live_process_cpu_ms"], 0)
                    self.assertEqual(row["affinity_after"]["background"]["allowed_logical_cpus"], [logical_cpu])
                    self.assertGreaterEqual(row["accounting_end_overrun_ms"], 0)

    def test__given_foreground_exceeding_window__when_background_stops_at_fixed_end__then_no_foreground_jobs_are_dropped(self):
        with runner.one_cpu_budget() as logical_cpu:
            corpus = FakeCorpus(delay=0.06)
            row = runner.trial(corpus, "cpu", logical_cpu=logical_cpu, background_spec=self.background(),
                               accounting_factory=Accounting, duration=0.06, period=0.02, deadline=0.015)
        self.assertEqual(row["completed_jobs"], 3)
        self.assertEqual([index for _, index in corpus.calls], [7, 0, 1, 2])
        self.assertEqual(row["deadline_misses"], 3)
        self.assertGreater(row["foreground_window_overrun_ms"], 100)
        self.assertLess(row["background"]["finish_overrun_ms"], row["foreground_window_overrun_ms"])

    def test__given_output_mismatch__when_foreground_runs__then_trial_fails_and_background_process_is_reaped(self):
        original = {process.pid for process in multiprocessing.active_children()}
        with runner.one_cpu_budget() as logical_cpu:
            with self.assertRaisesRegex(AssertionError, "output mismatch"):
                runner.trial(FakeCorpus(fail=True), "cpu", logical_cpu=logical_cpu,
                             background_spec=self.background(), accounting_factory=Accounting,
                             duration=0.06, period=0.02, deadline=0.015)
        self.assertEqual({process.pid for process in multiprocessing.active_children()}, original)

    def test__given_wrong_background_checksum__when_worker_warms_up__then_ready_barrier_rejects_it(self):
        with runner.one_cpu_budget() as logical_cpu:
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                runner.Background(logical_cpu, {"iterations": 1000, "expected_hex": "00"*32})

    def test__given_threads_with_different_priorities__when_auditing_budget__then_measurement_is_rejected(self):
        with patch.object(runner, "thread_masks", return_value={10: [1], 11: [1]}), \
                patch.object(runner.os, "getpriority", side_effect=lambda which, tid: int(tid == 11)), \
                patch.object(runner.os, "sched_getscheduler", return_value=0):
            with self.assertRaisesRegex(AssertionError, "unequal scheduling"):
                runner.audit_affinity(10, 1)
        with self.assertRaisesRegex(AssertionError, "unequal scheduling"):
            runner.equal_priorities({"runner": {"nice": 0, "scheduler_policy": 0},
                                     "background": {"nice": 1, "scheduler_policy": 0}})

    def test__given_pending_final_decoder_fence__when_trial_closes__then_published_counters_are_checked_after_drain(self):
        class Session:
            _process = types.SimpleNamespace(pid=os.getpid())

            def __init__(self):
                self.closed = False
                self.stats = {"pending_fences": 1}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True
                self.stats = {"pending_fences": 0}

        class Corpus(FakeCorpus):
            workloads = {runner.GROUP+"/merged": {"case": {"output_frames": {"output": 64}}}}

            def session(self, key):
                return Session()

            def submit(self, active, key, index):
                self.cpu(key, index)

        def checked(active, case, requests, copies):
            self.assertTrue(active.closed)
            self.assertEqual((requests, copies), (4, 4))
            return active.stats

        with runner.one_cpu_budget() as logical_cpu:
            row = runner.trial(Corpus(), "nvdec-merged", logical_cpu=logical_cpu,
                               background_spec=self.background(), accounting_factory=Accounting,
                               counter_check=checked, duration=0.06, period=0.02, deadline=0.015)
        self.assertEqual(row["decoder_stats_before_close"]["pending_fences"], 1)
        self.assertEqual(row["decoder_stats_after_close"]["pending_fences"], 0)

    def test__given_calibration__when_selecting_fixed_work__then_checksum_and_last_measurement_match_selected_iterations(self):
        result = runner.calibrate(target_seconds=0.0005)
        self.assertEqual(result["iterations"], result["calibration"][-1]["iterations"])
        self.assertEqual(result["expected_hex"], runner.chunk(result["iterations"]).hex())
        self.assertEqual(len(result["calibration"]), 3)

    def test__given_fake_validated_corpus__when_running_report_orchestration__then_every_mode_rotates_and_final_stability_is_required(self):
        created = []

        class Session:
            _process = types.SimpleNamespace(pid=os.getpid())
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

        class Corpus:
            fail_verify = False
            wrong_source = False

            def __init__(self, *args, **kwargs):
                self.identity = {"fake_test_corpus": True}
                self.bundle = {"source_sha256": {"codec_compute/workloads/guard_fusion/concurrency.py":
                                                 "wrong" if self.wrong_source else runner.digest(Path(runner.__file__).read_bytes())}}
                self.workloads = {runner.GROUP+"/"+variant: {"case": {"costs": {"total_pictures": 65}}}
                                  for variant in ("baseline", "merged")}
                self.cpu_calls, self.nv_calls = [], []
                self.closed = self.verified = False
                created.append(self)

            def cpu(self, key, index):
                self.cpu_calls.append((key, index))

            def session(self, key):
                return Session()

            def submit(self, active, key, index, full=False):
                self.nv_calls.append((key, index, full))

            def verify_stable(self):
                self.verified = True
                if self.fail_verify:
                    raise AssertionError("test frozen-source mismatch")

            def close(self):
                self.closed = True

        class Loader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                module.Corpus, module.Accounting = Corpus, Accounting
                module.counts = checked_counts

        def checked_counts(active, case, requests, copies):
            self.assertTrue(active.closed)
            self.assertEqual((requests, copies), (8, 520))
            return {"requests": requests, "copied_frames": copies, "pending_fences": 0}

        specification = importlib.util.spec_from_loader("_fake_concurrency_runtime", Loader())

        def fake_trial(corpus, mode, **kwargs):
            jobs = [] if mode == "background-only" else [{"release_to_completion_ms": 10}]*12
            if mode == "cpu":
                jobs[0] = {"release_to_completion_ms": 800}
            return {"mode": mode, "background": {"chunks_per_second": 100-runner.MODES.index(mode)},
                    "foreground_jobs": jobs, "completed_jobs": len(jobs), "deadline_misses": int(mode == "cpu"),
                    "foreground_process_cpu_ms": 10, "background_live_process_cpu_ms": 100,
                    "combined_process_cpu_ms": 110}

        with tempfile.TemporaryDirectory(prefix="concurrency-report-test-") as directory, \
                patch.object(runner.importlib.util, "spec_from_file_location", return_value=specification), \
                patch.object(runner, "trial", side_effect=fake_trial), \
                patch.object(runner, "calibrate", return_value=self.background()), redirect_stdout(io.StringIO()):
            output = Path(directory)/"report.json"
            args = dict(cpu_helper="unused", pcm_helper="unused", nvdec_helper="unused", build_info="unused")
            report = runner.run("unused", output, repeats=5, **args)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["runner_affinity_restored"])
            self.assertEqual(len(report["trials"]), 20)
            for repeat in range(5):
                self.assertEqual(tuple(row["mode"] for row in report["trials"] if row["repeat"] == repeat), runner.rotation(repeat))
            self.assertTrue(created[0].verified and created[0].closed)
            self.assertEqual(len(created[0].cpu_calls), 8)
            self.assertEqual(len(created[0].nv_calls), 16)
            self.assertTrue(all(call[2] for call in created[0].nv_calls))
            self.assertEqual(report["summary"]["cpu"]["foreground_jobs"], 60)
            self.assertEqual(report["deadline_qualification"]["cpu"]["status"], "missed")
            self.assertEqual(report["deadline_qualification"]["cpu"]["misses"], 5)
            self.assertEqual(report["validation"]["decoder_stats_after_close"]["merged"]["pending_fences"], 0)
            Corpus.fail_verify = True
            with self.assertRaisesRegex(AssertionError, "frozen-source mismatch"):
                runner.run("unused", output, repeats=1, **args)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")
            self.assertTrue(created[-1].closed)
            Corpus.wrong_source = True
            with self.assertRaisesRegex(AssertionError, "source differs from frozen bundle"):
                runner.run("unused", output, repeats=1, **args)
            self.assertEqual(created[-1].cpu_calls, [])
            self.assertTrue(created[-1].closed)

    def test__given_standalone_script__when_requesting_help__then_python38_syntax_and_no_ir_imports_are_required(self):
        path = Path(runner.__file__)
        ast.parse(path.read_text(), feature_version=(3, 8))
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.startswith("codec_compute"):
                raise AssertionError("Unexpected IR package import")
            return original_import(name, *args, **kwargs)

        with patch.object(sys, "argv", [str(path), "--help"]), patch("builtins.__import__", side_effect=checked_import), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                runpy.run_path(str(path), run_name="__main__")
        self.assertEqual(exited.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
