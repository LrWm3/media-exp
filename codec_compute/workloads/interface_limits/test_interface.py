"""Local decoder export and fake-worker tests; no NVDEC hardware claim."""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import benchmark, build, capture, prepare, runtime


class FakeSession:
    """Model one delayed fence, selected copies and exact prepared real frames."""
    protocol_version = 2

    def __init__(self, workload):
        self.w = workload
        self.closed = False
        self.events = []
        self._process = mock.Mock(pid=12345)
        self.stats = dict(requests=0, submitted_frames=0, decoded_frames=0, copied_frames=0,
                          fence_pictures=0, decoded_fences=0, pending_fences=0,
                          pipeline_starts=1, fence_bytes=0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, units, fence, selected):
        if self.closed or fence != self.w["fence"]:
            raise AssertionError("Wrong fence or closed worker")
        job = next(job for job in self.w["jobs"] if tuple(units) == job["units"])
        frames = job["frames"]
        previous = self.stats["pending_fences"]
        self.stats["requests"] += 1
        self.stats["submitted_frames"] += len(frames)+1
        self.stats["decoded_frames"] += len(frames)+previous
        self.stats["decoded_fences"] += previous
        self.stats["fence_pictures"] += 1
        self.stats["fence_bytes"] += len(fence)
        self.stats["pending_fences"] = 1
        result = frames if selected is None else tuple(frames[i] for i in selected)
        self.stats["copied_frames"] += len(result)
        return result

    def submit(self, units, selected=None, *, fence):
        self.events.append(("full" if selected is None else "selected", self.w["jobs"].index(
            next(job for job in self.w["jobs"] if tuple(units) == job["units"]))))
        return self.request(units, fence, selected)

    def complete(self, units, *, fence):
        self.events.append(("complete", self.w["jobs"].index(
            next(job for job in self.w["jobs"] if tuple(units) == job["units"]))))
        return self.request(units, fence, ())

    def close(self):
        if not self.closed:
            self.stats["decoded_frames"] += self.stats["pending_fences"]
            self.stats["decoded_fences"] += self.stats["pending_fences"]
            self.stats["pending_fences"] = 0
            self.closed = True


class FakeAccounting:
    def __init__(self, pid=None):
        self.tick = 0

    def snapshot(self):
        self.tick += 0.001
        return {"parent_cpu_s": self.tick, "parent_user_s": self.tick, "parent_system_s": 0}


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="media-interface-tests-")
        cls.root = Path(cls.temp.name)
        cls.helpers, cls.bundle = cls.root/"helpers", cls.root/"bundle"
        build.build(cls.helpers)
        cls.export = prepare.prepare(cls.bundle, cpu_helper=cls.helpers/"libcpu.so",
                                     planes_helper=cls.helpers/"libcpu_planes.so", size=32, depths=(1, 2))
        cls.arguments = dict(cpu_helper=cls.helpers/"libcpu.so", planes_helper=cls.helpers/"libcpu_planes.so",
                             pcm_helper=cls.helpers/"libpcm.so", nvdec_helper=Path(__file__),
                             build_info=cls.helpers/"build.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def corpus(self):
        corpus = runtime.Corpus(self.bundle, **self.arguments)
        self.addCleanup(corpus.close)
        return corpus

    def test__given_three_layouts_and_four_inputs__when_exported__then_all_local_decoders_and_native_outputs_agree(self):
        self.assertEqual(self.export["status"], "pass")
        self.assertIn("ffmpeg", self.export["decoders"])
        self.assertEqual(self.export["validation"]["streams"], 60)
        self.assertEqual(self.export["validation"]["local_decoder_runs"], 60*len(self.export["decoders"]))
        self.assertEqual(self.export["validation"]["paired_complete_i420"], 40)
        corpus = self.corpus()
        self.assertEqual(len(corpus.groups), 5)
        for variants in corpus.groups.values():
            for index in range(4):
                corpus.cpu(variants["rows-1"], index)
        corpus.verify_stable()

    def test__given_paired_layouts__when_grouping_slices__then_pictures_and_full_i420_stay_equal(self):
        corpus = self.corpus()
        for variants in corpus.groups.values():
            a, b, c = [corpus.workloads[variants[v]]["case"]["costs"] for v in runtime.LAYOUTS]
            self.assertEqual(a["total_pictures"], c["total_pictures"])
            self.assertGreaterEqual(a["prediction_slices"], b["prediction_slices"])
            self.assertGreaterEqual(b["prediction_slices"], c["prediction_slices"])
        chain = corpus.groups["32/resident-chain-2"]
        self.assertGreater(corpus.workloads[chain["rows-1"]]["case"]["costs"]["prediction_slices"],
                           corpus.workloads[chain["full"]]["case"]["costs"]["prediction_slices"])

    def test__given_weaker_either_cpu_build__when_loading__then_comparison_is_rejected(self):
        original = json.loads((self.helpers/"build.json").read_text())
        for name in ("cpu", "cpu_planes"):
            with self.subTest(name=name):
                weak = copy.deepcopy(original)
                weak["flags"][name].remove("-funroll-loops")
                path = self.root/(name+"-weak.json")
                path.write_text(json.dumps(weak))
                with self.assertRaisesRegex(AssertionError, "Both CPU baselines"):
                    runtime.Corpus(self.bundle, **dict(self.arguments, build_info=path))

    def test__given_forged_native_motion__when_auditing__then_semantic_graph_binding_rejects_it(self):
        case = copy.deepcopy(self.export["cases"]["32/motion-interpolation/full"])
        case = json.loads(json.dumps(case))
        case["cpu"]["field_a"][0][0] += 1
        with self.assertRaisesRegex(AssertionError, "semantic graph"):
            runtime.audit_cpu(self.bundle, "32/motion-interpolation/full", case)

    def test__given_changed_layout_or_reference__when_pairing__then_mismatch_is_rejected(self):
        _, bundle, workloads = runtime.resident.load(self.bundle)
        key = "32/resident-chain-2/full"
        workloads[key]["case"]["writer"]["prediction_slice_rows"] = 1
        with self.assertRaisesRegex(AssertionError, "Writer layout"):
            runtime.audit_pairs(self.bundle, bundle, workloads)
        workloads[key]["case"]["writer"]["prediction_slice_rows"] = 0
        frames = workloads[key]["jobs"][0]["frames"]
        workloads[key]["jobs"][0]["frames"] = frames[:-1]+(bytes(len(frames[-1])),)
        with self.assertRaisesRegex(AssertionError, "complete I420"):
            runtime.audit_pairs(self.bundle, bundle, workloads)

    def test__given_reused_row_tail_or_manifest__when_auditing_full_layout__then_mislabel_is_rejected(self):
        _, bundle, workloads = runtime.resident.load(self.bundle)
        key = "32/resident-chain-2/full"
        row = workloads["32/resident-chain-2/rows-1"]
        original = workloads[key]["case"]["manifest"]
        workloads[key]["case"]["manifest"] = row["case"]["manifest"]
        with self.assertRaisesRegex(AssertionError, "Saved manifest"):
            runtime.audit_pairs(self.bundle, bundle, workloads)
        workloads[key]["case"]["manifest"] = original
        workloads[key]["tail"] = row["tail"]
        with self.assertRaisesRegex(AssertionError, "Actual prediction NAL"):
            runtime.audit_pairs(self.bundle, bundle, workloads)

    def test__given_reordered_input_names__when_binding_cpu_spec__then_reject(self):
        case = copy.deepcopy(self.export["cases"]["32/motion-interpolation/full"])
        case["input_names"] = ["b", "a"]
        with self.assertRaisesRegex(AssertionError, "ordered full-u8"):
            runtime.audit_cpu(self.bundle, "32/motion-interpolation/full", case)

    def test__given_native_and_prepared_inputs__when_submitting__then_only_native_reencodes_every_plane(self):
        corpus = self.corpus()
        key = "32/motion-interpolation/rows-1"
        w = corpus.workloads[key]
        with mock.patch.object(w["encoder"], "input_luma", wraps=w["encoder"].input_luma) as encode:
            active = FakeSession(w)
            corpus.submit(active, key, 0, mode="native-host")
            self.assertEqual(encode.call_count, 2)
            corpus.submit(active, key, 1, mode="prepared-host")
            self.assertEqual(encode.call_count, 2)
            corpus.complete(active, key, 2)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(active.stats["copied_frames"], 2)
            active.close()

    def test__given_returned_guard_corruption__when_checking_host_result__then_full_i420_fails(self):
        corpus = self.corpus()
        key = "32/resident-chain-2/full"
        active = FakeSession(corpus.workloads[key])
        original = active.submit
        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            return (b"\x01"+result[0][1:],)+result[1:]
        with mock.patch.object(active, "submit", side_effect=corrupted):
            with self.assertRaisesRegex(AssertionError, "full I420"):
                corpus.submit(active, key, 0)
        active.close()

    def test__given_completion_mode__when_timed__then_zero_copies_and_postflight_drain_are_explicit(self):
        corpus = self.corpus()
        key = "32/resident-chain-2/full"
        active = FakeSession(corpus.workloads[key])
        with mock.patch.object(corpus, "session", return_value=active), mock.patch.object(benchmark.runtime, "Accounting", FakeAccounting):
            result = benchmark.trial(corpus, key, "prepared-complete", 0)
        self.assertEqual(active.events, [("complete", 3)]+[("complete", i) for i in range(4)]+[("full", i) for i in range(4)])
        self.assertEqual(result["timed_copied_frames"], 0)
        self.assertEqual(result["postflight_full_i420_jobs"], 4)
        self.assertEqual(result["stats_after_timing_before_final_drain"]["pending_fences"], 1)
        after = result["stats_after_close_including_warmup_and_postflight"]
        self.assertEqual(after["pending_fences"], 0)
        self.assertEqual(after["requests"], 9)
        self.assertEqual(after["copied_frames"], 12)
        self.assertTrue(active.closed)

    def test__given_old_protocol_or_unexpected_completion_result__when_completing__then_fail(self):
        corpus = self.corpus()
        key = "32/control/full"
        active = FakeSession(corpus.workloads[key])
        active.protocol_version = 1
        with self.assertRaisesRegex(RuntimeError, "protocol v2"):
            corpus.complete(active, key, 0)
        active.protocol_version = 2
        with mock.patch.object(active, "complete", return_value=(b"unexpected",)):
            with self.assertRaisesRegex(AssertionError, "unexpected host results"):
                corpus.complete(active, key, 0)
        active.close()

    def test__given_rotated_repeats__when_ordering__then_every_layout_mode_occurs_and_cpu_occurs_once(self):
        groups = {"512/control": {}, "512/resident-chain-64": {}}
        orders = [benchmark.ordered_trials(groups, repeat) for repeat in range(5)]
        self.assertEqual(len({tuple(order) for order in orders}), 5)
        for order in orders:
            self.assertEqual(len(order), 20)
            self.assertEqual(len(set(order)), 20)
            for group in groups:
                self.assertEqual(sum(g == group and mode == "cpu" for g, _, mode in order), 1)

    def test__given_fake_worker_full_run__when_report_passes__then_scope_counters_and_provenance_are_present(self):
        corpus = self.corpus()
        sessions = []
        def session(key):
            active = FakeSession(corpus.workloads[key])
            sessions.append(active)
            return active
        output = self.root/"fake-report.json"
        with mock.patch.object(benchmark.runtime, "Corpus", return_value=corpus), \
                mock.patch.object(corpus, "session", side_effect=session), \
                mock.patch.object(benchmark.runtime, "Accounting", FakeAccounting):
            report = benchmark.run(self.bundle, output, **self.arguments, repeats=1)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["validation"]["full_nvdec_preflight"]), 15)
        self.assertIn("Unequal-interface", report["protocol"]["prepared-complete"])
        self.assertIn("No readback", report["protocol"]["prepared-complete"])
        self.assertEqual(report["bundle_sha256"], runtime.common.digest((self.bundle/"bundle.json").read_bytes()))
        self.assertTrue(all(active.closed for active in sessions))
        self.assertTrue(all(len(record["modes"]) == 10 for record in report["groups"].values()))

    def test__given_saved_capture__when_rendering__then_pixels_and_references_are_exact(self):
        from PIL import Image
        output = self.root/"capture"
        result = capture.capture(self.bundle, output)
        self.assertEqual(len(result["panels"]), 4)
        self.assertEqual(result["display"]["scale"], 1)
        self.assertFalse(result["display"]["contrast_stretch"])
        for panel in result["panels"]:
            with Image.open(output/panel["path"]) as image:
                self.assertEqual(runtime.common.digest(image.tobytes()), panel["active_luma_sha256"])
        self.assertEqual(result["panels"][0]["active_luma_sha256"], result["panels"][1]["active_luma_sha256"])

    def test__given_portable_runners__when_parsed_for_python38__then_no_modern_syntax_is_required(self):
        for name in ("build.py", "runtime.py", "benchmark.py", "capture.py"):
            ast.parse((Path(__file__).parent/name).read_text(), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
