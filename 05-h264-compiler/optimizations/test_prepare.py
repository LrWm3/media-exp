"""Saved graph replay, exact stream costs, and portable comparison artifacts."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("optimization_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test__given_saved_nested_gather_parameters__when_replaying__then_hashable_graph_preserves_permutation(self):
        path = prepare.COMPILER / "native_ops" / "streams" / "tile-permutation.json"
        manifest = json.loads(path.read_text())
        program = prepare.compile_saved_graph(manifest)
        board = [[(y // 16) * 4 + x // 16 for x in range(64)] for y in range(64)]
        actual = program.reference(board)["output"]
        expected = [[15 - ((y // 16) * 4 + x // 16) for x in range(64)] for y in range(64)]
        self.assertEqual(actual, expected)
        gather = next(item for item in program.graph if item.op == "gather")
        self.assertIsInstance(gather.params[0], tuple)
        self.assertTrue(all(isinstance(pair, tuple) for pair in gather.params[0]))

    def test__given_binary_pcm_input__when_extracting__then_only_zero_and_255_become_binary_values(self):
        manifest = {"width": 16, "height": 16, "guard": 16, "pictures": 2,
                    "input_range": [0, 1], "seed_encoding": "binary 0/255"}
        raw = bytearray(48 * 48 * 3 // 2)
        for y in range(16):
            for x in range(16):
                raw[(y + 16) * 48 + x + 16] = 255 * ((x + y) % 2)
        self.assertEqual(prepare.extract_inputs(raw, manifest, False),
                         [[(x + y) % 2 for x in range(16)] for y in range(16)])
        raw[16 * 48 + 16] = 1
        with self.assertRaisesRegex(AssertionError, "nonbinary PCM"):
            prepare.extract_inputs(raw, manifest, False)

    def test__given_mixed_annexb_prefixes__when_splitting_and_counting__then_segments_and_prediction_bytes_are_exact(self):
        prefix = b"\x00\x00\x00\x01"
        short = b"\x00\x00\x01"
        headers = prefix + b"\x67sps" + short + b"\x68pps"
        inputs = prefix + b"\x65first\x00\x00\x03\x01" + short + b"\x61second"
        tail = prefix + b"\x61prediction-one" + short + b"\x61prediction-two"
        segment = headers + inputs + tail
        self.assertEqual(prepare.split_segments(segment * 2), [segment, segment])
        manifest = {"pictures": 3, "inputs": [{"name": "a"}, {"name": "b"}],
                    "refresh_frames": [], "stats": {"peak_live_values": 2}}
        costs = prepare.stream_costs(segment, manifest)
        self.assertEqual(costs["prediction_slices"], 2)
        self.assertEqual(costs["prediction_bytes"], len(tail))
        self.assertEqual(costs["stream_bytes"], len(segment))
        self.assertEqual(costs["input_pictures"], 2)
        with self.assertRaisesRegex(ValueError, "SPS"):
            prepare.split_segments(inputs + tail)
        with self.assertRaisesRegex(ValueError, "IDR"):
            prepare.split_segments(headers + tail)

    def test__given_case_groups__when_selecting__then_all_42_original_jobs_remain_represented(self):
        cases = prepare.selected_cases()
        self.assertEqual(len(cases), 11)
        self.assertEqual(sum(case.segments for case in cases), 42)
        self.assertEqual(len(prepare.selected_cases(["native"])), 6)
        self.assertEqual([item.name for item in prepare.selected_cases(["native-affine"])], ["native-affine"])
        mnist, = prepare.selected_cases(["mnist"])
        self.assertEqual(mnist.items_per_segment * mnist.segments, 16)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            prepare.selected_cases(["missing"])

    def test__given_saved_life_graph__when_disabling_only_exact_search__then_682_pictures_become_402(self):
        saved, = prepare.selected_cases(["life"])
        manifest = json.loads((prepare.REPOSITORY / saved.manifest).read_text())
        baseline = prepare.compile_saved_graph(manifest, exact_search=False)
        optimized = prepare.compile_saved_graph(manifest, exact_search=True)
        self.assertEqual(baseline.pictures, 682)
        self.assertEqual(optimized.pictures, 402)
        self.assertFalse(baseline.stats["exact_search_enabled"])
        self.assertTrue(optimized.stats["exact_search_enabled"])
        self.assertEqual(baseline.reference_mode, optimized.reference_mode)
        self.assertEqual(baseline.slice_mode, optimized.slice_mode)
        self.assertEqual(baseline.graph_outputs.keys(), optimized.graph_outputs.keys())

    @unittest.skipUnless(prepare.available("ffmpeg"), "FFmpeg unavailable")
    def test__given_committed_add_graph__when_preparing_comparison__then_portable_hashes_outputs_and_baselines_match(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            suite = prepare.prepare(output, ["native-saturating-add"], ["ffmpeg"])
            self.assertEqual(json.loads((output / "suite.json").read_text()), suite)
            self.assertEqual(suite["summary"]["local_backends"]["ffmpeg"], {"status": "pass", "cases": 1})
            case, = suite["cases"]
            self.assertTrue(case["comparison"]["named_outputs_equal"])
            self.assertTrue(case["comparison"]["independent_graph_equal"])
            for key in ("stream", "expected"):
                self.assertEqual(prepare.sha256((output / case[key]).read_bytes()), case[key + "_sha256"])
            for named in case["named_outputs"].values():
                self.assertEqual(prepare.sha256((output / named["path"]).read_bytes()), named["sha256"])
            representative = suite["representatives"]["native-saturating-add"]
            baseline, optimized = representative["baseline"], representative["optimized"]
            self.assertEqual((output / baseline["expected"]).read_bytes(),
                             (output / optimized["expected"]).read_bytes())
            self.assertEqual(prepare.sha256((output / baseline["stream"]).read_bytes()), baseline["stream_sha256"])
            self.assertEqual(case["comparison"]["before"]["prediction_slices"], 36)
            self.assertLess(case["comparison"]["after"]["prediction_slices"], 36)

    @unittest.skipUnless(prepare.available("ffmpeg"), "FFmpeg unavailable")
    def test__given_no_search_baseline__when_preparing_comparison__then_fresh_stream_costs_and_full_raw_replace_history(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            suite = prepare.prepare(output, ["native-saturating-add"], ["ffmpeg"], baseline="no-exact-search")
            self.assertEqual(json.loads((output / "suite.json").read_text()), suite)
            self.assertEqual(suite["baseline"], "no-exact-search")
            self.assertEqual(suite["summary"]["baseline"], "no-exact-search")
            self.assertIn("exact_search=False vs exact_search=True", suite["comparison"])
            case, = suite["cases"]
            comparison = case["comparison"]
            representative = suite["representatives"][case["name"]]
            baseline = representative["baseline"]
            stream = (output / baseline["stream"]).read_bytes()
            raw = (output / baseline["expected"]).read_bytes()
            manifest = json.loads((output / comparison["baseline_manifest"]).read_text())
            self.assertFalse(manifest["stats"]["exact_search_enabled"])
            self.assertEqual(baseline["stream"], comparison["baseline_stream"])
            self.assertEqual(comparison["before"], prepare.stream_costs(stream, manifest))
            self.assertEqual(comparison["before"]["prediction_slices"], comparison["after"]["prediction_slices"])
            self.assertLess(comparison["before"]["prediction_slices"], 36)
            self.assertEqual(prepare.sha256(stream), baseline["stream_sha256"])
            self.assertEqual(prepare.sha256(stream), comparison["baseline_segment_sha256"])
            self.assertEqual(prepare.sha256(raw), baseline["expected_sha256"])
            self.assertEqual(prepare.sha256(raw), comparison["baseline_decoded_sha256"])
            self.assertEqual(len(raw), manifest["pictures"] * representative["frame_bytes"])
            self.assertEqual(len(raw), baseline["expected_bytes"])
            self.assertEqual(baseline["pictures"], manifest["pictures"])
            self.assertEqual(raw, prepare.decode(output / baseline["stream"], "ffmpeg"))
            self.assertTrue(comparison["named_outputs_equal"])
            self.assertTrue(comparison["independent_graph_equal"])
            historical = (prepare.REPOSITORY / case["source_stream"]).read_bytes()
            self.assertEqual(prepare.sha256(historical), comparison["historical_segment_sha256"])
            self.assertEqual(prepare.sha256(historical), suite["historical_source_sha256"][case["source_stream"]])
            self.assertNotEqual(comparison["historical_segment_sha256"], comparison["baseline_segment_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
