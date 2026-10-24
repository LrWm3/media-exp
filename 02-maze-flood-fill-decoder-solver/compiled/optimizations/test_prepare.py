"""Portable maze bundle proof, selection mapping, and decoder readback accounting."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("maze_optimization_prepare", "prepare.py")
benchmark = load("maze_optimization_benchmark", "benchmark.py")


class PreparationTests(unittest.TestCase):
    def test__given_fixture_suite__when_building__then_exhaustive_and_long_distance_cases_remain_present(self):
        cases = {name: (maze, steps) for name, maze, steps in prepare.selected_cases()}
        self.assertEqual(len(cases), 10)
        self.assertEqual(cases["exhaustive"][0].width, 64)
        self.assertEqual(cases["exhaustive"][1], 3)
        maze, steps = cases["long-corridor"]
        _, solution = prepare.bfs_reference(maze, steps)
        self.assertEqual(solution["distances"][29][30], 463)
        self.assertEqual(len(solution["path"]), 464)
        self.assertEqual(solution["converged_at"], 464)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            prepare.selected_cases(["missing"])

    def test__given_paired_demo_kernels__when_compiling__then_265_and_214_pictures_share_54_named_pictures(self):
        for compiler, pictures in ((prepare.compile_baseline, 265), (prepare.compile_maze, 214)):
            with self.subTest(pictures=pictures):
                program = compiler(32, 32, 52)
                self.assertEqual(program.pictures, pictures)
                self.assertEqual(len(set(program.output_frames.values())), 54)
                self.assertEqual(program.output_frames["output"], program.output_frames["step_52"])

    def test__given_portable_runner__when_parsing_for_python38__then_no_compiler_imports_are_required(self):
        tree = ast.parse(Path(benchmark.__file__).read_text(), feature_version=(3, 8))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertIn("decoder_core.decode", imports)
        self.assertFalse(any(name and (name.startswith("h264") or "maze" in name) for name in imports))

    @unittest.skipUnless(prepare.available("ffmpeg"), "FFmpeg unavailable")
    def test__given_tiny_paired_bundle__when_running_four_modes__then_hashes_captures_and_actual_readbacks_match(self):
        maze = prepare.parse_ascii("S..G")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "bundle"
            suite = prepare.prepare(root, local_backends=["ffmpeg"], cases=[("demo", maze, 4)])
            self.assertEqual(json.loads((root / "bundle.json").read_text()), suite)
            case, = suite["cases"]
            self.assertEqual(case["solution"]["distances"][0][3], 3)
            self.assertEqual(case["solution"]["converged_at"], 4)
            self.assertTrue(case["comparison"]["all_states_match_bfs"])
            for variant in case["variants"].values():
                self.assertEqual(variant["selected_frames"], sorted(set(variant["output_frames"].values())))
                self.assertLess(len(variant["selected_frames"]), variant["pictures"])
                manifest = json.loads((root / variant["manifest"]).read_text())
                self.assertEqual(manifest["stream_sha256"], variant["stream_sha256"])
                self.assertEqual(len(manifest["prediction_sha256"]), 64)
            with patch.object(benchmark, "decode_batch", wraps=benchmark.decode_batch) as decode_batch:
                report = benchmark.benchmark(root, Path(directory) / "results", backend="ffmpeg", repeats=1)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(decode_batch.call_count, 12)
            self.assertTrue(all(len(call.args[0]) == 1 for call in decode_batch.call_args_list))
            for mode in report["benchmark"]["modes"].values():
                item = case["variants"][mode["variant"]]
                self.assertEqual(mode["decoded_frames"], item["pictures"])
                self.assertEqual(mode["copied_frames"], item["pictures"])
                expected_count = len(item["selected_frames"]) if mode["readback"] == "outputs" else item["pictures"]
                self.assertEqual(mode["returned_bytes"], expected_count * case["frame_bytes"])
                self.assertTrue(mode["full_i420_byte_equality"])
            capture = report["capture"]
            raw = (Path(directory) / "results" / capture["path"]).read_bytes()
            self.assertEqual(benchmark.digest(raw), capture["sha256"])
            self.assertEqual(capture["sha256"], case["variants"]["optimized"]["selected_sha256"])
            self.assertEqual(capture["selected_frames"], case["variants"]["optimized"]["selected_frames"])
            # Corruption must be rejected before any decoder process starts.
            stream = root / case["variants"]["baseline"]["stream"]
            stream.write_bytes(stream.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                benchmark.load_bundle(root)

    @unittest.skipUnless(prepare.available("ffmpeg"), "FFmpeg unavailable")
    def test__given_named_output_aliases__when_selection_drops_intermediate_state__then_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            report = prepare.prepare(root, local_backends=["ffmpeg"],
                                     cases=[("tiny", prepare.parse_ascii("S.G"), 2)])
            report["cases"][0]["variants"]["optimized"]["selected_frames"].pop(1)
            (root / "bundle.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "exactly all named"):
                benchmark.load_bundle(root)

    def test__given_bundle_relative_path__when_escaping_root__then_access_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes"):
                benchmark.checked_file(Path(directory), "../outside", "unused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
