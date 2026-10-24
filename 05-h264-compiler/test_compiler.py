"""Compiler invariants checked against unmodified decoders and integer semantics."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import h264
from decoder_core.decode import available


class CompilerTests(unittest.TestCase):
    def image(self, size=16):
        rng = random.Random(4)
        return [[rng.randrange(2) for _ in range(size)] for _ in range(size)]

    def test__given_shifted_or__when_fusing__then_two_prediction_operations_and_all_decoders_match(self):
        def kernel(x):
            return h264.bit_or(h264.shift(x, 1, 0), h264.shift(x, 0, 1))
        program = h264.compile(kernel, 16, 16)
        self.assertEqual(program.pictures, 4)  # seed, normalization, B sum, P OR
        self.assertEqual(len(program.operations[1].sources), 2)
        self.assertEqual(program.operations[2].denominator, 7)
        self.assertEqual(program.operations[2].sources[0].weight, 64)
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                if not available(backend):
                    self.skipTest(f"{backend} unavailable")
                program.verify(self.image(), backend)

    def test__given_all_u8_values_and_cutoffs__when_thresholding__then_every_comparison_is_exact(self):
        program = h264.compile(lambda x: {str(t): h264.threshold(x, t) for t in range(-1, 258)},
                               16, 16, input_range=(0, 255))
        image = [[y*16+x for x in range(16)] for y in range(16)]
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                if not available(backend):
                    self.skipTest(f"{backend} unavailable")
                outputs = program.verify(image, backend)
                for cutoff, output in outputs.items():
                    self.assertEqual(output, [[int(v >= int(cutoff)) for v in row] for row in image])

    def test__given_opposing_nested_shifts__when_optimizing__then_intermediate_border_cropping_is_preserved(self):
        image = self.image()
        for row in image:
            row[0] = 1
        def kernel(x):
            return h264.threshold(h264.shift(h264.shift(x, 1, 0), -1, 0), 1)
        optimized = h264.compile(kernel, 16, 16)
        plain = h264.compile(kernel, 16, 16, optimize=False)
        result = optimized.verify(image)["output"]
        self.assertEqual(result, plain.verify(image)["output"])
        self.assertTrue(all(row[0] == 0 for row in result))

    def test__given_nonzero_bias_and_shift__when_fusing__then_outside_read_is_zero_before_bias(self):
        def kernel(x):
            return {"output": h264.weighted(h264.shift(x, 1), bias=5),
                    "shifted_constant": h264.shift(h264.weighted(x, 0, 1), 1)}
        program = h264.compile(kernel, 16, 16)
        result = program.verify(self.image())
        self.assertTrue(all(row[-1] == 5 for row in result["output"]))
        self.assertTrue(all(row[-1] == 0 for row in result["shifted_constant"]))

    def test__given_duplicate_and_dead_expressions__when_compiling__then_only_required_common_values_remain(self):
        def kernel(x):
            h264.weighted(x, 100, 10)  # deliberately unused
            a = h264.bit_or(x, h264.shift(x, 1))
            b = h264.bit_or(x, h264.shift(x, 1))
            return {"output": h264.bit_and(a, b), "a": a, "b": b}
        program = h264.compile(kernel, 16, 16)
        self.assertGreater(program.stats["cse_eliminated"], 0)
        self.assertEqual(program.stats["dead_nodes"], 1)
        self.assertEqual(program.output_frames["a"], program.output_frames["b"])
        program.verify(self.image())

    def test__given_long_lived_input__when_unrolling__then_refreshes_keep_every_reference_legal(self):
        def kernel(x):
            current = h264.repeat(70, h264.bit_not, x)
            return h264.bit_or(x, current)
        program = h264.compile(kernel, 16, 16)
        self.assertGreater(len(program.refreshes), 0)
        self.assertTrue(all(1 <= frame-s.frame <= 16 for frame, op in enumerate(program.operations, 1) for s in op.sources))
        self.assertEqual(program.verify(self.image())["output"], self.image())

    def test__given_long_static_graph__when_tracing__then_python_recursion_depth_is_not_a_limit(self):
        program = h264.compile(lambda x: h264.repeat(1500, h264.bit_not, x), 16, 16)
        self.assertEqual(program.pictures, 1502)
        self.assertEqual(program.verify(self.image())["output"], self.image())

    def test__given_signed_dot_and_large_bias__when_lowering__then_all_supported_u8_terms_remain_exact(self):
        image = self.image()
        def kernel(x):
            terms = [h264.shift(x, i%15, i//15) for i in range(196)]
            return {"output": h264.binary_dot(terms, [-1]*196, -20),
                    "bias255": h264.weighted(x, -1, 255),
                    "positive": h264.binary_dot(terms, [1]*196, 20)}
        program = h264.compile(kernel, 16, 16)
        program.verify(image)

    def test__given_input_changes__when_reusing_tail__then_compilation_needs_no_host_pixel_arithmetic(self):
        program = h264.compile(lambda x: h264.bit_or(x, h264.shift(x, 1)), 16, 16)
        first, second = self.image(), [[0]*16 for _ in range(16)]
        with patch("h264.backend.evaluate", side_effect=AssertionError("host inference")):
            first_stream, second_stream = program.encode(first), program.encode(second)
        self.assertEqual(first_stream[len(program.writer.seed(first)):], second_stream[len(program.writer.seed(second)):])
        self.assertEqual(program.decode(second)["output"], second)

    def test__given_saved_program__when_inspecting_manifest__then_graph_and_reference_frames_describe_executable(self):
        program = h264.compile(lambda x: {"output": h264.bit_not(x), "input": x}, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = program.save(Path(directory)/"test.h264", self.image())
            manifest = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(manifest["output_frames"], program.output_frames)
            self.assertEqual(manifest["pictures"], program.pictures)
            self.assertTrue(manifest["graph"])
            self.assertTrue(manifest["prediction_sha256"])
        program.verify(self.image())

    def test__given_unsafe_arithmetic_or_control_flow__when_compiling__then_no_approximation_is_silently_used(self):
        cases = [lambda x: h264.weighted(x, 256), lambda x: h264.combine(x, x, 64, 1),
                 lambda x: h264.shift(x, 128), lambda x: h264.threshold(x, .5),
                 lambda x: h264.weighted(x, -1), lambda x: h264.repeat(-1, h264.bit_not, x)]
        for function in cases:
            with self.subTest(function=function), self.assertRaises(h264.CompileError):
                h264.compile(function, 16, 16)
        with self.assertRaises(h264.CompileError):
            h264.compile(lambda x: h264.add(x, x), 16, 16, input_range=(0, 255))
        with self.assertRaises(h264.CompileError):
            h264.compile(lambda x: h264.weighted(x, 127, -253), 16, 16, input_range=(3, 4))
        with self.assertRaises(TypeError):
            h264.compile(lambda x: x if x else h264.bit_not(x), 16, 16)
        with self.assertRaises(TypeError):
            h264.compile(lambda x: x == 1, 16, 16)

    def test__given_excess_live_values__when_scheduling__then_compiler_fails_instead_of_losing_references(self):
        def kernel(x):
            values = [h264.bit_or(x, h264.shift(x, offset)) for offset in range(1, 20)]
            total = values[0]
            for value in values[1:]:
                total = h264.add(total, value)
            for value in values:
                total = h264.add(total, value)
            return total
        with self.assertRaisesRegex(h264.CompileError, "live values"):
            h264.compile(kernel, 16, 16)

    def test__given_wrong_seed_or_missing_pictures__when_running__then_validation_fails(self):
        program = h264.compile(lambda x: x, 16, 16)
        with self.assertRaises(ValueError):
            program.encode([[0]])
        with self.assertRaises(ValueError):
            program.encode([[2]*16 for _ in range(16)])
        with self.assertRaisesRegex(AssertionError, "picture count"):
            program.read(b"")

    def test__given_colliding_manifest_path__when_saving__then_existing_file_is_not_overwritten(self):
        program = h264.compile(lambda x: x, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"program.json"
            path.write_text("existing content")
            with self.assertRaisesRegex(ValueError, "collides"):
                program.save(path, self.image())
            self.assertEqual(path.read_text(), "existing content")
        with self.assertRaises(h264.CompileError):
            h264.compile(lambda x: x, width=16.0, height=16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
