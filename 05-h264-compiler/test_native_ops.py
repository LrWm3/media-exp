"""Native affine arithmetic, multiple inputs, and persistent decoder references."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h264
from decoder_core.decode import available


class NativeCompilerTests(unittest.TestCase):
    def backends(self):
        return [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]

    def test__given_every_u8_pair__when_saturating_add_runs__then_one_b_picture_matches_all_65536_sums(self):
        program = h264.compile(h264.saturating_add, 64, 64, input_ranges={"a": (0, 255), "b": (0, 255)})
        self.assertEqual(program.pictures, 3)
        self.assertEqual(program.output_frames, {"output": 2})
        self.assertEqual(len(program.operations), 1)
        self.assertEqual([source.weight for source in program.operations[0].sources], [2, 2])
        for chunk in range(16):
            a = [[(chunk*4096+y*64+x)//256 for x in range(64)] for y in range(64)]
            b = [[(y*64+x)%256 for x in range(64)] for y in range(64)]
            expected = [[min(255, av+bv) for av, bv in zip(ar, br)] for ar, br in zip(a, b)]
            for backend in self.backends():
                with self.subTest(chunk=chunk, backend=backend):
                    self.assertEqual(program.verify({"a": a, "b": b}, backend)["output"], expected)

    def test__given_signed_weights_and_offsets__when_saturating__then_clipping_occurs_after_the_complete_sum(self):
        unary = [(2, -128), (-1, 127), (127, -127), (-128, 127), (0, -128), (1, 127)]
        binary = [(1, -1, 127), (63, -64, 127), (-64, 63, -128), (-1, -1, 127), (2, 3, -127)]
        def kernel(a, b):
            results = {f"u{i}": h264.saturating_weighted(a, w, bias) for i, (w, bias) in enumerate(unary)}
            results.update({f"b{i}": h264.saturating_combine(a, b, wa, wb, bias)
                            for i, (wa, wb, bias) in enumerate(binary)})
            return results
        program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        a = [[y*16+x for x in range(16)] for y in range(16)]
        b = [[(value*73+41)%256 for value in row] for row in a]
        for backend in self.backends():
            with self.subTest(backend=backend):
                program.verify({"a": a, "b": b}, backend)

    def test__given_long_chain_and_two_original_inputs__when_refreshing__then_both_inputs_remain_available(self):
        def kernel(a, b):
            value = h264.saturating_add(a, b)
            for i in range(75):
                value = h264.saturating_combine(value, a if i%2 else b, wb=-1, bias=67)
            return {"output": value, "original_a": a, "original_b": b}
        program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        image = {"a": [[y*16+x for x in range(16)] for y in range(16)],
                 "b": [[(x*17+y*23)%256 for x in range(16)] for y in range(16)]}
        self.assertGreater(len(program.refreshes), 0)
        self.assertTrue(all(1 <= frame-source.frame <= 16
                            for frame, operation in enumerate(program.operations, program.input_count)
                            for source in operation.sources))
        for backend in self.backends():
            with self.subTest(backend=backend):
                result = program.verify(image, backend)
                self.assertEqual(result["original_a"], image["a"])
                self.assertEqual(result["original_b"], image["b"])

    def test__given_fifteen_independent_inputs__when_loaded_and_reused__then_scheduler_handles_initial_picture_ages(self):
        ranges = {f"x{i}": (0, 1) for i in range(15)}
        def kernel(**images):
            values = list(images.values())
            total = h264.add(values[0], values[1])
            for value in values[2:]+values[:1]:
                total = h264.add(total, value)
            return total
        program = h264.compile(kernel, 16, 16, input_ranges=ranges)
        images = {name: [[i%2]*16 for _ in range(16)] for i, name in enumerate(ranges)}
        self.assertTrue(program.refreshes)
        self.assertFalse(program.binary_input)  # Named binary inputs are literal 0/1.
        for backend in self.backends():
            with self.subTest(backend=backend):
                self.assertEqual(program.verify(images, backend)["output"], [[7]*16 for _ in range(16)])

    def test__given_changed_inputs__when_reusing_predictions__then_only_pcm_loading_changes(self):
        program = h264.compile(h264.saturating_add, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        first = {"a": [[200]*16 for _ in range(16)], "b": [[100]*16 for _ in range(16)]}
        second = {"a": [[10]*16 for _ in range(16)], "b": [[20]*16 for _ in range(16)]}
        with patch("h264.backend.evaluate", side_effect=AssertionError("CPU inference invoked")):
            tail = program.tail
            self.assertTrue(program.encode(first).endswith(tail))
            self.assertTrue(program.encode(second).endswith(tail))
            self.assertEqual(program.decode(second)["output"], [[30]*16 for _ in range(16)])
        with tempfile.TemporaryDirectory() as directory:
            path = program.save(Path(directory)/"add.h264", first)
            manifest = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual([item["frame"] for item in manifest["inputs"]], [0, 1])
        self.assertEqual(manifest["instructions"][0]["frame"], 2)
        self.assertEqual([source["frame"] for source in manifest["instructions"][0]["sources"]], [0, 1])

    def test__given_input_aliases_or_unused_inputs__when_decoding__then_named_outputs_refer_to_correct_seeds(self):
        program = h264.compile(lambda a, b: {"same": b}, 16, 16,
                               input_ranges={"a": (0, 1), "b": (100, 200)})
        self.assertEqual(program.pictures, 2)
        self.assertEqual(program.output_frames, {"same": 1})
        self.assertFalse(program.tail)
        images = {"b": [[150]*16 for _ in range(16)], "a": [[1]*16 for _ in range(16)]}
        for backend in self.backends():
            with self.subTest(backend=backend):
                self.assertEqual(program.verify(images, backend)["same"], images["b"])

    def test__given_unsupported_native_parameters__when_compiling_or_loading__then_failure_is_explicit(self):
        functions = [lambda x: h264.sample(x, qx=.5), lambda x: h264.sample(x, qy=509),
                     lambda x: h264.gather(x, [(0, 0)]*2),
                     lambda x: h264.saturating_weighted(x, 128),
                     lambda x: h264.saturating_weighted(x, 1, 128),
                     lambda x: h264.saturating_combine(x, x, 64, 1),
                     lambda x: h264.saturating_combine(x, x, 1, 1, -129)]
        for function in functions:
            with self.subTest(function=function), self.assertRaises(h264.CompileError):
                h264.compile(function, 16, 16, input_range=(0, 255))
        for ranges in ({}, {f"x{i}": (0, 255) for i in range(16)}, {"bad name": (0, 255)}):
            with self.subTest(ranges=ranges), self.assertRaises(h264.CompileError):
                h264.compile(lambda **inputs: next(iter(inputs.values())), input_ranges=ranges)
        program = h264.compile(h264.saturating_add, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        for inputs in ({"a": [[0]*16]*16}, {"a": [[0]*16]*16, "b": [[256]*16]*16}, [[0]*16]*16):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                program.encode(inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
