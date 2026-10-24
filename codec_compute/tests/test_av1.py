"""Real AV1 decoder tests; expected arithmetic exists only in test code."""
import random
import shutil
import unittest
from unittest.mock import patch

from codec_compute.av1 import AV1Step, compile_program
from codec_compute.av1.bootstrap import bridge_path


def available():
    try:
        bridge_path()
        return bool(shutil.which("ffmpeg"))
    except RuntimeError:
        return False


def rows(data, width):
    return [list(data[i:i + width]) for i in range(0, len(data), width)]


def average(left, right):
    return [[(a + b + 1) // 2 for a, b in zip(lrow, rrow)]
            for lrow, rrow in zip(left, right)]


def translate(source, dx, dy):
    height, width = len(source), len(source[0])
    return [[source[min(height - 1, max(0, y + dy))]
                   [min(width - 1, max(0, x + dx))]
             for x in range(width)] for y in range(height)]


@unittest.skipUnless(available(), "Run python3 -m codec_compute.av1.bootstrap; FFmpeg is required")
class AV1NativeTests(unittest.TestCase):
    def test__given_changed_inputs__when_reusing_compound_tail__then_both_decoders_match_every_frame(self):
        steps = (AV1Step("copied", "copy", ("a",)),
                 AV1Step("mean", "average", ("copied", "b")),
                 AV1Step("shifted", "translate", ("mean",), -7, 4),
                 AV1Step("result", "average", ("shifted", "a")))
        program = compile_program(32, 32, ("a", "b"), steps)
        fixed_tail = program.compute_tail
        rng = random.Random(2031)
        patterns = (bytes(1024), bytes([255]) * 1024, bytes(range(256)) * 4,
                    bytes(255 * (i % 2) for i in range(1024)),
                    bytes(rng.randrange(256) for _ in range(1024)))
        for index, a in enumerate(patterns):
            b = patterns[(index + 2) % len(patterns)]
            inputs = {"a": rows(a, 32), "b": rows(b, 32)}
            expected = dict(inputs)
            expected["copied"] = inputs["a"]
            expected["mean"] = average(expected["copied"], inputs["b"])
            expected["shifted"] = translate(expected["mean"], -7, 4)
            expected["result"] = average(expected["shifted"], inputs["a"])
            for decoder in ("libdav1d", "libaom-av1"):
                with self.subTest(pattern=index, decoder=decoder):
                    execution = program.run(inputs, decoder=decoder)
                    self.assertEqual(execution.frames, expected)
                    self.assertTrue(execution.stream.endswith(fixed_tail))
                    self.assertEqual(program.compute_tail, fixed_tail)

    def test__given_runtime_inputs__when_assembling__then_serializer_receives_only_input_frames(self):
        import codec_compute.av1 as native
        program = compile_program(16, 16, ("a", "b"),
                                  (AV1Step("mean", "average", ("a", "b")),))
        original_serializer = native._serialize
        with patch.object(native, "_serialize", wraps=original_serializer) as serializer:
            first = program.encode({"a": bytes(256), "b": bytes([255]) * 256})
            second = program.encode({"a": bytes([10]) * 256, "b": bytes([11]) * 256})
        self.assertNotEqual(first, second)
        for call in serializer.call_args_list:
            self.assertEqual(len(call.args[2]), 2)
            self.assertTrue(all(instruction[0] == 0 for instruction in call.args[2]))
        self.assertEqual(program.decode(first).frames["mean"], [[128] * 16] * 16)
        self.assertEqual(program.decode(second).frames["mean"], [[11] * 16] * 16)

    def test__given_translation_boundaries__when_decoded__then_edge_samples_replicate(self):
        for width, height in ((16, 16), (64, 48)):
            source = [[(y * 71 + x * 19) % 256 for x in range(width)] for y in range(height)]
            for dx, dy in ((0, 0), (-16, -16), (16, 16), (-1, 1)):
                with self.subTest(width=width, height=height, dx=dx, dy=dy):
                    program = compile_program(width, height, ("a",),
                                              (AV1Step("out", "translate", ("a",), dx, dy),))
                    self.assertEqual(program.run({"a": source}).frames["out"],
                                     translate(source, dx, dy))

    def test__given_many_stages__when_reusing_dead_reference_slots__then_all_intermediates_remain_observable(self):
        steps = []
        previous = "a"
        for index in range(20):
            name = f"stage{index}"
            steps.append(AV1Step(name, "average", (previous, "b")))
            previous = name
        program = compile_program(16, 16, ("a", "b"), steps)
        a = [[(x * 71 + y * 13) % 256 for x in range(16)] for y in range(16)]
        b = [[(x * 17 + y * 29) % 256 for x in range(16)] for y in range(16)]
        got = program.run({"a": a, "b": b}).frames
        expected = a
        for index in range(20):
            expected = average(expected, b)
            self.assertEqual(got[f"stage{index}"], expected)
        self.assertEqual(len(got), 22)

    def test__given_same_source_compound__when_decoded__then_identity_is_exact(self):
        program = compile_program(16, 16, ("a",), (AV1Step("out", "average", ("a", "a")),))
        source = bytes(range(256))
        self.assertEqual(program.run({"a": source}).frames["out"], rows(source, 16))

    def test__given_bad_planes__when_encoding__then_validation_precedes_native_work(self):
        program = compile_program(16, 16, ("a",), (AV1Step("out", "copy", ("a",)),))
        for inputs in ({}, {"a": bytes(255)}, {"a": [[-1] * 16] * 16},
                       {"a": [[256] * 16] * 16}, {"a": [[True] * 16] * 16}):
            with self.subTest(inputs=str(inputs)[:60]), self.assertRaises(ValueError):
                program.encode(inputs)


class AV1ValidationTests(unittest.TestCase):
    def test__given_unsupported_programs__when_compiling__then_rejection_is_explicit(self):
        cases = ((8, 16, ("a",), ()),
                 (16, 16, ("a",), (AV1Step("out", "affine", ("a",)),)),
                 (16, 16, ("a",), (AV1Step("out", "translate", ("a",), 17, 0),)),
                 (16, 16, ("a",), (AV1Step("out", "average", ("a", "missing")),)),
                 (16, 16, ("a",), (AV1Step("a", "copy", ("a",)),)),
                 (16, 16, tuple(str(i) for i in range(9)), ()))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                compile_program(*case)

    def test__given_nine_simultaneously_live_values__when_scheduling__then_capacity_error_precedes_build(self):
        names = tuple(f"input{i}" for i in range(8))
        steps = (AV1Step("intermediate", "copy", (names[0],)),) + tuple(
            AV1Step(f"out{i}", "copy", (name,)) for i, name in enumerate(names))
        with self.assertRaisesRegex(ValueError, "reference capacity"):
            compile_program(16, 16, names, steps)


if __name__ == "__main__":
    unittest.main()
