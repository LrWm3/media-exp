"""Real AV1 decoder tests; expected arithmetic exists only in test code."""
from pathlib import Path
import random
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch

from codec_compute.av1 import AV1Program, AV1Step, compile_program
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

    def test__given_all_u8_pairs__when_compound_average_decodes__then_both_decoders_match_exhaustively(self):
        program = compile_program(256, 256, ("left", "right"),
                                  (AV1Step("mean", "average", ("left", "right")),))
        left = bytes(a for a in range(256) for _ in range(256))
        right = bytes(range(256)) * 256
        expected = [[(a + b + 1) // 2 for b in range(256)] for a in range(256)]
        stream = program.encode({"left": left, "right": right})
        for decoder in ("libdav1d", "libaom-av1"):
            with self.subTest(decoder=decoder):
                frames = program.decode(stream, decoder=decoder).frames
                self.assertEqual(frames["left"], rows(left, 256))
                self.assertEqual(frames["right"], rows(right, 256))
                self.assertEqual(frames["mean"], expected)

    def test__given_eight_live_references__when_replacing_each_slot__then_retained_values_and_tail_stay_exact(self):
        import codec_compute.av1 as native
        names = tuple(f"input{i}" for i in range(8))
        steps = [AV1Step(f"copy{i}", "copy", (name,)) for i, name in enumerate(names)]
        steps += [AV1Step(f"mean{i}", "average", (f"copy{i}", "copy7")) for i in range(7)]
        previous = "mean0"
        for index in range(1, 7):
            output = f"reduce{index}"
            steps.append(AV1Step(output, "average", (previous, f"mean{index}")))
            previous = output
        steps.append(AV1Step("result", "average", (previous, "copy7")))
        program = compile_program(32, 32, names, steps)
        rng = random.Random(381)
        inputs = {name: bytes(rng.randrange(256) for _ in range(1024)) for name in names}
        expected = {name: rows(plane, 32) for name, plane in inputs.items()}
        for step in steps:
            expected[step.output] = (expected[step.sources[0]] if step.kind == "copy" else
                                     average(expected[step.sources[0]], expected[step.sources[1]]))
        stream = program.encode(inputs)
        for decoder in ("libdav1d", "libaom-av1"):
            with self.subTest(decoder=decoder):
                self.assertEqual(program.decode(stream, decoder=decoder).frames, expected)
        # Audit the pinned encoder with real inputs only in this test. Runtime
        # production code never sends compute instructions alongside real data.
        full = native._serialize(32, 32, program._instructions,
                                 tuple(inputs[name] for name in names))
        self.assertEqual(full, stream)
        self.assertEqual(b"".join(native._packets(full)[8:]), program.compute_tail)
        self.assertEqual(native._packets(full)[:8], native._packets(program.encode_inputs(inputs)))

        traced = subprocess.run(
            [shutil.which("ffmpeg"), "-v", "trace", "-i", "pipe:0", "-c:v", "copy",
             "-bsf:v", "trace_headers", "-f", "null", "-"], input=stream,
            capture_output=True, check=True, timeout=60)
        fields = {}
        for line in traced.stderr.decode().splitlines():
            if "trace_headers" not in line:
                continue
            match = re.search(r"\]\s+\d+\s+(\w+)\s+.*=\s+(\d+)$", line)
            if match:
                fields.setdefault(match.group(1), []).append(int(match.group(2)))
        frame_count = len(program.frame_names)
        self.assertEqual(fields["base_q_idx"], [0] * frame_count)
        self.assertEqual(fields["disable_cdf_update"], [1] * frame_count)
        self.assertEqual(fields["primary_ref_frame"], [7] * (frame_count - 1))
        self.assertEqual(fields["refresh_frame_flags"],
                         [1 << instruction[1] for instruction in program._instructions[1:]])
        for field in ("enable_order_hint", "enable_cdef", "enable_restoration",
                      "frame_id_numbers_present_flag"):
            self.assertTrue(fields[field])
            self.assertEqual(set(fields[field]), {0})

    def test__given_bad_planes__when_encoding__then_validation_precedes_native_work(self):
        program = compile_program(16, 16, ("a",), (AV1Step("out", "copy", ("a",)),))
        for inputs in ({}, {"a": bytes(255)}, {"a": [[-1] * 16] * 16},
                       {"a": [[256] * 16] * 16}, {"a": [[True] * 16] * 16}):
            with self.subTest(inputs=str(inputs)[:60]), self.assertRaises(ValueError):
                program.encode(inputs)


class AV1ValidationTests(unittest.TestCase):
    @staticmethod
    def decoder_fixture():
        return AV1Program(16, 16, ("a",), (), ("a",), b"", b"",
                          ((0, 0, 0, 0, 0, 0),))

    def test__given_decoder_error_with_success_status__when_decoding__then_error_is_fatal(self):
        program = self.decoder_fixture()
        result = subprocess.CompletedProcess(["ffmpeg"], 0, bytes(256), b"corrupt frame")
        with patch("codec_compute.av1.shutil.which", return_value="ffmpeg"), \
             patch("codec_compute.av1.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "corrupt frame"):
                program.decode(b"stream")
        self.assertIn("-xerror", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test__given_stalled_native_tools__when_invoked__then_timeouts_are_explicit(self):
        import codec_compute.av1 as native
        program = self.decoder_fixture()
        with patch("codec_compute.av1.shutil.which", return_value="ffmpeg"), \
             patch("codec_compute.av1.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            with self.assertRaisesRegex(RuntimeError, "decoder.*60-second timeout"):
                program.decode(b"stream")
        with patch("codec_compute.av1.bootstrap.bridge_path", return_value=Path("bridge")), \
             patch("codec_compute.av1.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("bridge", 180)) as run:
            with self.assertRaisesRegex(RuntimeError, "serializer.*180-second timeout"):
                native._serialize(16, 16, program._instructions, (bytes(256),))
        self.assertEqual(run.call_args.kwargs["timeout"], 180)

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
