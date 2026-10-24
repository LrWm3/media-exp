"""Real-decoder checks of the complete exact Boolean atlas and recurrent tails."""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas import build_atlas
from program import Program
from decoder_core.decode import available, decode


def boards(width=16, height=16):
    rng = random.Random(8080)
    return [
        {"a": [[((x + y) % 4) // 2 for x in range(width)] for y in range(height)],
         "b": [[(x + y) % 2 for x in range(width)] for y in range(height)]},
        {name: [[rng.randrange(2) for _ in range(width)] for _ in range(height)]
         for name in ("a", "b")},
        {name: [[0] * width for _ in range(height)] for name in ("a", "b")},
        {name: [[1] * width for _ in range(height)] for name in ("a", "b")},
    ]


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_atlas()["functions"]
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        if not cls.backends:
            raise unittest.SkipTest("No real H.264 decoder available")

    def decoded(self, program, inputs, backend, stream=None):
        with tempfile.TemporaryDirectory(prefix="compute-limits-test-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(stream if stream is not None else b"".join(program.encode(item) for item in inputs))
            raw = decode(path, backend)
        stride = program.pictures * program.writer.width * program.writer.height * 3 // 2
        self.assertEqual(len(raw), stride * len(inputs))
        return [raw[index * stride:(index + 1) * stride] for index in range(len(inputs))]

    def assert_decoders(self, program, inputs):
        expected, reference = [program.reference(item) for item in inputs], None
        for backend in self.backends:
            with self.subTest(backend=backend):
                raw = self.decoded(program, inputs, backend)
                self.assertEqual([program.read(item) for item in raw], expected)
                if reference is None:
                    reference = raw
                else:
                    self.assertEqual(raw, reference, "Every luma/chroma byte must agree between decoders")
        return reference

    def test__given_all_boolean_functions__when_decoding_both_modes__then_truth_and_exact_costs_match(self):
        inputs = boards()
        for case in self.cases:
            for packed in (False, True):
                with self.subTest(function=case["name"], packed=packed):
                    program = Program(case, 16, 16, packed=packed)
                    minimum = case["packed_minimum" if packed else "scalar_minimum"]
                    self.assertEqual(len(program.operations), minimum)
                    self.assertEqual(len(program.steps), len(case["plan"]))
                    self.assertEqual(program.pictures, 2 + minimum)
                    self.assertEqual(program.costs["logical_instructions"], case["scalar_minimum"])
                    self.assertEqual(program.costs["prediction_slices"], program.tail.count(b"\x00\x00\x00\x01"))
                    self.assertEqual(program.reference(inputs[0])["output"][0][:4], case["truth"])
                    self.assert_decoders(program, inputs)

    def test__given_recurrent_rounds__when_pictures_exceed_sliding_window__then_original_b_and_all_steps_survive(self):
        inputs = boards()[:2]
        for rounds in (5, 40):
            for case in self.cases:
                for packed in (False, True):
                    with self.subTest(rounds=rounds, function=case["name"], packed=packed):
                        program = Program(case, 16, 16, packed=packed, rounds=rounds)
                        minimum = case["packed_minimum" if packed else "scalar_minimum"]
                        self.assertEqual(len(program.operations), rounds * minimum)
                        self.assertEqual(len(program.steps), rounds * len(case["plan"]))
                        self.assertEqual(program.costs["computational_evaluations"], rounds * 16 * 16)
                        self.assertLessEqual(len(set(program.reference_slots.values())), 4)
                        self.assertEqual(sorted(program.reference_slots), list(range(program.pictures)))
                        self.assert_decoders(program, inputs)

    def test__given_independent_xor_branches__when_packing__then_two_steps_share_one_picture(self):
        for mask in (6, 9):
            program = Program(self.cases[mask], packed=True, rounds=5)
            self.assertEqual(len(program.tiles), 2)
            for index in range(5):
                first, second, final = program.steps[3 * index:3 * index + 3]
                self.assertEqual(first["frame"], second["frame"])
                self.assertEqual((first["tile"], second["tile"], final["tile"]), (0, 1, 0))
                self.assertEqual(final["frame"], first["frame"] + 1)
                self.assertEqual(program.frame_tiles[final["frame"]], (0,))
        for case in self.cases:
            if case["id"] not in (6, 9):
                scalar, packed = Program(case), Program(case, packed=True)
                self.assertEqual(len(packed.tiles), 1)
                self.assertEqual((scalar.writer.width, scalar.writer.height),
                                 (packed.writer.width, packed.writer.height))
                self.assertEqual(scalar.tail, packed.tail)

    def test__given_identity_aliases__when_rounds_repeat__then_no_compute_picture_is_emitted(self):
        for mask, frame in ((10, 1), (12, 0)):
            for packed in (False, True):
                program = Program(self.cases[mask], 16, 16, packed=packed, rounds=40)
                self.assertEqual(program.pictures, 2)
                self.assertEqual(program.tail, b"")
                self.assertEqual(program.output_frames, {"output": frame})
                inputs = boards()[:1]
                raw = self.decoded(program, inputs, self.backends[0])[0]
                stride = program.writer.width * program.writer.height * 3 // 2
                self.assertEqual(program.read(raw[frame * stride:(frame + 1) * stride],
                                              selected_frames=[frame]), program.reference(inputs[0]))

    def test__given_disabled_host_oracles__when_inputs_change__then_frozen_prediction_tail_still_computes(self):
        inputs = boards()[:2]
        for mask in (0, 6, 7, 9, 15):
            program = Program(self.cases[mask], 16, 16, packed=True, rounds=5)
            expected = program.reference(inputs[1])
            first_prefix = program.encode_inputs(inputs[0])
            first_stream = program.encode(inputs[0])
            with patch.object(Program, "reference", side_effect=AssertionError("Host truth oracle invoked")), \
                    patch("atlas.evaluate_plan", side_effect=AssertionError("Host plan evaluator invoked")):
                changed = program.encode_inputs(inputs[1])
                self.assertNotEqual(changed, first_prefix)
                self.assertEqual(program.encode(inputs[1]), changed + first_stream[len(first_prefix):])
                for backend in self.backends:
                    with self.subTest(mask=mask, backend=backend):
                        raw = self.decoded(program, [inputs[1]], backend,
                                           changed + first_stream[len(first_prefix):])[0]
                        self.assertEqual(program.read(raw), expected)

    def test__given_corrupted_provided_planes__when_reading__then_guards_chroma_and_unused_tiles_are_rejected(self):
        program = Program(self.cases[6], 16, 16, packed=True)
        inputs = boards()[:1]
        raw = self.decoded(program, inputs, self.backends[0])[0]
        width, height = program.writer.width, program.writer.height
        pixels, stride = width * height, width * height * 3 // 2
        final = program.output_frames["output"]
        unused = final * stride + 16 * width + program.tiles[1]["x"] + 16
        for location in (0, pixels, unused):
            with self.subTest(location=location):
                damaged = bytearray(raw)
                damaged[location] ^= 1
                with self.assertRaises(AssertionError):
                    program.read(damaged)
        selected = raw[final * stride:(final + 1) * stride]
        self.assertEqual(program.read(selected, selected_frames=[final]), program.reference(inputs[0]))
        for selection in ([0], [final, final], [final, 0], [True], [-1], []):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                program.read(selected, selected_frames=selection)

    def test__given_non_boolean_or_wrong_shape_inputs__when_encoding__then_invalid_domains_are_rejected(self):
        program = Program(self.cases[6], 16, 16)
        for invalid in ({}, {"a": [[0]], "b": [[0]]},
                        {"a": [[2] * 16 for _ in range(16)], "b": boards()[0]["b"]}):
            with self.assertRaises(ValueError):
                program.encode(invalid)


if __name__ == "__main__":
    unittest.main()
