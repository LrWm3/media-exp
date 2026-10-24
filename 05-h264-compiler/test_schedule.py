"""Retained-slot lifetimes, baseline controls, and observable compiler costs."""

import unittest

import h264
from decoder_core.decode import available


class ScheduleTests(unittest.TestCase):
    def test__given_long_lived_inputs__when_slots_are_reused__then_every_read_is_resident(self):
        def kernel(a, b):
            value = a
            for index in range(80):
                value = h264.saturating_combine(value, a if index % 2 else b, wb=-1, bias=67)
            return {"output": value, "a": a, "b": b}

        program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        legacy = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)},
                              reference_mode="sliding", slice_mode="macroblock")
        residents = {program.reference_slots[frame]: frame for frame in range(2)}
        for frame, operation in enumerate(program.operations, 2):
            for source in operation.sources:
                self.assertEqual(residents[program.reference_slots[source.frame]], source.frame)
            residents[program.reference_slots[frame]] = frame
        self.assertEqual(program.refreshes, [])
        self.assertGreater(len(legacy.refreshes), 0)
        self.assertEqual(program.pictures + len(legacy.refreshes), legacy.pictures)
        inputs = {"a": [[y*16+x for x in range(16)] for y in range(16)],
                  "b": [[(y*73+x*17)%256 for x in range(16)] for y in range(16)]}
        expected = legacy.verify(inputs)
        for backend in ("ffmpeg", "vaapi", "openh264"):
            if available(backend):
                with self.subTest(backend=backend):
                    self.assertEqual(program.verify(inputs, backend), expected)

    def test__given_program_manifest__when_reporting_costs__then_counts_match_encoded_stream(self):
        program = h264.compile(lambda x: h264.bit_or(x, h264.shift(x, 1)), 32, 32)
        image = [[(x+y)%2 for x in range(32)] for y in range(32)]
        stream = program.encode(image)
        costs = program.manifest()["costs"]
        self.assertEqual(costs["total_pictures"], sum(costs[key] for key in
                         ("input_pictures", "normalization_pictures", "arithmetic_pictures", "refresh_pictures")))
        self.assertEqual(costs["prediction_bytes"], len(stream)-len(program.encode_inputs(image)))
        self.assertEqual(costs["prediction_slices"], program.tail.count(b"\x00\x00\x00\x01"))
        self.assertEqual(program.manifest()["reference_slots"], program.reference_slots)
        self.assertEqual(program.encode(image), stream)

    def test__given_invalid_backend_controls__when_compiling__then_configuration_fails_explicitly(self):
        for options in ({"reference_mode": "unknown"}, {"slice_mode": "unknown"}):
            with self.subTest(options=options), self.assertRaises(h264.CompileError):
                h264.compile(lambda x: x, **options)


if __name__ == "__main__":
    unittest.main()
