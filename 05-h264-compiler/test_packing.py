"""Generic packed schedules retain integer semantics and physical references."""

import unittest

import h264
from decoder_core.decode import available


class PackingTests(unittest.TestCase):
    def check(self, program, inputs):
        for backend in ("ffmpeg", "openh264", "vaapi"):
            if available(backend):
                with self.subTest(backend=backend):
                    program.verify(inputs, backend)

    def test__given_named_u8_inputs__when_packing_independent_arithmetic__then_both_tiles_feed_later_predictions(self):
        def kernel(a, b):
            first = h264.saturating_weighted(a, 2, -9)
            second = h264.saturating_weighted(b, -1, 127)
            return {"a": a, "b": b, "first": first, "second": second,
                    "sum": h264.saturating_add(first, second),
                    "difference": h264.saturating_combine(first, second, 1, -1)}
        original = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)})
        packed = h264.pack(original, tiles=2)
        self.assertEqual(packed.output_frames["first"], packed.output_frames["second"])
        self.assertNotEqual(packed.output_regions["first"]["tile"], packed.output_regions["second"]["tile"])
        image = [[y*16+x for x in range(16)] for y in range(16)]
        self.check(packed, {"a": image, "b": [list(reversed(row)) for row in image]})

    def test__given_shifts_at_halo_limit__when_packing_biased_rectangles__then_other_tiles_never_leak(self):
        def kernel(x):
            first = h264.saturating_weighted(x, 1, 80)
            second = h264.saturating_weighted(x, -1, 127)
            return {"first": first, "second": second,
                    "left": h264.shift(first, -16), "right": h264.shift(second, 16),
                    "up": h264.shift(first, dy=-16), "down": h264.shift(second, dy=16),
                    "corner": h264.shift(second, -16, 16)}
        packed = h264.pack(h264.compile(kernel, 32, 16, input_range=(0, 255)), tiles=4)
        self.check(packed, [[(y*31+x*7) % 256 for x in range(32)] for y in range(16)])

    def test__given_reference_used_after_many_pictures__when_repacking__then_retained_slots_preserve_it(self):
        def kernel(x):
            first = h264.bit_not(x)
            result = h264.repeat(35, lambda state: h264.bit_or(h264.shift(state, 1), x), first)
            return {"first": first, "output": result}
        packed = h264.pack(h264.compile(kernel, 16, 16), tiles=2)
        self.assertGreater(packed.pictures, 16)
        self.assertTrue(any(frame-source.frame > 16
                            for frame, op in enumerate(packed.operations, packed.input_count)
                            for child in op.block_operations for source in child.sources))
        self.check(packed, [[int(x == y) for x in range(16)] for y in range(16)])

    def test__given_literal_identity__when_packing_without_predictions__then_input_is_named_output(self):
        packed = h264.pack(h264.compile(lambda x: x, 16, 16, input_range=(0, 255)), tiles=2)
        self.assertEqual(packed.pictures, 1)
        self.assertEqual(packed.tail, b"")
        # The local GStreamer OpenH264 wrapper rejects this one-picture stream;
        # FFmpeg and VA-API support a stream containing just the input picture.
        for backend in ("ffmpeg", "vaapi"):
            if available(backend):
                packed.verify([[y*16+x for x in range(16)] for y in range(16)], backend)

    def test__given_quarter_units_at_integer_locations__when_packing__then_motion_is_normalized_exactly(self):
        original = h264.compile(lambda x: {"one": h264.sample(x, qx=4),
                                           "edge": h264.sample(x, qx=-64, qy=64)},
                                32, 32, input_range=(0, 255))
        packed = h264.pack(original, tiles=2)
        self.check(packed, [[(y*32+x) % 256 for x in range(32)] for y in range(32)])

    def test__given_aliases__when_mapping_frames__then_original_dependencies_always_precede_consumers(self):
        def kernel(x):
            values = [h264.bit_or(x, h264.shift(x, dx)) for dx in range(1, 9)]
            return {str(index): value for index, value in enumerate(values)}
        original = h264.compile(kernel, 16, 16)
        for capacity in (1, 2, 3, 4, 8):
            packed = h264.pack(original, tiles=capacity)
            for frame, op in enumerate(packed.operations, packed.input_count):
                for child in (op,) + op.block_operations:
                    self.assertTrue(all(source.frame < frame for source in child.sources))
            self.check(packed, [[(x+y) % 2 for x in range(16)] for y in range(16)])

    def test__given_invalid_readback__when_extracting__then_missing_or_duplicate_pictures_fail(self):
        packed = h264.pack(h264.compile(lambda x: h264.bit_not(x), 16, 16), tiles=2)
        for selection in ((), (-1,), (True,), (0, 0), (packed.pictures,), (0,), (1, 0)):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                packed.read(b"", selected_frames=selection)
        with self.assertRaises(AssertionError):
            packed.read(b"")

    def test__given_unrepresentable_canvas__when_packing__then_compilation_rejects_geometry_or_slice_rate(self):
        program = h264.compile(lambda x: h264.bit_not(x), 256, 128)
        with self.assertRaises(h264.CompileError):
            h264.pack(program, tiles=16)
        with self.assertRaises(h264.CompileError):
            h264.pack(h264.pack(h264.compile(lambda x: x, 16, 16)))

    def test__given_canvas_exceeding_pcm_bound__when_packing__then_compilation_rejects_before_input_encoding(self):
        program = h264.compile(lambda x: h264.weighted(x, bias=1), 64, 64, input_range=(0, 100))
        for tiles in (4, 8):
            with self.subTest(tiles=tiles), self.assertRaisesRegex(h264.CompileError, "access-unit"):
                h264.pack(program, tiles=tiles)


if __name__ == "__main__":
    unittest.main()
