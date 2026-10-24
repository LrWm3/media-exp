"""Explicit High levels obey level budgets and decode byte exactly."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture


def nal_bytes(count, kind=5):
    """Synthetic escaped NAL for testing byte budgets, not picture decoding."""
    return b"\x00\x00\x00\x01" + bytes([kind]) + b"\x55"*(count-1)


class LevelLimitTests(unittest.TestCase):
    def test__given_default_or_explicit_level31__when_encoding__then_existing_stream_bytes_are_unchanged(self):
        board = [[16*y+x for x in range(16)] for y in range(16)]
        default, explicit = Writer(16, 16), Writer(16, 16, level_idc=31)
        self.assertEqual(default.seed_luma(board), explicit.seed_luma(board))
        self.assertEqual(hashlib.sha256(default.seed_luma(board)).hexdigest(),
                         "bc8f01c02af464b79488ad7dc87d3f874bbbf0aa2fcea15eafa63326c0a276ec")
        operation = Operation("average", (Source(0), Source(0)))
        self.assertEqual(default.prediction(1, operation), explicit.prediction(1, operation))
        higher = [(Writer(16, 16, level_idc=level), level) for level in (41, 51)]
        for writer, level in ((default, 31), (explicit, 31), *higher):
            sps = writer.parameter_sets().split(b"\x00\x00\x00\x01")[1]
            self.assertEqual((sps[1], sps[3]), (100, level))
            self.assertEqual(default.prediction(1, operation), writer.prediction(1, operation))

    def test__given_unsupported_or_malformed_level__when_constructing__then_explicit_selection_is_required(self):
        for level in (None, True, False, 3.1, 31.0, 41.0, 51.0, "51", 0, 30, 32, 40, 42, 50, 52, [], {}):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "level_idc must be 31, 41, or 51"):
                Writer(16, 16, level_idc=level)

    def test__given_oversize_geometry_or_reference_storage__when_constructing__then_selected_level_limits_apply(self):
        for level, axis_width, square in ((31, 2688, 512), (41, 4080, 704), (51, 8672, 1696)):
            with self.subTest(level=level, limit="axis"), self.assertRaisesRegex(ValueError, "per-axis"):
                Writer(axis_width, 16, level_idc=level, slice_mode="row")
            with self.subTest(level=level, limit="DPB"), self.assertRaisesRegex(ValueError, "decoded-picture buffer"):
                Writer(square, square, level_idc=level, slice_mode="row")
        # With the current sixteen-frame declaration the DPB bound is tighter
        # than MaxFS/MaxMBPS. Verify those independent checks also remain valid
        # when a caller explicitly reduces that declaration.
        with self.assertRaisesRegex(ValueError, "frame-size"):
            Writer(1024, 1024, level_idc=31, slice_mode="row", reference_capacity=1)
        with self.assertRaisesRegex(ValueError, "macroblock processing rate"):
            Writer(1024, 1024, level_idc=41, fps=60, slice_mode="row", reference_capacity=1)

    def test__given_uniform_and_fragmented_slices__when_emitting__then_actual_slice_rate_uses_selected_level(self):
        with self.assertRaisesRegex(ValueError, "slice rate"):
            Writer(256, 256, level_idc=31)
        self.assertEqual(Writer(256, 256, level_idc=41).slice_count(Operation("copy", (Source(0),))), 324)
        with self.assertRaisesRegex(ValueError, "slice rate"):
            Writer(272, 272, level_idc=41)
        self.assertEqual(Writer(512, 512, level_idc=51).slice_count(Operation("copy", (Source(0),))), 1156)
        with self.assertRaisesRegex(ValueError, "slice rate"):
            Writer(560, 560, level_idc=51)
        children = tuple(Operation("alternating header", (Source(0, offset=1+index % 2),)) for index in range(400))
        operation = Operation("fragmented", (Source(0),), block_operations=children)
        rejected = Writer(320, 320, level_idc=41, fps=24, slice_mode="row")
        self.assertEqual(rejected.slice_count(operation), 442)
        with patch.object(rejected, "header", wraps=rejected.header) as header:
            with self.assertRaisesRegex(ValueError, "slice rate"):
                rejected.prediction(1, operation)
            header.assert_not_called()
        allowed = Writer(320, 320, level_idc=41, fps=23, slice_mode="row")
        self.assertEqual(allowed.prediction(1, operation).count(b"\x00\x00\x00\x01"), 442)

    def test__given_fragmented_level51_fields__when_emitting__then_actual_slices_obey_rate_limit(self):
        children = tuple(Operation("alternating header", (Source(0, offset=1+index % 2),)) for index in range(36**2))
        operation = Operation("fragmented", (Source(0),), block_operations=children)
        rejected = Writer(576, 576, level_idc=51, slice_mode="row")
        self.assertEqual(rejected.slice_count(operation), 1370)
        with patch.object(rejected, "header", wraps=rejected.header) as header:
            with self.assertRaisesRegex(ValueError, "slice rate"):
                rejected.prediction(1, operation)
            header.assert_not_called()
        allowed = Writer(576, 576, level_idc=51, fps=29, slice_mode="row")
        self.assertEqual(allowed.prediction(1, operation).count(b"\x00\x00\x00\x01"), 1370)

    def test__given_access_units_at_exact_byte_boundaries__when_counting__then_start_codes_are_excluded_from_mincr_limits(self):
        for level, initial in ((31, 60279), (41, 274336), (51, 1097346)):
            writer = Writer(256, 256, level_idc=level, fps=1, slice_mode="row")
            data = nal_bytes(initial)
            with self.subTest(level=level):
                self.assertEqual(writer._checked_access_unit(0, data, "test"), data)
                with self.assertRaisesRegex(ValueError, "initial access-unit"):
                    writer._checked_access_unit(0, data+b"\x55", "test")
        writer = Writer(16, 16, fps=30)
        data = nal_bytes(345600)
        self.assertEqual(writer._checked_access_unit(1, data, "test"), data)
        with self.assertRaisesRegex(ValueError, "subsequent access-unit"):
            writer._checked_access_unit(1, data+b"\x55", "test")
        writer = Writer(16, 16, fps=30, level_idc=41)
        with self.assertRaisesRegex(ValueError, "subsequent access-unit"):
            writer._checked_access_unit(1, nal_bytes(1572865), "test")
        writer = Writer(512, 512, fps=30, level_idc=51, slice_mode="row")
        with self.assertRaisesRegex(ValueError, "subsequent access-unit"):
            writer._checked_access_unit(1, nal_bytes(6291457), "test")

    def test__given_explicit_higher_rate_budgets__when_counting_vcl_and_nal_bytes__then_parameter_bytes_do_not_consume_vcl_budget(self):
        for level, vcl_limit, nal_limit in ((41, 260416, 312500), (51, 1250000, 1500000)):
            writer = Writer(16, 16, level_idc=level)
            with self.subTest(level=level):
                vcl = nal_bytes(vcl_limit)
                self.assertEqual(writer._checked_access_unit(1, vcl, "test"), vcl)
                with self.assertRaisesRegex(ValueError, "VCL bitrate"):
                    writer._checked_access_unit(1, vcl+b"\x55", "test")
                nonvcl = nal_bytes(nal_limit-4, kind=6)  # Include four-byte start code.
                self.assertEqual(writer._checked_access_unit(1, nonvcl, "test"), nonvcl)
                with self.assertRaisesRegex(ValueError, "NAL bitrate"):
                    writer._checked_access_unit(1, nonvcl+b"\x55", "test")
        with self.assertRaisesRegex(ValueError, "VCL bitrate"):
            Writer(256, 256, level_idc=41, fps=60, slice_mode="row").input_luma([[0]*256 for _ in range(256)], 0)

    def test__given_picture_larger_than_inferred_cpb__when_emitting__then_vcl_and_byte_stream_capacity_are_checked(self):
        writer = Writer(16, 16, fps=1)
        with self.assertRaisesRegex(ValueError, "coded-picture buffer"):
            writer._checked_access_unit(1, nal_bytes(2187501), "test")
        with self.assertRaisesRegex(ValueError, "coded-picture buffer"):
            writer._checked_access_unit(1, nal_bytes(2624997, kind=6), "test")

    def test__given_large_declared_motion_bound__when_predicting__then_level_component_limits_still_apply(self):
        for level in (31, 41, 51):
            writer = Writer(16, 16, level_idc=level, motion_bits=16)
            for source in (Source(0, qx=-8192), Source(0, qx=8191), Source(0, qy=-2048), Source(0, qy=2047)):
                with self.subTest(level=level, legal=source):
                    self.assertTrue(writer.prediction(1, Operation("legal motion", (source,))))
            for source in (Source(0, qx=-8193), Source(0, qx=8192), Source(0, qy=-2049), Source(0, qy=2048)):
                with self.subTest(level=level, illegal=source), self.assertRaisesRegex(ValueError, "component limit"):
                    writer.prediction(1, Operation("illegal motion", (source,)))

    def test__given_identical_256_inputs_and_lower_fps__when_retaining_level31__then_initial_au_limit_still_rejects(self):
        board = [[0]*256 for _ in range(256)]
        for fps in (1, 15, 30):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "initial access-unit"):
                Writer(256, 256, fps=fps, slice_mode="row").input_luma(board, 0)
        writer = Writer(256, 256, level_idc=41, slice_mode="row")
        self.assertLess(len(writer.input_luma(board, 0))-12, 274336)
        self.assertLess(len(writer.input_luma(board, 1))-4, 1572864)


class LargerLevelDecoderTests(unittest.TestCase):
    def test__given_arbitrary_large_pcm_inputs__when_using_explicit_higher_levels__then_available_decoders_match_all_i420_bytes(self):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        for size, level in ((256, 41), (512, 51)):
            self.assert_inputs(size, level, backends)

    def assert_inputs(self, size, level, backends):
        fixtures = (("zero", lambda x, y: 0), ("255", lambda x, y: 255),
                    ("alternating", lambda x, y: 255*((x+y) % 2)),
                    ("full-byte", lambda x, y: (17*x+31*y) % 256))
        for label, pixel in fixtures:
            a = [[pixel(x, y) for x in range(size)] for y in range(size)]
            b = [[255-value for value in row] for row in a]
            average = [[(x+y+1)//2 for x, y in zip(left, right)] for left, right in zip(a, b)]
            difference = [[max(0, (a[y][x+1] if x+1 < size else 0)
                                  - (b[y+1][x] if y+1 < size else 0))
                           for x in range(size)] for y in range(size)]
            expected = b"".join(i420_picture(board) for board in (a, b, average, difference))
            for mode in ("macroblock", "row"):
                writer = Writer(size, size, level_idc=level, slice_mode=mode,
                                reference_slots={0: 0, 1: 1, 2: 2, 3: 0})
                stream = writer.input_luma(a, 0) + writer.input_luma(b, 1)
                stream += writer.prediction(2, Operation("average", (Source(0), Source(1))))
                stream += writer.prediction(3, Operation("motion difference", (Source(0, 2, dx=1), Source(1, -2, dy=1))))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"program.h264"
                    path.write_bytes(stream)
                    for backend in backends:
                        with self.subTest(size=size, level=level, fixture=label, slices=mode, decoder=backend):
                            self.assertEqual(decode(path, backend), expected)


if __name__ == "__main__":
    unittest.main()
