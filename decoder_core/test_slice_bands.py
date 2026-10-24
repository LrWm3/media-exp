"""Row-band motion prediction checked against independent full-I420 pixels."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture
from decoder_core.test_guard_fusion import predict


class SliceBandTests(unittest.TestCase):
    def assert_decoded(self, streams, pictures):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        expected = b"".join(i420_picture(board) for board in pictures)
        with tempfile.TemporaryDirectory() as directory:
            for label, stream in streams.items():
                path = Path(directory)/(str(label)+".h264")
                path.write_bytes(stream)
                for backend in backends:
                    with self.subTest(band=label, decoder=backend):
                        self.assertEqual(decode(path, backend), expected)

    def test__given_default_and_one_row__when_encoding__then_historical_bytes_are_preserved(self):
        board = [[(x*37+y*71) % 256 for x in range(32)] for y in range(32)]
        operations = (Operation("p", (Source(0, qx=1, qy=-2),)),
                      Operation("b", (Source(0, weight=3), Source(1, weight=-1)), denominator=1))
        for enabled in (False, True):
            streams = []
            for options in ({}, {"prediction_slice_rows": 1}):
                writer = Writer(32, 32, slice_mode="row", merge_zero_guards=enabled, **options)
                streams.append(writer.input_luma(board, 0)+b"".join(
                    writer.prediction(frame, op) for frame, op in enumerate(operations, 1)))
            self.assertEqual(streams[0], streams[1])
            self.assertEqual(hashlib.sha256(streams[0]).hexdigest(),
                             "71e0aa1f577a76f98ac5bd2b04fca9c77adc8c437c35aa6adf2d37a90f5aa0ed" if enabled else
                             "8851a04498737cc4d83a3b01651c422119a44eed37b9dac94849950b12c1aed7")

    def test__given_invalid_band_arguments__when_constructing__then_validation_rejects_them(self):
        for value in (None, True, False, -1, 1.0, "2", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                Writer(16, 16, prediction_slice_rows=value)
        for rows in (0, 1, 2, 4, 8):
            with self.assertRaisesRegex(ValueError, "decoded-picture buffer"):
                Writer(512, 512, level_idc=31, slice_mode="row", prediction_slice_rows=rows)

    def test__given_three_sizes_and_uniform_headers__when_grouping_rows__then_counts_and_inputs_remain_exact(self):
        for size, level in ((128, 31), (256, 41), (512, 51)):
            op = Operation("blend", (Source(0, qx=1, qy=2), Source(0, qx=-1, qy=-2)))
            baseline = Writer(size, size, level_idc=level, slice_mode="row", merge_zero_guards=True)
            for rows in (1, 2, 4, 8, 0, 10**20):
                writer = Writer(size, size, level_idc=level, slice_mode="row", merge_zero_guards=True,
                                prediction_slice_rows=rows)
                effective = min(rows or writer.mb_height, writer.mb_height)
                expected = (writer.mb_height+effective-1)//effective
                with self.subTest(size=size, rows=rows):
                    self.assertEqual(writer.slice_count(op), expected)
                    self.assertEqual(writer.prediction(1, op).count(b"\0\0\0\1"), expected)
                    self.assertEqual(writer.parameter_sets(), baseline.parameter_sets())
                    runs = list(writer._prediction_runs(op))
                    self.assertEqual([address for run in runs for address, _ in run],
                                     list(range(writer.mb_width*writer.mb_height)))
                    for run in runs:
                        self.assertEqual(run[0][0] % writer.mb_width, 0)
                        self.assertLessEqual(len(run), effective*writer.mb_width)

    def test__given_neighbor_vectors__when_predicting__then_median_substitution_and_slice_availability_apply(self):
        writer = Writer(32, 32)  # Four coded macroblocks per row.
        both = lambda a, b: (a, b)
        history = {0: both((1, 8), (-1, -8)), 1: both((7, 2), (-7, -2)),
                   2: both((5, 6), (-5, -6)), 4: both((3, 4), (-3, -4))}
        self.assertEqual(writer._motion_prediction(5, history, 0), (5, 4))
        self.assertEqual(writer._motion_prediction(5, history, 1), (-5, -4))
        # Top-right C is outside the picture, so D substitutes for it.
        right = {2: both((2, 7), (2, 7)), 3: both((8, 3), (8, 3)),
                 6: both((5, 5), (5, 5))}
        self.assertEqual(writer._motion_prediction(7, right, 0), (5, 5))
        self.assertEqual(writer._motion_prediction(5, {4: history[4]}, 0), (3, 4))
        self.assertEqual(writer._motion_prediction(4, {0: history[0]}, 0), (1, 8))
        self.assertEqual(writer._motion_prediction(4, {3: right[3]}, 0), (0, 0))
        self.assertEqual(writer._motion_prediction(8, {}, 0), (0, 0))

    def test__given_all_phases_and_signed_p_b_fields__when_decoding_every_band_size__then_full_i420_matches(self):
        width, height = 80, 64
        pictures = [[[((x*71+y*37+x*y) % 256) for x in range(width)] for y in range(height)],
                    [[255*((x+3*y) % 7 < 3) for x in range(width)] for y in range(height)]]
        operations = []
        for arity in (1, 2):
            for denominator in range(8):
                sources = ((Source(0, weight=(1, 3, 127, -128, -3, 65, 0, 127)[denominator]),)
                           if arity == 1 else (Source(0, weight=127 if denominator == 7 else 3),
                                               Source(denominator % 2, weight=0 if denominator == 7 else -1)))
                parent = Operation("field", sources, denominator)
                blocks = []
                for index in range(width*height//256):
                    fx, fy = index % 4, (index//4) % 4
                    moving = (replace(sources[0], qx=fx-4, qy=fy-4),)
                    if arity == 2:
                        moving += (replace(sources[1], qx=7-fx, qy=-1-fy),)
                    blocks.append(replace(parent, sources=moving))
                op = replace(parent, block_operations=tuple(blocks))
                operations.append(op)
                pictures.append(predict(pictures, op))
        slots = {frame: (frame if frame < 2 else 2) for frame in range(len(pictures))}
        streams = {}
        for rows in (1, 2, 4, 8, 0):
            writer = Writer(width, height, slice_mode="row", merge_zero_guards=True,
                            reference_slots=slots, prediction_slice_rows=rows)
            streams[rows] = b"".join(writer.input_luma(board, frame) for frame, board in enumerate(pictures[:2]))
            streams[rows] += b"".join(writer.prediction(frame, op) for frame, op in enumerate(operations, 2))
        self.assertLess(len(streams[0]), len(streams[1]))
        self.assert_decoded(streams, pictures)

    def test__given_bias_mixed_headers_or_disabled_guards__when_requesting_bands__then_existing_rows_are_preserved(self):
        parent = Operation("parent", (Source(0, weight=3), Source(1, weight=-1)), 1)
        biased = replace(parent, sources=(Source(0, offset=1), Source(1, offset=-1)))
        mixed = replace(parent, block_operations=(parent,)*3+(replace(parent, denominator=2),))
        for mode, enabled, op in (("row", True, biased), ("row", True, mixed),
                                   ("row", False, parent), ("macroblock", True, parent)):
            baseline = Writer(32, 32, slice_mode=mode, merge_zero_guards=enabled)
            for rows in (0, 2, 4, 8):
                writer = Writer(32, 32, slice_mode=mode, merge_zero_guards=enabled, prediction_slice_rows=rows)
                with self.subTest(mode=mode, enabled=enabled, operation=op.label, rows=rows):
                    self.assertEqual(writer._prediction_band_rows(op), 1)
                    self.assertEqual(writer.prediction(2, op), baseline.prediction(2, op))
                    self.assertEqual(writer.slice_count(op), baseline.slice_count(op))

    def test__given_retained_chains_and_changing_inputs__when_decoding_bands__then_each_cropped_round_remains_exact(self):
        width, height, rounds = 32, 48, 24
        operations = []
        for frame in range(1, rounds+1):
            if frame % 5 == 0:
                op = Operation("bias fallback", (Source(frame-1, weight=-1, offset=127),))
            elif frame % 2:
                op = Operation("offsets", (Source(frame-1, qx=1, qy=-1),),
                               block_offsets=((-4, 0), (2, -1), (0, 3), (-2, 1), (3, 2), (-1, -3)))
            else:
                op = Operation("retained input", (Source(frame-1, qx=1, qy=2), Source(0, qx=-1, qy=-2)))
            operations.append(op)
        slots = {frame: int(frame != 0) for frame in range(rounds+1)}
        tails = {}
        for fixture in range(2):
            pictures = [[[((x*71+y*37+fixture*83) % 256) for x in range(width)] for y in range(height)]]
            for op in operations:
                pictures.append(predict(pictures, op))
            streams = {}
            for rows in (1, 2, 4, 8, 0):
                writer = Writer(width, height, slice_mode="row", reference_slots=slots,
                                merge_zero_guards=True, prediction_slice_rows=rows)
                tail = b"".join(writer.prediction(frame, op) for frame, op in enumerate(operations, 1))
                self.assertEqual(tails.setdefault(rows, tail), tail)
                streams[rows] = writer.input_luma(pictures[0], 0)+tail
            self.assert_decoded(streams, pictures)

    def test__given_malformed_children__when_grouping__then_original_sources_are_still_validated(self):
        valid = Operation("valid", (Source(1),))
        writer = Writer(32, 16, slice_mode="row", merge_zero_guards=True, prediction_slice_rows=0)
        for field, value in (("frame", True), ("frame", 1.0), ("weight", True), ("offset", 0.0),
                             ("qx", 16), ("dy", False)):
            malformed = replace(valid, sources=(replace(Source(1), **{field: value}),))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                writer.prediction(2, replace(valid, block_operations=(valid, malformed)))

    def test__given_component_endpoints_and_visible_blocks__when_predicting_across_rows__then_motion_limits_remain_exact(self):
        pictures = [[[((x*71+y*37) % 256) for x in range(32)] for y in range(32)]]
        fields = ((8191, 2047), (-8192, -2048), (1, -2), (0, 0))
        operations = []
        for arity in (1, 2):
            parent = Operation("endpoints", (Source(0),)*arity)
            blocks = []
            for index, (qx, qy) in enumerate(fields):
                sources = (Source(0, qx=qx, qy=qy),)
                if arity == 2:
                    bx, by = fields[(index+2) % len(fields)]
                    sources += (Source(0, qx=bx, qy=by),)
                blocks.append(replace(parent, sources=sources))
            op = replace(parent, block_operations=tuple(blocks))
            operations.append(op)
            pictures.append(predict(pictures, op))
        streams = {}
        for rows in (1, 2, 0):
            writer = Writer(32, 32, slice_mode="row", merge_zero_guards=True,
                            motion_bits=14, prediction_slice_rows=rows)
            streams[rows] = writer.input_luma(pictures[0], 0)+b"".join(
                writer.prediction(frame, op) for frame, op in enumerate(operations, 1))
            for source in (Source(0, qx=8192), Source(0, qx=-8193),
                           Source(0, qy=2048), Source(0, qy=-2049)):
                with self.subTest(rows=rows, source=source), self.assertRaisesRegex(ValueError, "component limit"):
                    writer.prediction(1, Operation("out of range", (source,)))
        self.assert_decoded(streams, pictures)


if __name__ == "__main__":
    unittest.main()
