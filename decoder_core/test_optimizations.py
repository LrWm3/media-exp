"""Grouped slices and long-term slots checked against unmodified decoders."""

from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode


class WriterOptimizationTests(unittest.TestCase):
    def decoded(self, writer, streams, pictures):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        pixels = writer.width * writer.height
        frame_bytes = pixels * 3 // 2
        with tempfile.TemporaryDirectory() as directory:
            baseline = None
            for backend in backends:
                for name, stream in streams.items():
                    with self.subTest(backend=backend, stream=name):
                        path = Path(directory) / (name + ".h264")
                        path.write_bytes(stream)
                        raw = decode(path, backend)
                        self.assertEqual(len(raw), pictures * frame_bytes)
                        for start in range(0, len(raw), frame_bytes):
                            luma = raw[start:start + pixels]
                            self.assertEqual(raw[start + pixels:start + frame_bytes], bytes([128]) * (pixels // 2))
                            self.assertEqual(luma[:writer.width * 16], bytes(writer.width * 16))
                            self.assertEqual(luma[-writer.width * 16:], bytes(writer.width * 16))
                            for y in range(16, writer.height - 16):
                                row = luma[y * writer.width:(y + 1) * writer.width]
                                self.assertEqual(row[:16] + row[-16:], bytes(32))
                        if baseline is None:
                            baseline = raw
                        else:
                            self.assertEqual(raw, baseline)
                        yield backend, name, raw

    def test__given_random_weighted_p_b_and_block_gathers__when_grouping_rows__then_all_decoders_match_full_i420(self):
        rng = random.Random(8124)
        width, height = 96, 64
        boards = [[[rng.randrange(256) for _ in range(width)] for _ in range(height)] for _ in range(2)]
        operations = []
        for frame in range(2, 14):
            sources = tuple(Source(rng.randrange(frame), weight=rng.randrange(-3, 8),
                                   offset=rng.randrange(-50, 51), dx=rng.randrange(-6, 7),
                                   dy=rng.randrange(-6, 7), qx=rng.randrange(-3, 4),
                                   qy=rng.randrange(-3, 4)) for _ in range(1 + frame % 2))
            operations.append(Operation("random weighted prediction", sources, frame % 3))
        for frame in range(14, 16):
            offsets = tuple((rng.randrange(-80, 81), rng.randrange(-80, 81)) for _ in range(24))
            operations.append(Operation("varying quarter-pixel block gather",
                                        (Source(frame - 14, weight=3, offset=7, dx=1, qy=-2),), 2, offsets))
        streams = {}
        for mode in ("macroblock", "row"):
            writer = Writer(width, height, motion_bits=8, slice_mode=mode)
            stream = b"".join(writer.input_luma(board, frame) for frame, board in enumerate(boards))
            stream += b"".join(writer.prediction(frame, op) for frame, op in enumerate(operations, 2))
            streams[mode] = stream
        self.assertLess(len(streams["row"]), len(streams["macroblock"]))
        self.assertEqual(len(list(self.decoded(writer, streams, 16))),
                         2 * sum(available(name) for name in ("ffmpeg", "openh264", "vaapi")))

    def test__given_eight_by_six_macroblocks__when_grouping_motion_slices__then_forty_eight_become_fourteen(self):
        operation = Operation("motion", (Source(0, dx=1, qy=1),))
        macroblocks = Writer(96, 64)
        rows = Writer(96, 64, slice_mode="row")
        for writer, count in ((macroblocks, 48), (rows, 14)):
            with self.subTest(mode=writer.slice_mode):
                encoded = writer.prediction(1, operation)
                self.assertEqual(writer.slice_count(operation), count)
                self.assertEqual(encoded.count(b"\x00\x00\x00\x01"), count)
        # Matching zero-weight headers can include guard and active macroblocks
        # in the same row, including transitions between different motion vectors.
        zero = replace(operation, sources=(replace(operation.sources[0], weight=0),))
        self.assertEqual(rows.slice_count(zero), 6)
        board = [[x + y for x in range(96)] for y in range(64)]
        streams = {writer.slice_mode: writer.seed_luma(board) + writer.prediction(1, zero)
                   for writer in (macroblocks, rows)}
        list(self.decoded(rows, streams, 2))

    def test__given_long_lived_input_and_reused_slots__when_decoding_both_slice_modes__then_old_and_replacement_pictures_survive(self):
        rng = random.Random(151)
        width, height = 32, 32
        first = [[rng.randrange(200) for _ in range(width)] for _ in range(height)]
        second = [[rng.randrange(200) for _ in range(width)] for _ in range(height)]
        slots = {frame: 1 for frame in range(24)}
        slots[0] = 0
        slots[22] = 0
        operations = [(frame, Operation("reuse current source slot", (Source(frame - 1, offset=1),)))
                      for frame in range(2, 21)]
        operations += [
            (21, Operation("read input after twenty pictures", (Source(0), Source(20)))),
            (22, Operation("overwrite original input after reading it", (Source(0),))),
            (23, Operation("read both replacements", (Source(21), Source(22)))),
        ]
        streams = {}
        for mode in ("macroblock", "row"):
            writer = Writer(width, height, reference_slots=slots, slice_mode=mode)
            streams[mode] = (writer.seed_luma(first) + writer.input_luma(second, 1)
                             + b"".join(writer.prediction(frame, op) for frame, op in operations))
        twentieth = [[value + 19 for value in row] for row in second]
        twenty_first = [[(a + b + 1) // 2 for a, b in zip(arow, brow)]
                        for arow, brow in zip(first, twentieth)]
        twenty_third = [[(a + b + 1) // 2 for a, b in zip(arow, brow)]
                        for arow, brow in zip(twenty_first, first)]
        for _, _, raw in self.decoded(writer, streams, 24):
            frame_bytes = writer.width * writer.height * 3 // 2
            for frame, expected in ((0, first), (20, twentieth), (21, twenty_first),
                                    (22, first), (23, twenty_third)):
                start = frame * frame_bytes
                actual = [list(raw[start + y * writer.width + 16:start + y * writer.width + writer.width - 16])
                          for y in range(16, writer.height - 16)]
                self.assertEqual(actual, expected)

    def test__given_all_sixteen_slots__when_replacing_slot_fifteen__then_prediction_reads_prior_occupant(self):
        slots = {frame: min(frame, 15) for frame in range(18)}
        writer = Writer(16, 16, reference_slots=slots, slice_mode="row")
        stream = b"".join(writer.input_luma([[frame * 7] * 16 for _ in range(16)], frame)
                          for frame in range(16))
        stream += writer.prediction(16, Operation("full DPB overwrite", (Source(15, offset=2),)))
        stream += writer.prediction(17, Operation("oldest and newest", (Source(0), Source(16))))
        for _, _, raw in self.decoded(writer, {"full-slots": stream}, 18):
            size = writer.width * writer.height * 3 // 2
            self.assertEqual(raw[16 * size + 16 * writer.width + 16], 107)
            self.assertEqual(raw[17 * size + 16 * writer.width + 16], 54)

    def test__given_copied_reference_plan__when_encoding_repeatedly_out_of_order__then_bytes_and_plan_remain_immutable(self):
        slots = {0: 0, 1: 1, 2: 1, 3: 0}
        writer = Writer(16, 16, reference_slots=slots, slice_mode="row")
        operation = Operation("old input", (Source(0, qx=1),))
        expected = writer.prediction(3, operation)
        slots[0] = 15
        slots[2] = 0
        with self.assertRaises(TypeError):
            writer.reference_slots[0] = 15
        writer.prediction(2, Operation("reuse", (Source(1),)))
        writer.seed_luma([[7] * 16 for _ in range(16)])
        self.assertEqual(writer.prediction(3, operation), expected)
        self.assertEqual(dict(writer.reference_slots), {0: 0, 1: 1, 2: 1, 3: 0})

    def test__given_stale_future_missing_or_legacy_old_reference__when_encoding__then_writer_rejects_it(self):
        writer = Writer(16, 16, reference_slots={0: 0, 1: 1, 2: 1, 3: 0})
        with self.assertRaisesRegex(ValueError, "resident"):
            writer.prediction(3, Operation("stale", (Source(1),)))
        for source in (Source(-1), Source(3), Source(4), Source(True), Source(1.0)):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "precede"):
                writer.prediction(3, Operation("invalid source", (source,)))
        with self.assertRaisesRegex(ValueError, "missing"):
            writer.prediction(4, Operation("missing destination", (Source(3),)))
        with self.assertRaisesRegex(ValueError, "missing"):
            writer.input_luma([[0] * 16 for _ in range(16)], 4)
        with self.assertRaisesRegex(ValueError, "outside DPB"):
            Writer(16, 16).prediction(17, Operation("legacy age limit", (Source(0),)))

    def test__given_invalid_slot_plan_or_slice_mode__when_constructing_writer__then_writer_rejects_it(self):
        invalid_plans = ({}, {1: 0}, {0: 1}, {0: 0, 2: 1}, {0: 0, -1: 1},
                         {0: 0, 1: 16}, {0: 0, 1: -1}, {0: 0, 1: True},
                         {False: 0}, {0: 0, "1": 1}, {0: 0, 1: 1.0}, [0, 1])
        for slots in invalid_plans:
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                Writer(16, 16, reference_slots=slots)
        with self.assertRaisesRegex(ValueError, "slice_mode"):
            Writer(16, 16, slice_mode="picture")

    def test__given_invalid_prediction_parameters__when_encoding_either_mode__then_writer_rejects_it(self):
        operations = [Operation("invalid count", ()), Operation("invalid count", (Source(0),) * 3)]
        operations += [Operation("invalid denominator", (Source(0),), denominator)
                       for denominator in (-1, 8, True, 1.5)]
        operations += [Operation("invalid weight", (Source(0, weight=weight),))
                       for weight in (-129, 128, True, 0.0, 1.5)]
        for mode in ("macroblock", "row"):
            writer = Writer(16, 16, slice_mode=mode)
            for operation in operations:
                with self.subTest(mode=mode, operation=operation), self.assertRaises(ValueError):
                    writer.prediction(1, operation)
            for frame in (-1, True, 1.5):
                with self.subTest(frame=frame), self.assertRaises(ValueError):
                    writer.prediction(frame, Operation("invalid frame", (Source(0),)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
