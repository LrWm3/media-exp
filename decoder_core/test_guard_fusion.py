"""Uniform zero-offset row headers preserve finite guards and full I420."""

from dataclasses import replace
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from codec_compute.sampling import interpolate_h264
from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture


def predict(pictures, operation):
    """Semantic six-tap reads followed by independent weighted arithmetic."""
    height, width = len(pictures[0]), len(pictures[0][0])
    planes = [bytes(value for row in board for value in row) for board in pictures]
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            index = (y//16)*(width//16)+x//16
            op = operation.block_operations[index] if operation.block_operations else operation
            values = []
            for source in op.sources:
                qx, qy = source.qx, source.qy
                if operation.block_offsets:
                    ox, oy = operation.block_offsets[index]
                    qx, qy = qx+ox, qy+oy
                values.append(interpolate_h264(planes[source.frame], width, height,
                              4*(x+source.dx)+qx, 4*(y+source.dy)+qy))
            denominator = op.denominator
            if len(values) == 1:
                rounding = (1 << (denominator-1)) if denominator else 0
                value = ((op.sources[0].weight*values[0]+rounding) >> denominator) + op.sources[0].offset
            else:
                value = ((sum(source.weight*sample for source, sample in zip(op.sources, values))
                          + (1 << denominator)) >> (denominator+1))
                value += (sum(source.offset for source in op.sources)+1) >> 1
            row.append(min(255, max(0, value)))
        result.append(row)
    return result


class GuardFusionTests(unittest.TestCase):
    def assert_decoded(self, streams, pictures):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        expected = b"".join(i420_picture(board) for board in pictures)
        with tempfile.TemporaryDirectory() as folder:
            for label, stream in streams.items():
                path = Path(folder)/(label+".h264")
                path.write_bytes(stream)
                for backend in backends:
                    with self.subTest(stream=label, decoder=backend):
                        self.assertEqual(decode(path, backend), expected)

    def test__given_default_or_false_flag__when_encoding__then_legacy_bytes_are_preserved(self):
        board = [[(x*37+y*71) % 256 for x in range(32)] for y in range(32)]
        operations = (Operation("p", (Source(0, qx=1, qy=-2),)),
                      Operation("b", (Source(0, weight=3), Source(1, weight=-1)), denominator=1))
        streams = []
        for options in ({}, {"merge_zero_guards": False}):
            writer = Writer(32, 32, slice_mode="row", **options)
            streams.append(writer.input_luma(board, 0)+b"".join(
                writer.prediction(frame, op) for frame, op in enumerate(operations, 1)))
        self.assertEqual(streams[0], streams[1])
        self.assertEqual(hashlib.sha256(streams[0]).hexdigest(),
                         "8851a04498737cc4d83a3b01651c422119a44eed37b9dac94849950b12c1aed7")

    def test__given_nonboolean_flag__when_constructing__then_writer_rejects_it(self):
        for value in (None, 0, 1, "false", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "merge_zero_guards must be Boolean"):
                Writer(16, 16, merge_zero_guards=value)

    def test__given_uniform_headers__when_merging_rows__then_counts_drop_without_crossing_rows(self):
        for size, level, old, new in ((128, 31, 26, 10), (256, 41, 50, 18), (512, 51, 98, 34)):
            for sources in ((Source(0, qx=3),), (Source(0, weight=3), Source(0, weight=-1))):
                operation = Operation("uniform", sources)
                for enabled, expected in ((False, old), (True, new)):
                    writer = Writer(size, size, slice_mode="row", level_idc=level, merge_zero_guards=enabled)
                    with self.subTest(size=size, sources=sources, enabled=enabled):
                        self.assertEqual(writer.slice_count(operation), expected)
                        self.assertEqual(writer.prediction(1, operation).count(b"\0\0\0\1"), expected)
                        for run in writer._prediction_runs(operation):
                            self.assertEqual(len({address//writer.mb_width for address, _ in run}), 1)
        operation = Operation("macroblock", (Source(0),))
        old = Writer(32, 32)
        new = Writer(32, 32, merge_zero_guards=True)
        self.assertFalse(new.zero_guard_merge_eligible(operation))
        self.assertEqual(new.prediction(1, operation), old.prediction(1, operation))
        self.assertEqual(new.slice_count(operation), 16)

    def test__given_biased_or_mixed_headers__when_enabled__then_forced_zero_guards_remain_byte_identical(self):
        parent = Operation("parent", (Source(0, weight=3), Source(1, weight=-1)), 1)
        blocks = (parent,)*4
        variants = [replace(parent, sources=(Source(0, offset=1), Source(1, offset=-1)))]
        changes = (replace(parent, denominator=2),
                   replace(parent, sources=tuple(reversed(parent.sources))),
                   replace(parent, sources=(Source(0, weight=2), parent.sources[1])),
                   replace(parent, sources=(Source(0, weight=3, offset=1), parent.sources[1])),
                   replace(parent, sources=(Source(1, weight=3), parent.sources[1])))
        for changed in changes:
            variants.extend((replace(parent, block_operations=blocks[:-1]+(changed,)),
                             replace(parent, block_operations=(changed,)*4)))
        old = Writer(32, 32, slice_mode="row")
        new = Writer(32, 32, slice_mode="row", merge_zero_guards=True)
        for operation in variants:
            with self.subTest(operation=operation):
                self.assertFalse(new.zero_guard_merge_eligible(operation))
                self.assertEqual(new.prediction(2, operation), old.prediction(2, operation))
        pictures = [[[(x*37+y*71+frame*91) % 256 for x in range(32)] for y in range(32)] for frame in range(2)]
        stream = b"".join(new.input_luma(board, frame) for frame, board in enumerate(pictures))
        for frame, operation in enumerate(variants, 2):
            pictures.append(predict(pictures, operation))
            stream += new.prediction(frame, operation)
        self.assert_decoded({"fallback": stream}, pictures)

    def test__given_fractional_block_fields_and_all_denominators__when_decoding__then_weighted_pixels_guards_and_chroma_match(self):
        size = 64
        randomizer = random.Random(8317)
        pictures = [[[randomizer.randrange(256) for _ in range(size)] for _ in range(size)],
                    [[255*((x+3*y) % 7 < 3) for x in range(size)] for y in range(size)]]
        operations = []
        for arity in (1, 2):
            for denominator in range(8):
                sources = ((Source(0, weight=(1, 3, 127, -128, -3, 65, 0, 127)[denominator]),)
                           if arity == 1 else (Source(0, weight=127 if denominator == 7 else 3),
                                               Source(denominator % 2, weight=0 if denominator == 7 else -1)))
                parent = Operation("weighted", sources, denominator)
                children = []
                for index in range(16):
                    fx, fy = index % 4, index//4
                    moving = (replace(sources[0], qx=fx-4, qy=fy-4),)
                    if arity == 2:
                        moving += (replace(sources[1], qx=3-fx, qy=3-fy),)
                    children.append(replace(parent, sources=moving))
                operation = replace(parent, block_operations=tuple(children))
                operations.append(operation)
                pictures.append(predict(pictures, operation))
        slots = {frame: (frame if frame < 2 else 2) for frame in range(len(pictures))}
        streams = {}
        for enabled in (False, True):
            writer = Writer(size, size, slice_mode="row", reference_slots=slots, merge_zero_guards=enabled)
            streams[str(enabled)] = b"".join(writer.input_luma(board, frame) for frame, board in enumerate(pictures[:2]))
            streams[str(enabled)] += b"".join(writer.prediction(frame, op) for frame, op in enumerate(operations, 2))
            if enabled:
                for op in operations:
                    self.assertEqual(writer.slice_count(op), size//16+2)
                    for run in writer._prediction_runs(op):
                        for address, block in run:
                            y, x = divmod(address, writer.mb_width)
                            if x in (0, writer.mb_width-1) or y in (0, writer.mb_height-1):
                                self.assertEqual([writer._motion(s) for s in block.sources], [(0, 0)]*len(op.sources))
                                self.assertEqual([s.weight for s in block.sources], [s.weight for s in op.sources])
        self.assertLess(len(streams["True"]), len(streams["False"]))
        self.assert_decoded(streams, pictures)

    def test__given_offset_fields_bias_fallback_and_retained_chain__when_decoding__then_every_picture_matches(self):
        size, rounds = 32, 24
        pictures = [[[((x*71+y*37) % 256) for x in range(size)] for y in range(size)]]
        operations = []
        for frame in range(1, rounds+1):
            if frame % 5 == 0:
                op = Operation("biased fallback", (Source(frame-1, weight=-1, offset=127),))
            elif frame % 2:
                op = Operation("fractional offsets", (Source(frame-1, qx=1, qy=-1),),
                               block_offsets=((-4, 0), (2, -1), (0, 3), (-2, 1)))
            else:
                op = Operation("retained input blend", (Source(frame-1, qx=1, qy=2), Source(0, qx=-1, qy=-2)))
            operations.append(op)
            pictures.append(predict(pictures, op))
        slots = {frame: int(frame != 0) for frame in range(rounds+1)}
        streams = {}
        for enabled in (False, True):
            writer = Writer(size, size, slice_mode="row", reference_slots=slots, merge_zero_guards=enabled)
            streams[str(enabled)] = writer.seed_luma(pictures[0])+b"".join(
                writer.prediction(frame, op) for frame, op in enumerate(operations, 1))
        self.assert_decoded(streams, pictures)

    def test__given_malformed_original_sources__when_merging__then_validation_cannot_be_hidden_by_equal_keys(self):
        writer = Writer(32, 16, slice_mode="row", merge_zero_guards=True)
        valid = Operation("valid", (Source(1),))
        invalid = [replace(valid, sources=(replace(Source(1), **{field: value}),))
                   for field, values in (("frame", (True, 1.0, -1, 2)),
                                         ("weight", (True, 1.0, 128)),
                                         ("offset", (False, 0.0, -129)),
                                         ("qx", (True, 1.0, 16)),
                                         ("dy", (True, 0.0, -5))) for value in values]
        invalid += [replace(valid, denominator=value) for value in (False, 0.0, 8)]
        for child in invalid:
            for op in (replace(valid, block_operations=(valid, child)),
                       replace(child, block_operations=(valid, valid))):
                with self.subTest(operation=op), self.assertRaises(ValueError):
                    writer.prediction(2, op)


if __name__ == "__main__":
    unittest.main()
