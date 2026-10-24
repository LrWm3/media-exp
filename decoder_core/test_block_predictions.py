"""Per-macroblock weighted predictions checked against unmodified decoders."""

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode


def predict_blocks(pictures, operation):
    """Independent integer-motion weighted-prediction oracle over active pixels."""
    height, width = len(pictures[0]), len(pictures[0][0])
    result = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            block = operation.block_operations[(y // 16) * (width // 16) + x // 16]
            values = []
            for source in block.sources:
                sx, sy = x + source.dx, y + source.dy
                values.append(pictures[source.frame][sy][sx]
                              if 0 <= sx < width and 0 <= sy < height else 0)
            denominator = block.denominator
            if len(values) == 1:
                rounding = (1 << (denominator - 1)) if denominator else 0
                value = ((block.sources[0].weight * values[0] + rounding) >> denominator)
                value += block.sources[0].offset
            else:
                value = (sum(source.weight * sample for source, sample in zip(block.sources, values))
                         + (1 << denominator)) >> (denominator + 1)
                value += (sum(source.offset for source in block.sources) + 1) >> 1
            result[y][x] = max(0, min(255, value))
    return result


def i420_picture(board):
    width, height = len(board[0]) + 32, len(board) + 32
    luma = bytearray(width * 16)
    for row in board:
        luma.extend(bytes(16) + bytes(row) + bytes(16))
    luma.extend(bytes(width * 16))
    return bytes(luma) + bytes([128]) * (width * height // 2)


class BlockPredictionTests(unittest.TestCase):
    def decode_streams(self, streams, expected=None):
        backends = [name for name in ("ffmpeg", "vaapi", "openh264") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        baseline = expected
        with tempfile.TemporaryDirectory() as directory:
            for name, stream in streams.items():
                path = Path(directory) / (name + ".h264")
                path.write_bytes(stream)
                for backend in backends:
                    with self.subTest(stream=name, backend=backend):
                        raw = decode(path, backend)
                        if baseline is None:
                            baseline = raw
                        self.assertEqual(raw, baseline)
        return baseline

    def test__given_tiled_p_and_b_weights_bias_denominators_references_and_motion__when_decoding__then_pixels_and_zero_separators_match(self):
        width, height = 96, 64
        pictures = [[[0 if y // 16 == 2 or x // 16 in (2, 5)
                      else (3 * x + 5 * y + 31 * frame) % 180 + 20
                      for x in range(width)] for y in range(height)] for frame in range(2)]
        operations = []
        for arity in (1, 2):
            children = []
            for my in range(height // 16):
                for mx in range(width // 16):
                    if my == 2 or mx in (2, 5):
                        children.append(Operation("zero separator", (Source(0, weight=0),) * arity))
                        continue
                    group = (my * 3 + mx // 3) % 4
                    source = Source((0, arity)[group % 2],
                                    weight=(1, 3, -2, 2)[group],
                                    offset=(7, -30, 90, 1)[group],
                                    dx=-(mx % 3), dy=-int(my == 1))
                    sources = (source,)
                    if arity == 2:
                        sources += (Source((2, 0)[group % 2],
                                           weight=(2, -1, 3, 1)[group],
                                           offset=(-5, 40, -20, 2)[group],
                                           dx=mx % 3, dy=int(my == 3)),)
                    children.append(Operation("tile", sources, (0, 1, 2, 1)[group]))
            operation = Operation("packed picture", (Source(0, weight=5, offset=70),) * arity,
                                  3, block_operations=tuple(children))
            operations.append(operation)
            pictures.append(predict_blocks(pictures, operation))
        streams = {}
        for mode in ("macroblock", "row"):
            writer = Writer(width, height, slice_mode=mode)
            streams[mode] = b"".join(writer.input_luma(board, frame)
                                    for frame, board in enumerate(pictures[:2]))
            streams[mode] += b"".join(writer.prediction(frame, operation)
                                     for frame, operation in enumerate(operations, 2))
        self.assertLess(len(streams["row"]), len(streams["macroblock"]))
        self.decode_streams(streams, b"".join(i420_picture(board) for board in pictures))

    def test__given_distinct_quarter_vectors_in_one_run__when_decoding__then_motion_prediction_and_linear_interior_match(self):
        board = [[20 + 4 * x + 4 * y for x in range(32)] for y in range(16)]
        children = (Operation("left", (Source(0, qx=-3, qy=1),)),
                    Operation("right", (Source(0, qx=3, qy=-1),)))
        operation = Operation("quarter motion", (Source(0),), block_operations=children)
        streams = {}
        for mode in ("macroblock", "row"):
            writer = Writer(32, 16, slice_mode=mode)
            streams[mode] = writer.seed_luma(board) + writer.prediction(1, operation)
        raw = self.decode_streams(streams)
        self.assertEqual(len(raw), writer.width * writer.height * 3)
        start = writer.width * writer.height * 3 // 2
        for y in range(4, 12):
            for x in range(4, 28):
                self.assertEqual(raw[start + (y + 16) * writer.width + x + 16],
                                 board[y][x] + (-2 if x < 16 else 2))

    def test__given_per_block_long_term_references_and_slot_replacement__when_decoding__then_old_inputs_and_current_occupants_survive(self):
        slots = {frame: 1 for frame in range(22)}
        slots[0] = slots[21] = 0
        pictures = [[[frame * 7 + x % 16 if x // 16 != 1 else 0
                      for x in range(48)] for _ in range(16)] for frame in range(20)]
        zero = Operation("separator", (Source(0, weight=0),))
        operations = (
            Operation("old and recent", (Source(19),), block_operations=(
                Operation("old input", (Source(0, offset=3),)), zero,
                Operation("recent input", (Source(19, offset=2),)))),
            Operation("read before replacement", (Source(20),), block_operations=(
                Operation("old input", (Source(0),)), zero,
                Operation("prior output", (Source(20, weight=3, offset=-4),), 1))),
        )
        for operation in operations:
            pictures.append(predict_blocks(pictures, operation))
        streams = {}
        for mode in ("macroblock", "row"):
            writer = Writer(48, 16, reference_slots=slots, slice_mode=mode)
            streams[mode] = b"".join(writer.input_luma(board, frame)
                                    for frame, board in enumerate(pictures[:20]))
            streams[mode] += b"".join(writer.prediction(frame, operation)
                                     for frame, operation in enumerate(operations, 20))
        self.decode_streams(streams, b"".join(i420_picture(board) for board in pictures))

    def test__given_equal_headers_and_different_motion__when_grouping_rows__then_only_header_changes_split_runs(self):
        writer = Writer(32, 16, slice_mode="row")
        first = Operation("first", (Source(0, weight=2, offset=3),), 1)
        second = replace(first, sources=(replace(first.sources[0], dx=1, qy=-1),))
        operation = Operation("parent", (Source(0),), block_operations=(first, second))
        self.assertEqual(writer.slice_count(operation), 5)
        self.assertEqual(writer.prediction(2, operation).count(b"\x00\x00\x00\x01"), 5)
        changes = (replace(second, denominator=2),
                   replace(second, sources=(replace(second.sources[0], frame=1),)),
                   replace(second, sources=(replace(second.sources[0], weight=3),)),
                   replace(second, sources=(replace(second.sources[0], offset=4),)))
        for changed in changes:
            with self.subTest(changed=changed):
                split = replace(operation, block_operations=(first, changed))
                self.assertEqual(writer.slice_count(split), 6)
                self.assertEqual(writer.prediction(2, split).count(b"\x00\x00\x00\x01"), 6)
        zero = Operation("zero", (Source(0, weight=0),))
        self.assertEqual(writer.slice_count(replace(operation, block_operations=(zero, zero))), 3)

    def test__given_short_nested_conflicting_or_mixed_arity_blocks__when_predicting__then_writer_rejects_structure(self):
        child = Operation("child", (Source(0),))
        parent = replace(child, block_operations=(child, child))
        invalid = (
            replace(parent, block_operations=(child,)),
            replace(parent, block_operations=(child,) * 3),
            replace(parent, block_offsets=((0, 0), (0, 0))),
            replace(parent, block_operations=(child, replace(child, block_operations=(child,)))),
            replace(parent, block_operations=(child, replace(child, block_offsets=((0, 0),)))),
            replace(parent, block_operations=(child, replace(child, sources=(Source(0), Source(0))))),
            replace(parent, sources=(Source(0), Source(0))),
            replace(parent, block_operations=(child, None)),
        )
        for mode in ("macroblock", "row"):
            writer = Writer(32, 16, slice_mode=mode)
            for operation in invalid:
                with self.subTest(mode=mode, operation=operation), self.assertRaises(ValueError):
                    writer.prediction(1, operation)

    def test__given_invalid_child_weights_denominators_and_motion__when_predicting__then_every_child_is_validated(self):
        child = Operation("child", (Source(0),))
        invalid = [replace(child, denominator=value) for value in (-1, 8, True, 1.5)]
        for field in ("weight", "offset"):
            invalid += [replace(child, sources=(replace(Source(0), **{field: value}),))
                        for value in (-129, 128, True, 1.5)]
        for field in ("dx", "dy", "qx", "qy"):
            invalid += [replace(child, sources=(replace(Source(0), **{field: value}),))
                        for value in (True, 0.5, 16, -17)]
        invalid += [replace(child, sources=()), replace(child, sources=(Source(0),) * 3)]
        for mode in ("macroblock", "row"):
            writer = Writer(32, 16, slice_mode=mode)
            for last in invalid:
                operation = replace(child, block_operations=(child, last))
                with self.subTest(mode=mode, child=last), self.assertRaises(ValueError):
                    writer.prediction(1, operation)

    def test__given_invalid_child_reference_or_guard_reference__when_predicting__then_actual_headers_validate_all_frames(self):
        for mode in ("macroblock", "row"):
            writer = Writer(32, 16, slice_mode=mode)
            child = Operation("valid", (Source(16),))
            for reference in (-1, 17, 18, True, 1.5, 0):
                invalid = replace(child, sources=(Source(reference),))
                for operation in (replace(child, block_operations=(child, invalid)),
                                  replace(invalid, block_operations=(child, child))):
                    with self.subTest(mode=mode, operation=operation), self.assertRaises(ValueError):
                        writer.prediction(17, operation)
            writer = Writer(32, 16, slice_mode=mode, reference_slots={0: 0, 1: 1, 2: 1, 3: 0})
            parent = Operation("resident guard", (Source(0),), block_operations=(
                Operation("resident", (Source(2),)), Operation("expired", (Source(1),))))
            with self.assertRaisesRegex(ValueError, "resident"):
                writer.prediction(3, parent)
            for reference in (True, 1.0):
                parent = Operation("valid guard", (Source(0),), block_operations=(
                    Operation("integer reference", (Source(1),)),
                    Operation("equal key, invalid type", (Source(reference),))))
                with self.subTest(mode=mode, reference=reference), self.assertRaisesRegex(ValueError, "precede"):
                    writer.prediction(2, parent)

    def test__given_fragmented_rows_above_actual_slice_rate__when_predicting__then_writer_rejects_before_any_header_is_written(self):
        children = tuple(Operation("alternating header", (Source(0, offset=index % 2 + 1),))
                         for index in range(100))
        operation = Operation("fragmented", (Source(0),), block_operations=children)
        writer = Writer(160, 160, fps=15, slice_mode="row")
        self.assertEqual(writer.slice_count(operation), 122)
        with patch.object(writer, "header", wraps=writer.header) as header:
            with self.assertRaisesRegex(ValueError, "slice rate"):
                writer.prediction(1, operation)
            header.assert_not_called()
        allowed = Writer(160, 160, fps=14, slice_mode="row")
        self.assertEqual(allowed.prediction(1, operation).count(b"\x00\x00\x00\x01"), 122)
        boundary_children = list(children)
        boundary_children[1] = boundary_children[0]
        boundary = replace(operation, block_operations=tuple(boundary_children))
        self.assertEqual(writer.slice_count(boundary) * writer.fps, 1800)
        self.assertEqual(writer.prediction(1, boundary).count(b"\x00\x00\x00\x01"), 120)

    def test__given_legacy_positional_api_and_no_block_predictions__when_encoding__then_existing_bytes_are_unchanged(self):
        expected = {"macroblock": "0afeb1b20699fe2f1f59ec0f46fa62c391daec0e8cbeba16fad4b745d1796816",
                    "row": "ca03068a289db2d1b1f86e0e50c3be16f48c30166b1e2293a496c8855fe35d57"}
        operations = (
            Operation("p", (Source(0, 3, -7, 1, -1, 2, 1),), 2),
            Operation("b", (Source(0, 3, 9), Source(1, -2, 12, qx=-3)), 1),
            Operation("gather", (Source(2, 2, 7),), 1,
                      ((64, 64), (-64, 64), (64, -64), (-64, -64))),
        )
        for operation in operations:
            self.assertEqual(operation.block_operations, ())
        for mode, digest in expected.items():
            writer = Writer(32, 32, motion_bits=8, slice_mode=mode)
            stream = writer.seed_luma([[x + 2 * y for x in range(32)] for y in range(32)])
            stream += b"".join(writer.prediction(frame, operation)
                               for frame, operation in enumerate(operations, 1))
            with self.subTest(mode=mode):
                self.assertEqual(hashlib.sha256(stream).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
