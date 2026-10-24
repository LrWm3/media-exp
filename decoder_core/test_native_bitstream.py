"""Native input and motion syntax checked against unmodified H.264 decoders."""

from dataclasses import replace
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode


def legacy_input_luma(writer, board, frame):
    """Original bit-at-a-time PCM path, retained as an independent packing oracle."""
    bits = writer.header(0, frame)
    for my in range(writer.mb_height):
        for mx in range(writer.mb_width):
            bits.ue(25)
            bits.align()
            for y in range(my * 16, my * 16 + 16):
                for x in range(mx * 16, mx * 16 + 16):
                    inside = 16 <= x < writer.width - 16 and 16 <= y < writer.height - 16
                    bits.u(board[y - 16][x - 16] if inside else 0, 8)
            for _ in range(128):
                bits.u(128, 8)
    # Keep the former NAL packing here, independent of nal_from_rbsp.
    escaped, zeros = bytearray(), 0
    for value in bits.finish():
        if zeros == 2 and value <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(value)
        zeros = zeros + 1 if value == 0 else 0
    kind = 5 if frame == 0 else 1
    prefix = writer.parameter_sets() if frame == 0 else b""
    return prefix + b"\x00\x00\x00\x01" + bytes([0x60 | kind]) + escaped


class NativeBitstreamTests(unittest.TestCase):
    def decoded(self, writer, stream):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.h264"
            path.write_bytes(stream)
            for backend in backends:
                with self.subTest(backend=backend):
                    raw = decode(path, backend)
                    pixels = writer.width * writer.height
                    picture_bytes = pixels * 3 // 2
                    self.assertEqual(len(raw) % picture_bytes, 0)
                    pictures = []
                    for start in range(0, len(raw), picture_bytes):
                        luma = raw[start:start + pixels]
                        self.assertEqual(raw[start + pixels:start + picture_bytes], bytes([128]) * (pixels // 2))
                        for y in range(writer.height):
                            for x in range(writer.width):
                                if x < 16 or x >= writer.width - 16 or y < 16 or y >= writer.height - 16:
                                    self.assertEqual(luma[y * writer.width + x], 0)
                        pictures.append([list(luma[y * writer.width + 16:y * writer.width + writer.width - 16])
                                         for y in range(16, writer.height - 16)])
                    yield backend, pictures

    def test__given_legacy_inputs__when_loading_frame_zero__then_seed_bytes_and_positional_api_are_unchanged(self):
        writer = Writer(16, 16)
        image = [[y * 16 + x for x in range(16)] for y in range(16)]
        binary = [[(x + y) % 2 for x in range(16)] for y in range(16)]
        self.assertEqual(hashlib.sha256(writer.seed_luma(image)).hexdigest(),
                         "bc8f01c02af464b79488ad7dc87d3f874bbbf0aa2fcea15eafa63326c0a276ec")
        self.assertEqual(hashlib.sha256(writer.seed(binary)).hexdigest(),
                         "cefc050d986909594a9b29176b2c75401127551bbed25cd04b91908fef1d82df")
        self.assertEqual(writer.seed_luma(image), writer.input_luma(image, 0))
        source = Source(1, 2, 3, 4, 5)
        self.assertEqual((source.frame, source.weight, source.offset, source.dx, source.dy, source.qx, source.qy),
                         (1, 2, 3, 4, 5, 0, 0))
        self.assertEqual(Operation("legacy", (source,), 2).block_offsets, ())

    def test__given_random_and_escape_heavy_pcm__when_packing_bytes__then_both_input_frames_match_legacy_bit_encoding(self):
        rng = random.Random(741)
        escape_pattern = (0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 255)
        for width, height in ((16, 16), (32, 16), (32, 32), (64, 64)):
            writer = Writer(width, height)
            boards = (
                [[0] * width for _ in range(height)],
                [[255] * width for _ in range(height)],
                [[escape_pattern[(x + y) % len(escape_pattern)] for x in range(width)] for y in range(height)],
                [[rng.randrange(256) for _ in range(width)] for _ in range(height)],
            )
            for pattern, board in enumerate(boards):
                for frame in (0, 1):
                    with self.subTest(width=width, height=height, pattern=pattern, frame=frame):
                        self.assertEqual(writer.input_luma(board, frame), legacy_input_luma(writer, board, frame))

    def test__given_picture_number_boundaries__when_packing_pcm_header__then_legacy_bytes_match_without_extra_stop_bits(self):
        writer = Writer(16, 32, fps=60, motion_bits=9)
        board = [[(x * 59 + y * 13) % 256 for x in range(16)] for y in range(32)]
        for frame in (0, 1, 127, 32767, 32768, 65535, 65536):
            with self.subTest(frame=frame):
                self.assertEqual(writer.input_luma(board, frame), legacy_input_luma(writer, board, frame))

    def test__given_two_u8_inputs__when_summing_and_reading_both_again__then_every_decoder_retains_inputs_and_clips_sum(self):
        writer = Writer(16, 16)
        rng = random.Random(37)
        first = [[rng.randrange(256) for _ in range(16)] for _ in range(16)]
        second = [[rng.randrange(256) for _ in range(16)] for _ in range(16)]
        subsequent = writer.input_luma(second, 1)
        self.assertEqual([part[0] & 31 for part in subsequent.split(b"\x00\x00\x00\x01")[1:]], [1])
        stream = writer.input_luma(first, 0) + subsequent
        stream += writer.prediction(2, Operation("sum", (Source(0, 2), Source(1, 2))))
        stream += writer.prediction(3, Operation("first again", (Source(0),)))
        stream += writer.prediction(4, Operation("second again", (Source(1),)))
        summed = [[min(255, a + b) for a, b in zip(arow, brow)] for arow, brow in zip(first, second)]
        self.assertTrue(any(value == 255 for row in summed for value in row))
        self.assertTrue(any(value < 255 for row in summed for value in row))
        for _, pictures in self.decoded(writer, stream):
            self.assertEqual(pictures, [first, second, summed, first, second])

    def test__given_mixed_integer_and_quarter_motion__when_decoding_linear_ramp__then_interpolated_interior_matches(self):
        writer = Writer(16, 16)
        image = [[16 + 4 * x + 4 * y for x in range(16)] for y in range(16)]
        # (1.25, -0.5) pixels on this plane adds exactly three luma levels.
        source = Source(0, dx=1, dy=-1, qx=1, qy=2)
        stream = writer.seed_luma(image) + writer.prediction(1, Operation("quarter", (source,)))
        baseline = None
        for _, pictures in self.decoded(writer, stream):
            self.assertEqual(len(pictures), 2)
            self.assertEqual(pictures[0], image)
            for y in range(4, 12):
                for x in range(4, 12):
                    self.assertEqual(pictures[1][y][x], image[y][x] + 3)
            if baseline is None:
                baseline = pictures
            else:
                self.assertEqual(pictures, baseline)

    def test__given_four_block_offsets__when_decoding__then_row_major_blocks_swap_and_weighting_survives(self):
        writer = Writer(32, 32, motion_bits=8)
        image = [[((x // 16) + 2 * (y // 16)) * 40 + (x % 16) + (y % 16)
                  for x in range(32)] for y in range(32)]
        source = Source(0, weight=2, offset=7, dx=1, dy=1, qx=1, qy=1)
        # Source contributes (5, 5) quarter pixels; each combined MV swaps diagonally.
        offsets = ((59, 59), (-69, 59), (59, -69), (-69, -69))
        operation = Operation("swap", (source,), 1, offsets)
        stream = writer.seed_luma(image) + writer.prediction(1, operation)
        expected = [[image[(y + 16) % 32][(x + 16) % 32] + 7 for x in range(32)] for y in range(32)]
        for _, pictures in self.decoded(writer, stream):
            self.assertEqual(pictures, [image, expected])

    def test__given_invalid_frame_or_luma__when_loading_input__then_writer_rejects_it(self):
        writer = Writer(16, 16)
        image = [[0] * 16 for _ in range(16)]
        for frame in (-1, 0.5, True):
            with self.subTest(frame=frame), self.assertRaisesRegex(ValueError, "frame"):
                writer.input_luma(image, frame)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            writer.input_luma([[0]], 1)
        for value in (-1, 256, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "samples"):
                writer.input_luma([[value] * 16 for _ in range(16)], 1)

    def test__given_noninteger_motion__when_predicting__then_writer_rejects_every_component(self):
        writer = Writer(16, 16)
        for component in ("dx", "dy", "qx", "qy"):
            for value in (0.5, True):
                with self.subTest(component=component, value=value), self.assertRaisesRegex(ValueError, "integers"):
                    writer.prediction(1, Operation("invalid", (replace(Source(0), **{component: value}),)))

    def test__given_boundary_motion__when_predicting__then_combined_quarter_vector_controls_range(self):
        writer = Writer(16, 16, motion_bits=4)
        for source in (Source(0, qx=-16, qy=15), Source(0, dx=4, qx=-1), Source(0, dy=-5, qy=4)):
            with self.subTest(source=source):
                self.assertTrue(writer.prediction(1, Operation("valid", (source,))))
        for source in (Source(0, qx=16), Source(0, qy=-17), Source(0, dx=3, qx=4)):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "SPS bound"):
                writer.prediction(1, Operation("invalid", (source,)))

    def test__given_invalid_block_offsets__when_predicting__then_writer_rejects_shape_sources_and_combined_range(self):
        writer = Writer(32, 16, motion_bits=4)
        cases = (
            Operation("short", (Source(0),), block_offsets=((0, 0),)),
            Operation("bipred", (Source(0), Source(0)), block_offsets=((0, 0), (0, 0))),
            Operation("fraction", (Source(0),), block_offsets=((0.5, 0), (0, 0))),
            Operation("bool", (Source(0),), block_offsets=((True, 0), (0, 0))),
            Operation("pair", (Source(0),), block_offsets=((0, 0, 0), (0, 0))),
            Operation("shape", (Source(0),), block_offsets=(0, 0)),
            Operation("range", (Source(0, qx=15),), block_offsets=((1, 0), (0, 0))),
        )
        for operation in cases:
            with self.subTest(label=operation.label), self.assertRaises(ValueError):
                writer.prediction(1, operation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
