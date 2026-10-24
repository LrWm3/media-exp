"""Normative integer interpolation and decoder checks for fractional motion."""

import random
import unittest

import h264
from h264.graph import Image
from h264.verify import evaluate, interpolate
from decoder_core.decode import available


class InterpolationOracleTests(unittest.TestCase):
    def test__given_six_tap_neighborhood__when_sampling_all_phases__then_normative_values_match(self):
        # H.264 8.4.2.2.1, equations 8-241..8-261. At integer (2,2),
        # G=118, H=229, M=10, b=177, h=33, m=239, s=103, j=132.
        board = [[6, 240, 57, 201, 72, 22],
                 [71, 57, 118, 71, 75, 16],
                 [31, 69, 118, 229, 211, 104],
                 [47, 60, 10, 206, 174, 101],
                 [105, 169, 200, 186, 126, 107],
                 [111, 213, 30, 26, 92, 182]]
        expected = [[118, 148, 177, 203],
                    [76, 105, 155, 208],
                    [33, 83, 132, 186],
                    [22, 68, 118, 171]]
        plane = bytes(value for row in board for value in row)
        actual = [[interpolate(plane, 6, 6, 8+fx, 8+fy)
                   for fx in range(4)] for fy in range(4)]
        self.assertEqual(actual, expected)

    def test__given_diagonal_half_pel__when_filtering__then_intermediate_values_keep_precision(self):
        plane = bytearray(36)
        plane[2*6+2] = 1
        self.assertEqual(interpolate(plane, 6, 6, 10, 8), 1)
        self.assertEqual(interpolate(plane, 6, 6, 10, 10), 0)
        plane[2*6+2] = plane[2*6+3] = 255
        self.assertEqual(interpolate(plane, 6, 6, 10, 8), 255)
        self.assertEqual(interpolate(plane, 6, 6, 10, 10), 199)

    def test__given_negative_coordinates__when_sampling__then_floor_and_zero_guard_are_used(self):
        # One impulse: six-tap half samples beside it round to 159; their
        # diagonal product rounds to 100. Quarter averages round upward.
        plane = bytes([255])
        self.assertEqual(interpolate(plane, 1, 1, -1, 0), 207)
        self.assertEqual(interpolate(plane, 1, 1, 0, -1), 207)
        self.assertEqual(interpolate(plane, 1, 1, -2, -2), 100)
        self.assertEqual(interpolate(plane, 1, 1, -1, -1), 159)
        self.assertEqual(interpolate(plane, 1, 1, -3, -3), 0)
        self.assertEqual(interpolate(plane, 1, 1, -6, 0), 0)
        for qx, qy in ((-1000, 0), (1000, 0), (0, -1000), (0, 1000)):
            with self.subTest(position=(qx, qy)):
                self.assertEqual(interpolate(plane, 1, 1, qx, qy), 0)

    def test__given_edge_samples__when_padding_explicitly__then_every_phase_remains_equal(self):
        width, height, padding = 5, 4, 4
        plane = bytes((x*47+y*89) % 256 for y in range(height) for x in range(width))
        padded_width, padded_height = width+2*padding, height+2*padding
        padded = bytearray(padded_width*padded_height)
        for y in range(height):
            padded[(y+padding)*padded_width+padding:(y+padding)*padded_width+padding+width] = plane[y*width:(y+1)*width]
        for y_q in range(-11, 4*height+12):
            for x_q in range(-11, 4*width+12):
                with self.subTest(position=(x_q, y_q)):
                    self.assertEqual(interpolate(plane, width, height, x_q, y_q),
                                     interpolate(padded, padded_width, padded_height,
                                                 x_q+4*padding, y_q+4*padding))

    def test__given_named_boards__when_evaluating_saturation__then_inputs_and_clipping_match(self):
        a = Image("input", (), ("a",), 0, 255)
        b = Image("input", (), ("b",), 0, 255)
        weighted = Image("saturating_weighted", (a,), (2, -100), 0, 255)
        combined = Image("saturating_combine", (a, b), (2, -1, -20), 0, 255)
        boards = {"b": [[30, 250, 20, 0]], "a": [[0, 100, 200, 255]]}
        outputs = evaluate([a, b, weighted, combined], {"weighted": 2, "combined": 3}, boards)
        self.assertEqual(outputs["weighted"], [[0, 100, 255, 255]])
        self.assertEqual(outputs["combined"], [[0, 0, 255, 255]])

    def test__given_per_block_offsets__when_evaluating_gather__then_blocks_use_row_major_order(self):
        board = [[1+(x//16)+2*(y//16) for x in range(32)] for y in range(32)]
        source = Image("input", (), ("image",), 0, 255)
        gathered = Image("gather", (source,), (((64, 64), (-64, 64), (64, -64), (-64, -64)),), 0, 255)
        output = evaluate([source, gathered], {"output": 1}, board)["output"]
        self.assertEqual(output, [[4-(x//16)-2*(y//16) for x in range(32)] for y in range(32)])


class InterpolationDecoderTests(unittest.TestCase):
    @staticmethod
    def image(width=32, height=32):
        rng = random.Random(241)
        return [[rng.randrange(256) for _ in range(width)] for _ in range(height)]

    def verify_decoders(self, program, board):
        for backend in ("ffmpeg", "vaapi", "openh264"):
            with self.subTest(backend=backend):
                if not available(backend):
                    self.skipTest(f"{backend} unavailable")
                program.verify(board, backend)

    def test__given_all_fractional_phases_and_negative_motion__when_decoding__then_oracle_matches(self):
        offsets = [(fx, fy) for fy in range(4) for fx in range(4)]
        offsets += [(fx-4, fy-4) for fy in range(4) for fx in range(4)]
        offsets += [(67, -67), (-67, 67), (508, 508), (-508, -508)]
        def kernel(x):
            return {str(offset): h264.sample(x, qx=offset[0], qy=offset[1]) for offset in offsets}
        program = h264.compile(kernel, 32, 32, input_range=(0, 255))
        self.verify_decoders(program, self.image())

    def test__given_fractional_gather_per_macroblock__when_decoding__then_each_phase_matches(self):
        offsets = [(fx-4, fy-4) for fy in range(4) for fx in range(4)]
        program = h264.compile(lambda x: h264.gather(x, offsets=offsets), 64, 64, input_range=(0, 255))
        self.verify_decoders(program, self.image(64, 64))

    def test__given_opposing_fractional_samples__when_optimizing__then_rounding_and_cropping_survive(self):
        def kernel(x):
            sampled = h264.sample(x, qx=1, qy=-3)
            return {"nested": h264.sample(sampled, qx=-1, qy=3),
                    "threshold": h264.threshold(sampled, 128),
                    "weighted": h264.saturating_weighted(sampled, w=2, bias=-37)}
        board = [[255 if (x+y) % 3 else 0 for x in range(32)] for y in range(32)]
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                program = h264.compile(kernel, 32, 32, input_range=(0, 255), optimize=optimize)
                self.verify_decoders(program, board)


if __name__ == "__main__":
    unittest.main(verbosity=2)
