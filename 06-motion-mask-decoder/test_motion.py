"""Compare unmodified decoders with an independent pixel-level motion oracle."""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import frame
from kernel import compile_motion, h264, morphology, motion_kernel
from reference import motion, square_filter
from decoder_core.decode import available, decode


def board(value=0, width=32, height=32):
    return [[value] * width for _ in range(height)]


def rectangle(image, left, top, right, bottom, value):
    for y in range(top, bottom):
        for x in range(left, right):
            image[y][x] = value


def compile_difference(width=64, height=64, cutoff=32):
    def kernel(previous, current):
        outputs = motion_kernel(previous, current, cutoff)
        # Exercise production arithmetic; dead-code elimination omits morphology.
        return {name: outputs[name] for name in ("difference", "raw_mask")}
    return h264.compile(kernel, width, height,
                        input_ranges={"previous": (0, 255), "current": (0, 255)})


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        if not cls.backends:
            raise unittest.SkipTest("No FFmpeg, OpenH264, or VAAPI decoder available")
        cls.program = compile_motion(width=32, height=32)

    def assert_decodes(self, program, inputs, expected):
        for backend in self.backends:
            with self.subTest(backend=backend):
                # Do not use Program.verify: expected pixels come from a separate oracle.
                self.assertEqual(program.decode(inputs, backend), expected)

    def test__given_every_u8_pair__when_decoding_difference_and_threshold__then_all_65536_pairs_match(self):
        program = compile_difference()
        for chunk in range(16):
            previous = [[(chunk * 4096 + y * 64 + x) // 256 for x in range(64)] for y in range(64)]
            current = [[(y * 64 + x) % 256 for x in range(64)] for y in range(64)]
            difference = [[abs(a - b) for a, b in zip(old, new)] for old, new in zip(previous, current)]
            expected = {"difference": difference,
                        "raw_mask": [[int(value >= 32) for value in row] for row in difference]}
            with self.subTest(chunk=chunk):
                self.assert_decodes(program, {"previous": previous, "current": current}, expected)

    def test__given_both_signs_at_cutoff_boundaries__when_thresholding__then_equality_counts_as_change(self):
        for cutoff in (1, 32, 128, 255):
            pairs = [(old, old + sign * distance)
                     for distance in (cutoff - 1, cutoff, cutoff + 1)
                     for sign in (-1, 1) for old in (0, 64, 128, 255)
                     if 0 <= old + sign * distance <= 255]
            previous = [[pairs[(y * 16 + x) % len(pairs)][0] for x in range(16)] for y in range(16)]
            current = [[pairs[(y * 16 + x) % len(pairs)][1] for x in range(16)] for y in range(16)]
            difference = [[abs(a - b) for a, b in zip(old, new)] for old, new in zip(previous, current)]
            expected = {"difference": difference,
                        "raw_mask": [[int(value >= cutoff) for value in row] for row in difference]}
            with self.subTest(cutoff=cutoff):
                self.assert_decodes(compile_difference(16, 16, cutoff),
                                    {"previous": previous, "current": current}, expected)

    def test__given_all_512_local_patterns__when_applying_morphology__then_every_neighborhood_matches(self):
        image = board(width=128, height=64)
        for bits in range(512):
            left, top = bits % 32 * 4, bits // 32 * 4
            for bit in range(9):
                image[top + bit // 3][left + bit % 3] = (bits >> bit) & 1
        expected = {"eroded": square_filter(image, True), "dilated": square_filter(image, False)}
        for bits in range(512):
            x, y = bits % 32 * 4 + 1, bits // 32 * 4 + 1
            self.assertEqual(expected["eroded"][y][x], int(bits == 511))
            self.assertEqual(expected["dilated"][y][x], int(bits != 0))
        program = h264.compile(lambda mask: {"eroded": morphology(mask, True),
                                             "dilated": morphology(mask, False)},
                               128, 64, input_ranges={"mask": (0, 1)})
        self.assert_decodes(program, {"mask": image}, expected)

    def test__given_changes_at_image_edges__when_applying_morphology__then_outside_pixels_are_zero(self):
        program = h264.compile(lambda mask: {"eroded": morphology(mask, True),
                                             "dilated": morphology(mask, False)},
                               32, 32, input_ranges={"mask": (0, 1)})
        corners = board()
        for x, y in ((0, 0), (31, 0), (0, 31), (31, 31)):
            corners[y][x] = 1
        for name, image in (("empty", board()), ("full", board(1)), ("corners", corners)):
            expected = {"eroded": square_filter(image, True), "dilated": square_filter(image, False)}
            if name == "full":
                interior = board()
                rectangle(interior, 1, 1, 31, 31, 1)
                self.assertEqual(expected["eroded"], interior)
                self.assertEqual(expected["dilated"], image)
            with self.subTest(case=name):
                self.assert_decodes(program, {"mask": image}, expected)

    def test__given_unchanged_and_random_frames__when_running_full_pipeline__then_all_stages_match(self):
        rng = random.Random(56319)
        images = [[[rng.randrange(256) for _ in range(32)] for _ in range(32)] for _ in range(4)]
        pairs = [("unchanged", images[0], images[0]),
                 ("random_first", images[0], images[1]),
                 ("random_second", images[2], images[3]),
                 ("full_change", board(), board(255))]
        for name, previous, current in pairs:
            expected = motion(previous, current)
            if name == "unchanged":
                self.assertEqual(expected["output"], board())
            elif name == "full_change":
                interior = board()
                rectangle(interior, 1, 1, 31, 31, 1)
                self.assertEqual(expected["output"], interior)
            with self.subTest(case=name):
                self.assert_decodes(self.program, {"previous": previous, "current": current}, expected)

    def test__given_isolated_impulses_and_one_hole__when_opening_then_closing__then_noise_clears_and_hole_fills(self):
        previous, current = board(80), board(80)
        rectangle(current, 8, 8, 24, 24, 200)
        current[16][16] = 80
        impulses = ((2, 2), (29, 2), (2, 29), (29, 29))
        for x, y in impulses:
            current[y][x] = 200
        expected = motion(previous, current)
        cleaned = board()
        rectangle(cleaned, 8, 8, 24, 24, 1)
        self.assertEqual(expected["output"], cleaned)
        self.assertEqual(expected["opened"][16][16], 0)
        self.assertEqual(expected["output"][16][16], 1)
        for x, y in impulses:
            self.assertEqual(expected["raw_mask"][y][x], 1)
            self.assertEqual(expected["opened"][y][x], 0)
        self.assert_decodes(self.program, {"previous": previous, "current": current}, expected)

    def test__given_translated_rectangle__when_detecting_motion__then_changed_strips_exclude_unchanged_overlap(self):
        previous, current = board(80), board(80)
        rectangle(previous, 6, 8, 18, 24, 200)
        rectangle(current, 10, 8, 22, 24, 200)
        expected = motion(previous, current)
        changed = board()
        rectangle(changed, 6, 8, 10, 24, 1)
        rectangle(changed, 18, 8, 22, 24, 1)
        self.assertEqual(expected["raw_mask"], changed)
        self.assertEqual(expected["output"], changed)
        self.assertEqual(sum(map(sum, expected["output"])), 128)
        self.assertEqual(expected["output"][16][14], 0)
        self.assert_decodes(self.program, {"previous": previous, "current": current}, expected)

    def test__given_deterministic_noisy_video__when_decoding_frame_pairs__then_all_stages_match_pixel_oracle(self):
        program = compile_motion()
        for index in (0, 6):
            previous, current = frame(index), frame(index + 1)
            with self.subTest(index=index):
                self.assert_decodes(program, {"previous": previous, "current": current}, motion(previous, current))

    def test__given_changed_frame_pairs__when_reusing_prediction_tail__then_new_masks_need_no_host_evaluator(self):
        first = {"previous": board(80), "current": board(80)}
        second = {"previous": board(80), "current": board(80)}
        rectangle(second["current"], 8, 8, 24, 24, 200)
        expected = motion(second["previous"], second["current"])
        with patch("h264.backend.evaluate", side_effect=AssertionError("Host graph evaluator invoked")):
            program = compile_motion(width=32, height=32)
            first_prefix, second_prefix = program.encode_inputs(first), program.encode_inputs(second)
            first_stream, second_stream = program.encode(first), program.encode(second)
            tail = first_stream[len(first_prefix):]
            self.assertTrue(tail)
            self.assertEqual(tail, program.tail)
            self.assertEqual(second_stream[len(second_prefix):], tail)
            self.assertNotEqual(first_prefix, second_prefix)
            with tempfile.TemporaryDirectory(prefix="motion-tail-test-") as directory:
                path = Path(directory) / "reused.h264"
                path.write_bytes(second_prefix + tail)
                for backend in self.backends:
                    with self.subTest(backend=backend):
                        self.assertEqual(program.read(decode(path, backend)), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
