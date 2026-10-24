"""Integration tests run the generated instructions through real decoders."""

from pathlib import Path
import tempfile
import unittest

from h264 import Writer
from life import seed_board
from program import compile_stream, instructions
from verify import available, check, decode


def all_neighborhoods():
    # Isolated 3x3 patches encode every possible center + eight-neighbor input.
    board = [[0] * 128 for _ in range(64)]
    for pattern in range(512):
        x, y = (pattern % 32) * 4, (pattern // 32) * 4
        for bit in range(9):
            board[y + bit // 3][x + bit % 3] = (pattern >> bit) & 1
    return board


class DecoderTests(unittest.TestCase):
    def assert_decodes(self, board, generations, backend="ffmpeg", stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        if stream is None:
            stream, _ = compile_stream(board, generations)
        with tempfile.TemporaryDirectory(prefix="life-test-") as directory:
            path = Path(directory) / "test.h264"
            path.write_bytes(stream)
            return check(decode(path, backend), board, generations)

    def test__given_all_512_neighborhoods__when_decoding__then_every_rule_matches(self):
        self.assert_decodes(all_neighborhoods(), 1)

    def test__given_patterns_and_random_edges__when_decoding_many_generations__then_all_cells_match(self):
        for pattern in ("empty", "block", "blinker", "glider", "r-pentomino", "random"):
            with self.subTest(pattern=pattern):
                self.assert_decodes(seed_board(pattern, 32, 48), 25)

    def test__given_replacement_seed__when_reusing_identical_instructions__then_decoder_computes_new_future(self):
        first = seed_board("glider", 32, 32)
        second = seed_board("random", 32, 32, 7331)
        writer = Writer(32, 32)
        stream, _ = compile_stream(first, 12)
        tail = stream[len(writer.seed(first)):]
        replacement, _ = compile_stream(second, 12)
        self.assertEqual(tail, replacement[len(writer.seed(second)):])
        self.assert_decodes(second, 12, stream=writer.seed(second) + tail)

    def test__given_zero_generations__when_decoding__then_seed_is_exact(self):
        self.assert_decodes(seed_board("random", 16, 16), 0)

    def test__given_level_limit_violations__when_generating__then_nonconforming_streams_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "per-axis"):
            Writer(2688, 16, fps=1)
        with self.assertRaisesRegex(ValueError, "slice rate"):
            Writer(96, 96)
        with self.assertRaisesRegex(ValueError, "access-unit"):
            Writer(256, 128, fps=5).seed(seed_board("empty", 256, 128))

    def test__given_long_program__when_allocating_references__then_every_operand_fits_sliding_window(self):
        program = instructions(5000)
        self.assertEqual(len(program), 70000)
        for frame, op in enumerate(program, 1):
            self.assertTrue(all(1 <= frame - source.frame <= Writer.MAX_REFS for source in op.sources))

    def test__given_exhaustive_inputs__when_decoding_on_intel_hardware__then_all_rules_match(self):
        self.assert_decodes(all_neighborhoods(), 1, "vaapi")

    def test__given_random_edges__when_decoding_on_intel_hardware__then_25_generations_match(self):
        self.assert_decodes(seed_board("random", 32, 48, 99), 25, "vaapi")

    def test__given_exhaustive_inputs__when_decoding_with_openh264__then_all_rules_match(self):
        self.assert_decodes(all_neighborhoods(), 1, "openh264")


if __name__ == "__main__":
    unittest.main(verbosity=2)
