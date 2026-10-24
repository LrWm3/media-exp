"""Shared-compiler Life integration tests using actual decoder output."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from life_kernel import EXPERIMENT, compile_life, generation_frames, verify_generations
from decoder_core.decode import available, decode
from life import seed_board, step


def all_neighborhoods():
    board = [[0] * 128 for _ in range(64)]
    for pattern in range(512):
        x, y = (pattern % 32) * 4, (pattern // 32) * 4
        for bit in range(9):
            board[y + bit // 3][x + bit % 3] = (pattern >> bit) & 1
    return board


class CompiledLifeTests(unittest.TestCase):
    def assert_decodes(self, board, generations, backend="ffmpeg", optimize=True):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        program = compile_life(len(board[0]), len(board), generations, optimize=optimize)
        outputs = program.verify(board, backend=backend)
        expected = board
        for generation in range(generations + 1):
            with self.subTest(generation=generation):
                self.assertEqual(outputs[f"generation_{generation}"], expected)
            expected = step(expected)
        verify_generations(outputs, board, generations)
        return program, outputs

    def test__given_all_512_neighborhoods__when_decoding_three_generations__then_every_cell_matches(self):
        self.assert_decodes(all_neighborhoods(), 3)

    def test__given_random_edges_and_known_patterns__when_decoding__then_every_generation_matches(self):
        for pattern in ("empty", "block", "blinker", "glider", "r-pentomino", "random"):
            with self.subTest(pattern=pattern):
                self.assert_decodes(seed_board(pattern, 32, 16, 7331), 5)

    def test__given_one_compiled_program__when_replacing_the_seed__then_decoder_computes_each_future(self):
        if not available("ffmpeg"):
            self.skipTest("ffmpeg unavailable")
        program = compile_life(16, 16, 4)
        operations = list(program.operations)
        for pattern in ("glider", "random"):
            board = seed_board(pattern, 16, 16, 99)
            outputs = program.verify(board)
            verify_generations(outputs, board, 4)
            self.assertEqual(program.operations, operations)

    def test__given_zero_generations__when_decoding__then_named_outputs_equal_input(self):
        board = seed_board("random", 16, 16)
        _, outputs = self.assert_decodes(board, 0)
        self.assertEqual(outputs["output"], board)

    def test__given_unoptimized_graph__when_decoding__then_oracle_still_matches(self):
        self.assert_decodes(seed_board("random", 16, 16), 2, optimize=False)

    def test__given_original_and_compiled_variants__when_decoding_identical_seed__then_all_states_match(self):
        board = seed_board("random", 16, 16, 71)
        _, outputs = self.assert_decodes(board, 4)
        with tempfile.TemporaryDirectory(prefix="compiled-life-parity-") as directory:
            path = Path(directory) / "original.h264"
            subprocess.run([sys.executable, str(EXPERIMENT / "run.py"), "--pattern", "random",
                            "--random-seed", "71", "--width", "16", "--height", "16",
                            "--generations", "4", "--verify", "--output", str(path)],
                           cwd=EXPERIMENT, check=True, capture_output=True, timeout=180)
            raw = decode(path)
        coded_width, frame_size = 48, 48 * 48 * 3 // 2
        original = [b"".join(raw[generation * 14 * frame_size + (y + 16) * coded_width + 16:
                                    generation * 14 * frame_size + (y + 16) * coded_width + 32]
                              for y in range(16)) for generation in range(5)]
        self.assertEqual(generation_frames(outputs, 4), original)

    def test__given_exhaustive_inputs__when_decoding_with_openh264__then_every_rule_matches(self):
        self.assert_decodes(all_neighborhoods(), 1, backend="openh264")

    def test__given_exhaustive_inputs__when_decoding_with_vaapi__then_every_rule_matches(self):
        self.assert_decodes(all_neighborhoods(), 1, backend="vaapi")

    def test__given_negative_budget__when_compiling__then_error_identifies_invalid_generations(self):
        with self.assertRaisesRegex(ValueError, "generations"):
            compile_life(16, 16, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
