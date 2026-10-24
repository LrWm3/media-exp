"""Real-decoder tests against independent BFS, including exhaustive local inputs."""

from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from codec import available, decode, Writer
from flood_program import compile_stream, instructions
from maze import Maze, demo_maze, parse_ascii
from solve import extract, verify


def exhaustive_maze():
    cells = [[False] * 64 for _ in range(64)]
    starts = []
    for index, pattern in enumerate(product((0, 128, 255), repeat=5)):
        cx, cy = (index % 16) * 4 + 1, (index // 16) * 4 + 1
        for (dx, dy), value in zip(((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)), pattern):
            x, y = cx + dx, cy + dy
            cells[y][x] = value != 0
            if value == 255:
                starts.append((x, y))
    return Maze(tuple(tuple(row) for row in cells), tuple(starts))


class FloodTests(unittest.TestCase):
    def solve(self, maze, steps, backend="ffmpeg", stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        if stream is None:
            stream, _ = compile_stream(maze, steps)
        with tempfile.TemporaryDirectory(prefix="flood-test-") as directory:
            path = Path(directory) / "maze.h264"
            path.write_bytes(stream)
            result = extract(decode(path, backend), maze, steps)
        verify(result, maze)
        return result

    def test__given_all_243_neighborhoods__when_decoded_by_three_backends__then_each_rule_matches(self):
        maze = exhaustive_maze()
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                self.solve(maze, 1, backend)

    def test__given_demo_with_sealed_room__when_decoded_by_three_backends__then_distances_and_route_match(self):
        maze = demo_maze()
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                result = self.solve(maze, 60, backend)
                self.assertEqual(len(result.path) - 1, 50)
                self.assertEqual(result.converged_at, 51)
                self.assertIsNone(result.distances[4][28])

    def test__given_open_boundaries_and_multiple_starts__when_decoding__then_distances_use_nearest_start_without_wrap(self):
        maze = Maze(tuple((True,) * 16 for _ in range(16)), ((0, 0), (15, 15)))
        result = self.solve(maze, 20)
        self.assertEqual(result.distances[0][15], 15)
        self.assertEqual(result.distances[7][7], 14)

    def test__given_diagonal_only_connection__when_flooding__then_goal_remains_unreachable(self):
        maze = parse_ascii("S#\n#G")
        result = self.solve(maze, 3)
        self.assertEqual(result.path, [])
        self.assertEqual(result.converged_at, 1)

    def test__given_short_budget__when_reading_results__then_unvisited_cells_are_unresolved(self):
        maze = Maze(tuple((True,) * 16 for _ in range(16)), ((0, 0),), (15, 15))
        result = self.solve(maze, 2)
        self.assertIsNone(result.converged_at)
        self.assertIsNone(result.distances[15][15])
        self.assertEqual(result.distances[0][2], 2)
        self.assertEqual(result.path, [])

    def test__given_corridor_longer_than_255__when_decoding__then_distance_does_not_overflow_luma(self):
        cells = [[False] * 32 for _ in range(32)]
        for row in range(15):
            y = row * 2 + 1
            cells[y][1:31] = [True] * 30
            if row < 14:
                cells[y + 1][30 if row % 2 == 0 else 1] = True
        maze = Maze(tuple(tuple(row) for row in cells), ((1, 1),), (30, 29))
        result = self.solve(maze, 464)
        self.assertEqual(result.distances[29][30], 463)
        self.assertEqual(len(result.path), 464)

    def test__given_different_mazes__when_reusing_identical_instruction_bytes__then_decoder_computes_new_distances(self):
        first, second = demo_maze(7), demo_maze(42)
        writer = Writer(32, 32)
        stream, _ = compile_stream(first, 80)
        tail = stream[len(writer.seed_luma(first.initial_luma())):]
        other, _ = compile_stream(second, 80)
        self.assertEqual(tail, other[len(writer.seed_luma(second.initial_luma())):])
        self.solve(second, 80, stream=writer.seed_luma(second.initial_luma()) + tail)

    def test__given_random_walls__when_decoding__then_every_intermediate_flood_state_matches_bfs(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                cells = [[rng.random() > 0.35 for _ in range(32)] for _ in range(16)]
                cells[0][0] = True
                maze = Maze(tuple(tuple(row) for row in cells), ((0, 0),))
                self.solve(maze, 32)

    def test__given_compiler_and_distance_extractor__when_bfs_is_disabled__then_neither_requires_the_oracle(self):
        maze = demo_maze()
        with patch("maze.bfs_distances", side_effect=AssertionError("oracle called")), \
                patch("solve.bfs_distances", side_effect=AssertionError("oracle called")):
            stream, _ = compile_stream(maze, 12)
            with tempfile.TemporaryDirectory(prefix="flood-no-oracle-") as directory:
                path = Path(directory) / "maze.h264"
                path.write_bytes(stream)
                result = extract(decode(path), maze, 12)
        self.assertEqual(result.distances[1][1], 0)
        self.assertGreater(sum(d is not None for row in result.distances for d in row), 1)

    def test__given_zero_steps__when_decoding__then_only_sources_have_distance_zero(self):
        result = self.solve(demo_maze(), 0)
        self.assertEqual(sum(d is not None for row in result.distances for d in row), 1)
        self.assertIsNone(result.converged_at)

    def test__given_invalid_input__when_loading_or_compiling__then_errors_are_clear(self):
        for text in ("...", "S.\n#", "SX", "SGG"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ascii(text)
        with self.assertRaises(ValueError):
            instructions(-1)
        with self.assertRaises(ValueError):
            Writer(16, 16).seed_luma([[256] * 16 for _ in range(16)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
