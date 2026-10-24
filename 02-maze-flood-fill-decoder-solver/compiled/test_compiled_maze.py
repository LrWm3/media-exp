"""Compiled flood integration tests against BFS and the original bitstream."""

from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from maze_kernel import compile_maze, extract_solution
from decoder_core.decode import available, decode
from flood_program import compile_stream
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


class CompiledMazeTests(unittest.TestCase):
    def solve(self, maze, steps, backend="ffmpeg", optimize=True):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        program = compile_maze(maze.width, maze.height, steps, optimize=optimize)
        outputs = program.verify(maze.initial_luma(), backend=backend)
        solution = extract_solution(outputs, maze, steps)
        verify(solution, maze)
        return solution

    def test__given_optimized_flood__when_compiling__then_setup_and_iterations_have_expected_picture_costs(self):
        for steps in (1, 2, 52):
            with self.subTest(steps=steps):
                program = compile_maze(32, 32, steps)
                self.assertEqual(program.pictures, 6 + 4 * steps)
                self.assertEqual(program.stats["reference_copies"], 0)
                self.assertEqual(set(program.output_frames),
                                 {"output", "passages", *[f"step_{i}" for i in range(steps + 1)]})
                for step in range(2, steps + 1):
                    self.assertEqual(program.output_frames[f"step_{step}"]
                                     - program.output_frames[f"step_{step - 1}"], 4)

    def test__given_all_243_neighborhoods__when_decoding_three_steps__then_every_state_matches_bfs(self):
        self.solve(exhaustive_maze(), 3)

    def test__given_demo_and_sealed_room__when_decoding_until_convergence__then_distances_and_path_match(self):
        maze = demo_maze()
        solution = self.solve(maze, 52)
        self.assertEqual(len(solution.path) - 1, 50)
        self.assertEqual(solution.converged_at, 51)
        self.assertIsNone(solution.distances[4][28])

    def test__given_open_edges_and_multiple_starts__when_decoding__then_nearest_source_wins_without_wrap(self):
        maze = Maze(tuple((True,) * 16 for _ in range(16)), ((0, 0), (15, 15)))
        solution = self.solve(maze, 16)
        self.assertEqual(solution.distances[0][15], 15)
        self.assertEqual(solution.distances[7][7], 14)
        self.assertEqual(solution.converged_at, 16)

    def test__given_diagonal_goal__when_flooding__then_goal_stays_unreachable(self):
        solution = self.solve(parse_ascii("S#\n#G"), 3)
        self.assertEqual(solution.path, [])
        self.assertEqual(solution.converged_at, 1)

    def test__given_isolated_and_connected_starts__when_flooding__then_sources_persist_and_sealed_room_stays_unreached(self):
        maze = parse_ascii("S#S..S##..\n########..\n########G.")
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                solution = self.solve(maze, 4, backend=backend)
                self.assertEqual(solution.converged_at, 2)
                self.assertEqual(solution.distances[0][:6], [0, None, 0, 1, 1, 0])
                self.assertIsNone(solution.distances[2][8])
                self.assertEqual(solution.path, [])
                for state in solution.states:
                    for x, y in maze.starts:
                        self.assertEqual(state[y * maze.width + x], 255)

    def test__given_short_budget__when_extracting__then_unvisited_goal_stays_unresolved(self):
        maze = Maze(tuple((True,) * 16 for _ in range(16)), ((0, 0),), (15, 15))
        solution = self.solve(maze, 2)
        self.assertIsNone(solution.converged_at)
        self.assertIsNone(solution.distances[15][15])
        self.assertEqual(solution.distances[0][2], 2)

    def test__given_random_walls__when_decoding__then_all_intermediate_states_match_bfs(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                cells = [[rng.random() > 0.35 for _ in range(16)] for _ in range(16)]
                cells[0][0] = True
                self.solve(Maze(tuple(tuple(row) for row in cells), ((0, 0),)), 12)

    def test__given_corridor_longer_than_255__when_decoding__then_first_arrival_distance_does_not_overflow_luma(self):
        cells = [[False] * 32 for _ in range(32)]
        for row in range(15):
            y = row * 2 + 1
            cells[y][1:31] = [True] * 30
            if row < 14:
                cells[y + 1][30 if row % 2 == 0 else 1] = True
        maze = Maze(tuple(tuple(row) for row in cells), ((1, 1),), (30, 29))
        solution = self.solve(maze, 464)
        self.assertEqual(solution.distances[29][30], 463)
        self.assertEqual(len(solution.path), 464)
        self.assertEqual(solution.converged_at, 464)

    def test__given_original_and_compiled_variants__when_decoding_same_maze__then_all_states_and_distances_match(self):
        maze = demo_maze()
        solution = self.solve(maze, 12)
        stream, _ = compile_stream(maze, 12)
        with tempfile.TemporaryDirectory(prefix="compiled-maze-parity-") as directory:
            path = Path(directory) / "original.h264"
            path.write_bytes(stream)
            original = extract(decode(path), maze, 12)
        self.assertEqual(solution, original)

    def test__given_disabled_bfs_oracle__when_compiling_and_extracting__then_results_still_come_from_decoder(self):
        if not available("ffmpeg"):
            self.skipTest("ffmpeg unavailable")
        maze = demo_maze()
        with patch("maze.bfs_distances", side_effect=AssertionError("oracle called")), \
                patch("solve.bfs_distances", side_effect=AssertionError("oracle called")):
            program = compile_maze(32, 32, 8)
            outputs = program.decode(maze.initial_luma())
            solution = extract_solution(outputs, maze, 8)
        self.assertEqual(solution.distances[1][1], 0)
        self.assertGreater(sum(value is not None for row in solution.distances for value in row), 1)

    def test__given_one_compiled_program__when_replacing_maze__then_both_decoded_floods_match_bfs(self):
        if not available("ffmpeg"):
            self.skipTest("ffmpeg unavailable")
        program = compile_maze(32, 32, 10)
        operations = list(program.operations)
        for seed in (7, 42):
            maze = demo_maze(seed)
            outputs = program.verify(maze.initial_luma())
            verify(extract_solution(outputs, maze, 10), maze)
            self.assertEqual(program.operations, operations)

    def test__given_zero_budget__when_decoding__then_only_starts_have_distance_zero(self):
        program = compile_maze(32, 32, 0)
        self.assertEqual(program.pictures, 5)
        self.assertEqual(program.stats["dead_nodes"], 1)
        solution = self.solve(demo_maze(), 0)
        self.assertEqual(sum(value is not None for row in solution.distances for value in row), 1)
        self.assertIsNone(solution.converged_at)

    def test__given_unoptimized_graph__when_decoding__then_all_states_match_bfs(self):
        self.solve(demo_maze(), 3, optimize=False)
        self.solve(parse_ascii("S#S..S##..\n########..\n########G."), 4, optimize=False)

    def test__given_demo_outputs__when_selecting_readback__then_all_decoders_match_full_readback_and_bfs(self):
        maze = demo_maze()
        program = compile_maze(32, 32, 52)
        indices = program.output_picture_indices
        self.assertEqual(list(indices), sorted(set(program.output_frames.values())))
        self.assertEqual(len(indices), 54)
        self.assertEqual(program.output_frames["output"], program.output_frames["step_52"])
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                if not available(backend):
                    self.skipTest(f"{backend} unavailable")
                all_outputs = program.verify(maze.initial_luma(), backend=backend, readback="all")
                selected_outputs = program.verify(maze.initial_luma(), backend=backend, readback="outputs")
                self.assertEqual(selected_outputs, all_outputs)
                solution = extract_solution(selected_outputs, maze, 52)
                verify(solution, maze)
                self.assertEqual(solution.converged_at, 51)
                self.assertEqual(len(solution.path) - 1, 50)

    def test__given_exhaustive_inputs__when_decoding_with_openh264__then_every_rule_matches(self):
        self.solve(exhaustive_maze(), 1, backend="openh264")

    def test__given_exhaustive_inputs__when_decoding_with_vaapi__then_every_rule_matches(self):
        self.solve(exhaustive_maze(), 1, backend="vaapi")

    def test__given_negative_budget__when_compiling__then_error_identifies_invalid_steps(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            compile_maze(16, 16, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
