"""Saved-stream CLI readback preserves distances and reports actual copies."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from maze_kernel import compile_maze
from maze import demo_maze, bfs_distances
from decoder_core.decode import available


class RunnerTests(unittest.TestCase):
    @unittest.skipUnless(available("ffmpeg"), "FFmpeg unavailable")
    def test__given_saved_demo__when_readback_mode_changes__then_distances_match_and_counts_are_honest(self):
        maze = demo_maze()
        distances = bfs_distances(maze)
        with tempfile.TemporaryDirectory(prefix="maze-cli-") as directory:
            results, streams = [], []
            for mode in ("outputs", "all"):
                path = Path(directory) / (mode + ".h264")
                command = [sys.executable, str(Path(__file__).with_name("run.py")),
                           "--steps", "52", "--verify", "--output", str(path)]
                if mode == "all":
                    command.extend(("--readback", "all"))
                subprocess.run(command, check=True, capture_output=True, text=True)
                result = json.loads(path.with_suffix(".distances.json").read_text())
                self.assertEqual(result["distances"], [
                    [distances[y][x] if cell else -1 for x, cell in enumerate(row)]
                    for y, row in enumerate(maze.cells)])
                self.assertEqual(result["pictures"], 214)
                self.assertEqual(result["readback"]["mode"], mode)
                # FFmpeg copies the full stream before compatibility selection.
                self.assertEqual(result["readback"]["decoded_pictures"], 214)
                self.assertEqual(result["readback"]["copied_pictures"], 214)
                self.assertEqual(result["readback"]["returned_pictures"], 54 if mode == "outputs" else 214)
                self.assertEqual(result["goal_distance"], 50)
                self.assertEqual(result["converged_at"], 51)
                results.append(result)
                streams.append(path.read_bytes())
            self.assertEqual(streams[0], streams[1])
            self.assertEqual(results[0]["path"], results[1]["path"])
            self.assertEqual(results[0]["readback"]["selected_frames"],
                             list(compile_maze(32, 32, 52).output_picture_indices))


if __name__ == "__main__":
    unittest.main()
