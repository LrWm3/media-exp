#!/usr/bin/env python3
"""Render the paired run's actual selected NVDEC output and decoded route."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from maze_kernel import compile_maze, extract_solution
from maze import Maze
from solve import verify
# maze_kernel adds the compiler's directory, which also contains run.py.
sys.path.insert(0, str(HERE.parent))
from run import solution_document
from capture import render


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_capture(bundle, results, output):
    bundle_bytes = (bundle / "bundle.json").read_bytes()
    report = json.loads((results / "results.json").read_text())
    if report["status"] != "pass" or report["bundle_sha256"] != digest(bundle_bytes):
        raise ValueError("Decoder results do not verify this bundle")
    case = next(case for case in json.loads(bundle_bytes)["cases"] if case["name"] == "demo")
    record = report["capture"]
    raw = (results / record["path"]).read_bytes()
    if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
        raise ValueError("Selected capture differs from verified output")
    source = case["input"]
    maze = Maze(tuple(tuple(row) for row in source["cells"]),
                tuple(tuple(point) for point in source["starts"]),
                tuple(source["goal"]) if source["goal"] is not None else None)
    program = compile_maze(maze.width, maze.height, case["steps"])
    if digest(program.encode(maze.initial_luma())) != record["stream_sha256"]:
        raise ValueError("Current program differs from the captured stream")
    outputs = program.read(raw, selected_frames=record["selected_frames"])
    solution = extract_solution(outputs, maze, case["steps"])
    verify(solution, maze)
    readback = {"mode": "outputs", "decoded_pictures": record["decoded_frames"],
                "copied_pictures": record["copied_frames"],
                "returned_pictures": len(record["selected_frames"]),
                "copied_bytes": record["copied_frames"] * record["frame_bytes"],
                "returned_bytes": len(raw), "selected_frames": record["selected_frames"]}
    document = solution_document(program, maze, solution, case["steps"], report["backend"], readback)
    document["captured_i420_sha256"] = digest(raw)
    output.mkdir(parents=True, exist_ok=True)
    (output / "maze.distances.json").write_text(json.dumps(document, indent=2) + "\n")
    render(solution, maze, output / "maze.gif", report["backend"])
    # Select a held frame showing the route recovered from actual arrival times.
    last_state = solution.converged_at if solution.converged_at is not None else case["steps"]
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output / "maze.gif"),
                    "-vf", f"select=eq(n\\,{last_state + 2})", "-frames:v", "1", "-update", "1",
                    "-y", str(output / "maze-capture.png")], check=True, timeout=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "assets")
    args = parser.parse_args()
    make_capture(args.bundle, args.results, args.output)
