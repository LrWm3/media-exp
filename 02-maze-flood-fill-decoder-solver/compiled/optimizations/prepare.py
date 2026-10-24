#!/usr/bin/env python3
"""Prepare portable paired maze streams, full decoder references, and BFS proofs."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMPILED = HERE.parent
EXPERIMENT = COMPILED.parent
REPOSITORY = EXPERIMENT.parent
sys.path.insert(0, str(COMPILED))

from maze_kernel import compile_maze, extract_solution
import h264
from decoder_core.decode import available, decode
from maze import CARDINALS, Maze, bfs_distances, demo_maze, parse_ascii
from solve import verify


BASELINE_SOURCE = {
    "commit": "67caef8f4f6febece83918c23e6ba79f10bb3b13",
    "path": "02-maze-flood-fill-decoder-solver/compiled/maze_kernel.py",
    "sha256": "ad4014c82d1256ea179dadc908c49d394639447bd3d636277b0d0a659ccff572",
    "description": "Original compile_maze body reproduced below; both variants use the current compiler",
}


def compile_baseline(width, height, steps):
    """Benchmark-only original self-plus-four-neighbors OR, then passage AND."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    @h264.kernel
    def flood(seed):
        passages = h264.threshold(seed, 1)
        reached = h264.threshold(seed, 129)
        states = [reached]
        for _ in range(steps):
            expanded = reached
            for dx, dy in CARDINALS:
                expanded = h264.bit_or(expanded, h264.shift(reached, dx=dx, dy=dy))
            reached = h264.bit_and(expanded, passages)
            states.append(reached)
        return {"output": reached,
                **{f"step_{i}": state for i, state in enumerate(states)},
                "passages": passages}

    return h264.compile(flood, width=width, height=height,
                        input_range=(0, 255), optimize=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plane(board):
    return bytes(value for row in board for value in row)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_bytes(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": digest(data), "bytes": len(data)}


def exhaustive_maze():
    cells = [[False] * 64 for _ in range(64)]
    starts = []
    for index, pattern in enumerate(product((0, 128, 255), repeat=5)):
        cx, cy = (index % 16) * 4 + 1, (index // 16) * 4 + 1
        for (dx, dy), value in zip(((0, 0), *CARDINALS), pattern):
            x, y = cx + dx, cy + dy
            cells[y][x] = value != 0
            if value == 255:
                starts.append((x, y))
    return Maze(tuple(tuple(row) for row in cells), tuple(starts))


def long_corridor():
    """Same 463-hop snake as the original solver's overflow regression."""
    cells = [[False] * 32 for _ in range(32)]
    for row in range(15):
        y = row * 2 + 1
        cells[y][1:31] = [True] * 30
        if row < 14:
            cells[y + 1][30 if row % 2 == 0 else 1] = True
    return Maze(tuple(tuple(row) for row in cells), ((1, 1),), (30, 29))


def selected_cases(names=None):
    open_cells = tuple((True,) * 16 for _ in range(16))
    cases = [("demo", demo_maze(), 52),
             ("exhaustive", exhaustive_maze(), 3),
             ("isolated-start", parse_ascii("S#\n#G"), 4),
             ("multi-start", Maze(open_cells, ((0, 0), (15, 15))), 16),
             ("short-budget", Maze(open_cells, ((0, 0),), (15, 15)), 2),
             ("zero", demo_maze(), 0)]
    for seed in range(3):
        rng = random.Random(seed)
        cells = [[rng.random() > 0.35 for _ in range(16)] for _ in range(16)]
        cells[0][0] = True
        cases.append(("random-%d" % seed,
                      Maze(tuple(tuple(row) for row in cells), ((0, 0),)), 12))
    cases.append(("long-corridor", long_corridor(), 464))
    if names is not None:
        unknown = set(names) - {name for name, _, _ in cases}
        if unknown:
            raise ValueError("Unknown cases: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case[0] in names]
    return cases


def bfs_reference(maze, steps):
    """Independent BFS determines every named plane, distance, and route."""
    distances = bfs_distances(maze)
    named = {"passages": [[int(cell) for cell in row] for row in maze.cells]}
    for step in range(steps + 1):
        named["step_%d" % step] = [[int(value is not None and value <= step)
                                   for value in row] for row in distances]
    named["output"] = named["step_%d" % steps]
    bounded = [[value if value is not None and value <= steps else None
                for value in row] for row in distances]
    last_arrival = max(value for row in distances for value in row if value is not None)
    path = []
    if maze.goal is not None:
        x, y = maze.goal
        if bounded[y][x] is not None:
            path.append([x, y])
            while bounded[y][x] > 0:
                x, y = next((x + dx, y + dy) for dx, dy in CARDINALS
                            if 0 <= x + dx < maze.width and 0 <= y + dy < maze.height
                            and bounded[y + dy][x + dx] == bounded[y][x] - 1)
                path.append([x, y])
            path.reverse()
    return named, {"distances": bounded, "path": path,
                   "converged_at": last_arrival + 1 if last_arrival < steps else None}


def prepare(output, names=None, local_backends=("ffmpeg", "vaapi", "openh264"), cases=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(names) if cases is None else cases
    backends = {backend: {"status": "pending" if available(backend) else "unavailable",
                          "variants": 0} for backend in local_backends}
    if "ffmpeg" not in backends or backends["ffmpeg"]["status"] == "unavailable":
        raise RuntimeError("FFmpeg is required for full-picture reference bytes")
    source_paths = [HERE / "prepare.py", HERE / "benchmark.py", COMPILED / "maze_kernel.py",
                    EXPERIMENT / "maze.py", EXPERIMENT / "solve.py",
                    REPOSITORY / "decoder_core" / "decode.py",
                    REPOSITORY / "decoder_core" / "bitstream.py",
                    *sorted((REPOSITORY / "05-h264-compiler" / "h264").glob("*.py"))]
    report = {"format": "h264-maze-optimization-bundle-0.1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
              "source_sha256": {str(path.relative_to(REPOSITORY)): digest(path.read_bytes())
                                for path in source_paths},
              "baseline_source": BASELINE_SOURCE, "local_backends": backends,
              "comparison": "original five-prediction flood vs four-prediction flood; same current compiler",
              "oracle": "Independent breadth-first search; all states, bounded distances, convergence, and route",
              "cases": []}
    programs = {}
    for name, maze, steps in cases:
        print("PREPARE %s: %dx%d, %d steps" % (name, maze.width, maze.height, steps), flush=True)
        expected_named, expected_solution = bfs_reference(maze, steps)
        case = {"name": name, "steps": steps, "width": maze.width, "height": maze.height,
                "input": {"maze_ascii": maze.ascii(), "cells": [list(row) for row in maze.cells],
                          "starts": [list(point) for point in maze.starts],
                          "goal": list(maze.goal) if maze.goal is not None else None},
                "solution": expected_solution, "expected_named_outputs": {}, "variants": {}}
        for label, board in expected_named.items():
            case["expected_named_outputs"][label] = save_bytes(
                output, "expected/%s/named/%s.plane" % (name, label), plane(board))
        observed_outputs = {}
        for variant, compiler in (("baseline", compile_baseline), ("optimized", compile_maze)):
            key = variant, maze.width, maze.height, steps
            if key not in programs:
                programs[key] = compiler(maze.width, maze.height, steps)
            program = programs[key]
            manifest = program.manifest()
            selection = sorted(set(program.output_frames.values()))
            frame_bytes = program.writer.width * program.writer.height * 3 // 2
            case.update(coded_width=program.writer.width, coded_height=program.writer.height,
                        guard=manifest["guard"], frame_bytes=frame_bytes)
            stream_relative = "streams/%s/%s.h264" % (name, variant)
            program.save(output / stream_relative, maze.initial_luma())
            stream = (output / stream_relative).read_bytes()
            manifest_relative = "streams/%s/%s.json" % (name, variant)
            raw = decode(output / stream_relative, "ffmpeg")
            if len(raw) != program.pictures * frame_bytes:
                raise AssertionError("Full reference picture count mismatch")
            actual_named = program.read(raw)
            if actual_named != expected_named:
                raise AssertionError("Named outputs differ from BFS: %s/%s" % (name, variant))
            solution = extract_solution(actual_named, maze, steps)
            verify(solution, maze)
            if {"distances": solution.distances, "path": [list(point) for point in solution.path],
                "converged_at": solution.converged_at} != expected_solution:
                raise AssertionError("Decoded solution differs from BFS: %s/%s" % (name, variant))
            observed_outputs[variant] = actual_named
            for backend, status in backends.items():
                if status["status"] == "unavailable":
                    continue
                other = raw if backend == "ffmpeg" else decode(output / stream_relative, backend)
                if other != raw:
                    raise AssertionError("Full I420 differs: %s/%s/%s" % (name, variant, backend))
                status["variants"] += 1
            expected_info = save_bytes(output, "expected/%s/%s.i420" % (name, variant), raw)
            selected_raw = b"".join(raw[index * frame_bytes:(index + 1) * frame_bytes]
                                    for index in selection)
            case["variants"][variant] = {
                "stream": stream_relative, "stream_sha256": digest(stream),
                "stream_bytes": len(stream), "manifest": manifest_relative,
                "manifest_sha256": digest((output / manifest_relative).read_bytes()),
                "expected": expected_info["path"], "expected_sha256": expected_info["sha256"],
                "expected_bytes": expected_info["bytes"], "pictures": program.pictures,
                "costs": program.costs, "selected_frames": selection,
                "selected_sha256": digest(selected_raw), "selected_bytes": len(selected_raw),
                "output_frames": program.output_frames,
                "named_outputs": {label: {"frame": frame, "selected_position": selection.index(frame),
                                           "sha256": digest(plane(actual_named[label]))}
                                  for label, frame in program.output_frames.items()}}
            print("PASS %s/%s: %d pictures, %d named pictures; full bytes + BFS" %
                  (name, variant, program.pictures, len(selection)), flush=True)
        if observed_outputs["baseline"] != observed_outputs["optimized"]:
            raise AssertionError("Paired named outputs differ: " + name)
        case["comparison"] = {"named_outputs_equal": True, "solutions_equal": True,
                              "all_states_match_bfs": True}
        report["cases"].append(case)
    for status in backends.values():
        if status["status"] == "pending":
            status["status"] = "pass"
    report["case_count"] = len(report["cases"])
    write_json(output / "bundle.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "out" / "bundle")
    parser.add_argument("--cases", nargs="+", help="Restrict fixture names; default includes all ten cases")
    parser.add_argument("--local-backends", nargs="+", choices=("ffmpeg", "vaapi", "openh264"),
                        default=["ffmpeg", "vaapi", "openh264"])
    args = parser.parse_args()
    try:
        selected_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    report = prepare(args.output, args.cases, args.local_backends)
    print(json.dumps({"case_count": report["case_count"], "local_backends": report["local_backends"]}))


if __name__ == "__main__":
    main()
