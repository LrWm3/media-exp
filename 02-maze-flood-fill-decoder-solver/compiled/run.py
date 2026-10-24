#!/usr/bin/env python3
"""Compile symbolic flood, decode its saved stream, and extract actual arrivals."""

import argparse
import json
import os
from pathlib import Path

from maze_kernel import EXPERIMENT, compile_maze, extract_solution
from capture import render
from decoder_core.decode import decode_batch
from maze import demo_maze, parse_ascii
from solve import verify


def decode_saved(program, path, backend, readback="outputs"):
    """Decode the saved stream, returning named planes and actual copy counts."""
    if readback not in ("all", "outputs"):
        raise ValueError("readback must be 'all' or 'outputs'")
    result = decode_batch([path], backend, width=program.writer.width,
                          height=program.writer.height, pictures_per_job=program.pictures,
                          selected_frames=program.output_picture_indices if readback == "outputs" else None)
    outputs = program.read(b"".join(result.frames[0]), selected_frames=result.selected_frames)
    return outputs, {"mode": readback, "decoded_pictures": result.decoded_frames,
                     "copied_pictures": result.copied_frames,
                     "returned_pictures": len(result.frames[0]),
                     "copied_bytes": result.copied_frames * result.frame_bytes,
                     "returned_bytes": sum(map(len, result.frames[0])),
                     "selected_frames": list(result.selected_frames)}


def solution_document(program, maze, solution, steps, backend, readback):
    complete = solution.converged_at is not None
    reached = sum(distance is not None for row in solution.distances for distance in row)
    goal_status = ("not_requested" if maze.goal is None else "reached" if solution.path
                   else "unreachable" if complete else "unresolved")
    return {
        "decoder": backend, "width": maze.width, "height": maze.height,
        "steps": steps, "pictures": program.pictures, "output_frames": program.output_frames,
        "readback": readback,
        "converged_at": solution.converged_at, "complete": complete,
        "reached_cells": reached, "unvisited_cells": sum(map(sum, maze.cells)) - reached,
        "starts": maze.starts, "goal": maze.goal, "goal_status": goal_status,
        "goal_distance": len(solution.path) - 1 if solution.path else None,
        "path": solution.path,
        "distance_legend": {"-1": "wall", "null": "unreachable if complete, otherwise unresolved",
                            "nonnegative": "shortest hop distance from nearest start"},
        "distances": [[solution.distances[y][x] if cell else -1 for x, cell in enumerate(row)]
                      for y, row in enumerate(maze.cells)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--maze", type=Path)
    inputs.add_argument("--random-seed", type=int)
    parser.add_argument("--steps", type=int, help="Expansion budget; default open-cell count guarantees convergence")
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi", "nvdec"), default="ffmpeg")
    parser.add_argument("--helper", type=Path, help="Optional path to the NVDEC decoder helper")
    parser.add_argument("--readback", choices=("outputs", "all"), default="outputs",
                        help="Return completed flood states and passages, or every intermediate picture")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out/maze.h264")
    parser.add_argument("--verify", action="store_true", help="Compare every decoded state and first-arrival distance with BFS")
    parser.add_argument("--preview", action="store_true", help="Render decoded wavefront and route; implies verification")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()
    if args.helper:
        os.environ["H264_NVDEC_DECODER"] = str(args.helper.resolve())
    try:
        maze = (demo_maze(args.random_seed) if args.random_seed is not None
                else parse_ascii((args.maze or EXPERIMENT / "mazes/demo.txt").read_text()))
        steps = args.steps if args.steps is not None else sum(map(sum, maze.cells))
        program = compile_maze(maze.width, maze.height, steps,
                               optimize=not args.no_optimize)
        program.save(args.output, maze.initial_luma())
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}: {args.output.stat().st_size:,} bytes; {program.pictures} decoded pictures; {steps} steps", flush=True)
    outputs, readback = decode_saved(program, args.output, args.decoder, args.readback)
    print(f"Readback ({args.readback}): {readback['returned_pictures']} returned pictures; "
          f"{readback['copied_pictures']} copied by {args.decoder}", flush=True)
    solution = extract_solution(outputs, maze, steps)
    if args.verify or args.preview:
        verify(solution, maze)
        print(f"PASS ({args.decoder}): {steps + 1} decoded states and arrival distances match independent BFS")
    result = solution_document(program, maze, solution, steps, args.decoder, readback)
    args.output.with_suffix(".distances.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Reached {result['reached_cells']} cells; "
          f"{'unreachable' if result['complete'] else 'unresolved'} {result['unvisited_cells']}; "
          f"convergence step {solution.converged_at}")
    if solution.path:
        print(f"Goal reached in {len(solution.path) - 1} steps; route read from decoded distances")
    if args.preview:
        render(solution, maze, args.output.with_suffix(".gif"), args.decoder)
        print(f"Capture from decoded states: {args.output.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
