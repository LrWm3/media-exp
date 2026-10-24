#!/usr/bin/env python3
"""Compile symbolic flood, decode its saved stream, and extract actual arrivals."""

import argparse
import json
from pathlib import Path

from maze_kernel import EXPERIMENT, compile_maze, extract_solution
from capture import render
from decoder_core.decode import decode
from maze import demo_maze, parse_ascii
from solve import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--maze", type=Path)
    inputs.add_argument("--random-seed", type=int)
    parser.add_argument("--steps", type=int, help="Expansion budget; default open-cell count guarantees convergence")
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out/maze.h264")
    parser.add_argument("--verify", action="store_true", help="Compare every decoded state and first-arrival distance with BFS")
    parser.add_argument("--preview", action="store_true", help="Render decoded wavefront and route; implies verification")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()
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
    outputs = program.read(decode(args.output, args.decoder))
    solution = extract_solution(outputs, maze, steps)
    if args.verify or args.preview:
        verify(solution, maze)
        print(f"PASS ({args.decoder}): {steps + 1} decoded states and arrival distances match independent BFS")
    complete = solution.converged_at is not None
    reached = sum(distance is not None for row in solution.distances for distance in row)
    unvisited = sum(map(sum, maze.cells)) - reached
    goal_status = ("not_requested" if maze.goal is None else "reached" if solution.path
                   else "unreachable" if complete else "unresolved")
    result = {
        "decoder": args.decoder, "width": maze.width, "height": maze.height,
        "steps": steps, "output_frames": program.output_frames,
        "converged_at": solution.converged_at, "complete": complete,
        "reached_cells": reached, "unvisited_cells": unvisited,
        "starts": maze.starts, "goal": maze.goal, "goal_status": goal_status,
        "goal_distance": len(solution.path) - 1 if solution.path else None,
        "path": solution.path,
        "distance_legend": {"-1": "wall", "null": "unreachable if complete, otherwise unresolved",
                            "nonnegative": "shortest hop distance from nearest start"},
        "distances": [[solution.distances[y][x] if cell else -1 for x, cell in enumerate(row)]
                      for y, row in enumerate(maze.cells)],
    }
    args.output.with_suffix(".distances.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Reached {reached} cells; {'unreachable' if complete else 'unresolved'} {unvisited}; convergence step {solution.converged_at}")
    if solution.path:
        print(f"Goal reached in {len(solution.path) - 1} steps; route read from decoded distances")
    if args.preview:
        render(solution, maze, args.output.with_suffix(".gif"), args.decoder)
        print(f"Capture from decoded states: {args.output.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
