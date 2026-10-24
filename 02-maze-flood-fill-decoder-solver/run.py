#!/usr/bin/env python3
"""Compile a maze into H.264, decode it, and read reachability and hop distances."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from capture import render
from codec import decode
from flood_program import compile_stream, PICTURES_PER_STEP
from maze import demo_maze, parse_ascii
from solve import extract, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--maze", type=Path, help="ASCII maze: # wall, . passage, S start, G goal")
    inputs.add_argument("--random-seed", type=int, help="Generate another 32x32 example")
    parser.add_argument("--steps", type=int, help="Expansion budget; default = open-cell count (guarantees convergence)")
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("out/maze.h264"))
    parser.add_argument("--verify", action="store_true", help="Compare every decoded flood state with independent BFS")
    parser.add_argument("--preview", action="store_true", help="Capture decoded wavefront and reconstructed shortest path as GIF")
    args = parser.parse_args()
    try:
        if args.random_seed is not None:
            maze = demo_maze(args.random_seed)
        else:
            path = args.maze or Path(__file__).parent / "mazes/demo.txt"
            maze = parse_ascii(path.read_text())
        steps = args.steps if args.steps is not None else sum(map(sum, maze.cells))
        stream, program = compile_stream(maze, steps)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stream)
    args.output.with_suffix(".program.json").write_text(json.dumps({
        "width": maze.width, "height": maze.height, "steps": steps,
        "pictures_per_step": PICTURES_PER_STEP, "maze": maze.ascii(),
        "pictures": [{"frame": 0, "label": "I_PCM maze and starts"}] +
                    [{"frame": frame, **asdict(op)} for frame, op in enumerate(program, 1)],
    }, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(stream):,} bytes; {len(program) + 1} pictures; {steps} steps", flush=True)
    solution = extract(decode(args.output, args.decoder), maze, steps)
    if args.verify:
        verify(solution, maze)
        print(f"PASS ({args.decoder}): all {steps + 1} states and first-arrival distances match BFS")
    complete = solution.converged_at is not None
    reached = sum(d is not None for row in solution.distances for d in row)
    unvisited = sum(map(sum, maze.cells)) - reached
    print(f"Reached {reached} cells; {'unreachable' if complete else 'unresolved'} {unvisited}; convergence step {solution.converged_at}")
    if maze.goal is None:
        goal_status = "not_requested"
    elif solution.path:
        goal_status = "reached"
        print(f"Goal reached in {len(solution.path) - 1} steps; shortest route extracted from decoded distances")
    else:
        goal_status = "unreachable" if complete else "unresolved"
        print(f"Goal {goal_status}")
    result = {
        "decoder": args.decoder, "width": maze.width, "height": maze.height, "steps": steps,
        "converged_at": solution.converged_at, "complete": complete,
        "reached_cells": reached, "unvisited_cells": unvisited,
        "starts": maze.starts, "goal": maze.goal, "goal_status": goal_status,
        "goal_distance": len(solution.path) - 1 if solution.path else None,
        "path": solution.path,
        "distance_legend": {"-1": "wall", "null": "unreachable if complete, otherwise unresolved", "nonnegative": "shortest hop distance from nearest start"},
        "distances": [[solution.distances[y][x] if cell else -1 for x, cell in enumerate(row)]
                      for y, row in enumerate(maze.cells)],
    }
    args.output.with_suffix(".distances.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.preview:
        render(solution, maze, args.output.with_suffix(".gif"), args.decoder)
        print(f"Capture from decoded states: {args.output.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
