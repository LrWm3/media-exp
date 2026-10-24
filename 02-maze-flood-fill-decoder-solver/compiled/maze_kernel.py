"""Four-connected maze flood expressed using the shared symbolic compiler."""

from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "05-h264-compiler"))

import h264
from maze import CARDINALS
from solve import Solution


def compile_maze(width, height, steps, *, optimize=True):
    """Trace geometry and budget; maze contents remain literal runtime luma.

    With P = passages and S = starts, terrain = P + S has values 0/1/2.
    Thresholding terrain plus the binary neighbor union at 2 implements
    F(R) = S union (P intersection neighbors(R)). Starts therefore persist,
    including isolated starts. F is monotone and R0 = S is contained in F(S),
    so every subsequent reached set contains its predecessor. Inductively,
    Rt contains exactly cells at BFS distance <= t: each non-start needs a
    reached predecessor, while every start remains reached without one.

    Three binary ORs combine four shifted states; the existing exact search
    fuses the final add/threshold into one prediction picture.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    @h264.kernel
    def flood(seed):
        passages = h264.threshold(seed, 1)
        starts = h264.threshold(seed, 129)
        terrain = h264.add(passages, starts)
        reached = starts
        states = [reached]
        for _ in range(steps):
            neighbors = [h264.shift(reached, dx=dx, dy=dy) for dx, dy in CARDINALS]
            neighbor_union = neighbors[0]
            for neighbor in neighbors[1:]:
                neighbor_union = h264.bit_or(neighbor_union, neighbor)
            reached = h264.threshold(h264.add(neighbor_union, terrain), 2)
            states.append(reached)
        return {"output": reached,
                **{f"step_{i}": state for i, state in enumerate(states)},
                "passages": passages}

    return h264.compile(flood, width=width, height=height,
                        input_range=(0, 255), optimize=optimize)


def extract_solution(outputs, maze, steps):
    """Read first arrivals and a route from decoder outputs, without running BFS."""
    distances = [[None] * maze.width for _ in range(maze.height)]
    states = []
    converged_at = None
    passages = [[int(cell) for cell in row] for row in maze.cells]
    if outputs["passages"] != passages:
        raise AssertionError("Decoded passages differ from input maze")
    for step in range(steps + 1):
        grid = outputs[f"step_{step}"]
        if len(grid) != maze.height or any(len(row) != maze.width for row in grid):
            raise AssertionError(f"Wrong decoded grid dimensions at step {step}")
        state = bytearray()
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value not in (0, 1) or (value and not maze.cells[y][x]):
                    raise AssertionError(f"Invalid reachability at step {step}: ({x}, {y})")
                if value and distances[y][x] is None:
                    distances[y][x] = step
                if not value and distances[y][x] is not None:
                    raise AssertionError(f"Reachability regressed at step {step}: ({x}, {y})")
                state.append(255 if value else 128 if maze.cells[y][x] else 0)
        state = bytes(state)
        if step == 0 and state != bytes(value for row in maze.initial_luma() for value in row):
            raise AssertionError("Decoded starts differ from input maze")
        if states and state == states[-1] and converged_at is None:
            converged_at = step
        if converged_at is not None and state != states[-1]:
            raise AssertionError("Flood changed after convergence")
        states.append(state)
    if outputs["output"] != outputs[f"step_{steps}"]:
        raise AssertionError("Final output differs from final flood state")
    path = []
    if maze.goal is not None:
        x, y = maze.goal
        if distances[y][x] is not None:
            path.append((x, y))
            while distances[y][x] > 0:
                previous = [(x + dx, y + dy) for dx, dy in CARDINALS
                            if 0 <= x + dx < maze.width and 0 <= y + dy < maze.height
                            and distances[y + dy][x + dx] == distances[y][x] - 1]
                if not previous:
                    raise AssertionError("Decoded distances contain no predecessor")
                x, y = previous[0]
                path.append((x, y))
            if (x, y) not in maze.starts:
                raise AssertionError("Decoded route did not end at a start")
            path.reverse()
    return Solution(states, distances, path, converged_at)
