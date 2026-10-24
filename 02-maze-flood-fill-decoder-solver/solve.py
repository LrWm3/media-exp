"""Read distances and a route from actual decoded flood states."""

from dataclasses import dataclass

from flood_program import PICTURES_PER_STEP
from maze import CARDINALS, bfs_distances


@dataclass
class Solution:
    states: list[bytes]
    distances: list[list[int | None]]
    path: list[tuple[int, int]]
    converged_at: int | None


def extract(raw, maze, steps):
    width, height = maze.width + 32, maze.height + 32
    luma_size = width * height
    frame_size = luma_size * 3 // 2
    count = 1 + steps * PICTURES_PER_STEP
    if len(raw) != frame_size * count:
        raise AssertionError(f"Expected {count} pictures / {frame_size * count} bytes, got {len(raw)} bytes")
    # Check every temporary too: no data may enter the guard or chroma planes.
    neutral = bytes([128]) * (luma_size // 2)
    for frame in range(count):
        offset = frame * frame_size
        if raw[offset + luma_size:offset + frame_size] != neutral:
            raise AssertionError(f"Chroma changed in picture {frame}")
        luma = raw[offset:offset + luma_size]
        if any(luma[:16 * width]) or any(luma[-16 * width:]):
            raise AssertionError(f"Guard changed in picture {frame}")
        for y in range(16, height - 16):
            if any(luma[y * width:y * width + 16]) or any(luma[(y + 1) * width - 16:(y + 1) * width]):
                raise AssertionError(f"Guard changed in picture {frame}")
    states, converged_at = [], None
    distances = [[None] * maze.width for _ in range(maze.height)]
    initial = bytes(value for row in maze.initial_luma() for value in row)
    for step in range(steps + 1):
        offset = step * PICTURES_PER_STEP * frame_size
        state = b"".join(raw[offset + (y + 16) * width + 16:offset + (y + 16) * width + 16 + maze.width]
                         for y in range(maze.height))
        if step == 0 and state != initial:
            raise AssertionError("Decoded initial state differs from maze")
        for index, value in enumerate(state):
            x, y = index % maze.width, index // maze.width
            if value not in (0, 128, 255) or (value == 0) != (not maze.cells[y][x]):
                raise AssertionError(f"Invalid cell or wall at step {step}: ({x}, {y}) = {value}")
            if states and states[-1][index] == 255 and value != 255:
                raise AssertionError(f"Reachability regressed at step {step}: ({x}, {y})")
            if value == 255 and distances[y][x] is None:
                distances[y][x] = step
        if states and state == states[-1] and converged_at is None:
            converged_at = step
        if converged_at is not None and state != states[-1]:
            raise AssertionError("Flood changed after convergence")
        states.append(state)
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
                raise AssertionError("Route did not end at a start")
            path.reverse()
    return Solution(states, distances, path, converged_at)


def verify(solution, maze):
    """Compare every cell at every completed step with independent CPU BFS."""
    expected = bfs_distances(maze)
    for step, state in enumerate(solution.states):
        correct = bytes(0 if not maze.cells[y][x] else 255 if expected[y][x] is not None
                        and expected[y][x] <= step else 128
                        for y in range(maze.height) for x in range(maze.width))
        if state != correct:
            index = next(i for i, (a, b) in enumerate(zip(state, correct)) if a != b)
            raise AssertionError(f"BFS mismatch at step {step}, cell ({index % maze.width}, {index // maze.width})")
    for y in range(maze.height):
        for x in range(maze.width):
            distance = expected[y][x]
            observed = distance if distance is not None and distance < len(solution.states) else None
            if solution.distances[y][x] != observed:
                raise AssertionError("First-arrival distances disagree with BFS")
