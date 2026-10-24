"""ASCII inputs, deterministic fixtures, and an independent BFS oracle."""

from collections import deque
from dataclasses import dataclass
import random

CARDINALS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Maze:
    cells: tuple[tuple[bool, ...], ...]
    starts: tuple[tuple[int, int], ...]
    goal: tuple[int, int] | None = None

    def __post_init__(self):
        if not self.cells or not self.cells[0] or any(len(row) != len(self.cells[0]) for row in self.cells):
            raise ValueError("Maze must be a nonempty rectangle")
        if not self.starts:
            raise ValueError("Maze needs at least one start")
        for x, y in self.starts + ((self.goal,) if self.goal is not None else ()):
            if not (0 <= x < self.width and 0 <= y < self.height and self.cells[y][x]):
                raise ValueError("Starts and goal must be inside open cells")
        if len(set(self.starts)) != len(self.starts):
            raise ValueError("Duplicate start")

    @property
    def width(self):
        return len(self.cells[0])

    @property
    def height(self):
        return len(self.cells)

    def initial_luma(self):
        starts = set(self.starts)
        return [[255 if (x, y) in starts else 128 if cell else 0
                 for x, cell in enumerate(row)] for y, row in enumerate(self.cells)]

    def ascii(self):
        starts = set(self.starts)
        return "\n".join("".join("S" if (x, y) in starts else "G" if (x, y) == self.goal
                                else "." if cell else "#" for x, cell in enumerate(row))
                         for y, row in enumerate(self.cells)) + "\n"


def parse_ascii(text):
    rows = text.strip("\n\r").splitlines()
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ASCII maze must be rectangular")
    if any(char not in "#. SG" for row in rows for char in row):
        raise ValueError("Use # walls, . or space passages, S starts, and optional G goal")
    starts = tuple((x, y) for y, row in enumerate(rows) for x, char in enumerate(row) if char == "S")
    goals = [(x, y) for y, row in enumerate(rows) for x, char in enumerate(row) if char == "G"]
    if len(goals) > 1:
        raise ValueError("At most one goal is supported")
    width, height = (len(rows[0]) + 15) // 16 * 16, (len(rows) + 15) // 16 * 16
    cells = tuple(tuple(y < len(rows) and x < len(rows[0]) and rows[y][x] != "#"
                        for x in range(width)) for y in range(height))
    return Maze(cells, starts, goals[0] if goals else None)


def demo_maze(seed=7):
    """Randomized Prim maze plus a deliberately sealed 3x3 room."""
    rng = random.Random(seed)
    cells = [[False] * 32 for _ in range(32)]
    cells[1][1] = True
    frontier = []

    def extend(x, y):
        for dx, dy in CARDINALS:
            nx, ny = x + dx * 2, y + dy * 2
            if 1 <= nx <= 23 and 1 <= ny <= 29 and not cells[ny][nx]:
                frontier.append((x, y, nx, ny))

    extend(1, 1)
    while frontier:
        x, y, nx, ny = frontier.pop(rng.randrange(len(frontier)))
        if cells[ny][nx]:
            continue
        cells[ny][nx] = cells[(y + ny) // 2][(x + nx) // 2] = True
        extend(nx, ny)
    for y in range(3, 6):
        for x in range(27, 30):
            cells[y][x] = True
    return Maze(tuple(tuple(row) for row in cells), ((1, 1),), (23, 29))


def bfs_distances(maze):
    """Verification only; the compiler and decoded-distance extraction never call it."""
    distance = [[None] * maze.width for _ in range(maze.height)]
    queue = deque(maze.starts)
    for x, y in maze.starts:
        distance[y][x] = 0
    while queue:
        x, y = queue.popleft()
        for dx, dy in CARDINALS:
            nx, ny = x + dx, y + dy
            if (0 <= nx < maze.width and 0 <= ny < maze.height and maze.cells[ny][nx]
                    and distance[ny][nx] is None):
                distance[ny][nx] = distance[y][x] + 1
                queue.append((nx, ny))
    return distance
