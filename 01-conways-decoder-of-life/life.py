"""Seed fixtures and an independent oracle, used only for verification."""

import random


def seed_board(name, width, height, random_seed=1):
    board = [[0] * width for _ in range(height)]
    if name == "random":
        rng = random.Random(random_seed)
        return [[int(rng.random() < 0.3) for _ in range(width)] for _ in range(height)]
    patterns = {
        "glider": [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)],
        "blinker": [(0, 0), (1, 0), (2, 0)],
        "block": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "r-pentomino": [(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
        "empty": [],
    }
    if name not in patterns:
        raise ValueError(f"Unknown pattern: {name}")
    for x, y in patterns[name]:
        board[height // 2 - 1 + y][width // 2 - 1 + x] = 1
    return board


def step(board):
    height, width = len(board), len(board[0])
    result = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            count = sum(board[ny][nx]
                        for ny in range(max(0, y - 1), min(height, y + 2))
                        for nx in range(max(0, x - 1), min(width, x + 2))
                        if nx != x or ny != y)
            result[y][x] = int(count == 3 or (board[y][x] and count == 2))
    return result
