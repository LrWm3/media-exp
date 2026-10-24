"""Deterministic synthetic video: moving objects, mild noise, isolated impulses."""

import random


def frame(index, width=96, height=64, noisy=True):
    # Bounce a bright rectangle horizontally and a dark disk vertically.
    travel = width-30
    phase = (index*4) % (2*travel)
    rectangle_x = 5 + min(phase, 2*travel-phase)
    disk_y = 20 + round(10*((index % 12)-6)/6)
    rng = random.Random(7319+index)
    board = []
    for y in range(height):
        row = []
        for x in range(width):
            value = 70 + x*60//width + y*30//height
            if width//2-2 <= x <= width//2+2:
                value = 145  # Static scene structure must not trigger motion.
            if rectangle_x <= x < rectangle_x+18 and height-26 <= y < height-10:
                value = 230
            if (x-(width-22))**2+(y-disk_y)**2 <= 8**2:
                value = 15
            if noisy:
                value += rng.randrange(-3, 4)
            row.append(max(0, min(255, value)))
        board.append(row)
    if noisy:
        # Deliberately isolated high-amplitude changes, unlike broad object edges.
        for _ in range(24):
            x, y = rng.randrange(width), rng.randrange(height)
            board[y][x] = 255-board[y][x]
    return board


def sequence(jobs=24, width=96, height=64):
    return [frame(index, width, height) for index in range(jobs+1)]
