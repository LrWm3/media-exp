"""Independent pixel oracle: no graph, lowering, codec formulas, or decoder imports."""


def square_filter(board, erode):
    height, width = len(board), len(board[0])
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            neighbors = [board[sy][sx] if 0 <= sx < width and 0 <= sy < height else 0
                         for sy in range(y-1, y+2) for sx in range(x-1, x+2)]
            row.append(int(all(neighbors) if erode else any(neighbors)))
        result.append(row)
    return result


def motion(previous, current, cutoff=32):
    difference = [[abs(a-b) for a, b in zip(old, new)] for old, new in zip(previous, current)]
    raw = [[int(value >= cutoff) for value in row] for row in difference]
    eroded = square_filter(raw, True)
    opened = square_filter(eroded, False)
    cleaned = square_filter(square_filter(opened, False), True)
    return {"output": cleaned, "difference": difference, "raw_mask": raw,
            "eroded": eroded, "opened": opened}
