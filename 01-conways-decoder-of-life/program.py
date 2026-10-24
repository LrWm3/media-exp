"""Compile a seed-independent, finite Life program to H.264 instructions."""

from h264 import Operation, Source, Writer

FRAMES_PER_GENERATION = 14
NEIGHBORS = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)


def instructions(generations):
    if generations < 0:
        raise ValueError("Generations cannot be negative")
    program = []

    def emit(label, *sources, denominator=0):
        program.append(Operation(label, sources, denominator))
        return len(program)

    state = 0
    for generation in range(1, generations + 1):
        prefix = f"g{generation}"
        cell = emit(f"{prefix}: binary cells", Source(state, 1, -1), denominator=7)
        # Explicit B prediction divides the weighted sum by two, hence weights 2.
        dx, dy = NEIGHBORS[0]
        ex, ey = NEIGHBORS[1]
        count = emit(f"{prefix}: neighbors 1-2", Source(cell, 2, dx=dx, dy=dy), Source(cell, 2, dx=ex, dy=ey))
        for number, (dx, dy) in enumerate(NEIGHBORS[2:], 3):
            count = emit(f"{prefix}: neighbors 1-{number}", Source(count, 2), Source(cell, 2, dx=dx, dy=dy))
        score = emit(f"{prefix}: 2n+c", Source(count, 4), Source(cell, 2))
        lower = emit(f"{prefix}: clip(score-4)", Source(score, 1, -4))
        upper = emit(f"{prefix}: clip(score-7)", Source(score, 1, -7))
        window = emit(f"{prefix}: clip(lower-4*upper)", Source(lower, 2), Source(upper, -8))
        amplified = emit(f"{prefix}: amplify x127", Source(window, 127))
        state = emit(f"{prefix}: generation", Source(amplified, 3))
    return program


def compile_stream(board, generations, fps=30):
    writer = Writer(len(board[0]), len(board), fps)
    program = instructions(generations)
    stream = bytearray(writer.seed(board))
    for frame, operation in enumerate(program, 1):
        stream.extend(writer.prediction(frame, operation))
    return bytes(stream), program
