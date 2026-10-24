"""A maze-independent program: one breadth-first expansion per five pictures."""

from codec import Operation, Source, Writer

PICTURES_PER_STEP = 5


def instructions(steps):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    program = []

    def emit(label, *sources):
        program.append(Operation(label, sources))
        return len(program)

    state = 0
    for step in range(1, steps + 1):
        prefix = f"step {step}"
        reached = emit(f"{prefix}: reached cells", Source(state, 1, -128))
        north_south = emit(f"{prefix}: north OR south",
                           Source(reached, 5, dy=-1), Source(reached, 5, dy=1))
        east_west = emit(f"{prefix}: east OR west",
                         Source(reached, 5, dx=-1), Source(reached, 5, dx=1))
        neighbors = emit(f"{prefix}: any reached neighbor",
                         Source(north_south, 2), Source(east_west, 2))
        state = emit(f"{prefix}: block walls and preserve reached cells",
                     Source(neighbors, 1, -128), Source(state, 4, -128))
    return program


def compile_stream(maze, steps, fps=30):
    writer = Writer(maze.width, maze.height, fps)
    program = instructions(steps)
    stream = bytearray(writer.seed_luma(maze.initial_luma()))
    for frame, operation in enumerate(program, 1):
        stream.extend(writer.prediction(frame, operation))
    return bytes(stream), program
