"""Conway's rule expressed using the shared symbolic H.264 compiler."""

from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(ROOT))
# The original experiment also has h264.py. Prefer the compiler package.
sys.path.insert(0, str(ROOT / "05-h264-compiler"))

import h264

NEIGHBORS = tuple((dx, dy) for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1) if dx or dy)


def compile_life(width, height, generations, *, optimize=True):
    """Build a seed-independent graph, retaining every generation as an output."""
    if not isinstance(generations, int) or generations < 0:
        raise ValueError("generations must be a nonnegative integer")

    @h264.kernel
    def life(seed):
        states = [seed]
        current = seed
        for _ in range(generations):
            neighbors = [h264.shift(current, dx=dx, dy=dy)
                         for dx, dy in NEIGHBORS]
            count = neighbors[0]
            for neighbor in neighbors[1:]:
                count = h264.add(count, neighbor)
            born = h264.equal(count, 3)
            survives = h264.bit_and(current, h264.equal(count, 2))
            current = h264.bit_or(born, survives)
            states.append(current)
        # Compile the final dependency chain before exporting intermediate names.
        return {"output": current,
                **{f"generation_{i}": state for i, state in enumerate(states)}}

    return h264.compile(life, width=width, height=height,
                        input_range=(0, 1), optimize=optimize)


def generation_frames(outputs, generations):
    """Convert decoded logical cells to display bytes; no Life computation here."""
    return [bytes(value * 255 for row in outputs[f"generation_{i}"] for value in row)
            for i in range(generations + 1)]


def verify_generations(outputs, board, generations):
    """Check actual decoded outputs against the original independent oracle."""
    from life import step

    expected = board
    for generation in range(generations + 1):
        if outputs[f"generation_{generation}"] != expected:
            raise AssertionError(f"Life oracle mismatch at generation {generation}")
        if generation < generations:
            expected = step(expected)
    if outputs["output"] != expected:
        raise AssertionError("Final output differs from the final generation")
