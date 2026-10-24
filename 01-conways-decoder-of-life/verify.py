"""Compare decoded pixels against an independent Life oracle."""

from h264 import available, decode
from life import step
from program import FRAMES_PER_GENERATION

def check(raw, board, generations):
    height, width = len(board), len(board[0])
    coded_width, coded_height = width + 32, height + 32
    luma_size = coded_width * coded_height
    frame_size = luma_size * 3 // 2
    count = 1 + generations * FRAMES_PER_GENERATION
    if len(raw) != frame_size * count:
        raise AssertionError(f"Expected {count} frames ({frame_size * count} bytes), got {len(raw)} bytes")
    # Guard and neutral chroma are invariants for every temporary picture too.
    guard_indices = [y * coded_width + x for y in range(coded_height) for x in range(coded_width)
                     if x < 16 or x >= width + 16 or y < 16 or y >= height + 16]
    neutral_chroma = bytes([128]) * (luma_size // 2)
    for frame in range(count):
        start = frame * frame_size
        if raw[start + luma_size:start + frame_size] != neutral_chroma:
            raise AssertionError(f"Non-neutral chroma in picture {frame}")
        if any(raw[start + i] for i in guard_indices):
            raise AssertionError(f"Nonzero guard in picture {frame}")
    expected = board
    generation_luma = []
    for generation in range(generations + 1):
        start = generation * FRAMES_PER_GENERATION * frame_size
        luma = bytes(raw[start + (y + 16) * coded_width + x + 16]
                     for y in range(height) for x in range(width))
        expected_luma = bytes(value * 255 for row in expected for value in row)
        if luma != expected_luma:
            wrong = next(i for i, (a, b) in enumerate(zip(luma, expected_luma)) if a != b)
            raise AssertionError(f"Generation {generation}, cell ({wrong % width}, {wrong // width}): "
                                 f"decoder={luma[wrong]}, expected={expected_luma[wrong]}")
        generation_luma.append(luma)
        if generation < generations:
            expected = step(expected)
    return generation_luma
