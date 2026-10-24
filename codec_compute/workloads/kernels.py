"""Useful image tasks, with explicit arithmetic and border semantics.

These are ordinary IR clients. They neither emit codec instructions nor choose
different mathematics for different targets. Unsupported combinations fail.
"""

from dataclasses import dataclass

import codec_compute as media


def positive_change(a, b):
    """Brightness increases in current frame a relative to reference b."""
    return media.sub_sat(a, b)


def absolute_change(a, b):
    magnitude = media.absdiff(a, b)
    return {"output": magnitude, "mask": media.threshold(magnitude, 24)}


def temporal_envelope(a, b):
    low, high = media.minimum(a, b), media.maximum(a, b)
    return {"low": low, "high": high, "output": media.sub_sat(high, low)}


def temporal_average(a, b):
    return media.average(a, b)


CROSS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def dilate(image, *, border):
    result = image
    for dx, dy in CROSS:
        result = media.bit_or(result, media.shift(image, dx, dy, border=border))
    return result


def smooth(image, *, border):
    # Each average rounds separately. This intentionally is not a single
    # rounding of a [1,2,1] x [1,2,1] convolution.
    horizontal = media.average(media.average(media.shift(image, -1, border=border),
                                             media.shift(image, 1, border=border)), image)
    return media.average(media.average(media.shift(horizontal, dy=-1, border=border),
                                       media.shift(horizontal, dy=1, border=border)), horizontal)


def local_range(image):
    neighbors = [image, *(media.shift(image, dx, dy, border="zero") for dx, dy in CROSS)]
    low, high = image, image
    for neighbor in neighbors[1:]:
        low = media.minimum(low, neighbor)
        high = media.maximum(high, neighbor)
    return {"low": low, "high": high, "output": media.sub_sat(high, low)}


@dataclass(frozen=True)
class Workload:
    name: str
    kernel: object
    inputs: tuple
    description: str
    border: str | None = None

    @property
    def input_ranges(self):
        return dict(self.inputs)


BYTE_PAIR = (("a", (0, 255)), ("b", (0, 255)))
BYTE_IMAGE = (("image", (0, 255)),)
BINARY_IMAGE = (("image", (0, 1)),)

WORKLOADS = (
    Workload("positive-change", positive_change, BYTE_PAIR,
             "One-sided brightness increase: max(current-reference,0)."),
    Workload("absolute-change", absolute_change, BYTE_PAIR,
             "Absolute frame difference and a >=24 change mask."),
    Workload("temporal-envelope", temporal_envelope, BYTE_PAIR,
             "Two-frame minimum, maximum, and observed range; no persistent host-side state."),
    Workload("temporal-average", temporal_average, BYTE_PAIR,
             "Two-frame blend with exact half-up rounding."),
    Workload("dilate-zero", lambda image: dilate(image, border="zero"), BINARY_IMAGE,
             "Expand a binary mask by one four-connected step, including its center.", "zero"),
    Workload("dilate-clamp", lambda image: dilate(image, border="clamp"), BINARY_IMAGE,
             "Expand a binary mask by one four-connected step, including its center.", "clamp"),
    Workload("smooth-zero", lambda image: smooth(image, border="zero"), BYTE_IMAGE,
             "Separable three-tap smoothing with nested rounded averages.", "zero"),
    Workload("smooth-clamp", lambda image: smooth(image, border="clamp"), BYTE_IMAGE,
             "Separable three-tap smoothing with nested rounded averages.", "clamp"),
    Workload("local-range", local_range, BYTE_IMAGE,
             "Five-pixel cross erosion, dilation, and local max-minus-min contrast.", "zero"),
)


def fixtures(workload, variant=0, size=32):
    """Deterministic scene/change and boundary-rich fixtures; never results."""
    if workload.inputs == BYTE_PAIR:
        background = [[32+(x*3+y*5) % 64 for x in range(size)] for y in range(size)]
        previous = [[210 if 5 <= x < 16 and 9 <= y < 24 else background[y][x]
                     for x in range(size)] for y in range(size)]
        current = [[210 if 12 <= x < 23 and 6 <= y < 21 else background[y][x]
                    for x in range(size)] for y in range(size)]
        if variant:
            previous = [[(x*17+y*31) % 256 for x in range(size)] for y in range(size)]
            current = [[255-value for value in row] for row in previous]
        return {"a": current, "b": previous}
    if workload.inputs == BINARY_IMAGE:
        image = [[int((x, y) in ((0, 0), (size-1, size-1), (1, size//2)) or
                      (10 <= x < 14 and 9 <= y < 23) or (9 <= x < 23 and 17 <= y < 21))
                  for x in range(size)] for y in range(size)]
        if variant:
            image = [[int((x*7+y*11) % 13 == 0) for x in range(size)] for y in range(size)]
    else:
        image = [[max(0, min(255, (180 if 8 <= x < 24 and 8 <= y < 24 else 48)
                                      + ((x*29+y*17) % 67)-33))
                  for x in range(size)] for y in range(size)]
        if variant:
            image = [[(x*37+y*19) % 256 for x in range(size)] for y in range(size)]
    return {"image": image}


def reference(workload, inputs):
    """Task formulas independent of the IR interpreter and target lowering."""
    name = workload.name
    if len(inputs) == 2:
        a, b = inputs["a"], inputs["b"]
        def pointwise(fn):
            return [[fn(x, y) for x, y in zip(first, second)] for first, second in zip(a, b)]
        if name == "positive-change":
            return {"output": pointwise(lambda x, y: max(x-y, 0))}
        if name == "absolute-change":
            return {"output": pointwise(lambda x, y: abs(x-y)),
                    "mask": pointwise(lambda x, y: int(abs(x-y) >= 24))}
        if name == "temporal-envelope":
            return {"low": pointwise(min), "high": pointwise(max),
                    "output": pointwise(lambda x, y: abs(x-y))}
        if name == "temporal-average":
            return {"output": pointwise(lambda x, y: (x+y+1)//2)}
        raise ValueError("Unknown two-input workload")
    board = inputs["image"]
    height, width = len(board), len(board[0])
    def sample(source, x, y):
        if workload.border == "clamp":
            x, y = min(width-1, max(0, x)), min(height-1, max(0, y))
        return source[y][x] if 0 <= x < width and 0 <= y < height else 0
    if name.startswith("smooth-"):
        horizontal = [[(((sample(board, x-1, y)+sample(board, x+1, y)+1)//2)+board[y][x]+1)//2
                       for x in range(width)] for y in range(height)]
        return {"output": [[(((sample(horizontal, x, y-1)+sample(horizontal, x, y+1)+1)//2)
                             +horizontal[y][x]+1)//2 for x in range(width)] for y in range(height)]}
    extrema = {key: [[fn(sample(board, x+dx, y+dy) for dx, dy in ((0, 0), *CROSS))
                       for x in range(width)] for y in range(height)]
               for key, fn in (("low", min), ("high", max))}
    if name.startswith("dilate-"):
        return {"output": extrema["high"]}
    if name == "local-range":
        return {**extrema, "output": [[hi-lo for lo, hi in zip(low, high)]
                                      for low, high in zip(extrema["low"], extrema["high"])]}
    raise ValueError("Unknown spatial workload")
