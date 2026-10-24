"""Frame differences and binary morphology expressed only as compiler operations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05-h264-compiler"))
import h264


def morphology(image, erode):
    """Separable 3x3 square, with zero padding at each intermediate boundary."""
    cutoff = 3 if erode else 1
    horizontal = h264.threshold(h264.add(h264.add(h264.shift(image, dx=-1), image),
                                        h264.shift(image, dx=1)), cutoff)
    return h264.threshold(h264.add(h264.add(h264.shift(horizontal, dy=-1), horizontal),
                                  h264.shift(horizontal, dy=1)), cutoff)


def motion_kernel(previous, current, cutoff=32):
    positive = h264.saturating_combine(current, previous, wa=1, wb=-1)
    negative = h264.saturating_combine(previous, current, wa=1, wb=-1)
    # At each pixel only one difference can be positive, so their sum fits u8.
    difference = h264.saturating_add(positive, negative)
    raw_mask = h264.threshold(difference, cutoff)
    eroded = morphology(raw_mask, erode=True)
    opened = morphology(eroded, erode=False)
    # Opening removes isolated changes. Closing fills small holes/gaps.
    expanded = morphology(opened, erode=False)
    cleaned = morphology(expanded, erode=True)
    return {"output": cleaned, "difference": difference, "raw_mask": raw_mask,
            "eroded": eroded, "opened": opened}


def compile_motion(width=96, height=64, cutoff=32):
    if type(cutoff) is not int or not 1 <= cutoff <= 255:
        raise ValueError("cutoff must be an integer in 1..255")
    return h264.compile(lambda previous, current: motion_kernel(previous, current, cutoff),
                        width=width, height=height,
                        input_ranges={"previous": (0, 255), "current": (0, 255)})
