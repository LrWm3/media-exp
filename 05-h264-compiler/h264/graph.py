"""Symbolic u8 image operations. Python runs once to build a finite graph."""

from dataclasses import dataclass
from contextvars import ContextVar


class CompileError(ValueError):
    pass


_created = ContextVar("h264_created", default=None)


@dataclass(frozen=True, eq=False)
class Image:
    op: str
    args: tuple
    params: tuple
    lo: int
    hi: int

    def __bool__(self):
        raise TypeError("Symbolic image cannot control Python flow; use image operations and static repeat")

    def __int__(self):
        raise TypeError("Symbolic image has no host pixel value")

    def __eq__(self, other):
        raise TypeError("Use h264.equal(image, integer), not Python comparisons")

    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__
    __hash__ = object.__hash__

    def __add__(self, other):
        return add(self, other)


def integer(value, name):
    if type(value) is not int:
        raise CompileError(f"{name} must be a compile-time integer")
    return value


def node(op, args=(), params=(), lo=0, hi=255):
    if any(not isinstance(arg, Image) for arg in args):
        raise CompileError("Image operands must be symbolic images")
    result = Image(op, tuple(args), tuple(params), lo, hi)
    if _created.get() is not None:
        _created.get().append(result)
    return result


def Input(name="image", value_range=(0, 1)):
    lo, hi = value_range
    integer(lo, "input minimum")
    integer(hi, "input maximum")
    if not 0 <= lo <= hi <= 255:
        raise CompileError("Input range must lie in [0,255]")
    return node("input", params=(str(name),), lo=lo, hi=hi)


def kernel(function):
    """Mark a Python graph-building function; arbitrary Python is not translated."""
    return function


def shift(x, dx=0, dy=0):
    integer(dx, "dx")
    integer(dy, "dy")
    if max(abs(dx), abs(dy)) > 127:
        raise CompileError("This backend supports integer motion offsets within +/-127 pixels")
    return node("shift", (x,), (dx, dy), min(0, x.lo), x.hi)


def bounds(x, weight):
    return min(x.lo*weight, x.hi*weight), max(x.lo*weight, x.hi*weight)


def affine(args, weights, bias, op):
    integer(bias, "bias")
    intervals = [bounds(x, integer(w, "weight")) for x, w in zip(args, weights)]
    lo = sum(pair[0] for pair in intervals) + bias
    hi = sum(pair[1] for pair in intervals) + bias
    if not 0 <= lo <= hi <= 255:
        raise CompileError(f"{op} range [{lo},{hi}] exceeds u8; use a biased sum with a proven range")
    return node(op, args, (*weights, bias), lo, hi)


def weighted(x, w=1, bias=0):
    """Exact w*x+bias; reject possible overflow instead of silently clipping."""
    return affine((x,), (w,), bias, "weighted")


def combine(a, b, wa=1, wb=1, bias=0):
    """Exact wa*a+wb*b+bias, without implicit averaging or saturation."""
    return affine((a, b), (wa, wb), bias, "combine")


def add(a, b):
    return combine(a, b)


def threshold(x, t):
    integer(t, "threshold")
    return node("threshold", (x,), (t,), int(t <= x.lo), int(t <= x.hi))


def binary(x):
    if not isinstance(x, Image) or not 0 <= x.lo <= x.hi <= 1:
        raise CompileError("Boolean operation needs a proven binary image")
    return x


def bit_not(x):
    return weighted(binary(x), -1, 1)


def bit_or(a, b):
    return threshold(add(binary(a), binary(b)), 1)


def bit_and(a, b):
    return threshold(add(binary(a), binary(b)), 2)


def equal(x, value):
    integer(value, "comparison")
    return bit_and(threshold(x, value), bit_not(threshold(x, value+1)))


def repeat(count, function, value):
    integer(count, "repeat count")
    if count < 0:
        raise CompileError("repeat count cannot be negative")
    for _ in range(count):
        value = function(value)
    return value


def binary_dot(terms, weights, threshold):
    terms, weights = tuple(terms), tuple(weights)
    if not 1 <= len(terms) == len(weights) <= 255:
        raise CompileError("Binary dot requires 1..255 matching terms and weights")
    if any(type(w) is not int or w not in (-1, 1) for w in weights):
        raise CompileError("Binary weights must be -1 or +1")
    for term in terms:
        binary(term)
    integer(threshold, "dot threshold")
    bias = weights.count(-1)
    total = weighted(terms[0], weights[0], bias)
    for term, weight in zip(terms[1:], weights[1:]):
        total = combine(total, term, wb=weight)
    # This helper builds ordinary graph operations; the backend knows no layers.
    return globals()["threshold"](total, bias+threshold)


def conv2d_binary(x, weights, threshold):
    rows = tuple(tuple(row) for row in weights)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise CompileError("conv2d_binary currently supports a 3x3 kernel")
    return binary_dot([shift(x, dx-1, dy-1) for dy in range(3) for dx in range(3)],
                      [w for row in rows for w in row], threshold)


def postorder(roots):
    """Iterative traversal also handles thousands of unrolled iterations."""
    ordered, done = [], set()
    stack = [(root, False) for root in reversed(list(roots))]
    while stack:
        current, expanded = stack.pop()
        if current in done:
            continue
        if expanded:
            done.add(current)
            ordered.append(current)
        else:
            stack.append((current, True))
            stack.extend((arg, False) for arg in reversed(current.args))
    return ordered


def trace(function, input_range):
    created = []
    token = _created.set(created)
    try:
        source = Input(value_range=input_range)
        result = function(source)
    finally:
        _created.reset(token)
    outputs = {"output": result} if isinstance(result, Image) else result
    if not isinstance(outputs, dict) or not outputs or any(not isinstance(k, str) or not isinstance(v, Image) for k, v in outputs.items()):
        raise CompileError("Kernel must return an image or a nonempty mapping of names to images")
    if any(n is not source for n in postorder(outputs.values()) if n.op == "input"):
        raise CompileError("Kernel must use its supplied single input image")
    return source, outputs, len(created)
