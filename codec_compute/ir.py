"""Exact image computations, independent of codec instructions and resources."""

from collections.abc import Mapping
from dataclasses import dataclass


class CompileError(ValueError):
    """A malformed computation or an unavailable exact target lowering."""


class UnsupportedOperation(CompileError):
    def __init__(self, target, operation, detail):
        self.target, self.operation, self.detail = target, operation, detail
        super().__init__(f"{target}: cannot lower {operation}: {detail}")


def integer(value, name):
    if type(value) is not int:
        raise CompileError(f"{name} must be an integer")
    return value


def clip8(value):
    return max(0, min(255, value))


@dataclass(frozen=True, eq=False)
class Image:
    op: str
    args: tuple
    params: tuple
    lo: int
    hi: int

    def __bool__(self):
        raise TypeError("An image cannot control Python flow; use static loops")

    def __eq__(self, other):
        raise TypeError("Image comparisons must use explicit image operations")

    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__
    __hash__ = object.__hash__

    def __add__(self, other):
        return add(self, other)


def _node(op, args, params, lo, hi):
    if any(not isinstance(arg, Image) for arg in args):
        raise CompileError("Operands must be symbolic images")
    return Image(op, tuple(args), tuple(params), lo, hi)


def copy(image):
    """Return identical pixels; no requirement to allocate another picture."""
    return _node("copy", (image,), (), image.lo, image.hi)


def affine(*images, weights=None, bias=0, saturating=False):
    """Integer sum(weights[i]*images[i])+bias, optionally clipped once to u8.

    Coefficient limits and source counts belong to target lowering. Strict
    arithmetic requires an interval proof that the result remains in u8.
    """
    if not images or any(not isinstance(image, Image) for image in images):
        raise CompileError("affine requires symbolic images")
    weights = (1,) * len(images) if weights is None else tuple(weights)
    if len(weights) != len(images):
        raise CompileError("affine needs one weight per operand")
    integer(bias, "bias")
    if type(saturating) is not bool:
        raise CompileError("saturating must be Boolean")
    intervals = []
    for image, weight in zip(images, weights):
        integer(weight, "weight")
        intervals.append(sorted((image.lo * weight, image.hi * weight)))
    lo = sum(pair[0] for pair in intervals) + bias
    hi = sum(pair[1] for pair in intervals) + bias
    if saturating:
        lo, hi = clip8(lo), clip8(hi)
    elif not 0 <= lo <= hi <= 255:
        raise CompileError(f"Exact affine range [{lo},{hi}] exceeds u8; request saturation explicitly")
    return _node("affine_sat" if saturating else "affine", images, (*weights, bias), lo, hi)


def add(a, b):
    return affine(a, b)


def add_sat(a, b):
    return affine(a, b, saturating=True)


def sub_sat(a, b):
    return affine(a, b, weights=(1, -1), saturating=True)


def shift(image, dx=0, dy=0, *, border="zero"):
    """Read image[y+dy,x+dx]; out-of-image positions use explicit border rules."""
    integer(dx, "dx")
    integer(dy, "dy")
    if border not in ("zero", "clamp"):
        raise CompileError("border must be 'zero' or 'clamp'")
    return _node("shift", (image,), (dx, dy, border),
                 min(0, image.lo) if border == "zero" else image.lo, image.hi)


def threshold(image, cutoff):
    integer(cutoff, "cutoff")
    return _node("threshold", (image,), (cutoff,), int(image.lo >= cutoff), int(image.hi >= cutoff))


def _binary(image):
    if not isinstance(image, Image) or not 0 <= image.lo <= image.hi <= 1:
        raise CompileError("Boolean operations require a proven 0/1 range")
    return image


def bit_not(image):
    return affine(_binary(image), weights=(-1,), bias=1)


def bit_and(a, b):
    return _node("and", (_binary(a), _binary(b)), (), a.lo & b.lo, a.hi & b.hi)


def bit_or(a, b):
    return _node("or", (_binary(a), _binary(b)), (), a.lo | b.lo, a.hi | b.hi)


def postorder(roots):
    visited, result = set(), []
    stack = [(root, False) for root in reversed(tuple(roots))]
    while stack:
        item, expanded = stack.pop()
        if item in visited:
            continue
        if expanded:
            visited.add(item)
            result.append(item)
        else:
            stack.append((item, True))
            stack.extend((arg, False) for arg in reversed(item.args))
    return tuple(result)


@dataclass(frozen=True)
class Graph:
    width: int
    height: int
    inputs: tuple
    outputs: tuple
    nodes: tuple

    def validate(self, images):
        names = {node.params[0] for node in self.inputs}
        if not isinstance(images, Mapping) or set(images) != names:
            raise ValueError(f"Supply exactly these named images: {sorted(names)}")
        for node in self.inputs:
            name = node.params[0]
            board = images[name]
            if len(board) != self.height or any(len(row) != self.width for row in board):
                raise ValueError(f"Input {name} must have shape {self.height}x{self.width}")
            if any(type(value) is not int or not node.lo <= value <= node.hi
                   for row in board for value in row):
                raise ValueError(f"Input {name} must contain integers in [{node.lo},{node.hi}]")

    def manifest(self):
        ids = {node: index for index, node in enumerate(self.nodes)}
        return {"format": "codec-compute-ir-1", "dtype": "u8", "shape": [self.height, self.width],
                "inputs": [{"name": node.params[0], "id": ids[node], "range": [node.lo, node.hi]}
                           for node in self.inputs],
                "outputs": {name: ids[node] for name, node in self.outputs},
                "nodes": [{"id": ids[node], "op": node.op, "args": [ids[arg] for arg in node.args],
                           "params": list(node.params), "range": [node.lo, node.hi]}
                          for node in self.nodes]}


def trace(function, *, width, height, input_ranges=None):
    if integer(width, "width") <= 0 or integer(height, "height") <= 0:
        raise CompileError("Image dimensions must be positive")
    ranges = {"image": (0, 255)} if input_ranges is None else input_ranges
    if not isinstance(ranges, Mapping) or not ranges:
        raise CompileError("input_ranges must be a nonempty mapping")
    inputs = []
    for name, bounds in ranges.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise CompileError("Input names must be Python identifiers")
        try:
            lo, hi = bounds
        except (TypeError, ValueError) as error:
            raise CompileError("Input ranges need two integer bounds") from error
        if not 0 <= integer(lo, "lower bound") <= integer(hi, "upper bound") <= 255:
            raise CompileError("Input ranges must lie within u8")
        inputs.append(Image("input", (), (name,), lo, hi))
    result = function(**{node.params[0]: node for node in inputs})
    if isinstance(result, Image):
        result = {"output": result}
    if not isinstance(result, Mapping) or not result:
        raise CompileError("Kernel must return an image or nonempty named image mapping")
    if any(not isinstance(name, str) or not name or not isinstance(node, Image)
           for name, node in result.items()):
        raise CompileError("Output names must be nonempty strings and values must be images")
    nodes = postorder((*inputs, *result.values()))
    declared = set(inputs)
    if any(node.op == "input" and node not in declared for node in nodes):
        raise CompileError("Kernel uses an input from a different graph")
    return Graph(width, height, tuple(inputs), tuple(result.items()), nodes)


def evaluate(graph, images):
    """Independent mathematical oracle; encoding and decoding never call this."""
    graph.validate(images)
    width, height, values = graph.width, graph.height, {}
    for node in graph.nodes:
        if node.op == "input":
            value = [sample for row in images[node.params[0]] for sample in row]
        elif node.op == "copy":
            value = list(values[node.args[0]])
        elif node.op in ("affine", "affine_sat"):
            weights, bias = node.params[:-1], node.params[-1]
            value = [sum(w * x for w, x in zip(weights, samples)) + bias
                     for samples in zip(*(values[arg] for arg in node.args))]
            if node.op == "affine_sat":
                value = [clip8(x) for x in value]
        elif node.op == "threshold":
            value = [int(x >= node.params[0]) for x in values[node.args[0]]]
        elif node.op in ("and", "or"):
            value = [a & b if node.op == "and" else a | b
                     for a, b in zip(values[node.args[0]], values[node.args[1]])]
        elif node.op == "shift":
            dx, dy, border = node.params
            source, value = values[node.args[0]], []
            for y in range(height):
                for x in range(width):
                    sx, sy = x + dx, y + dy
                    if border == "clamp":
                        sx, sy = min(width - 1, max(0, sx)), min(height - 1, max(0, sy))
                    value.append(source[sy * width + sx] if 0 <= sx < width and 0 <= sy < height else 0)
        else:
            raise CompileError(f"Unknown semantic operation: {node.op}")
        if any(not node.lo <= sample <= node.hi for sample in value):
            raise AssertionError(f"Oracle result violates range for {node.op}")
        values[node] = value
    return {name: [values[node][y * width:(y + 1) * width] for y in range(height)]
            for name, node in graph.outputs}
