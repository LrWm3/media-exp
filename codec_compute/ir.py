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


def _image(value):
    if not isinstance(value, Image):
        raise CompileError("Operands must be symbolic images")
    return value


def copy(image):
    """Return identical pixels; no requirement to allocate another picture."""
    _image(image)
    return _node("copy", (image,), (), image.lo, image.hi)


def affine(*images, weights=None, bias=0, saturating=False):
    """Integer sum(weights[i]*images[i])+bias, optionally clipped once to u8.

    Coefficient limits and source counts belong to target lowering. Strict
    arithmetic requires an interval proof that the result remains in u8.
    """
    if not images or any(not isinstance(image, Image) for image in images):
        raise CompileError("affine requires symbolic images")
    try:
        weights = (1,) * len(images) if weights is None else tuple(weights)
    except TypeError as error:
        raise CompileError("weights must be an iterable of integers") from error
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


def minimum(a, b):
    """Pointwise unsigned minimum; target realization is synthesized."""
    _image(a)
    _image(b)
    return _node("minimum", (a, b), (), min(a.lo, b.lo), min(a.hi, b.hi))


def maximum(a, b):
    """Pointwise unsigned maximum; target realization is synthesized."""
    _image(a)
    _image(b)
    return _node("maximum", (a, b), (), max(a.lo, b.lo), max(a.hi, b.hi))


def absdiff(a, b):
    """Pointwise |a-b| on unsigned bytes; target realization is synthesized."""
    _image(a)
    _image(b)
    lo = max(0, a.lo - b.hi, b.lo - a.hi)
    hi = max(abs(a.lo - b.hi), abs(a.hi - b.lo))
    return _node("absdiff", (a, b), (), lo, hi)


def average(a, b):
    """Exact rounded average: floor((A+B+1)/2), including full-range bytes.

    IR revision 2 keeps this operation intact: expanding through a u8 add
    would overflow or clip before division. The rounding rule is serialized.
    """
    _image(a)
    _image(b)
    return _node("average", (a, b), ("half_up",), (a.lo + b.lo + 1) // 2, (a.hi + b.hi + 1) // 2)


def shift(image, dx=0, dy=0, *, border="zero"):
    """Read image[y+dy,x+dx]; out-of-image positions use explicit border rules."""
    _image(image)
    integer(dx, "dx")
    integer(dy, "dy")
    if border not in ("zero", "clamp"):
        raise CompileError("border must be 'zero' or 'clamp'")
    return _node("shift", (image,), (dx, dy, border),
                 min(0, image.lo) if border == "zero" else image.lo, image.hi)


def threshold(image, cutoff):
    _image(image)
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


def bit_xor(a, b):
    """Boolean XOR on proven 0/1 images, not bytewise bit manipulation."""
    _binary(a)
    _binary(b)
    values = [x ^ y for x in range(a.lo, a.hi + 1) for y in range(b.lo, b.hi + 1)]
    return _node("xor", (a, b), (), min(values), max(values))


def postorder(roots):
    visited, active, result = set(), set(), []
    stack = [(root, False) for root in reversed(tuple(roots))]
    while stack:
        item, expanded = stack.pop()
        if not isinstance(item, Image):
            raise CompileError("Graph operands must be Image values")
        if not isinstance(item.args, tuple) or not isinstance(item.params, tuple):
            raise CompileError("Image operands and parameters must be immutable tuples")
        if item in visited:
            continue
        if expanded:
            active.remove(item)
            visited.add(item)
            result.append(item)
        else:
            if item in active:
                raise CompileError("Image graph must be acyclic")
            active.add(item)
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
        return {"format": "codec-compute-ir-2", "dtype": "u8", "shape": [self.height, self.width],
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
    result = (function(inputs[0]) if input_ranges is None
              else function(**{node.params[0]: node for node in inputs}))
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
    graph = Graph(width, height, tuple(inputs), tuple(result.items()), nodes)
    validate_graph(graph)
    return graph


def validate_graph(graph):
    """Check topology and independently rederive all advertised range proofs.

    Public IR objects are inspectable and constructible, but consumers must
    never trust user-authored bounds when selecting arithmetic instructions.
    """
    if not isinstance(graph, Graph):
        raise CompileError("Expected a Graph")
    if integer(graph.width, "width") <= 0 or integer(graph.height, "height") <= 0:
        raise CompileError("Image dimensions must be positive")
    if any(not isinstance(value, tuple) for value in (graph.inputs, graph.outputs, graph.nodes)):
        raise CompileError("Graph collections must be immutable tuples")
    if not graph.inputs or not graph.outputs:
        raise CompileError("Graph needs inputs and outputs")
    if any(not isinstance(node, Image) for node in graph.inputs):
        raise CompileError("Graph inputs must be Image values")
    declared, seen, input_names = set(graph.inputs), set(), set()
    if len(declared) != len(graph.inputs):
        raise CompileError("Graph inputs must be unique")
    for node in graph.nodes:
        if not isinstance(node, Image) or node in seen:
            raise CompileError("Graph nodes must be distinct Image values")
        if not isinstance(node.args, tuple) or not isinstance(node.params, tuple):
            raise CompileError("Image operands and parameters must be immutable tuples")
        if any(arg not in seen for arg in node.args):
            raise CompileError("Graph operands must precede their users")
        if type(node.lo) is not int or type(node.hi) is not int or not 0 <= node.lo <= node.hi <= 255:
            raise CompileError("Node range must lie within u8")
        if node.op == "input":
            if node not in declared or node.args or len(node.params) != 1:
                raise CompileError("Graph uses an undeclared or malformed input")
            name = node.params[0]
            if not isinstance(name, str) or not name.isidentifier() or name in input_names:
                raise CompileError("Graph input names must be unique identifiers")
            input_names.add(name)
        else:
            args, params = node.args, node.params
            try:
                if node.op == "copy" and len(args) == 1 and not params:
                    proof = copy(*args)
                elif node.op in ("affine", "affine_sat") and len(params) == len(args) + 1:
                    proof = affine(*args, weights=params[:-1], bias=params[-1], saturating=node.op == "affine_sat")
                elif node.op == "average" and len(args) == 2 and params == ("half_up",):
                    proof = average(*args)
                elif node.op == "shift" and len(args) == 1 and len(params) == 3:
                    proof = shift(*args, params[0], params[1], border=params[2])
                elif node.op == "threshold" and len(args) == 1 and len(params) == 1:
                    proof = threshold(*args, *params)
                elif node.op in ("and", "or", "xor", "minimum", "maximum", "absdiff") and len(args) == 2 and not params:
                    proof = {"and": bit_and, "or": bit_or, "xor": bit_xor,
                             "minimum": minimum, "maximum": maximum, "absdiff": absdiff}[node.op](*args)
                else:
                    raise CompileError(f"Unknown or malformed semantic operation: {node.op}")
            except (TypeError, AttributeError) as error:
                raise CompileError(f"Malformed semantic operation: {node.op}") from error
            if (proof.lo, proof.hi) != (node.lo, node.hi):
                raise CompileError(f"Invalid range proof for {node.op}: expected [{proof.lo},{proof.hi}]")
        seen.add(node)
    if not declared <= seen or any(node.op != "input" for node in declared):
        raise CompileError("Declared inputs must occur in the graph")
    names = set()
    for output in graph.outputs:
        if not isinstance(output, tuple) or len(output) != 2:
            raise CompileError("Graph outputs must be (name, image) tuples")
        name, node = output
        if not isinstance(name, str) or not name or name in names or node not in seen:
            raise CompileError("Outputs need unique names and defined image values")
        names.add(name)
    return graph


def evaluate(graph, images):
    """Independent mathematical oracle; encoding and decoding never call this."""
    validate_graph(graph)
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
        elif node.op == "average":
            value = [(a + b + 1) // 2 for a, b in zip(values[node.args[0]], values[node.args[1]])]
        elif node.op in ("and", "or", "xor", "minimum", "maximum", "absdiff"):
            operation = {"and": lambda a, b: a & b, "or": lambda a, b: a | b,
                         "xor": lambda a, b: a ^ b, "minimum": min, "maximum": max,
                         "absdiff": lambda a, b: abs(a - b)}[node.op]
            value = [operation(a, b) for a, b in zip(values[node.args[0]], values[node.args[1]])]
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
