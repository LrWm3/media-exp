"""Select exact P/B predictions and fuse single shifted reads, never nested shifts."""

from dataclasses import dataclass

from decoder_core.bitstream import Operation, Source
from .graph import CompileError, Image, postorder, bounds


@dataclass
class Instruction:
    result: int
    operation: Operation
    graph_id: int


def lower(source, outputs, optimize=True):
    graph, canonical, interned = [], {}, {}
    original = postorder(outputs.values())
    # Intern only identical expressions; rounding/clipping algebra is not rewritten.
    for item in original:
        args = tuple(canonical[arg] for arg in item.args)
        key = (item.op, tuple(id(arg) for arg in args), item.params, item.lo, item.hi)
        if optimize and key in interned:
            canonical[item] = interned[key]
        else:
            new = Image(item.op, args, item.params, item.lo, item.hi)
            canonical[item] = interned[key] = new
            graph.append(new)
    roots = {name: canonical[item] for name, item in outputs.items()}
    source = canonical.get(source)
    ids = {item: index for index, item in enumerate(graph)}
    instructions, produced, fused = [], {}, set()
    if source is not None:
        produced[source] = 0
    next_value = 1

    def shifted(arg):
        return optimize and arg.op == "shift"

    def dependencies(item):
        if item.op == "shift":
            return item.args  # Materialize inner shift: crop boundaries are observable.
        return tuple(arg.args[0] if shifted(arg) else arg for arg in item.args)

    def read(arg, fold=True):
        if fold and shifted(arg):
            fused.add(arg)
            return Source(produced[arg.args[0]], dx=arg.params[0], dy=arg.params[1])
        return Source(produced[arg])

    def emit(label, operands, weights, bias=0, denominator=0, graph_id=0):
        nonlocal next_value
        refs = tuple(Source(ref.frame, weight, bias, ref.dx, ref.dy) for ref, weight in zip(operands, weights))
        instructions.append(Instruction(next_value, Operation(label, refs, denominator), graph_id))
        next_value += 1
        return Source(next_value-1)

    def affine(item):
        weights, bias = item.params[:-1], item.params[-1]
        refs = [read(arg) for arg in item.args]
        limits = (-128, 127) if len(refs) == 1 else (-64, 63)
        if any(not limits[0] <= weight <= limits[1] for weight in weights):
            raise CompileError(f"{item.op} coefficient cannot be represented by a single prediction")
        if optimize and len(refs) == 1 and weights == (1,) and bias == 0 and not (refs[0].dx or refs[0].dy):
            return refs[0]
        first_bias = max(-128, min(127, bias))
        lo = sum(bounds(arg, w)[0] for arg, w in zip(item.args, weights)) + first_bias
        hi = sum(bounds(arg, w)[1] for arg, w in zip(item.args, weights)) + first_bias
        if not 0 <= lo <= hi <= 255:
            raise CompileError("Splitting affine bias would clip an intermediate; this form is not supported")
        hardware_weights = weights if len(refs) == 1 else tuple(2*w for w in weights)
        result = emit(item.op, refs, hardware_weights, first_bias, graph_id=ids[item])
        remaining = bias-first_bias
        while remaining:
            part = max(-128, min(127, remaining))
            result = emit("remaining affine bias", [result], [1], part, graph_id=ids[item])
            remaining -= part
        return result

    def compare(item):
        arg, t = item.args[0], item.params[0]
        ref = read(arg)
        if t <= arg.lo or t > arg.hi:
            return emit("constant threshold", [ref], [0], int(t <= arg.lo), graph_id=ids[item])
        # Search one legal rounded P operation, proving both threshold neighbors
        # and the highest possible value. Negative false values clip safely to0.
        for denominator in (7, 0, 1, 2, 3, 4, 5, 6):
            rounding = (1 << (denominator-1)) if denominator else 0
            for weight in range(1, 128):
                bias = 1-((weight*t+rounding) >> denominator)
                if (-128 <= bias <= 127 and
                    ((weight*(t-1)+rounding) >> denominator)+bias <= 0 and
                    ((weight*arg.hi+rounding) >> denominator)+bias <= 1):
                    return emit(f"threshold >= {t}", [ref], [weight], bias, denominator, ids[item])
        # Universal u8 comparison: shift cutoff to192, then ((x+64)>>7)-1.
        shift = 192-t
        while shift:
            part = max(-128, min(127, shift))
            ref = emit("threshold shift", [ref], [1], part, graph_id=ids[item])
            shift -= part
        return emit(f"threshold >= {t}", [ref], [1], -1, 7, ids[item])

    # Traverse physical dependencies, bypassing shifts folded into reads.
    stack = [(root, False) for root in reversed(list(roots.values()))]
    while stack:
        item, expanded = stack.pop()
        if item in produced:
            continue
        if not expanded:
            stack.append((item, True))
            stack.extend((arg, False) for arg in reversed(dependencies(item)) if arg not in produced)
            continue
        if item.op == "shift":
            ref = read(item.args[0], fold=False)
            ref = Source(ref.frame, dx=item.params[0], dy=item.params[1])
            result = emit("shift", [ref], [1], graph_id=ids[item])
        elif item.op in ("weighted", "combine"):
            result = affine(item)
        elif item.op == "threshold":
            result = compare(item)
        else:
            raise CompileError(f"Unsupported node: {item.op}")
        produced[item] = result.frame
    return instructions, {name: produced[item] for name, item in roots.items()}, graph, {
        "reachable_nodes_before_cse": len(original), "graph_nodes": len(graph),
        "cse_eliminated": len(original)-len(graph), "fused_shift_nodes": len(fused),
        "lowered_instructions": len(instructions)}, {name: ids[item] for name, item in roots.items()}
