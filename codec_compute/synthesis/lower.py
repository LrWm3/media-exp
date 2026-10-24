"""Discover scalar implementations, then realize them as existing compute IR.

Only the requested mathematical functions are named here. Their implementation
sequences come from bounded search over target primitives, never recipe tables.
Image layout, reference allocation, and bitstream constraints remain target
obligations after scalar synthesis succeeds.
"""

from operator import and_, or_, xor

from .. import ir
from .model import library_for, validate_plan
from .search import synthesize
from .verify import verify_record


GOALS = {"minimum": min, "maximum": max, "and": and_, "or": or_, "xor": xor}


class SynthesisError(ir.UnsupportedOperation):
    def __init__(self, target, operation, result):
        self.result = result
        super().__init__(target, operation,
                         f"scalar synthesis ended with {result.status}: "
                         f"tree bound {result.max_instructions}, candidate budget {result.max_candidates}; "
                         "this is not a codec impossibility result")


def realize_plan(plan, operands, *, library):
    """Realize a validated plan's primitive descriptors, without named goals.

    This is a mechanical translation, not an equivalence certificate. Callers
    integrating a requested function must first verify its plan and domains.
    Only arithmetic descriptors backed by our existing target IR are admitted.
    """
    validate_plan(plan, library)
    values = list(operands)
    if len(values) != plan.input_count or any(not isinstance(value, ir.Image) for value in values):
        raise ir.CompileError("Plan realization requires its declared symbolic image operands")
    for instruction in plan.instructions:
        primitive = library.by_name[instruction.primitive]
        args = tuple(values[index] for index in instruction.operands)
        if primitive.shift == 0:
            value = ir.affine(*args, weights=primitive.weights, bias=primitive.bias, saturating=True)
        elif primitive.weights == (1, 1) and primitive.shift == 1 and primitive.bias == 0:
            value = ir.average(*args)
        else:
            raise ir.UnsupportedOperation(library.target, primitive.name,
                                          "primitive descriptor has no exact compute-IR realization")
        values.append(value)
    return values[plan.output]


def _rebuild(node, args):
    if all(new is old for new, old in zip(args, node.args)):
        return node
    if node.op == "copy":
        return ir.copy(*args)
    if node.op in ("affine", "affine_sat"):
        if node.op == "affine":
            try:
                return ir.affine(*args, weights=node.params[:-1], bias=node.params[-1])
            except ir.CompileError:
                # Original operands and parameters were already validated;
                # only broader replacement intervals can invalidate this.
                pass
        # Synthesis loses interval correlations: Boolean XOR, for example,
        # expands to a sequence whose interval analysis conservatively says
        # [0,2]. Every replacement was equivalent on the original domains, so
        # a previously valid strict affine is still within u8 on all reachable
        # values. Clipping is therefore inactive there. Broader bounds remain
        # honest; we never forge the tighter original interval on new nodes.
        return ir.affine(*args, weights=node.params[:-1], bias=node.params[-1], saturating=True)
    if node.op == "average":
        return ir.average(*args)
    if node.op == "threshold":
        return ir.threshold(*args, *node.params)
    if node.op == "shift":
        return ir.shift(*args, *node.params[:2], border=node.params[2])
    raise ir.CompileError(f"Cannot rebuild semantic operation {node.op}")


def lower_graph(graph, *, target, max_instructions=3, max_candidates=50000):
    """Return a validated primitive graph and replayable scalar search records."""
    ir.validate_graph(graph)
    library = library_for(target)
    replacements, records = {}, []
    source_ids = {node: index for index, node in enumerate(graph.nodes)}
    for node in graph.nodes:
        args = tuple(replacements[arg] for arg in node.args)
        if node.op in GOALS:
            domains = tuple((arg.lo, arg.hi) for arg in node.args)
            result = synthesize(GOALS[node.op], library=library, input_ranges=domains,
                                max_instructions=max_instructions, max_candidates=max_candidates,
                                name=node.op)
            if not result.found:
                raise SynthesisError(library.target, node.op, result)
            # Verification records are evidence to replay, never assertions to
            # trust. Bind the selected plan, target descriptors and exact goal.
            verified = verify_record(result.plan, GOALS[node.op], domains, library=library,
                                     record=result.verification)
            if not verified["verified"]:
                raise ir.CompileError("Synthesis returned an unverified plan")
            replacement = realize_plan(result.plan, args, library=library)
            records.append({"source_node": source_ids[node], "operation": node.op,
                            "source_operands": [source_ids[arg] for arg in node.args],
                            "search": result.manifest(), "replacement": replacement})
        else:
            replacement = _rebuild(node, args)
        replacements[node] = replacement
    inputs = tuple(replacements[node] for node in graph.inputs)
    outputs = tuple((name, replacements[node]) for name, node in graph.outputs)
    nodes = ir.postorder((*inputs, *(node for _, node in outputs)))
    lowered = ir.Graph(graph.width, graph.height, inputs, outputs, nodes)
    ir.validate_graph(lowered)
    ids = {node: index for index, node in enumerate(nodes)}
    for record in records:
        # A zero-instruction alias can make an earlier synthesis dead. Preserve
        # its scalar record, but do not pretend it exists in the emitted graph.
        record["lowered_output_node"] = ids.get(record.pop("replacement"))
    return lowered, records
