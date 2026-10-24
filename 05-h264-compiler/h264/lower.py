"""Select exact predictions while preserving clipping and movement boundaries."""

from collections import Counter
from dataclasses import dataclass, replace

from decoder_core.bitstream import Operation, Source
from .graph import CompileError, Image, postorder, bounds


AFFINE = ("weighted", "combine", "saturating_weighted", "saturating_combine")


@dataclass
class Instruction:
    result: int
    operation: Operation
    graph_id: int


def _semantic_cost_guard(result, sources, outputs, optimize, exact_search):
    """Compare with the requested search mode, including early search fallbacks."""
    stats = result[3]
    if not stats["semantic_ir_rewrites"]:
        return result
    try:
        previous = lower(sources, outputs, optimize=optimize, exact_search=exact_search,
                         semantic_ir=False)
    except CompileError:
        # Dead arithmetic may be unsupported even though its simplified result
        # is legal. This case has no executable baseline cost to subtract.
        stats["semantic_ir_baseline_unavailable"] = True
        stats["semantic_ir_saved_instructions"] = None
        return result
    if len(result[0]) > len(previous[0]):
        previous[3].update({"semantic_ir_enabled": True,
                            "semantic_ir_rejected_rewrites": stats["semantic_ir_rewrites"]})
        return previous
    stats["semantic_ir_saved_instructions"] = len(previous[0])-len(result[0])
    return result


def lower(sources, outputs, optimize=True, exact_search=True, semantic_ir=True):
    from .ir import BooleanValues
    semantic = BooleanValues() if optimize and semantic_ir else None
    canonical, interned, original_keys, ranks = {}, {}, {}, {}
    original = postorder(outputs.values())
    rewrites = dict.fromkeys(("cse_eliminated", "commutative_cse_eliminated",
                             "double_not_eliminated", "direct_or_gates", "direct_and_gates",
                             "boolean_sum_nodes_bypassed", "boolean_operands_deduplicated",
                             "boolean_nodes_fused",
                             "affine_nodes_fused", "identity_affines_eliminated",
                             "threshold_negative_two_p"), 0)
    for original_id, item in enumerate(original):
        args, params = tuple(canonical[arg] for arg in item.args), item.params
        if (optimize and item.op == "weighted" and params == (-1, 1) and
                args[0].op == "weighted" and args[0].params == (-1, 1)):
            canonical[item] = args[0].args[0]
            rewrites["double_not_eliminated"] += 1
            continue
        original_key = (item.op, tuple(id(arg) for arg in args), params, item.lo, item.hi)
        # Coefficients travel with their operands: a+2*b equals 2*b+a, not 2*a+b.
        if optimize and item.op in ("combine", "saturating_combine"):
            pairs = sorted(zip(args, params[:-1]), key=lambda pair: ranks[pair[0]])
            args, weights = zip(*pairs)
            params = (*weights, params[-1])
        key = (item.op, tuple(id(arg) for arg in args), params, item.lo, item.hi)
        if optimize and key in interned:
            canonical[item] = interned[key]
            rewrites["cse_eliminated"] += 1
            if original_keys[key] != original_key:
                rewrites["commutative_cse_eliminated"] += 1
        else:
            value = Image(item.op, args, params, item.lo, item.hi)
            canonical[item] = interned[key] = semantic.rewrite(value, original_id) if semantic else value
            ranks.setdefault(canonical[item], len(ranks))
            original_keys[key] = original_key
    roots = {name: canonical[item] for name, item in outputs.items()}
    graph = postorder(roots.values())
    ids = {item: index for index, item in enumerate(graph)}
    uses = Counter(arg for item in graph for arg in item.args)
    named = set(roots.values())
    instructions, produced, fused, plans = [], {}, set(), {}
    exact_plans, proofs = {}, []
    search = None
    if optimize and exact_search:
        from .search import ExactSearch
        search = ExactSearch(graph, named, lambda sources, outputs: lower(
            sources, outputs, exact_search=False, semantic_ir=False))
    for index, source in enumerate(sources):
        if source in canonical:
            produced[canonical[source]] = index
    next_value = len(sources)

    def shifted(arg):
        return optimize and arg.op in ("shift", "sample")

    def disposable(item):
        return item not in named and uses[item] == 1

    def boolean_terms(item):
        if item.op != "threshold":
            return None
        # A materialized/named sum retains its ordinary arithmetic meaning.
        leaves, sums = [], 0
        pending = [item.args[0]]
        while pending:
            arg = pending.pop()
            if arg.op == "combine" and arg.params == (1, 1, 0) and disposable(arg):
                sums += 1
                pending.extend(reversed(arg.args))
            elif 0 <= arg.lo <= arg.hi <= 1:
                leaves.append(arg)
            else:
                return None
        if not sums or item.params[0] not in (1, len(leaves)):
            return None
        kind = "or" if item.params[0] == 1 else "and"
        return kind, leaves, sums

    def boolean_plan(item):
        terms = boolean_terms(item)
        if terms is None:
            return None
        kind, leaves, sums = terms
        pending, leaves, nested = list(reversed(leaves)), [], 0
        while pending:
            arg = pending.pop()
            inner = boolean_terms(arg) if disposable(arg) else None
            if inner is not None and inner[0] == kind:
                pending.extend(reversed(inner[1]))
                sums += inner[2]
                nested += 1
            else:
                leaves.append(arg)
        unique = tuple(dict.fromkeys(leaves))
        return kind, unique, (), nested, sums, len(leaves)-len(unique)

    def unclipped(item):
        if item.op not in AFFINE:
            return False
        intervals = [bounds(arg, weight) for arg, weight in zip(item.args, item.params[:-1])]
        lo = sum(pair[0] for pair in intervals)+item.params[-1]
        hi = sum(pair[1] for pair in intervals)+item.params[-1]
        return 0 <= lo <= hi <= 255

    def legal_affine(args, weights, bias, saturating):
        limits = (-128, 127) if len(args) == 1 else (-64, 63)
        if not 1 <= len(args) <= 2 or any(not limits[0] <= w <= limits[1] for w in weights):
            return False
        if len(args) == 2 and not -64 <= sum(weights) <= 64:
            return False  # B hardware weights are doubled at denominator zero.
        if saturating:
            return -128 <= bias <= 127
        first_bias = max(-128, min(127, bias))
        intervals = [bounds(arg, w) for arg, w in zip(args, weights)]
        return 0 <= sum(pair[0] for pair in intervals)+first_bias <= sum(pair[1] for pair in intervals)+first_bias <= 255

    def plan(item):
        if item in plans:
            return plans[item]
        replacement = search.choose(item) if search else None
        if replacement is not None:
            exact_plans[item] = replacement
            plans[item] = ("exact", replacement[0], (), 0, 0, 0)
            return plans[item]
        chosen = boolean_plan(item) if optimize else None
        if chosen is None:
            weights = item.params[:-1] if item.op in AFFINE else ()
            bias = item.params[-1] if item.op in AFFINE else 0
            args, count = item.args, 0
            saturating = item.op.startswith("saturating_")
            if optimize and item.op in AFFINE and legal_affine(args, weights, bias, saturating):
                # Only exact, unrounded arithmetic may disappear. Moving reads,
                # thresholds, observable clips, and shared/named values are barriers.
                while True:
                    for index, (arg, weight) in enumerate(zip(args, weights)):
                        if not disposable(arg) or not unclipped(arg):
                            continue
                        if not legal_affine(arg.args, arg.params[:-1], arg.params[-1], arg.op.startswith("saturating_")):
                            continue
                        pairs = list(zip(args, weights))
                        pairs[index:index+1] = [(source, weight*w) for source, w in zip(arg.args, arg.params[:-1])]
                        combined = {}
                        for source, coefficient in pairs:
                            combined[source] = combined.get(source, 0)+coefficient
                        new_args, new_weights = tuple(combined), tuple(combined.values())
                        new_bias = bias+weight*arg.params[-1]
                        if not legal_affine(new_args, new_weights, new_bias, saturating):
                            continue
                        args, weights, bias = new_args, new_weights, new_bias
                        count += 1
                        break
                    else:
                        break
            chosen = item.op, args, weights, bias, count, 0
        plans[item] = chosen
        return chosen

    def dependencies(item):
        args = plan(item)[1]
        if item.op in ("shift", "sample", "gather"):
            return args  # Crop/filter boundaries are observable.
        return tuple(arg.args[0] if shifted(arg) else arg for arg in args)

    def read(arg, fold=True):
        if fold and shifted(arg):
            fused.add(arg)
            fields = ("dx", "dy") if arg.op == "shift" else ("qx", "qy")
            return Source(produced[arg.args[0]], **dict(zip(fields, arg.params)))
        return Source(produced[arg])

    def emit(label, operands, weights, bias=0, denominator=0, graph_id=0, block_offsets=()):
        nonlocal next_value
        if len(operands) == 2 and not -128 <= sum(weights) <= (127 if denominator == 7 else 128):
            raise CompileError("B-prediction weight sum exceeds H.264 limits")
        refs = tuple(replace(ref, weight=weight, offset=bias) for ref, weight in zip(operands, weights))
        instructions.append(Instruction(next_value, Operation(label, refs, denominator, block_offsets), graph_id))
        next_value += 1
        return Source(next_value-1)

    def stationary(ref):
        return not (ref.dx or ref.dy or ref.qx or ref.qy)

    def affine(item):
        _, args, weights, bias, count, _ = plan(item)
        refs = [read(arg) for arg in args]
        limits = (-128, 127) if len(refs) == 1 else (-64, 63)
        if any(not limits[0] <= weight <= limits[1] for weight in weights):
            raise CompileError(f"{item.op} coefficient cannot be represented by a single prediction")
        rewrites["affine_nodes_fused"] += count
        if optimize and len(refs) == 1 and weights == (1,) and bias == 0 and stationary(refs[0]):
            rewrites["identity_affines_eliminated"] += 1
            return refs[0]
        saturating = item.op.startswith("saturating_")
        if saturating and not -128 <= bias <= 127:
            raise CompileError("Saturating affine bias must lie in [-128,127]; splitting would change clipping")
        first_bias = max(-128, min(127, bias))
        lo = sum(bounds(arg, w)[0] for arg, w in zip(args, weights)) + first_bias
        hi = sum(bounds(arg, w)[1] for arg, w in zip(args, weights)) + first_bias
        if not saturating and not 0 <= lo <= hi <= 255:
            raise CompileError("Splitting affine bias would clip an intermediate; this form is not supported")
        hardware_weights = weights if len(refs) == 1 else tuple(2*w for w in weights)
        result = emit(item.op, refs, hardware_weights, first_bias, graph_id=ids[item])
        remaining = bias-first_bias
        while remaining:
            part = max(-128, min(127, remaining))
            result = emit("remaining affine bias", [result], [1], part, graph_id=ids[item])
            remaining -= part
        return result

    def boolean(item):
        kind, args, _, nested, sums, duplicates = plan(item)
        rewrites["boolean_sum_nodes_bypassed"] += sums
        rewrites["boolean_operands_deduplicated"] += duplicates
        rewrites["boolean_nodes_fused"] += nested
        refs = [read(arg) for arg in args]
        result = refs[0]
        if len(refs) == 1 and not stationary(result):
            return emit(f"deduplicated {kind}", refs, [1], graph_id=ids[item])
        for ref in refs[1:]:
            weights, bias = ((1, 1), 0) if kind == "or" else ((2, 2), -1)
            result = emit(f"boolean {kind}", [result, ref], weights, bias, graph_id=ids[item])
            rewrites[f"direct_{kind}_gates"] += 1
        return result

    def exact(item):
        frontier, sequence, proof = exact_plans[item]
        refs = [read(arg) for arg in frontier]
        for step in sequence:
            refs.append(emit("exact scalar prediction", [refs[index] for index in step.inputs],
                             step.weights, step.bias, step.denominator, ids[item]))
        proofs.append(proof)
        return refs[-1]

    def compare(item):
        arg, t = item.args[0], item.params[0]
        ref = read(arg)
        if t <= arg.lo or t > arg.hi:
            return emit("constant threshold", [ref], [0], int(t <= arg.lo), graph_id=ids[item])
        # First prefer one exact rounded P prediction.
        for denominator in (7, 0, 1, 2, 3, 4, 5, 6):
            rounding = (1 << (denominator-1)) if denominator else 0
            for weight in range(1, 128):
                bias = 1-((weight*t+rounding) >> denominator)
                if (-128 <= bias <= 127 and
                    ((weight*(t-1)+rounding) >> denominator)+bias <= 0 and
                    ((weight*arg.hi+rounding) >> denominator)+bias <= 1):
                    return emit(f"threshold >= {t}", [ref], [weight], bias, denominator, ids[item])
        shift = 192-t
        fallback_shifts = (shift+126)//127 if shift > 0 else (-shift+127)//128
        if optimize and 0 <= t <= 127 and fallback_shifts+1 > 2:
            # clip(1-clip(t-x)) equals [x >= t] for every integer u8 x.
            ref = emit("threshold negative distance", [ref], [-1], t, graph_id=ids[item])
            rewrites["threshold_negative_two_p"] += 1
            return emit(f"threshold >= {t}", [ref], [-1], 1, graph_id=ids[item])
        # Baseline universal u8 comparison: shift cutoff to192, then ((x+64)>>7)-1.
        while shift:
            part = max(-128, min(127, shift))
            ref = emit("threshold shift", [ref], [1], part, graph_id=ids[item])
            shift -= part
        return emit(f"threshold >= {t}", [ref], [1], -1, 7, ids[item])

    # Iterative physical traversal handles thousands of unrolled graph nodes.
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
        elif item.op in ("sample", "gather"):
            ref = read(item.args[0], fold=False)
            offsets = ()
            if item.op == "sample":
                ref = replace(ref, qx=item.params[0], qy=item.params[1])
            else:
                offsets = item.params[0]
            result = emit(item.op, [ref], [1], graph_id=ids[item], block_offsets=offsets)
        elif plan(item)[0] == "exact":
            result = exact(item)
        elif plan(item)[0] in ("or", "and"):
            result = boolean(item)
        elif item.op in AFFINE:
            result = affine(item)
        elif item.op == "threshold":
            result = compare(item)
        else:
            raise CompileError(f"Unsupported node: {item.op}")
        produced[item] = result.frame
    # A frontier can change an older fusion opportunity. Check the whole graph
    # as well as each isolated cut so enabling search never adds instructions.
    if proofs:
        baseline = lower(sources, outputs, optimize=optimize, exact_search=False,
                         semantic_ir=semantic_ir)
        if len(instructions) >= len(baseline[0]):
            baseline[3]["exact_search_enabled"] = True
            baseline[3]["exact_search_attempts"] = search.attempts
            return _semantic_cost_guard(baseline, sources, outputs, optimize, exact_search) if semantic else baseline
    stats = {
        "reachable_nodes_before_cse": len(original), "graph_nodes": len(graph), **rewrites,
        "exact_search_enabled": bool(search), "exact_search_attempts": search.attempts if search else 0,
        "exact_search_rewrites": len(proofs), "exact_search_proofs": proofs,
        "exact_search_saved_instructions": (len(baseline[0]) - len(instructions)) if proofs else 0,
        "fused_shift_nodes": len(fused), "lowered_instructions": len(instructions),
        "semantic_ir_enabled": bool(semantic), "semantic_ir_saved_instructions": 0,
        **(semantic.stats() if semantic else BooleanValues().stats())}
    result = instructions, {name: produced[item] for name, item in roots.items()}, graph, stats, {
        name: ids[item] for name, item in roots.items()}
    return _semantic_cost_guard(result, sources, outputs, optimize, exact_search) if semantic else result
