"""Exact two-input Boolean atlas for explicit weighted P/B prediction.

No compiler search is imported here. The oracle, exhaustive one-instruction
certificate, witnesses, and structural minimality proof are independently
replayable using Python's standard library. See PROOFS.md for the model.
"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path


INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
NAMES = (
    "FALSE", "NOR", "NOT_A_AND_B", "NOT_A", "A_AND_NOT_B", "NOT_B",
    "XOR", "NAND", "AND", "XNOR", "B", "A_IMPLIES_B", "A",
    "B_IMPLIES_A", "OR", "TRUE",
)
ONE_INSTRUCTION_MASKS = tuple(mask for mask in range(16) if mask not in (6, 9))
PROOF = "PROOFS.md"


def truth_table(mask):
    """Return f(0,0), f(0,1), f(1,0), f(1,1), with bit i equal to f_i."""
    if type(mask) is not int or not 0 <= mask < 16:
        raise ValueError("Boolean mask must be an integer in [0,15]")
    return [(mask >> index) & 1 for index in range(4)]


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower},{upper}]")


def _step(inputs, weights, bias=0, denominator=0):
    return {"inputs": list(inputs), "weights": list(weights),
            "bias": bias, "denominator": denominator}


def evaluate_plan(plan, values, output_index):
    """Independent integer P/B oracle; intermediates can use all of [0,255].

    A one-source P operation shifts by d, while a two-source B operation shifts
    by d+1. Integer rounding precedes the signed offset and clipping. For B,
    bias denotes the effective offset; setting both signalled offsets to bias
    realizes it exactly. This also permits repeated source indices.
    """
    available = list(values)
    if len(available) != 2:
        raise ValueError("Exactly two input values are required")
    for value in available:
        _integer(value, 0, 255, "Input value")
    for step in plan:
        sources, weights = step["inputs"], step["weights"]
        if len(sources) not in (1, 2) or len(weights) != len(sources):
            raise ValueError("An instruction needs one or two matching operands")
        for source in sources:
            _integer(source, 0, len(available) - 1, "Source index")
        for weight in weights:
            _integer(weight, -128, 127, "Weight")
        bias, denominator = step["bias"], step["denominator"]
        _integer(bias, -128, 127, "Effective offset")
        _integer(denominator, 0, 7, "Denominator")
        numerator = sum(available[source] * weight
                        for source, weight in zip(sources, weights))
        shift = denominator + (len(sources) == 2)
        rounded = numerator if shift == 0 else (numerator + (1 << (shift - 1))) // (1 << shift)
        available.append(max(0, min(255, rounded + bias)))
    _integer(output_index, 0, len(available) - 1, "Output index")
    return available[output_index]


def _score(step):
    """Deterministic readable witness selection, unrelated to lower bounds."""
    return (len(step["inputs"]), sum(map(abs, step["weights"])),
            step["denominator"], abs(step["bias"]),
            tuple(step["inputs"]), tuple(step["weights"]), step["bias"])


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def enumerate_single_predictions():
    """Exhaust all legal weight/denominator classes, with exact reductions.

    For binary output containing a one, max(rounded_values)+bias must equal 1.
    Thus bias=1-max is the only candidate, and its output mask is the argmax
    mask. Constant zero has a separate explicit witness. Reversed B sources
    exchange weights; repeated B sources depend only on their sum [-256,254].
    These reductions are exhaustive, not a bounded synthesis heuristic.
    """
    witnesses = {0: _step([0], [0]), 15: _step([0], [0], 1)}
    counts = {"P": Counter(), "B_distinct": Counter(), "B_repeated": Counter()}

    def record(kind, rounded, sources, weights, denominator):
        maximum = max(rounded)
        bias = 1 - maximum
        if not -128 <= bias <= 127:
            raise AssertionError("The input-domain offset lemma was violated")
        mask = sum(1 << index for index, value in enumerate(rounded) if value == maximum)
        counts[kind][mask] += 1
        step = _step(sources, weights, bias, denominator)
        if mask not in witnesses or _score(step) < _score(witnesses[mask]):
            witnesses[mask] = step

    for source in (0, 1):
        for weight in range(-128, 128):
            for denominator in range(8):
                value = weight if denominator == 0 else (weight + (1 << (denominator - 1))) // (1 << denominator)
                rounded = (0, 0, value, value) if source == 0 else (0, value, 0, value)
                record("P", rounded, [source], [weight], denominator)

    for first in range(-128, 128):
        for second in range(-128, 128):
            for denominator in range(8):
                half, divisor = 1 << denominator, 1 << (denominator + 1)
                rounded = (0, (second + half) // divisor, (first + half) // divisor,
                           (first + second + half) // divisor)
                record("B_distinct", rounded, [0, 1], [first, second], denominator)

    for source in (0, 1):
        for total in range(-256, 255):
            first = max(-128, total - 127)
            second = total - first
            for denominator in range(8):
                value = (total + (1 << denominator)) // (1 << (denominator + 1))
                rounded = (0, 0, value, value) if source == 0 else (0, value, 0, value)
                record("B_repeated", rounded, [source, source], [first, second], denominator)

    # Enumerating all 65,536 raw offset pairs independently checks that the B
    # effective offset parameter represents the entire legal signed range.
    effective_offsets = {(first + second + 1) // 2
                         for first in range(-128, 128) for second in range(-128, 128)}
    if effective_offsets != set(range(-128, 128)):
        raise AssertionError("B effective-offset equivalence failed")
    actual_masks = tuple(sorted(witnesses))
    if actual_masks != ONE_INSTRUCTION_MASKS:
        raise AssertionError(f"Unexpected exhaustive coverage: {actual_masks}")
    for mask, step in witnesses.items():
        if [evaluate_plan([step], values, 2) for values in INPUTS] != truth_table(mask):
            raise AssertionError(f"Single-instruction witness failed for mask {mask}")
    report = {
        "method": "exhaustive legal weights and denominators with proved exact offset/source reductions",
        "source_reduction": "B(1,0) swaps weights; repeated B sources use every weight sum [-256,254]",
        "offset_reduction": "Any binary output containing one requires bias=1-max(rounded values); zero is explicit",
        "parameter_classes": {kind: sum(histogram.values()) for kind, histogram in counts.items()},
        "parameter_classes_total": sum(sum(histogram.values()) for histogram in counts.values()),
        "argmax_mask_histograms": {
            kind: {str(mask): histogram[mask] for mask in sorted(histogram)}
            for kind, histogram in counts.items()
        },
        "raw_effective_offset_parameterizations": {
            "P": 2 * 256 * 8 * 256,
            "B": 4 * 256 * 256 * 8 * 256,
        },
        "raw_B_offset_pairs_checked": 256 * 256,
        "effective_B_offsets": sorted(effective_offsets),
        "single_instruction_masks": list(actual_masks),
        "missing_single_instruction_masks": [6, 9],
        "witnesses_checked_on_all_four_inputs": len(witnesses),
        "witnesses": {str(mask): witnesses[mask] for mask in sorted(witnesses)},
    }
    report["sha256"] = _digest(report)
    return witnesses, report


def _packed_schedule(plan):
    levels, layers = [0, 0], []
    for index, step in enumerate(plan, 2):
        level = 1 + max(levels[source] for source in step["inputs"])
        levels.append(level)
        while len(layers) < level:
            layers.append([])
        layers[level - 1].append(index)
    return layers


def _is_unate(mask):
    table = truth_table(mask)
    for pairs in (((0, 2), (1, 3)), ((0, 1), (2, 3))):
        differences = [table[right] - table[left] for left, right in pairs]
        if min(differences) < 0 < max(differences):
            return False
    return True


def build_atlas():
    """Return a JSON-serializable atlas with proved minima for all 16 masks."""
    witnesses, certificate = enumerate_single_predictions()
    functions = []
    for mask in range(16):
        if mask in (10, 12):
            plan, output_index, scalar_minimum, packed_minimum = [], (1 if mask == 10 else 0), 0, 0
            proof_refs = [f"{PROOF}#zero-instructions"]
        elif mask == 6:
            plan = [_step([0, 1], [2, -2]), _step([0, 1], [-2, 2]), _step([2, 3], [2, 2])]
            output_index, scalar_minimum, packed_minimum = 4, 3, 2
            proof_refs = [f"{PROOF}#no-one-instruction-parity", f"{PROOF}#no-two-instruction-parity", f"{PROOF}#packed-pictures"]
        elif mask == 9:
            plan = [_step([0, 1], [2, 2], -1), _step([0, 1], [-2, -2], 1), _step([2, 3], [2, 2])]
            output_index, scalar_minimum, packed_minimum = 4, 3, 2
            proof_refs = [f"{PROOF}#no-one-instruction-parity", f"{PROOF}#no-two-instruction-parity", f"{PROOF}#packed-pictures"]
        else:
            plan, output_index, scalar_minimum, packed_minimum = [witnesses[mask]], 2, 1, 1
            proof_refs = [f"{PROOF}#zero-instructions", f"{PROOF}#exhaustive-single-instruction-certificate"]
        actual = [evaluate_plan(plan, values, output_index) for values in INPUTS]
        if actual != truth_table(mask):
            raise AssertionError(f"Atlas witness failed for mask {mask}")
        functions.append({
            "id": mask, "name": NAMES[mask], "truth": truth_table(mask),
            "plan": plan, "output_index": output_index,
            "scalar_minimum": scalar_minimum, "packed_minimum": packed_minimum,
            "minimum_status": "proved", "proof_refs": proof_refs,
            "packed_schedule": _packed_schedule(plan),
        })
    return {
        "schema_version": 1,
        "title": "Exact two-input Boolean weighted-prediction atlas",
        "truth_table_order": [list(values) for values in INPUTS],
        "truth_table_encoding": "mask=sum(f(a,b) << (2*a+b)); inputs A=index0, B=index1",
        "model": {
            "input_values": [0, 1], "input_state": "two original input planes already resident",
            "intermediate_values": [0, 255], "weights": [-128, 127],
            "offsets": [-128, 127], "denominators": [0, 7],
            "P": "clip_u8(round_shift(w*x,d)+bias)",
            "B": "clip_u8(round_shift(w0*x+w1*y,d+1)+(o0+o1+1)//2)",
            "round_shift": "n if shift=0, otherwise (n+2**(shift-1))//2**shift",
            "sources": "one or two original inputs or earlier instruction results; repeated sources permitted",
            "free_operations": "only original input aliases; constant generation costs one instruction",
            "excluded": ["residuals", "deblocking", "fractional motion/interpolation", "precomputed intermediate inputs", "data-dependent parameter or source selection"],
            "scalar_cost": "number of uniform scalar P/B operations",
            "packed_cost": "computation pictures after inputs; independent tiles may share a picture and read only earlier pictures",
            "packed_residency": "original source planes may occupy separated tiles in an input picture; per-tile legal parameters via slices are allowed",
            "scope": "local scalar predictions applied uniformly over arbitrary input pixels; no cross-pixel information",
        },
        "functions": functions,
        "single_instruction_certificate": certificate,
        "minimum_histograms": {
            "scalar": {"0": 2, "1": 12, "3": 2},
            "packed": {"0": 2, "1": 12, "2": 2},
        },
        "proof_document": PROOF,
    }


def verify_atlas(atlas):
    """Replay exhaustive certificate and witnesses; reject modified claims."""
    expected = build_atlas()
    if atlas != expected:
        raise ValueError("Atlas differs from independently replayed exhaustive certificate and proved minima")
    non_unate = [mask for mask in range(16) if not _is_unate(mask)]
    if non_unate != [6, 9]:
        raise AssertionError("Structural parity-proof hypotheses failed")
    return {
        "verified": True, "functions": 16, "truth_table_cases": 64,
        "single_prediction_classes": expected["single_instruction_certificate"]["parameter_classes_total"],
        "non_unate_masks": non_unate,
        "scalar_minima": expected["minimum_histograms"]["scalar"],
        "packed_minima": expected["minimum_histograms"]["packed"],
        "certificate_sha256": expected["single_instruction_certificate"]["sha256"],
        "lower_bound_basis": "analytic structural lemmas in PROOFS.md, not a bounded multi-step search",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the atlas JSON to this path instead of stdout")
    parser.add_argument("--verify", type=Path, metavar="ATLAS_JSON", help="Independently regenerate and verify an atlas JSON")
    args = parser.parse_args()
    result = verify_atlas(json.loads(args.verify.read_text())) if args.verify else build_atlas()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
