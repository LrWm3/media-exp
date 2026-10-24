"""Bounded, counterexample-guided enumeration of exact scalar expression trees.

The specification is an opaque positional callable. Names are metadata only:
there are no named-goal recipes or special cases. Primitive applications are
enumerated by tree size; equal sampled signatures retain the first smallest
representative. Every prospective match receives independent exhaustive
verification. A counterexample adds a sample and restarts enumeration from
scratch, so sampled equivalence never survives contradictory full-domain
observations.

The grammar contains one or two input leaves and the supplied unary/binary
primitives, with no implicit constants or conditionals. Repeated structural
subexpressions become shared instructions in the returned straight-line plan.
The bound counts applications in the expression tree before sharing. This is
not unrestricted DAG synthesis or a claim of globally optimal target cost.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import random
from typing import Callable

from .model import Instruction, Plan, Primitive, PrimitiveLibrary
from .verify import verify_plan


_GRAMMAR = (
    "input leaves; supplied unary/binary primitives; bounded expression-tree "
    "applications before structural sharing; no implicit constants or conditionals"
)
_MAX_INSTRUCTIONS = 8
_STATUS_DETAIL = {
    "found": "A candidate passed exhaustive verification over the declared input domain.",
    "instruction_bound": "No candidate was found within the declared expression-tree instruction bound.",
    "candidate_budget": "The candidate budget ended before bounded enumeration completed.",
}


@dataclass(frozen=True)
class SearchResult:
    status: str
    plan: Plan | None
    verification: dict | None
    name: str | None
    input_ranges: tuple[tuple[int, int], ...]
    max_instructions: int
    max_candidates: int
    candidates: int
    restarts: int
    sample_count: int
    verification_attempts: int
    counterexamples: tuple[tuple[int, ...], ...]
    library_fingerprint: str
    largest_tree_size_considered: int

    @property
    def found(self) -> bool:
        return self.status == "found"

    @property
    def rounds(self) -> int:
        return self.restarts + 1

    def manifest(self) -> dict:
        return {
            "format": "bounded-scalar-synthesis-1", "status": self.status,
            "detail": _STATUS_DETAIL[self.status], "name": self.name,
            "grammar": _GRAMMAR,
            "bound_metric": "primitive applications in expression tree before sharing",
            "prototype_max_instructions": _MAX_INSTRUCTIONS,
            "optimality": "no global native-instruction or target-cost optimality claim",
            "input_ranges": [list(bounds) for bounds in self.input_ranges],
            "max_instructions": self.max_instructions, "max_candidates": self.max_candidates,
            "candidates": self.candidates, "rounds": self.rounds, "restarts": self.restarts,
            "sample_count": self.sample_count, "verification_attempts": self.verification_attempts,
            "counterexamples": [list(values) for values in self.counterexamples],
            "library_fingerprint": self.library_fingerprint,
            "largest_tree_size_considered": self.largest_tree_size_considered,
            "plan": self.plan.manifest() if self.plan is not None else None,
            "verification": self.verification,
        }


@dataclass(frozen=True, eq=False)
class _Expression:
    serial: int
    signature: bytes
    primitive: Primitive | None = None
    operands: tuple[_Expression, ...] = ()
    input_index: int | None = None


def _ranges(input_ranges) -> tuple[tuple[int, int], ...]:
    if isinstance(input_ranges, Mapping):
        if any(not isinstance(name, str) or not name for name in input_ranges):
            raise ValueError("Input range names must be nonempty strings")
        input_ranges = input_ranges.values()
    try:
        ranges = tuple(tuple(bounds) for bounds in input_ranges)
    except TypeError as error:
        raise ValueError("Input ranges must be an ordered collection of inclusive bounds") from error
    if not 1 <= len(ranges) <= 2:
        raise ValueError("Scalar synthesis supports one or two inputs")
    if any(len(bounds) != 2 or any(type(value) is not int for value in bounds)
           or not 0 <= bounds[0] <= bounds[1] <= 255 for bounds in ranges):
        raise ValueError("Input ranges must be inclusive integer bounds within [0,255]")
    return ranges


def _initial_samples(ranges: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
    size = 1
    for lo, hi in ranges:
        size *= hi - lo + 1
    if size <= 64:
        return list(product(*(range(lo, hi + 1) for lo, hi in ranges)))
    anchors = [tuple(sorted({lo, hi, (lo + hi) // 2, min(lo + 1, hi), max(hi - 1, lo),
                             min(hi, max(lo, 127)), min(hi, max(lo, 128))}))
               for lo, hi in ranges]
    samples = list(product(*anchors))
    seen = set(samples)
    rng = random.Random(0)
    for _ in range(24):
        values = tuple(rng.randint(lo, hi) for lo, hi in ranges)
        if values not in seen:
            samples.append(values)
            seen.add(values)
    return samples


def _spec_value(function: Callable, values: tuple[int, ...]) -> int:
    try:
        result = function(*values)
    except Exception as error:
        raise ValueError(f"Specification failed for inputs {values!r}: {error}") from error
    if type(result) not in (int, bool) or not 0 <= result <= 255:
        raise ValueError(f"Specification must return an integer u8 value; inputs={values!r}, result={result!r}")
    return int(result)


def _to_plan(expression: _Expression, input_count: int) -> Plan:
    instructions: list[Instruction] = []
    assigned: dict[int, int] = {}

    def emit(node: _Expression) -> int:
        if node.input_index is not None:
            return node.input_index
        if node.serial not in assigned:
            operands = tuple(emit(operand) for operand in node.operands)
            assert node.primitive is not None
            assigned[node.serial] = input_count + len(instructions)
            instructions.append(Instruction(node.primitive.name, operands))
        return assigned[node.serial]

    output = emit(expression)
    return Plan(input_count, tuple(instructions), output)


def synthesize(function: Callable, *, library: PrimitiveLibrary,
               input_ranges=((0, 255), (0, 255)), max_instructions: int = 3,
               max_candidates: int = 50000, name: str | None = None) -> SearchResult:
    """Search a finite grammar and return an exhaustively verified plan or bound status.

    Candidate counts include evaluated input leaves and primitive expressions,
    including signatures discarded as duplicates and work repeated after a
    counterexample. The budget is global across all refinement rounds. Pure,
    deterministic specification callables are required; Boolean outputs become
    integer 0/1. The callable's name and source are never inspected. This
    prototype accepts tree instruction bounds from zero through eight.
    """
    if not callable(function):
        raise ValueError("Specification must be callable")
    if not isinstance(library, PrimitiveLibrary):
        raise ValueError("library must be an explicit PrimitiveLibrary")
    for label, value in (("max_instructions", max_instructions), ("max_candidates", max_candidates)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if max_instructions > _MAX_INSTRUCTIONS:
        raise ValueError(f"Prototype max_instructions must be at most {_MAX_INSTRUCTIONS}; larger tree bounds are unsupported")
    if name is not None and not isinstance(name, str):
        raise ValueError("name is optional string metadata")
    ranges = _ranges(input_ranges)
    samples = _initial_samples(ranges)
    seen_samples = set(samples)
    goal_cache: dict[tuple[int, ...], int] = {}
    candidates = restarts = verification_attempts = largest_tree_size = 0
    counterexamples: list[tuple[int, ...]] = []
    last_verification = None
    primitives = tuple(library)
    fingerprint = library.fingerprint()

    def result(status: str, plan: Plan | None = None) -> SearchResult:
        return SearchResult(status, plan, last_verification, name, ranges,
                            max_instructions, max_candidates, candidates, restarts,
                            len(samples), verification_attempts, tuple(counterexamples),
                            fingerprint, largest_tree_size)

    if max_candidates == 0:
        return result("candidate_budget")

    while True:
        for values in samples:
            if values not in goal_cache:
                goal_cache[values] = _spec_value(function, values)
        desired = bytes(goal_cache[values] for values in samples)
        layers: list[list[_Expression]] = [[] for _ in range(max_instructions + 1)]
        representatives: dict[bytes, _Expression] = {}
        restart = False

        def consider(expression: _Expression, tree_size: int) -> Plan | None:
            nonlocal restart, last_verification, verification_attempts
            if expression.signature in representatives:
                return None
            representatives[expression.signature] = expression
            layers[tree_size].append(expression)
            if expression.signature != desired:
                return None
            plan = _to_plan(expression, len(ranges))
            verification_attempts += 1
            last_verification = verify_plan(plan, function, ranges, library=library)
            if last_verification["verified"]:
                return plan
            counterexample = tuple(last_verification["counterexample"]["inputs"])
            if counterexample in seen_samples:
                raise ValueError("Specification or primitive semantics changed during synthesis: existing sample failed verification")
            samples.append(counterexample)
            seen_samples.add(counterexample)
            counterexamples.append(counterexample)
            restart = True
            return None

        for index in range(len(ranges)):
            if candidates >= max_candidates:
                return result("candidate_budget")
            candidates += 1
            expression = _Expression(len(representatives), bytes(values[index] for values in samples), input_index=index)
            plan = consider(expression, 0)
            if plan is not None:
                return result("found", plan)
            if restart:
                break

        for tree_size in range(1, max_instructions + 1):
            if restart:
                break
            for primitive in primitives:
                arity = len(primitive.weights)
                commutative = arity == 2 and primitive.weights[0] == primitive.weights[1]
                partitions = ((tree_size - 1,),) if arity == 1 else (
                    (left, tree_size - 1 - left) for left in range(tree_size))
                for partition in partitions:
                    for operands in product(*(layers[size] for size in partition)):
                        if commutative and operands[0].serial > operands[1].serial:
                            continue
                        if candidates >= max_candidates:
                            return result("candidate_budget")
                        candidates += 1
                        largest_tree_size = max(largest_tree_size, tree_size)
                        signature = bytes(Primitive.evaluate(primitive, *(operand.signature[index] for operand in operands))
                                          for index in range(len(samples)))
                        expression = _Expression(len(representatives), signature, primitive, operands)
                        plan = consider(expression, tree_size)
                        if plan is not None:
                            return result("found", plan)
                        if restart:
                            break
                    if restart:
                        break
                if restart:
                    break
        if not restart:
            return result("instruction_bound")
        restarts += 1
