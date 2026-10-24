"""Exhaustive scalar equivalence records, independent of synthesis traversal.

A verified record establishes equality with an opaque scalar specification over
exactly the recorded finite input domains, under the declared primitive model.
It does not establish codec bitstream conformance or decoder correctness.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import itertools
import json
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Plan, PrimitiveLibrary


VERIFICATION_SCHEMA = "codec-compute.scalar-verification.v1"


def _domains(input_ranges: object, count: int) -> list[dict[str, object]]:
    if isinstance(input_ranges, Mapping):
        entries = list(input_ranges.items())
        if any(not isinstance(name, str) or not name for name, _ in entries):
            raise ValueError("Input domain names must be nonempty strings")
    elif isinstance(input_ranges, Sequence) and not isinstance(input_ranges, (str, bytes)):
        entries = [(f"input{index}", bounds) for index, bounds in enumerate(input_ranges)]
    else:
        raise ValueError("Input domains require an ordered mapping or a sequence of bounds")
    if len(entries) != count:
        raise ValueError(f"Expected {count} inclusive input domains")
    result = []
    for name, bounds in entries:
        if (not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes))
                or len(bounds) != 2):
            raise ValueError(f"Input domain {name} requires inclusive (minimum, maximum) bounds")
        lo, hi = bounds
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= 255:
            raise ValueError(f"Input domain {name} must be an inclusive interval within [0, 255]")
        result.append({"name": name, "minimum": lo, "maximum": hi, "size": hi - lo + 1})
    return result


def _prepared_plan(plan: Plan, library: PrimitiveLibrary) -> tuple[
    tuple[tuple[int, ...], int, int, int, tuple[int, ...]], ...
]:
    """Validate references and snapshot descriptor arithmetic, never callbacks."""
    from .model import Plan, PrimitiveLibrary

    if not isinstance(plan, Plan) or not isinstance(library, PrimitiveLibrary):
        raise ValueError("Expected a Plan and PrimitiveLibrary")
    if type(plan.input_count) is not int or not 1 <= plan.input_count <= 2:
        raise ValueError("Exhaustive scalar verification supports one or two u8 inputs")
    registry = {}
    for primitive in library:
        if not isinstance(primitive.name, str) or not primitive.name or primitive.name in registry:
            raise ValueError("Primitive model IDs must be unique nonempty strings")
        weights = tuple(primitive.weights)
        if not 1 <= len(weights) <= 2 or any(type(weight) is not int for weight in weights):
            raise ValueError(f"Invalid primitive weights: {primitive.name}")
        if type(primitive.bias) is not int:
            raise ValueError(f"Invalid primitive bias: {primitive.name}")
        if type(primitive.shift) is not int or not 0 <= primitive.shift <= 31:
            raise ValueError(f"Invalid primitive divisor shift: {primitive.name}")
        registry[primitive.name] = (weights, primitive.bias, primitive.shift)
    prepared = []
    available = plan.input_count
    for instruction in plan.instructions:
        if not isinstance(instruction.primitive, str) or instruction.primitive not in registry:
            raise ValueError(f"Unknown primitive ID: {instruction.primitive!r}")
        weights, bias, shift = registry[instruction.primitive]
        operands = tuple(instruction.operands)
        if len(operands) != len(weights):
            raise ValueError(f"Primitive {instruction.primitive} requires {len(weights)} operands")
        if any(type(reference) is not int or not 0 <= reference < available for reference in operands):
            raise ValueError("Instruction operands must reference inputs or earlier instructions")
        rounding = (1 << (shift - 1)) if shift else 0
        prepared.append((weights, bias, rounding, 1 << shift, operands))
        available += 1
    if type(plan.output) is not int or not 0 <= plan.output < available:
        raise ValueError("Plan output must reference an input or an existing instruction")
    return tuple(prepared)


def _json_snapshot(value: object) -> object:
    """Snapshot manifests into portable, non-aliased JSON data."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Verification manifests must contain finite JSON data") from exc


def _u8(value: object, *, source: str, values: tuple[int, ...], allow_bool: bool = False) -> int:
    if allow_bool and type(value) is bool:
        return int(value)
    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError(f"{source} must return a u8 integer for inputs {values}; got {value!r}")
    return value


def verify_plan(plan: Plan, spec: Callable[..., int], input_ranges: object, *,
                library: PrimitiveLibrary) -> dict[str, object]:
    """Enumerate every domain tuple and record scalar-model equivalence.

    ``spec`` receives positional integers in domain order. Inclusive input
    ranges may be an ordered mapping or a sequence of ``(minimum, maximum)``
    pairs. Boolean specification results normalize to 0/1. Primitive arithmetic
    is independently calculated from declared weights, signed half-up division,
    post-division bias, and u8 clipping; executable evaluator overrides are ignored.
    Malformed models/plans/domains and invalid specification outputs raise
    ``ValueError`` rather than producing a false correctness certificate.

    Both passing and failing candidates are evaluated over the full Cartesian
    product. Digests therefore cover all states, even when an early mismatch
    exists. The first counterexample remains available to a CEGIS search.
    """
    if not callable(spec):
        raise ValueError("Scalar specification must be callable")
    prepared = _prepared_plan(plan, library)
    domains = _domains(input_ranges, plan.input_count)
    plan_manifest = _json_snapshot(plan.manifest())
    model_manifest = _json_snapshot(library.manifest())
    plan_fingerprint = plan.fingerprint()
    model_fingerprint = library.fingerprint()
    expected_digest, actual_digest = hashlib.sha256(), hashlib.sha256()
    first_counterexample = None
    mismatches = checked = 0
    ranges = [range(domain["minimum"], domain["maximum"] + 1) for domain in domains]
    for inputs in itertools.product(*ranges):
        try:
            expected_value = spec(*inputs)
        except Exception as exc:
            raise ValueError(f"Scalar specification failed for inputs {inputs}: {exc}") from exc
        expected = _u8(expected_value, source="Scalar specification", values=inputs, allow_bool=True)
        values = list(inputs)
        for weights, bias, rounding, divisor, operands in prepared:
            total = sum(weight * values[index] for weight, index in zip(weights, operands))
            # Python // is mathematical floor, including negative numerators.
            # Bias is added after rounding/division, and clipping occurs last.
            result = (total + rounding) // divisor + bias
            values.append(min(255, max(0, result)))
        actual = values[plan.output]
        expected_digest.update(bytes((expected,)))
        actual_digest.update(bytes((actual,)))
        checked += 1
        if expected != actual:
            mismatches += 1
            if first_counterexample is None:
                first_counterexample = {"inputs": list(inputs), "expected": expected, "actual": actual}
    total = prod(domain["size"] for domain in domains)
    verified = mismatches == 0
    return {
        "schema": VERIFICATION_SCHEMA,
        "method": "exhaustive_cartesian_enumeration",
        "scope": "scalar_primitive_model",
        "status": "verified" if verified else "failed",
        "verified": verified,
        "bitstream_verified": False,
        "target": library.target,
        "domains": domains,
        "input_order": [domain["name"] for domain in domains],
        "enumeration_order": "lexicographic_ascending_last_input_varies_fastest",
        "combinations_total": total,
        "combinations_checked": checked,
        "mismatch_count": mismatches,
        "primitive_model_fingerprint": model_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "primitive_model": model_manifest,
        "plan": plan_manifest,
        "hash_algorithm": "sha256",
        "hash_encoding": "one_u8_output_byte_per_domain_tuple",
        "hash_scope": "complete_domain",
        "expected_sha256": expected_digest.hexdigest(),
        "actual_sha256": actual_digest.hexdigest(),
        "counterexample": first_counterexample,
    }


def verify_record(plan: Plan, spec: Callable[..., int], input_ranges: object, *,
                  library: PrimitiveLibrary, record: Mapping[str, object]) -> dict[str, object]:
    """Recompute a record and reject any changed metadata or verification data.

    Fingerprints are checked against the supplied plan and primitive library;
    they are never accepted as substitutes for replaying every scalar state.
    This accepts a JSON round trip of either a passing or a failing record.
    ``spec`` remains a caller-supplied opaque callable, identified in the record
    by its complete domain output digest rather than a Python source hash.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Verification record must be a JSON object")
    saved = _json_snapshot(dict(record))
    _prepared_plan(plan, library)
    domains = _domains(input_ranges, plan.input_count)
    context = {
        "schema": VERIFICATION_SCHEMA,
        "target": library.target,
        "primitive_model_fingerprint": library.fingerprint(),
        "plan_fingerprint": plan.fingerprint(),
        "primitive_model": _json_snapshot(library.manifest()),
        "plan": _json_snapshot(plan.manifest()),
        "domains": domains,
    }
    for field, value in context.items():
        if saved.get(field) != value:
            raise ValueError(f"Verification record context mismatch: {field}")
    fresh = verify_plan(plan, spec, input_ranges, library=library)
    canonical_options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    if json.dumps(saved, **canonical_options) != json.dumps(fresh, **canonical_options):
        raise ValueError("Verification record does not match the recomputed exhaustive result")
    return fresh


__all__ = ["VERIFICATION_SCHEMA", "verify_plan", "verify_record"]
