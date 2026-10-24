"""Immutable, exact scalar primitive descriptions and straight-line programs.

These models describe pointwise unsigned-byte arithmetic. Motion, block layout,
reference allocation, input loading, and bitstream validity are separate target
obligations. Every divide uses mathematical floor, including negative values.
"""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer(value, name, lower=None, upper=None):
    if type(value) is not int or (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"Invalid integer {name}: {value!r}")
    return value


@dataclass(frozen=True)
class Primitive:
    """clip8(floor((sum(weights[i]*x[i])+round)/2**shift)+bias).

    ``round`` is zero for shift=0, otherwise 2**(shift-1). A primitive is an
    explicit instance of arithmetic, not a target-specific named-goal recipe.
    Cost is a declared estimate; it does not imply a global search optimum.
    """

    name: str
    weights: tuple[int, ...]
    bias: int = 0
    shift: int = 0
    cost: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Primitive names must be identifiers")
        try:
            weights = tuple(self.weights)
        except TypeError as error:
            raise ValueError("Primitive weights must be iterable") from error
        if not 1 <= len(weights) <= 2:
            raise ValueError("Scalar prototype supports unary and binary primitives")
        for weight in weights:
            _integer(weight, "weight")
        object.__setattr__(self, "weights", weights)
        _integer(self.bias, "bias")
        _integer(self.shift, "shift", 0, 31)
        _integer(self.cost, "cost", 1)

    @property
    def arity(self):
        return len(self.weights)

    @property
    def commutative(self):
        return self.arity == 2 and self.weights[0] == self.weights[1]

    def evaluate(self, *values):
        if len(values) != len(self.weights):
            raise ValueError(f"{self.name} needs {len(self.weights)} operands")
        for value in values:
            _integer(value, "primitive operand", 0, 255)
        total = sum(weight * value for weight, value in zip(self.weights, values))
        if self.shift:
            total = (total + (1 << (self.shift - 1))) // (1 << self.shift)
        return min(255, max(0, total + self.bias))

    def manifest(self):
        return {"name": self.name, "arity": self.arity, "weights": list(self.weights),
                "bias_after_rounding": self.bias, "divisor_shift": self.shift,
                "rounding": "floor(n/2**shift + 1/2) when shift>0; exact otherwise",
                "saturation": [0, 255], "integer_precision": "unbounded",
                "cost": self.cost, "commutative": self.commutative}


@dataclass(frozen=True)
class PrimitiveLibrary:
    target: str
    primitives: tuple[Primitive, ...]
    version: str = "scalar-u8-v1"

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Primitive library needs a target identifier")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Primitive library needs a semantic version")
        try:
            primitives = tuple(self.primitives)
        except TypeError as error:
            raise ValueError("Primitives must be iterable") from error
        if any(not isinstance(primitive, Primitive) for primitive in primitives):
            raise ValueError("Library entries must be exact Primitive descriptions")
        # Accept descriptions, never executable overrides hidden behind the
        # same manifest/fingerprint (e.g. a subclass with a different evaluator).
        primitives = tuple(Primitive(p.name, p.weights, p.bias, p.shift, p.cost) for p in primitives)
        if len({primitive.name for primitive in primitives}) != len(primitives):
            raise ValueError("Primitive names must be unique")
        object.__setattr__(self, "primitives", primitives)

    def __iter__(self):
        return iter(self.primitives)

    @property
    def by_name(self):
        return MappingProxyType({primitive.name: primitive for primitive in self.primitives})

    def manifest(self):
        return {"target": self.target, "version": self.version,
                "scope": "pointwise scalar u8; operands at identical image coordinates",
                "reference_model": "previous instruction values; physical scheduling checked separately",
                "primitives": [primitive.manifest() for primitive in self.primitives]}

    def fingerprint(self):
        return fingerprint(self.manifest())


@dataclass(frozen=True)
class Instruction:
    primitive: str
    operands: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.primitive, str) or not self.primitive:
            raise ValueError("Instruction needs a primitive identifier")
        try:
            operands = tuple(self.operands)
        except TypeError as error:
            raise ValueError("Instruction operands must be iterable") from error
        for operand in operands:
            _integer(operand, "operand index", 0)
        object.__setattr__(self, "operands", operands)

    def manifest(self):
        return {"primitive": self.primitive, "operands": list(self.operands)}


@dataclass(frozen=True)
class Plan:
    input_count: int
    instructions: tuple[Instruction, ...]
    output: int

    def __post_init__(self):
        _integer(self.input_count, "input count", 1, 2)
        _integer(self.output, "output index", 0)
        try:
            instructions = tuple(self.instructions)
        except TypeError as error:
            raise ValueError("Plan instructions must be iterable") from error
        if any(not isinstance(instruction, Instruction) for instruction in instructions):
            raise ValueError("Plan entries must be Instructions")
        object.__setattr__(self, "instructions", instructions)

    def manifest(self):
        return {"format": "scalar-plan-1", "input_count": self.input_count,
                "instructions": [instruction.manifest() for instruction in self.instructions],
                "output": self.output}

    def fingerprint(self):
        return fingerprint(self.manifest())


def validate_plan(plan, library):
    if not isinstance(plan, Plan) or not isinstance(library, PrimitiveLibrary):
        raise ValueError("Expected a Plan and PrimitiveLibrary")
    primitives = library.by_name
    for index, instruction in enumerate(plan.instructions, plan.input_count):
        primitive = primitives.get(instruction.primitive)
        if primitive is None:
            raise ValueError(f"Unknown primitive: {instruction.primitive}")
        if len(instruction.operands) != primitive.arity:
            raise ValueError(f"Wrong operand count for {primitive.name}")
        if any(operand >= index for operand in instruction.operands):
            raise ValueError("Plan operands must refer to inputs or earlier instructions")
    if plan.output >= plan.input_count + len(plan.instructions):
        raise ValueError("Plan output is undefined")
    return plan


def evaluate_plan(plan, values, library):
    validate_plan(plan, library)
    values = list(values)
    if len(values) != plan.input_count:
        raise ValueError("Plan input count differs from supplied values")
    for value in values:
        _integer(value, "plan input", 0, 255)
    primitives = library.by_name
    for instruction in plan.instructions:
        values.append(Primitive.evaluate(primitives[instruction.primitive],
                                         *(values[index] for index in instruction.operands)))
    return values[plan.output]


# This is the complete initial grammar. No min, max, or Boolean witnesses are
# stored. Their implementations must be discovered from these arithmetic tools.
_ARITHMETIC = (Primitive("sat_add", (1, 1)), Primitive("sat_sub", (1, -1)),
               Primitive("rounded_average", (1, 1), shift=1))
_LIBRARIES = {
    "h264": PrimitiveLibrary("h264", _ARITHMETIC),
    "hevc": PrimitiveLibrary("hevc", _ARITHMETIC),
    "av1": PrimitiveLibrary("av1", (_ARITHMETIC[2],)),
}


def library_for(target):
    canonical = "hevc" if target == "h265" else target
    try:
        return _LIBRARIES[canonical]
    except KeyError as error:
        raise ValueError(f"No scalar primitive library for target {target!r}") from error
