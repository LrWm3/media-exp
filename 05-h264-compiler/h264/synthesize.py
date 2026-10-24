"""Bounded exact search for binary functions of one or two u8 samples.

This is a finite projection/window grammar, not arbitrary superoptimization.
Costs are searched in order, but neither failure nor a returned cost proves
anything about programs outside this grammar:

* One instruction: any signed-byte P weight; B weights from
  ``ONE_STEP_BINARY_WEIGHTS``; all denominators and all useful signed offsets.
* Two instructions: a rounded scalar projection with coefficients in [-4, 4],
  clipped at a target-derived offset, followed by P(weight=-1, offset=1).
* Three instructions: two unrounded, clipped versions of such a projection,
  followed by B with positive first weight 1..4, negative second weight
  -128..-1, offset zero, and denominator 0..7. Hinge offsets lie within two
  integers of the target window's endpoints, adjusted for final rounding.

The final B weight is fitted by integer inequalities, rather than enumeration.
Every candidate is additionally evaluated over the entire Cartesian domain.
Only 1..2 input ranges and at most 64 domain points are supported. The search
does not use the graph, operation names, or application-specific patterns.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product


MAX_DOMAIN_SIZE = 64
MAX_INPUTS = 2
MAX_INSTRUCTIONS = 3
CACHE_SIZE = 512
PROJECTION_BOUND = 4
HINGE_RADIUS = 2
ONE_STEP_BINARY_WEIGHTS = tuple(sorted({*range(-8, 9), -128, -64, -32, -16,
                                        16, 32, 64, 127}))
_HINGE_DELTAS = (0, -1, 1, -2, 2)


@dataclass(frozen=True, slots=True)
class Step:
    """A weighted P (one source) or B (two sources) prediction.

    Indices address original inputs, then preceding results. ``bias`` is the
    signed offset placed on *each* source, so its effective B offset is also
    ``bias``. Weights are hardware weights, including B's extra factor of two
    in its divisor. Rounding precedes bias, and clipping follows bias.
    """

    inputs: tuple[int, ...]
    weights: tuple[int, ...]
    bias: int = 0
    denominator: int = 0

    def __post_init__(self):
        if (not isinstance(self.inputs, tuple) or not 1 <= len(self.inputs) <= 2
                or any(type(index) is not int or index < 0 for index in self.inputs)):
            raise ValueError("Step inputs must be one or two nonnegative integer indices")
        if (not isinstance(self.weights, tuple) or len(self.weights) != len(self.inputs)
                or any(type(weight) is not int or not -128 <= weight <= 127
                       for weight in self.weights)):
            raise ValueError("Step weights must be matching signed-byte integers")
        if type(self.bias) is not int or not -128 <= self.bias <= 127:
            raise ValueError("Step bias must be a signed-byte integer")
        if type(self.denominator) is not int or not 0 <= self.denominator <= 7:
            raise ValueError("Step denominator must be an integer from 0 through 7")


def _clip(value):
    return max(0, min(255, value))


def _rounded(value, denominator, sources):
    shift = denominator + (sources == 2)
    return (value + (1 << (shift - 1) if shift else 0)) >> shift


def evaluate_plan(plan: tuple[Step, ...], values: tuple[int, ...]) -> int:
    """Return the final sample using exact integer H.264 P/B semantics.

    Negative numerators use arithmetic right shift; clipping intermediate
    products or sums would be incorrect. Empty plans have no result.
    """
    available = list(values)
    if not available or any(type(value) is not int or not 0 <= value <= 255
                            for value in available):
        raise ValueError("Plan inputs must be u8 integers")
    if not plan:
        raise ValueError("An empty plan has no output")
    for step in plan:
        if not isinstance(step, Step):
            raise ValueError("Plans must contain Step instances")
        if any(index >= len(available) for index in step.inputs):
            raise ValueError("Step references an unavailable input or result")
        total = sum(available[index] * weight
                    for index, weight in zip(step.inputs, step.weights))
        available.append(_clip(_rounded(total, step.denominator, len(step.inputs))
                               + step.bias))
    return available[-1]


def _verified(plan, domain, target):
    return all(evaluate_plan(plan, values) == expected
               for values, expected in zip(domain, target))


def _fit_prediction(inputs, columns, weight_vectors, domain, target):
    # A positive output must be exactly one before clipping. Thus one positive
    # point determines the only possible offset for each weight/denominator.
    positive = tuple(index for index, value in enumerate(target) if value)
    negative = tuple(index for index, value in enumerate(target) if not value)
    anchor = positive[0]
    for weights in weight_vectors:
        totals = tuple(sum(weight * column[index]
                           for weight, column in zip(weights, columns))
                       for index in range(len(target)))
        for denominator in range(8):
            quantized = _rounded(totals[anchor], denominator, len(inputs))
            bias = 1 - quantized
            if not -128 <= bias <= 127:
                continue
            if any(_rounded(totals[index], denominator, len(inputs)) != quantized
                   for index in positive[1:]):
                continue
            if any(_rounded(totals[index], denominator, len(inputs)) + bias > 0
                   for index in negative):
                continue
            plan = (Step(inputs, weights, bias, denominator),)
            if _verified(plan, domain, target):
                return plan
    return None


def _projection_step(coefficients, bias, denominator=0):
    inputs = tuple(index for index, weight in enumerate(coefficients) if weight)
    multiplier = 2 if len(inputs) == 2 else 1
    weights = tuple(multiplier * coefficients[index] for index in inputs)
    return Step(inputs, weights, bias, denominator)


def _windows(domain, target):
    """Yield only projections whose truth labels form one contiguous window."""
    coefficients = product(range(-PROJECTION_BOUND, PROJECTION_BOUND + 1),
                           repeat=len(domain[0]))
    ordered = sorted((weights for weights in coefficients if any(weights)),
                     key=lambda weights: (sum(map(abs, weights)),
                                          sum(weight < 0 for weight in weights),
                                          tuple(-weight for weight in weights)))
    seen = set()
    for weights in ordered:
        projected = tuple(sum(weight * value for weight, value in zip(weights, values))
                          for values in domain)
        if projected in seen:
            continue
        seen.add(projected)
        first = min(value for value, label in zip(projected, target) if label)
        last = max(value for value, label in zip(projected, target) if label)
        if all((first <= value <= last) == bool(label)
               for value, label in zip(projected, target)):
            yield weights, projected, first, last


def _two_steps(windows, domain, target):
    for weights, projected, first, _ in windows:
        # Negated projections cover suffixes as well. Interior windows require
        # both hinges, and cannot pass this monotone two-instruction grammar.
        if first != min(projected):
            continue
        for denominator in range(8):
            quantized = tuple(_rounded(value, denominator, 1) for value in projected)
            bias = -max(value for value, label in zip(quantized, target) if label)
            if not -128 <= bias <= 127:
                continue
            if any(value + bias <= 0 for value, label in zip(quantized, target)
                   if not label):
                continue
            plan = (_projection_step(weights, bias, denominator),
                    Step((len(domain[0]),), (-1,), 1))
            if _verified(plan, domain, target):
                return plan
    return None


def _ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def _fit_negative_weight(left, right, target, divisor, positive_weight):
    """Fit k in [1,128] for clip((p*left-k*right+divisor/2)//divisor)."""
    lo, hi = 1, 128
    rounding = divisor // 2
    positive_min, positive_max = divisor - rounding, 2 * divisor - 1 - rounding
    negative_max = divisor - 1 - rounding
    for left_value, right_value, label in zip(left, right, target):
        value = positive_weight * left_value
        if not right_value:
            if (label and not positive_min <= value <= positive_max
                    or not label and value > negative_max):
                return None
            continue
        if label:
            lo = max(lo, _ceil_div(value - positive_max, right_value))
            hi = min(hi, (value - positive_min) // right_value)
        else:
            lo = max(lo, _ceil_div(value - negative_max, right_value))
        if lo > hi:
            return None
    return lo


def _three_steps(windows, domain, target):
    result_index = len(domain[0])
    for weights, projected, first, last in windows:
        for denominator in range(8):
            divisor = 1 << (denominator + 1)
            for positive_weight in range(1, PROJECTION_BOUND + 1):
                anchor = _ceil_div(divisor // 2, positive_weight) - first
                for left_delta in _HINGE_DELTAS:
                    left_bias = anchor + left_delta
                    if not -128 <= left_bias <= 127:
                        continue
                    left = tuple(_clip(value + left_bias) for value in projected)
                    for right_delta in _HINGE_DELTAS:
                        right_bias = -last + right_delta
                        if not -128 <= right_bias <= 127:
                            continue
                        right = tuple(_clip(value + right_bias) for value in projected)
                        negative_weight = _fit_negative_weight(
                            left, right, target, divisor, positive_weight)
                        if negative_weight is None:
                            continue
                        plan = (_projection_step(weights, left_bias),
                                _projection_step(weights, right_bias),
                                Step((result_index, result_index + 1),
                                     (positive_weight, -negative_weight), 0, denominator))
                        if _verified(plan, domain, target):
                            return plan
    return None


@lru_cache(maxsize=CACHE_SIZE)
def _synthesize(ranges, target, max_instructions):
    if not max_instructions:
        return None
    domain = tuple(product(*(range(lo, hi + 1) for lo, hi in ranges)))
    if len(set(target)) == 1:
        plan = (Step((0,), (0,), target[0]),)
        return plan if _verified(plan, domain, target) else None
    columns = tuple(zip(*domain))
    for index, column in enumerate(columns):
        plan = _fit_prediction((index,), (column,),
                               ((weight,) for weight in range(-128, 128)), domain, target)
        if plan is not None:
            return plan
    if len(ranges) == 2:
        plan = _fit_prediction((0, 1), columns,
                               product(ONE_STEP_BINARY_WEIGHTS, repeat=2), domain, target)
        if plan is not None:
            return plan
    if max_instructions == 1:
        return None
    windows = tuple(_windows(domain, target))
    plan = _two_steps(windows, domain, target)
    if plan is not None or max_instructions == 2:
        return plan
    return _three_steps(windows, domain, target)


def synthesize(ranges: tuple[tuple[int, int], ...], target: tuple[int, ...],
               max_instructions: int = 3) -> tuple[Step, ...] | None:
    """Find an exact plan within a bounded grammar, or return ``None``.

    Target entries follow ``itertools.product`` order, with the last input
    varying fastest. Invalid ranges, nonbinary targets, and domains larger
    than ``MAX_DOMAIN_SIZE`` raise ``ValueError``. A zero budget returns None;
    budgets above three still search only the documented three cost layers.
    Results are immutable and cached (including failures), up to CACHE_SIZE
    distinct range/table/budget keys. ``cache_info`` and ``cache_clear`` expose
    ordinary functools LRU diagnostics. No empty/alias plans are returned.
    """
    if not isinstance(ranges, (tuple, list)) or not 1 <= len(ranges) <= MAX_INPUTS:
        raise ValueError("Synthesis requires one or two input ranges")
    domain_size = 1
    normalized = []
    for pair in ranges:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or not 0 <= pair[0] <= pair[1] <= 255):
            raise ValueError("Each input range must be inclusive u8 integer bounds")
        domain_size *= pair[1] - pair[0] + 1
        normalized.append(tuple(pair))
    if domain_size > MAX_DOMAIN_SIZE:
        raise ValueError(f"Synthesis domain exceeds {MAX_DOMAIN_SIZE} points")
    if (not isinstance(target, (tuple, list)) or len(target) != domain_size
            or any(type(value) not in (int, bool) or value not in (0, 1) for value in target)):
        raise ValueError("Target must contain one binary value per domain point")
    if type(max_instructions) is not int or max_instructions < 0:
        raise ValueError("Instruction budget must be a nonnegative integer")
    return _synthesize(tuple(normalized), tuple(map(int, target)),
                       min(max_instructions, MAX_INSTRUCTIONS))


synthesize.cache_info = _synthesize.cache_info
synthesize.cache_clear = _synthesize.cache_clear
