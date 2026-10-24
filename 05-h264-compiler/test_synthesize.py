"""Proof checks for the bounded, application-independent prediction grammar."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import product
from math import floor
import unittest

from h264.synthesize import (CACHE_SIZE, MAX_DOMAIN_SIZE, Step, evaluate_plan,
                             synthesize)


class SynthesizeTests(unittest.TestCase):
    def assert_exact(self, ranges, predicate, budget=3):
        domain = tuple(product(*(range(lo, hi + 1) for lo, hi in ranges)))
        target = tuple(int(predicate(*values)) for values in domain)
        plan = synthesize(ranges, target, budget)
        self.assertIsNotNone(plan)
        self.assertLessEqual(len(plan), budget)
        self.assertEqual(tuple(evaluate_plan(plan, values) for values in domain), target)
        return plan

    def test__given_all_binary_truth_tables__when_searching__then_every_plan_is_exact(self):
        domain = tuple(product(range(2), repeat=2))
        for target in product(range(2), repeat=4):
            with self.subTest(target=target):
                plan = synthesize(((0, 1), (0, 1)), target)
                self.assertIsNotNone(plan)
                self.assertEqual(tuple(evaluate_plan(plan, values) for values in domain), target)

    def test__given_projected_window__when_searching__then_three_predictions_suffice(self):
        ranges = ((0, 8), (0, 1))
        predicate = lambda x, y: x == 3 or x == 2 and y == 1
        plan = self.assert_exact(ranges, predicate)
        self.assertEqual(len(plan), 3)
        domain = tuple(product(range(9), range(2)))
        target = tuple(int(predicate(*values)) for values in domain)
        self.assertIsNone(synthesize(ranges, target, 2))

    def test__given_unary_equalities_and_windows__when_searching__then_all_domain_points_match(self):
        for lower in range(9):
            for upper in range(lower, 9):
                with self.subTest(lower=lower, upper=upper):
                    self.assert_exact(((0, 8),), lambda x: lower <= x <= upper)
        self.assert_exact(((40, 55),), lambda x: 44 <= x <= 48)
        self.assert_exact(((0, MAX_DOMAIN_SIZE - 1),), lambda x: 12 <= x <= 43)

    def test__given_nontrivial_projection_coefficients__when_searching__then_windows_are_generic(self):
        cases = (
            (((0, 7), (0, 3)), lambda x, y: 4 <= x + 2 * y <= 7),
            (((0, 3), (0, 3)), lambda x, y: -1 <= 3 * x - 2 * y <= 2),
            (((0, 3), (0, 3)), lambda x, y: x == y),
        )
        for ranges, predicate in cases:
            with self.subTest(ranges=ranges, predicate=predicate):
                self.assert_exact(ranges, predicate)

    def test__given_simple_gates__when_searching_one_step__then_cost_budget_is_respected(self):
        for predicate in (lambda x, y: x | y, lambda x, y: x & y,
                          lambda x, y: not x, lambda x, y: x <= y):
            self.assert_exact(((0, 1), (0, 1)), predicate, 1)
        self.assert_exact(((0, 8),), lambda x: x >= 3, 2)
        self.assert_exact(((192, 255),), lambda x: x >= 201, 2)
        self.assertIsNone(synthesize(((0, 1),), (0, 1), 0))
        self.assertEqual(len(synthesize(((0, 1),), (0, 1), 100)), 1)

    def test__given_signed_predictions__when_evaluating__then_rounding_precedes_bias_and_clip(self):
        samples = (0, 1, 2, 127, 128, 254, 255)
        for sources in (1, 2):
            for denominator, weights, bias in product(
                    range(8), product((-128, -3, -1, 0, 1, 3, 127), repeat=sources),
                    (-128, -1, 0, 1, 127)):
                step = Step(tuple(range(sources)), weights, bias, denominator)
                for values in product(samples, repeat=sources):
                    total = sum(value * weight for value, weight in zip(values, weights))
                    shift = denominator + (sources == 2)
                    rounding = Fraction(1, 2) if shift else 0
                    expected = max(0, min(255, floor(Fraction(total, 1 << shift) + rounding) + bias))
                    self.assertEqual(evaluate_plan((step,), values), expected)
        # Early clipping of the negative product would change this result.
        self.assertEqual(evaluate_plan((Step((0,), (-1,), 1, 1),), (1,)), 1)
        self.assertEqual(evaluate_plan((Step((0, 1), (1, -4), 0, 1),), (5, 2)), 0)

    def test__given_clipped_intermediates__when_evaluating__then_later_steps_use_clipped_values(self):
        plan = (Step((0,), (127,), 127), Step((1,), (-1,), 127))
        self.assertEqual(evaluate_plan(plan, (255,)), 0)
        self.assertEqual(evaluate_plan(plan, (0,)), 0)
        plan = (Step((0,), (-128,), -128), Step((1,), (-1,), 1))
        self.assertEqual(evaluate_plan(plan, (255,)), 1)

    def test__given_repeated_tables__when_searching__then_results_are_cached_and_immutable(self):
        synthesize.cache_clear()
        plan = synthesize(((0, 2),), (0, 1, 0))
        self.assertIs(plan, synthesize(((0, 2),), (0, 1, 0)))
        self.assertEqual(synthesize.cache_info().hits, 1)
        self.assertEqual(synthesize.cache_info().maxsize, CACHE_SIZE)
        self.assertIsInstance(plan, tuple)
        with self.assertRaises(FrozenInstanceError):
            plan[0].bias = 0
        unsupported = (1, 0, 1, 0, 1)
        self.assertIsNone(synthesize(((0, 4),), unsupported))
        before = synthesize.cache_info().hits
        self.assertIsNone(synthesize(((0, 4),), unsupported))
        self.assertEqual(synthesize.cache_info().hits, before + 1)

    def test__given_cache_eviction__when_searching_again__then_deterministic_plans_remain_exact(self):
        synthesize.cache_clear()
        original = synthesize(((0, 8),), tuple(int(2 <= x <= 5) for x in range(9)))
        for value, label in product(range(256), range(2)):
            synthesize(((value, value),), (label,))
        self.assertEqual(synthesize.cache_info().currsize, CACHE_SIZE)
        plan = self.assert_exact(((0, 8),), lambda x: 2 <= x <= 5)
        self.assertEqual(plan, original)

    def test__given_malformed_problem__when_searching__then_inputs_are_rejected(self):
        cases = (
            ((), ()), (None, ()), (((0, 1),) * 3, (0,) * 8),
            (((1, 0),), ()), (((-1, 0),), (0, 1)), (((0, 256),), (0,) * 257),
            (((0, 1.0),), (0, 1)), (((False, 1),), (0, 1)),
            (((0, 64),), (0,) * 65), (((0, 8), (0, 8)), (0,) * 81),
            (((0, 1),), (0,)), (((0, 1),), (0, 2)), (((0, 1),), (0, 0.5)),
            (((0, 1),), None),
        )
        for ranges, target in cases:
            with self.subTest(ranges=ranges, target=target):
                with self.assertRaises(ValueError):
                    synthesize(ranges, target)
        for budget in (-1, True, 1.5, None):
            with self.assertRaises(ValueError):
                synthesize(((0, 1),), (0, 1), budget)

    def test__given_malformed_plan__when_constructing_or_evaluating__then_inputs_are_rejected(self):
        cases = (
            ((), ()), ((-1,), (1,)), ((True,), (1,)), ((0, 1, 2), (1, 1, 1)),
            ((0,), (128,)), ((0,), (-129,)), ((0,), (1.0,)), ((0,), ()),
            ((0,), (1,), 128), ((0,), (1,), 0, 8), ((0,), (1,), 0, True),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    Step(*arguments)
        for plan, values in (((), (1,)), ((Step((1,), (1,)),), (1,)),
                             ((Step((0,), (1,)),), (-1,)), (("bad",), (1,))):
            with self.assertRaises(ValueError):
                evaluate_plan(plan, values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
