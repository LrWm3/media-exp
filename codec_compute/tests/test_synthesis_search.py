"""Generic synthesis and counterexample refinement; no decoder execution here."""
import json
import unittest
from unittest.mock import patch

from codec_compute.synthesis.model import (
    Primitive, PrimitiveLibrary, evaluate_plan, library_for,
)
from codec_compute.synthesis.search import synthesize


class SynthesisSearchTests(unittest.TestCase):
    def test__given_full_u8_extrema__when_searching__then_two_instruction_plans_are_exhaustively_verified(self):
        for specification in (lambda a, b: min(a, b), lambda a, b: max(a, b)):
            with self.subTest(specification=specification):
                result = synthesize(specification, library=library_for("hevc"), max_instructions=2)
                self.assertTrue(result.found)
                self.assertLessEqual(len(result.plan.instructions), 2)
                self.assertTrue(result.verification["verified"])
                self.assertEqual(result.verification["combinations_checked"], 65536)
                self.assertEqual(result.verification["expected_sha256"], result.verification["actual_sha256"])
                self.assertEqual(result.verification["plan_fingerprint"], result.plan.fingerprint())

    def test__given_boolean_goals__when_searching_arithmetic_primitives__then_all_truth_tables_verify(self):
        for specification, bound in [(lambda a, b: bool(a and b), 2),
                                     (lambda a, b: a | b, 1), (lambda a, b: a ^ b, 3)]:
            with self.subTest(bound=bound):
                library = library_for("h264")
                result = synthesize(specification, library=library, input_ranges=((0, 1), (0, 1)),
                                    max_instructions=bound)
                self.assertEqual(result.status, "found")
                self.assertLessEqual(len(result.plan.instructions), bound)
                self.assertEqual(result.verification["combinations_checked"], 4)
                for a in (0, 1):
                    for b in (0, 1):
                        self.assertEqual(evaluate_plan(result.plan, (a, b), library), specification(a, b))

    def test__given_opaque_custom_goal__when_metadata_changes__then_search_never_reads_names_or_source(self):
        class Opaque:
            @property
            def __name__(self):
                raise AssertionError("search must not inspect specification names")

            @property
            def __code__(self):
                raise AssertionError("search must not inspect specification source")

            def __call__(self, a, b):
                return abs(a - b)

        library = library_for("hevc")
        first = synthesize(Opaque(), library=library, name="minimum")
        second = synthesize(Opaque(), library=library, name="unrelated_label")
        self.assertTrue(first.found)
        self.assertEqual(len(first.plan.instructions), 3)
        self.assertEqual(first.plan, second.plan)
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.verification, second.verification)
        self.assertEqual(first.verification["combinations_checked"], 65536)

    def test__given_inadequate_average_only_grammar__when_bounds_exhaust__then_result_reports_only_bounded_failure(self):
        result = synthesize(lambda a, b: min(a, b), library=library_for("av1"), max_instructions=3)
        self.assertEqual(result.status, "instruction_bound")
        self.assertFalse(result.found)
        self.assertIsNone(result.plan)
        self.assertLess(result.candidates, result.max_candidates)
        self.assertIn("within", result.manifest()["detail"])
        self.assertIn("no global", result.manifest()["optimality"])

    def test__given_small_candidate_budget__when_search_runs__then_budget_is_deterministic(self):
        outcomes = [synthesize(lambda a, b: min(a, b), library=library_for("hevc"), max_candidates=2)
                    for _ in range(2)]
        self.assertEqual(outcomes[0].manifest(), outcomes[1].manifest())
        self.assertEqual(outcomes[0].status, "candidate_budget")
        self.assertEqual(outcomes[0].candidates, 2)
        self.assertIsNone(outcomes[0].plan)
        empty_budget = synthesize(lambda a: a, library=PrimitiveLibrary("empty", ()),
                                  input_ranges=((0, 255),), max_candidates=0)
        self.assertEqual(empty_budget.status, "candidate_budget")
        self.assertEqual(empty_budget.candidates, 0)
        with patch("codec_compute.synthesis.search._spec_value") as specification:
            result = synthesize(lambda a: a, library=PrimitiveLibrary("empty", ()),
                                input_ranges=((0, 255),), max_candidates=0)
        specification.assert_not_called()
        self.assertEqual(result.status, "candidate_budget")

    def test__given_identity_without_primitives__when_searching_zero_instructions__then_input_alias_is_verified(self):
        result = synthesize(lambda a, b: b, library=PrimitiveLibrary("empty", ()),
                            input_ranges=((0, 255), (0, 255)), max_instructions=0)
        self.assertTrue(result.found)
        self.assertEqual(result.plan.instructions, ())
        self.assertEqual(result.plan.output, 1)
        self.assertEqual(result.verification["combinations_checked"], 65536)

    def test__given_sampled_aliases__when_exhaustive_checks_find_counterexamples__then_restart_recovers_valid_plan(self):
        initial = [(0, 0), (3, 3)]
        with patch("codec_compute.synthesis.search._initial_samples", side_effect=lambda _: list(initial)):
            result = synthesize(lambda a, b: min(a, b), library=library_for("hevc"),
                                input_ranges=((0, 3), (0, 3)), max_instructions=2)
        self.assertTrue(result.found)
        self.assertGreaterEqual(result.restarts, 2)
        self.assertIn((1, 0), result.counterexamples)
        self.assertIn((0, 1), result.counterexamples)
        self.assertEqual(result.verification_attempts, result.restarts + 1)
        self.assertEqual(result.sample_count, len(initial) + result.restarts)
        self.assertEqual(result.verification["combinations_checked"], 16)

    def test__given_counterexample_restarts__when_budget_runs_out__then_prior_rounds_count_against_total(self):
        with patch("codec_compute.synthesis.search._initial_samples", side_effect=lambda _: [(0, 0), (3, 3)]):
            result = synthesize(lambda a, b: min(a, b), library=library_for("hevc"),
                                input_ranges=((0, 3), (0, 3)), max_candidates=3)
        self.assertEqual(result.status, "candidate_budget")
        self.assertEqual(result.candidates, 3)
        self.assertEqual(result.restarts, 2)
        self.assertFalse(result.verification["verified"])
        self.assertIsNone(result.plan)

    def test__given_custom_unary_primitive__when_searching__then_library_descriptions_drive_enumeration(self):
        library = PrimitiveLibrary("custom", (Primitive("unfamiliar", (-1,), bias=17),))
        result = synthesize(lambda x: max(0, 17 - x), library=library,
                            input_ranges=((0, 255),), max_instructions=1)
        self.assertTrue(result.found)
        self.assertEqual(result.plan.instructions[0].primitive, "unfamiliar")
        self.assertEqual(result.verification["combinations_checked"], 256)

    def test__given_overridden_primitive_evaluator__when_searching__then_declared_descriptor_remains_authoritative(self):
        class PretendConstant(Primitive):
            def evaluate(self, *values):
                return 0

        library = PrimitiveLibrary("custom", (PretendConstant("invert", (-1,), bias=17),))
        result = synthesize(lambda x: max(0, 17 - x), library=library,
                            input_ranges=((0, 255),), max_instructions=1)
        self.assertTrue(result.found)
        self.assertTrue(result.verification["verified"])
        self.assertEqual(result.plan.instructions[0].primitive, "invert")

    def test__given_shared_subexpression__when_lifting_tree__then_repeated_node_emits_once(self):
        library = PrimitiveLibrary("custom", (Primitive("invert", (-1,), bias=100),
                                               Primitive("sum", (1, 1))))
        result = synthesize(lambda x: min(255, 2 * max(0, 100 - x)), library=library,
                            input_ranges=((0, 255),), max_instructions=3)
        self.assertTrue(result.found)
        self.assertEqual(len(result.plan.instructions), 2)
        self.assertEqual(result.plan.instructions[-1].operands, (1, 1))

    def test__given_valid_result__when_serializing__then_manifest_is_portable_and_binds_model(self):
        library = library_for("h264")
        result = synthesize(lambda a, b: a ^ b, library=library, input_ranges={"left": (0, 1), "right": (0, 1)})
        manifest = result.manifest()
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(manifest["library_fingerprint"], library.fingerprint())
        self.assertEqual(manifest["plan"], result.plan.manifest())
        self.assertEqual(manifest["verification_attempts"], 1)
        self.assertIn("before sharing", manifest["bound_metric"])

    def test__given_invalid_search_parameters__when_invoked__then_fail_explicitly(self):
        defaults = dict(library=library_for("hevc"))
        cases = [dict(max_instructions=-1), dict(max_instructions=True), dict(max_instructions=9),
                 dict(max_instructions=10000), dict(max_candidates=-1),
                 dict(max_candidates=1.5), dict(input_ranges=()), dict(input_ranges=((0, 256),)),
                 dict(input_ranges=((True, 1),)), dict(input_ranges=((3, 2),)),
                 dict(input_ranges=((0, 1),) * 3), dict(input_ranges={"": (0, 1)}),
                 dict(library=[]), dict(name=42)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                synthesize(lambda *values: 0, **(defaults | changes))
        with self.assertRaises(ValueError):
            synthesize(lambda a, b: 256, **defaults)
        with self.assertRaises(ValueError):
            synthesize(None, **defaults)
        with self.assertRaisesRegex(ValueError, "Specification failed for inputs"):
            synthesize(lambda a, b: a // 0, **defaults)


if __name__ == "__main__":
    unittest.main()
