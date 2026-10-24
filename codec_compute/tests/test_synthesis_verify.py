"""Independent exhaustive scalar verifier and replay-record checks."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from codec_compute.synthesis.model import Instruction, Plan, Primitive, PrimitiveLibrary
from codec_compute.synthesis.verify import verify_plan, verify_record


LIBRARY = PrimitiveLibrary("test-arithmetic", (
    Primitive("add", (1, 1)), Primitive("sub", (1, -1)),
    Primitive("average", (1, 1), shift=1),
))
FULL_DOMAIN = ((0, 255), (0, 255))
MINIMUM = Plan(2, (Instruction("sub", (0, 1)), Instruction("sub", (0, 2))), 3)
MAXIMUM = Plan(2, (Instruction("sub", (0, 1)), Instruction("add", (1, 2))), 3)
XOR = Plan(2, (Instruction("sub", (0, 1)), Instruction("sub", (1, 0)),
               Instruction("add", (2, 3))), 4)


class ExhaustiveVerificationTests(unittest.TestCase):
    def test__given_u8_minimum_and_maximum__when_verified__then_every_pair_matches(self):
        for plan, specification in ((MINIMUM, min), (MAXIMUM, max)):
            with self.subTest(specification=specification.__name__):
                record = verify_plan(plan, specification, FULL_DOMAIN, library=LIBRARY)
                expected = bytes(specification(a, b) for a in range(256) for b in range(256))
                self.assertEqual(record["status"], "verified")
                self.assertTrue(record["verified"])
                self.assertFalse(record["bitstream_verified"])
                self.assertEqual(record["scope"], "scalar_primitive_model")
                self.assertEqual(record["combinations_total"], 65536)
                self.assertEqual(record["combinations_checked"], 65536)
                self.assertEqual(record["mismatch_count"], 0)
                self.assertIsNone(record["counterexample"])
                self.assertEqual(record["expected_sha256"], hashlib.sha256(expected).hexdigest())
                self.assertEqual(record["actual_sha256"], record["expected_sha256"])

    def test__given_boolean_operations__when_verified__then_all_four_states_match(self):
        for plan, specification in ((MINIMUM, lambda a, b: bool(a and b)),
                                    (MAXIMUM, lambda a, b: bool(a or b)),
                                    (XOR, lambda a, b: bool(a) != bool(b))):
            with self.subTest(plan=plan):
                record = verify_plan(plan, specification, {"a": (0, 1), "b": (0, 1)}, library=LIBRARY)
                self.assertTrue(record["verified"])
                self.assertEqual(record["combinations_checked"], 4)
                self.assertEqual(record["input_order"], ["a", "b"])

    def test__given_average_rounding__when_verifying_all_pairs__then_odd_sums_round_up(self):
        plan = Plan(2, (Instruction("average", (0, 1)),), 2)
        record = verify_plan(plan, lambda a, b: (a + b + 1) // 2, FULL_DOMAIN, library=LIBRARY)
        self.assertTrue(record["verified"])
        self.assertEqual(record["combinations_checked"], 65536)

    def test__given_a_wrong_candidate__when_verification_fails__then_hashes_still_cover_full_domain(self):
        plan = Plan(2, (), 0)
        record = verify_plan(plan, min, FULL_DOMAIN, library=LIBRARY)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["verified"])
        self.assertEqual(record["counterexample"], {"inputs": [1, 0], "expected": 0, "actual": 1})
        self.assertEqual(record["combinations_checked"], 65536)
        self.assertEqual(record["mismatch_count"], 32640)
        self.assertEqual(record["hash_scope"], "complete_domain")
        self.assertEqual(record["expected_sha256"],
                         hashlib.sha256(bytes(min(a, b) for a in range(256) for b in range(256))).hexdigest())
        self.assertEqual(record["actual_sha256"],
                         hashlib.sha256(bytes(a for a in range(256) for _ in range(256))).hexdigest())

    def test__given_only_last_state_fails__when_verified__then_exhaustive_claim_catches_it(self):
        specification = lambda a, b: 0 if (a, b) == (255, 255) else a
        record = verify_plan(Plan(2, (), 0), specification, FULL_DOMAIN, library=LIBRARY)
        self.assertFalse(record["verified"])
        self.assertEqual(record["combinations_checked"], 65536)
        self.assertEqual(record["mismatch_count"], 1)
        self.assertEqual(record["counterexample"], {"inputs": [255, 255], "expected": 0, "actual": 255})

    def test__given_nonzero_ordered_domains__when_verified__then_bounds_and_enumeration_are_exact(self):
        seen = []
        def specification(left, right):
            seen.append((left, right))
            return min(left, right)
        record = verify_plan(MINIMUM, specification, {"z": (2, 4), "a": (7, 8)}, library=LIBRARY)
        self.assertEqual(seen, [(2, 7), (2, 8), (3, 7), (3, 8), (4, 7), (4, 8)])
        self.assertEqual(record["input_order"], ["z", "a"])
        self.assertEqual(record["combinations_total"], 6)
        self.assertTrue(record["verified"])

    def test__given_single_input_identity__when_no_instructions_exist__then_verification_needs_no_primitive(self):
        record = verify_plan(Plan(1, (), 0), lambda x: x, ((3, 9),),
                             library=PrimitiveLibrary("identity-only", ()))
        self.assertTrue(record["verified"])
        self.assertEqual(record["combinations_checked"], 7)

    def test__given_unavailable_shared_plan_evaluator__when_verifying__then_independent_traversal_succeeds(self):
        with patch("codec_compute.synthesis.model.evaluate_plan", side_effect=AssertionError("shared evaluator")), \
             patch("codec_compute.synthesis.model.validate_plan", side_effect=AssertionError("shared traversal")):
            record = verify_plan(MAXIMUM, max, ((0, 7), (0, 7)), library=LIBRARY)
        self.assertTrue(record["verified"])


class VerificationValidationTests(unittest.TestCase):
    def test__given_invalid_reference_or_primitive__when_verifying__then_candidate_is_rejected(self):
        cases = (Plan(2, (Instruction("unknown", (0, 1)),), 2),
                 Plan(2, (Instruction("sub", (0,)),), 2),
                 Plan(2, (Instruction("sub", (0, 2)),), 2),
                 Plan(2, (Instruction("sub", (0, 5)),), 2),
                 Plan(2, (), 2))
        for plan in cases:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                verify_plan(plan, min, FULL_DOMAIN, library=LIBRARY)
        for plan in (None, object(), {}):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                verify_plan(plan, min, FULL_DOMAIN, library=LIBRARY)

    def test__given_invalid_domains__when_verifying__then_exhaustive_scope_cannot_be_misstated(self):
        domains = ((), ((0, 1),), ((0, 256), (0, 1)), ((-1, 3), (0, 1)),
                   ((4, 3), (0, 1)), ((False, 1), (0, 1)), ((0.0, 1), (0, 1)),
                   ((0, 1, 2), (0, 1)), "00", {"": (0, 1), "b": (0, 1)})
        for domain in domains:
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                verify_plan(MINIMUM, min, domain, library=LIBRARY)

    def test__given_invalid_specification_values__when_verifying__then_no_certificate_is_returned(self):
        for invalid in (-1, 256, 1.0, "1", None):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "specification.*u8"):
                verify_plan(Plan(1, (), 0), lambda x: invalid, ((0, 1),), library=LIBRARY)
        with self.assertRaisesRegex(ValueError, "specification failed"):
            verify_plan(Plan(1, (), 0), lambda x: 1 // x, ((0, 1),), library=LIBRARY)

    def test__given_invalid_primitive_result__when_verifying__then_u8_model_contract_is_enforced(self):
        class InvalidPrimitive(Primitive):
            def evaluate(self, *values):
                return 256
        library = PrimitiveLibrary("broken", (InvalidPrimitive("broken", (1,)),))
        with self.assertRaisesRegex(ValueError, "Primitive broken.*u8"):
            verify_plan(Plan(1, (Instruction("broken", (0,)),), 1), lambda x: x,
                        ((0, 1),), library=library)


class VerificationReplayTests(unittest.TestCase):
    def setUp(self):
        self.ranges = {"a": (0, 4), "b": (0, 4)}
        self.record = verify_plan(MINIMUM, min, self.ranges, library=LIBRARY)

    def test__given_json_record__when_replaying_declared_models__then_full_record_is_reproduced(self):
        saved = json.loads(json.dumps(self.record))
        model = saved["primitive_model"]
        library = PrimitiveLibrary(model["target"], tuple(
            Primitive(item["name"], tuple(item["weights"]), item["bias_after_rounding"],
                      item["divisor_shift"], item["cost"]) for item in model["primitives"]),
            version=model["version"])
        data = saved["plan"]
        plan = Plan(data["input_count"], tuple(Instruction(item["primitive"], tuple(item["operands"]))
                                             for item in data["instructions"]), data["output"])
        ranges = {item["name"]: (item["minimum"], item["maximum"]) for item in saved["domains"]}
        replayed = verify_record(plan, min, ranges, library=library, record=saved)
        self.assertEqual(saved, replayed)
        self.assertEqual(saved["plan_fingerprint"], MINIMUM.fingerprint())
        self.assertEqual(saved["primitive_model_fingerprint"], LIBRARY.fingerprint())

    def test__given_changed_context__when_replaying__then_plan_model_and_domain_changes_are_rejected(self):
        changed_library = PrimitiveLibrary("different-target", LIBRARY.primitives)
        cases = ((MAXIMUM, self.ranges, LIBRARY), (MINIMUM, {"a": (0, 3), "b": (0, 4)}, LIBRARY),
                 (MINIMUM, self.ranges, changed_library))
        for plan, ranges, library in cases:
            with self.subTest(plan=plan, ranges=ranges, library=library.target), \
                 self.assertRaisesRegex(ValueError, "context mismatch"):
                verify_record(plan, min, ranges, library=library, record=self.record)

    def test__given_tampered_results_or_changed_spec__when_replaying__then_cached_hashes_are_not_trusted(self):
        for key, replacement in (("actual_sha256", "0" * 64), ("verified", False),
                                 ("combinations_checked", 24), ("combinations_total", 25.0),
                                 ("counterexample", {"inputs": [0, 0], "expected": 0, "actual": 1})):
            saved = deepcopy(self.record)
            saved[key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recomputed"):
                verify_record(MINIMUM, min, self.ranges, library=LIBRARY, record=saved)
        with self.assertRaisesRegex(ValueError, "recomputed"):
            verify_record(MINIMUM, max, self.ranges, library=LIBRARY, record=self.record)

    def test__given_failed_record__when_replayed__then_original_counterexample_remains_reproducible(self):
        failed = verify_plan(MINIMUM, max, self.ranges, library=LIBRARY)
        self.assertFalse(failed["verified"])
        self.assertEqual(verify_record(MINIMUM, max, self.ranges, library=LIBRARY, record=failed), failed)


if __name__ == "__main__":
    unittest.main()
