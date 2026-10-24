"""Exact arithmetic, exhaustive coverage, and Boolean minimum certificates."""

from copy import deepcopy
import unittest

from atlas import INPUTS, build_atlas, evaluate_plan, truth_table, verify_atlas


class AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = build_atlas()

    def test_given_all_boolean_masks_when_replaying_witnesses_then_all_four_inputs_match(self):
        self.assertEqual([item["id"] for item in self.atlas["functions"]], list(range(16)))
        for item in self.atlas["functions"]:
            with self.subTest(mask=item["id"]):
                actual = [evaluate_plan(item["plan"], values, item["output_index"]) for values in INPUTS]
                self.assertEqual(actual, truth_table(item["id"]))
                self.assertEqual(item["truth"], actual)
                self.assertEqual(len(item["plan"]), item["scalar_minimum"])
                self.assertEqual(len(item["packed_schedule"]), item["packed_minimum"])
                self.assertEqual(item["minimum_status"], "proved")

    def test_given_full_instruction_grammar_when_exhaustively_reduced_then_only_parity_masks_missing(self):
        certificate = self.atlas["single_instruction_certificate"]
        self.assertEqual(certificate["parameter_classes"], {"P": 4096, "B_distinct": 524288, "B_repeated": 8176})
        self.assertEqual(certificate["parameter_classes_total"], 536560)
        self.assertEqual(certificate["single_instruction_masks"], [mask for mask in range(16) if mask not in (6, 9)])
        self.assertEqual(certificate["missing_single_instruction_masks"], [6, 9])
        self.assertEqual(certificate["effective_B_offsets"], list(range(-128, 128)))
        self.assertEqual(certificate["raw_B_offset_pairs_checked"], 65536)
        for kind, histogram in certificate["argmax_mask_histograms"].items():
            self.assertEqual(sum(histogram.values()), certificate["parameter_classes"][kind])

    def test_given_aliases_constants_and_parity_when_checking_costs_then_exact_minima_match_proofs(self):
        for item in self.atlas["functions"]:
            mask = item["id"]
            expected_scalar = 0 if mask in (10, 12) else 3 if mask in (6, 9) else 1
            expected_packed = 0 if mask in (10, 12) else 2 if mask in (6, 9) else 1
            self.assertEqual(item["scalar_minimum"], expected_scalar)
            self.assertEqual(item["packed_minimum"], expected_packed)
            if mask in (0, 15):
                self.assertEqual(len(item["plan"]), 1)

    def test_given_packed_layers_when_checking_sources_then_every_read_comes_from_earlier_picture(self):
        for item in self.atlas["functions"]:
            available = {0, 1}
            for layer in item["packed_schedule"]:
                for result_index in layer:
                    self.assertTrue(set(item["plan"][result_index - 2]["inputs"]).issubset(available))
                available.update(layer)
            self.assertIn(item["output_index"], available)

    def test_given_signed_rounding_and_u8_values_when_evaluating_then_round_before_bias_and_clip(self):
        cases = [
            ([0], [-1], 5, 1, (1, 0), 5),
            ([0], [-3], 5, 1, (1, 0), 4),
            ([0, 1], [-1, 0], 5, 0, (1, 0), 5),
            ([0, 1], [-3, 0], 5, 0, (1, 0), 4),
            ([0], [127], 127, 0, (255, 0), 255),
            ([0], [-128], -128, 0, (255, 0), 0),
            ([0, 0], [127, 127], 0, 0, (255, 0), 255),
        ]
        for sources, weights, bias, denominator, values, expected in cases:
            with self.subTest(sources=sources, weights=weights, denominator=denominator):
                plan = [{"inputs": sources, "weights": weights, "bias": bias, "denominator": denominator}]
                self.assertEqual(evaluate_plan(plan, values, 2), expected)

    def test_given_nonbinary_intermediates_when_evaluating_then_full_u8_grammar_remains_available(self):
        plan = [{"inputs": [0, 1], "weights": [127, 127], "bias": 127, "denominator": 0},
                {"inputs": [2], "weights": [1], "bias": -128, "denominator": 0}]
        self.assertEqual([evaluate_plan(plan, values, 3) for values in INPUTS], [0, 63, 63, 126])

    def test_given_forward_reference_when_evaluating_then_invalid_dag_is_rejected(self):
        plan = [{"inputs": [2], "weights": [1], "bias": 0, "denominator": 0}]
        with self.assertRaises(ValueError):
            evaluate_plan(plan, (0, 1), 2)

    def test_given_saved_certificate_when_replaying_then_counts_coverage_and_claims_are_verified(self):
        report = verify_atlas(self.atlas)
        self.assertTrue(report["verified"])
        self.assertEqual(report["truth_table_cases"], 64)
        self.assertEqual(report["non_unate_masks"], [6, 9])

    def test_given_tampered_minimum_when_verifying_then_report_is_rejected(self):
        tampered = deepcopy(self.atlas)
        tampered["functions"][6]["scalar_minimum"] = 2
        with self.assertRaises(ValueError):
            verify_atlas(tampered)


if __name__ == "__main__":
    unittest.main()
