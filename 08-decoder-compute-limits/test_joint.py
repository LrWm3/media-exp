"""Construction/schedule counterexample with actual decoder validation."""

from pathlib import Path
import tempfile
import unittest

from atlas import INPUTS, build_atlas, evaluate_plan
from benchmark import load_bundle
from joint import METRICS, build_candidates, pareto_frontier, prepare_joint
from program import Program
from decoder_core.decode import available


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = build_candidates(build_atlas())

    def test_given_two_constructions_when_checking_all_inputs_then_both_attain_scalar_minimum(self):
        self.assertEqual(len(self.candidates), 6)
        for candidate in self.candidates:
            case = candidate["case"]
            with self.subTest(candidate=candidate["id"]):
                self.assertEqual(len(case["plan"]), 3)
                self.assertEqual(case["scalar_minimum"], 3)
                self.assertEqual([evaluate_plan(case["plan"], values, case["output_index"])
                                  for values in INPUTS], case["truth"])
                if candidate["construction"] == "serial":
                    self.assertEqual(evaluate_plan(case["plan"][:1], (1, 1), 2), 2)

    def test_given_equally_optimal_instruction_counts_when_scheduling_then_dependency_depths_differ(self):
        for candidate in self.candidates:
            with self.subTest(candidate=candidate["id"]):
                expected_depth = 3 if candidate["construction"] == "serial" else 2
                self.assertEqual(candidate["dependency_depth"], expected_depth)
                program = Program(candidate["case"], packed=candidate["packed"])
                expected_pictures = 2 if candidate["construction"] == "parallel" and candidate["packed"] else 3
                self.assertEqual(len(program.operations), expected_pictures)
                self.assertEqual(program.case["packed_minimum"], 2)

    def test_given_actual_emitted_costs_when_comparing_then_picture_and_area_winners_differ(self):
        for mask in (6, 9):
            records = []
            for candidate in self.candidates:
                if candidate["case"]["id"] != mask:
                    continue
                program = Program(candidate["case"], packed=candidate["packed"])
                costs = dict(program.costs)
                costs["prediction_coded_luma_samples"] = (program.writer.width * program.writer.height
                                                          * len(program.operations))
                records.append({"id": candidate["id"], "costs": costs})
            frontier = pareto_frontier(records)
            self.assertGreaterEqual(len(frontier), 2)
            picture_winner = min(records, key=lambda item: item["costs"]["prediction_pictures"])
            area_winner = min(records, key=lambda item: item["costs"]["prediction_coded_luma_samples"])
            self.assertNotEqual(picture_winner["id"], area_winner["id"])
            self.assertIn(picture_winner["id"], frontier)
            self.assertTrue(all(key in picture_winner["costs"] for key in METRICS))

    def test_given_dominated_candidate_when_finding_frontier_then_only_nondominated_candidates_remain(self):
        records = [{"id": "few-pictures", "costs": {"pictures": 2, "samples": 12}},
                   {"id": "few-samples", "costs": {"pictures": 3, "samples": 10}},
                   {"id": "dominated", "costs": {"pictures": 4, "samples": 12}}]
        self.assertEqual(pareto_frontier(records, ("pictures", "samples")), ["few-pictures", "few-samples"])

    def test_given_real_decoders_when_emitting_portable_candidates_then_every_fixture_matches_truth(self):
        if not available("ffmpeg"):
            self.skipTest("FFmpeg unavailable")
        decoders = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        with tempfile.TemporaryDirectory(prefix="decoder-joint-test-") as directory:
            root = Path(directory)
            report = prepare_joint(root, decoders, width=16, height=16)
            self.assertEqual(len(report["cases"]), 6)
            self.assertEqual(report["status"], "pass")
            bundle, _, workloads = load_bundle(root)
            self.assertEqual(len(workloads), 6)
            self.assertEqual(bundle["certificate_sha256"], report["certificate_sha256"])
            self.assertTrue(report["source_sha256"])
            self.assertTrue((root / "PROOFS.md").is_file())
            for record in report["cases"]:
                self.assertEqual(record["all_four_truth_oracle"], record["truth"])
                self.assertEqual(len(record["validation_fixtures"]), 2)
                for fixture in record["validation_fixtures"]:
                    self.assertEqual(fixture["truth_pairs_present"], [list(values) for values in INPUTS])
                    self.assertEqual(set(fixture["decoders"]), set(decoders))
                    self.assertTrue(all(check["status"] == "pass" for check in fixture["decoders"].values()))
            for decision in report["decisions"]:
                self.assertIsNone(decision["single_best_candidate"])
                self.assertGreaterEqual(len(decision["pareto_candidate_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
