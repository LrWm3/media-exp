"""Portable runner checks binding between streams, proofs, inputs and outputs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmark import benchmark, digest, load_bundle, run
from prepare import prepare


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="boolean-benchmark-test-")
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            metadata = prepare(cls.root, decoders=("ffmpeg",), throughput=False)
        # Exercise one constant and one packed nontrivial function, keeping the
        # measured runner test small while preparation verifies all witnesses.
        metadata["cases"] = [case for case in metadata["cases"]
                             if case["mask"] in (0, 6) and case["mode"] == "packed"]
        for case in metadata["cases"]:
            case["purpose"] = "throughput"
        cls.metadata = metadata
        (cls.root/"bundle.json").write_text(json.dumps(metadata))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test__given_packed_xor__when_selecting_output__then_one_picture_contains_the_same_verified_pixels(self):
        _, _, workloads = load_bundle(self.root)
        workload = next(value for value in workloads.values() if value["record"]["mask"] == 6)
        _, full, a = run(workload, "ffmpeg", "all")
        _, selected, b = run(workload, "ffmpeg", "outputs")
        self.assertEqual(a, b)
        self.assertEqual(full.decoded_frames, 4)
        self.assertEqual(len(selected.selected_frames), 1)
        # Software selection still copies all decoder pictures.
        self.assertEqual(selected.copied_frames, 4)

    def test__given_corrupt_proof_document__when_loading__then_certificate_binding_fails(self):
        path = self.root/self.metadata["proofs"]
        original = path.read_bytes()
        try:
            path.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bundle(self.root)
        finally:
            path.write_bytes(original)

    def test__given_changed_input_with_updated_hash__when_loading__then_decoded_seed_binding_still_fails(self):
        manifest_path = self.root/"bundle.json"
        original_manifest = manifest_path.read_bytes()
        metadata = json.loads(original_manifest)
        item = metadata["cases"][0]["inputs"]["a"]
        path = self.root/item["path"]
        original = path.read_bytes()
        changed = bytes([1-original[0]])+original[1:]
        try:
            path.write_bytes(changed)
            item["sha256"] = digest(changed)
            manifest_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Reference input"):
                load_bundle(self.root)
        finally:
            path.write_bytes(original)
            manifest_path.write_bytes(original_manifest)

    def test__given_timed_run__when_reporting_costs__then_physical_pixels_and_logical_evaluations_stay_distinct(self):
        with redirect_stdout(io.StringIO()):
            report = benchmark(self.root, self.root/"results", "ffmpeg", repeats=1, modes=("outputs",))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["totals"]["cases"], 2)
        xor = next(case for case in report["cases"].values() if case["mask"] == 6)
        timing = xor["timings"]["outputs"]
        self.assertEqual(timing["compute_luma_samples_per_evaluation"], 14)
        self.assertEqual(timing["all_luma_samples_per_evaluation"], 28)
        self.assertGreater(timing["recurrent_evaluations_per_second"], 0)
        self.assertEqual(timing["returned_pictures"], 1)
        self.assertEqual(timing["copied_pictures"], 4)


if __name__ == "__main__":
    unittest.main()
