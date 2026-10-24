"""CPU-only checks of frozen inputs, model loading and overlap accounting."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("concurrency", Path(__file__).with_name("concurrency.py"))
concurrency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(concurrency)


class ConcurrencyTests(unittest.TestCase):
    def model(self):
        return {"format": "binary-denoiser-v1", "padding": "zero",
                "activation": "sum >= integer threshold", "kernels": [[1] * 9 for _ in range(8)],
                "thresholds": [-3, -2, -1, 0, 1, 2, 3, 4], "mix_weights": [-1] * 8,
                "mix_threshold": -2}

    def test__given_model__when_loading__then_preserves_signed_weights_and_thresholds(self):
        values = concurrency.model_parameters(self.model())
        self.assertEqual(len(values), 89)
        self.assertEqual(values[72:80], [-3, -2, -1, 0, 1, 2, 3, 4])
        self.assertEqual(values[80:], [-1] * 8 + [-2])

    def test__given_invalid_semantics__when_loading__then_rejects_model(self):
        for key, value in (("padding", "wrap"), ("activation", "sum > threshold"),
                           ("mix_threshold", True), ("thresholds", [0] * 7),
                           ("mix_weights", [0] * 8)):
            with self.subTest(key=key):
                model = copy.deepcopy(self.model()); model[key] = value
                with self.assertRaises(ValueError):
                    concurrency.model_parameters(model)

    def test__given_selected_i420_frame__when_extracting__then_uses_guard_and_stride(self):
        frame = bytearray([255]) * (208 * 112 * 3 // 2)
        expected = bytes(index % 2 for index in range(1024))
        for y in range(32):
            frame[(y + 16) * 208 + 16:(y + 16) * 208 + 48] = expected[y * 32:(y + 1) * 32]
        self.assertEqual(concurrency.extract_final(frame), expected)
        with self.assertRaises(AssertionError):
            concurrency.extract_final(frame[:-1])

    def test__given_partial_units_and_gaps__when_counting_overlap__then_counts_only_intersections(self):
        stress = {"start_ns": 0, "end_ns": 10000000,
                  "units": [[0, 3000000, 2.8], [4000000, 6000000, 1.8], [7000000, 10000000, 2.8]]}
        result = concurrency.overlap_statistics(stress, {"start_ns": 2000000, "end_ns": 8000000})
        self.assertEqual(result["intersecting_units"], 3)
        self.assertEqual(result["fully_contained_units"], 1)
        self.assertEqual(result["host_interval_overlap_ms"], 4)
        self.assertAlmostEqual(result["inference_interval_covered_fraction"], 2 / 3)

    def test__given_no_complete_overlapping_unit__when_counting__then_avoids_rate_claim(self):
        stress = {"start_ns": 0, "end_ns": 10000000, "units": [[0, 10000000, 9]]}
        result = concurrency.overlap_statistics(stress, {"start_ns": 2000000, "end_ns": 3000000})
        self.assertEqual(result["intersecting_units"], 1)
        self.assertIsNone(result["contained_units_per_second"])
        empty = concurrency.overlap_statistics(stress, {"start_ns": 11000000, "end_ns": 12000000})
        self.assertEqual(empty["intersecting_units"], 0)
        self.assertEqual(empty["host_interval_overlap_ms"], 0)

    def test__given_tampered_or_escaping_file__when_loading__then_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root / "file").write_bytes(b"input")
            checksum = concurrency.digest(b"input")
            self.assertEqual(concurrency.checked_file(root, "file", checksum), b"input")
            with self.assertRaises(ValueError):
                concurrency.checked_file(root, "file", concurrency.digest(b"changed"))
            with self.assertRaises(ValueError):
                concurrency.checked_file(root, "../file", checksum)


if __name__ == "__main__":
    unittest.main()
