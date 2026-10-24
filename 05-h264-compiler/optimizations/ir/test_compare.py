"""Portable comparison retains packed locations and independent named outputs."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = load("ir_compare_tested", "compare.py")
validate = load("ir_validate_tested", "validate.py")


class ComparisonTests(unittest.TestCase):
    def test__given_untrusted_bundle_path__when_reading__then_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate.checked(Path(directory).resolve(), "../outside", "unused")

    @unittest.skipUnless(compare.available("ffmpeg"), "FFmpeg unavailable")
    def test__given_mixed_packed_outputs__when_validating__then_both_variants_and_tile_offsets_are_checked(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            report = compare.prepare(root, ["mixed-pb"], ["ffmpeg"])
            self.assertEqual(report["comparison_count"], 1)
            self.assertEqual(report["stream_count"], 2)
            case, = report["comparisons"]
            self.assertEqual(case["baseline"]["costs"]["pictures"], 5)
            self.assertEqual(case["optimized"]["costs"]["pictures"], 4)
            self.assertEqual(validate.validate(root, "ffmpeg")["stream_count"], 2)
            suite = json.loads((root / "suite.json").read_text())
            packed = suite["cases"][1]
            shifted = next(info for info in packed["named_outputs"].values() if info["x"] > 16)
            shifted["x"] += 1
            (root / "suite.json").write_text(json.dumps(suite))
            with self.assertRaisesRegex(AssertionError, "named output mismatch"):
                validate.validate(root, "ffmpeg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
