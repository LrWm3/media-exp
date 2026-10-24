"""Boundary checks for the portable benchmark protocol."""

import ast
from pathlib import Path
import tempfile
import unittest

from codec_compute.workloads import temporal_benchmark as runner
from codec_compute.workloads.benchmark import input_pair
from decoder_core.bitstream import Writer


class TemporalBenchmarkTests(unittest.TestCase):
    def test__given_boundary_pairs__when_computing_oracle__then_rounding_and_mask_are_exact(self):
        a, b = bytes([0, 255, 23, 24, 0, 1]), bytes([255, 0, 0, 0, 24, 0])
        self.assertEqual(runner.oracle("positive-change", a, b)["output"], bytes([0, 255, 23, 24, 0, 1]))
        self.assertEqual(runner.oracle("absolute-change", a, b)["mask"], bytes([1, 1, 0, 1, 1, 0]))
        self.assertEqual(runner.oracle("temporal-average", a, b)["output"], bytes([128, 128, 12, 12, 12, 1]))
        envelope = runner.oracle("temporal-envelope", a, b)
        self.assertEqual(envelope["output"], bytes(hi-lo for hi, lo in zip(envelope["high"], envelope["low"])))

    def test__given_unmodified_writer__when_postponing_annotations__then_input_prefix_is_identical(self):
        source = Path(runner.decoder.__file__).with_name("bitstream.py")
        compatible = runner.writer_class(source)
        params = dict(width=32, height=32, slice_mode="row", reference_slots={0: 0, 1: 1, 2: 0})
        original, portable = Writer(**params), compatible(**params)
        for index in range(2):
            inputs = input_pair(index, 32, 32)
            for frame, name in enumerate(("a", "b")):
                self.assertEqual(original.input_luma(inputs[name], frame), portable.input_luma(inputs[name], frame))
        ast.parse(Path(runner.__file__).read_text(), feature_version=(3, 8))

    def test__given_corrupt_guard_or_chroma__when_validating_i420__then_rejects_each(self):
        shape = {"coded_width": 48, "coded_height": 48, "width": 16, "height": 16,
                 "guard": 16, "frame_bytes": 48*48*3//2}
        frame = bytearray(48*48) + bytearray([128])*(48*48//2)
        self.assertGreater(runner.validate_guards([frame], shape)["guard_bytes"], 0)
        for position in (0, 48*48-1, 48*48, len(frame)-1):
            corrupted = bytearray(frame)
            corrupted[position] ^= 1
            with self.assertRaises(AssertionError):
                runner.validate_guards([corrupted], shape)

    def test__given_tampered_or_escaping_artifact__when_loading__then_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plane").write_bytes(b"abc")
            record = {"path": "plane", "bytes": 3, "sha256": runner.digest(b"abc")}
            self.assertEqual(runner.checked(root, record), b"abc")
            with self.assertRaises(ValueError):
                runner.checked(root, {**record, "sha256": "0"*64})
            with self.assertRaises(ValueError):
                runner.checked(root, {**record, "path": "../plane"})


if __name__ == "__main__":
    unittest.main()
