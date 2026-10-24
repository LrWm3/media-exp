"""Explicit B prediction weight-sum limits apply to uniform and tiled slices."""

from dataclasses import replace
import unittest

from decoder_core.bitstream import Operation, Source, Writer


class WeightLimitTests(unittest.TestCase):
    def test__given_b_weight_sum_outside_range__when_encoding_uniform_or_block_predictions__then_writer_rejects_it(self):
        for mode in ("macroblock", "row"):
            writer = Writer(32, 16, slice_mode=mode)
            parent = Operation("zero parent", (Source(0, 0), Source(0, 0)))
            for denominator in range(8):
                maximum = 127 if denominator == 7 else 128
                for weights in ((-128, -1), (-128, -128), (127, maximum-126), (127, 127)):
                    invalid = Operation("invalid sum", tuple(Source(0, weight) for weight in weights), denominator)
                    for operation in (invalid, replace(parent, block_operations=(parent, invalid))):
                        with self.subTest(mode=mode, denominator=denominator, weights=weights,
                                          blocks=bool(operation.block_operations)):
                            with self.assertRaisesRegex(ValueError, "weight sum"):
                                writer.slice_count(operation)
                            with self.assertRaisesRegex(ValueError, "weight sum"):
                                writer.prediction(1, operation)

    def test__given_boundary_b_sums_or_extreme_p_weights__when_encoding__then_writer_accepts_legal_parameters(self):
        for mode in ("macroblock", "row"):
            writer = Writer(16, 16, slice_mode=mode)
            for denominator in range(8):
                maximum = 127 if denominator == 7 else 128
                for weights in ((-128,), (127,), (-128, 0), (-64, -64), (127, maximum-127)):
                    operation = Operation("legal limit", tuple(Source(0, weight) for weight in weights), denominator)
                    with self.subTest(mode=mode, denominator=denominator, weights=weights):
                        self.assertGreater(writer.slice_count(operation), 0)
                        self.assertTrue(writer.prediction(1, operation))


if __name__ == "__main__":
    unittest.main()
