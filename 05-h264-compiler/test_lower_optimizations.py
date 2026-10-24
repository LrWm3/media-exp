"""Exact lowering rewrites checked against real decoded pixels and baseline graphs."""

import unittest

import h264
from decoder_core.decode import available
from h264.graph import trace
from h264.lower import lower


class LowerOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        if "ffmpeg" not in cls.backends:
            raise unittest.SkipTest("FFmpeg unavailable")

    def bits(self, bit=0):
        return [[((16*y+x) >> bit) & 1 for x in range(16)] for y in range(16)]

    def verify_decoders(self, program, inputs, expected):
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(program.verify(inputs, backend), expected)

    def test__given_two_and_three_binary_terms__when_selecting_gates__then_all_truth_tables_are_exact(self):
        inputs = {name: self.bits(bit) for bit, name in enumerate(("a", "b", "c"))}
        cases = (
            ("or2", lambda a, b, c: h264.bit_or(a, b), lambda a, b, c: a | b, 1, "or", 0),
            ("and2", lambda a, b, c: h264.bit_and(a, b), lambda a, b, c: a & b, 1, "and", 0),
            ("or3", lambda a, b, c: h264.threshold(h264.add(h264.add(a, b), c), 1),
             lambda a, b, c: a | b | c, 2, "or", 0),
            ("and3", lambda a, b, c: h264.threshold(h264.add(a, h264.add(b, c)), 3),
             lambda a, b, c: a & b & c, 2, "and", 0),
            ("or_duplicate", lambda a, b, c: h264.threshold(h264.add(h264.add(a, a), b), 1),
             lambda a, b, c: a | b, 1, "or", 1),
            ("and_duplicate", lambda a, b, c: h264.threshold(h264.add(a, h264.add(a, b)), 3),
             lambda a, b, c: a & b, 1, "and", 1),
            ("nested_or_duplicate", lambda a, b, c: h264.bit_or(a, h264.bit_or(a, b)),
             lambda a, b, c: a | b, 1, "or", 1),
            ("nested_and_duplicate", lambda a, b, c: h264.bit_and(a, h264.bit_and(a, b)),
             lambda a, b, c: a & b, 1, "and", 1),
            ("or_identity", lambda a, b, c: h264.bit_or(a, a), lambda a, b, c: a, 0, "or", 1),
            ("and_identity", lambda a, b, c: h264.bit_and(a, a), lambda a, b, c: a, 0, "and", 1),
        )
        for name, kernel, reference, gates, kind, duplicates in cases:
            with self.subTest(case=name):
                # Exercise this gate-selection pass independently of the
                # earlier semantic IR pass, which now proves these identities.
                program = h264.compile(kernel, 16, 16, input_ranges=dict.fromkeys(inputs, (0, 1)),
                                       semantic_ir=False)
                expected = {"output": [[reference(*(inputs[name][y][x] for name in inputs))
                                         for x in range(16)] for y in range(16)]}
                self.assertEqual(program.stats["lowered_instructions"], gates)
                self.assertEqual(program.stats[f"direct_{kind}_gates"], gates)
                self.assertEqual(program.stats["boolean_operands_deduplicated"], duplicates)
                for op in program.operations:
                    self.assertEqual(op.denominator, 0)
                    self.assertEqual([source.weight for source in op.sources], [1, 1] if kind == "or" else [2, 2])
                    self.assertTrue(all(source.offset == (0 if kind == "or" else -1) for source in op.sources))
                self.verify_decoders(program, inputs, expected)

    def test__given_named_or_shared_sum__when_thresholding__then_intermediate_arithmetic_remains(self):
        def named(a, b):
            total = h264.add(a, b)
            return {"sum": total, "or": h264.threshold(total, 1)}

        def shared(a, b):
            total = h264.add(a, b)
            return {"and": h264.threshold(total, 2), "or": h264.threshold(total, 1)}

        inputs = {"a": self.bits(), "b": self.bits(1)}
        for kernel, count in ((named, 2), (shared, 3)):
            with self.subTest(kernel=kernel.__name__):
                program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 1), "b": (0, 1)})
                baseline = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 1), "b": (0, 1)}, optimize=False)
                self.assertEqual(program.stats["lowered_instructions"], count)
                self.assertEqual(program.stats["boolean_sum_nodes_bypassed"], 0)
                self.verify_decoders(program, inputs, baseline.reference(inputs))

    def test__given_all_u8_cutoffs__when_using_negative_predictions__then_every_baseline_pixel_matches(self):
        def kernel(x):
            return {str(cutoff): h264.threshold(x, cutoff) for cutoff in range(-1, 258)}
        image = [[16*y+x for x in range(16)] for y in range(16)]
        expected = {str(cutoff): [[int(value >= cutoff) for value in row] for row in image]
                    for cutoff in range(-1, 258)}
        optimized = h264.compile(kernel, 16, 16, input_range=(0, 255))
        baseline = h264.compile(kernel, 16, 16, input_range=(0, 255), optimize=False)
        self.assertGreater(optimized.stats["threshold_negative_two_p"], 0)
        self.assertEqual(baseline.stats["threshold_negative_two_p"], 0)
        self.assertLess(optimized.stats["lowered_instructions"], baseline.stats["lowered_instructions"])
        self.verify_decoders(optimized, image, expected)
        self.verify_decoders(baseline, image, expected)
        for cutoff, count in ((1, 2), (32, 2), (64, 2), (65, 2), (127, 2), (192, 1)):
            with self.subTest(cutoff=cutoff):
                sources, outputs, _ = trace(lambda x: h264.threshold(x, cutoff), (0, 255))
                instructions, _, _, stats, _ = lower(sources, outputs)
                self.assertEqual(len(instructions), count)
                self.assertEqual(stats["threshold_negative_two_p"], int(cutoff <= 64))

    def test__given_double_not_and_named_inner__when_eliminating__then_named_values_still_decode(self):
        def kernel(x):
            inner = h264.bit_not(x)
            return {"inner": inner, "output": h264.bit_not(inner)}
        image = self.bits()
        program = h264.compile(kernel, 16, 16)
        self.assertEqual(program.stats["double_not_eliminated"], 1)
        self.assertEqual(program.stats["lowered_instructions"], 1)
        self.verify_decoders(program, image, {"inner": [[1-v for v in row] for row in image], "output": image})
        long_program = h264.compile(lambda x: h264.repeat(3000, h264.bit_not, x), 16, 16)
        self.assertEqual(long_program.stats["double_not_eliminated"], 1500)
        self.assertEqual(long_program.stats["lowered_instructions"], 0)
        self.verify_decoders(long_program, image, {"output": image})

    def test__given_commutative_sums__when_interning__then_coefficients_follow_their_operands(self):
        def kernel(a, b):
            return {"first": h264.combine(a, b, 1, 2),
                    "same": h264.combine(b, a, 2, 1),
                    "different": h264.combine(a, b, 2, 1)}
        inputs = {"a": self.bits(), "b": self.bits(1)}
        program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 1), "b": (0, 1)})
        baseline = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 1), "b": (0, 1)}, optimize=False)
        self.assertEqual(program.stats["commutative_cse_eliminated"], 1)
        self.assertEqual(program.output_frames["first"], program.output_frames["same"])
        self.assertNotEqual(program.output_frames["first"], program.output_frames["different"])
        self.verify_decoders(program, inputs, baseline.reference(inputs))

    def test__given_nested_affines__when_ranges_prove_fusion__then_exact_pixels_use_one_prediction(self):
        cases = (
            (lambda x: h264.weighted(h264.weighted(x, 2, 3), 3, 4), (0, 15), 6, 13),
            (lambda x: h264.weighted(h264.saturating_weighted(x, 2, 3), 3, 4), (0, 15), 6, 13),
        )
        image = [[x for x in range(16)] for _ in range(16)]
        for kernel, value_range, coefficient, bias in cases:
            program = h264.compile(kernel, 16, 16, input_range=value_range)
            self.assertEqual(program.stats["affine_nodes_fused"], 1)
            self.assertEqual(program.stats["lowered_instructions"], 1)
            self.verify_decoders(program, image, {"output": [[coefficient*v+bias for v in row] for row in image]})

    def test__given_clipping_rounding_or_illegal_fused_weight__when_optimizing__then_boundaries_remain(self):
        cases = (
            lambda x: h264.weighted(h264.saturating_weighted(x, 2, -1), 3, 2),
            lambda x: h264.weighted(h264.threshold(h264.add(x, h264.shift(x, 1)), 1), 2, 3),
            lambda x: h264.weighted(h264.weighted(x, 63), 3),
        )
        image = self.bits()
        for kernel in cases:
            program = h264.compile(kernel, 16, 16)
            baseline = h264.compile(kernel, 16, 16, optimize=False)
            self.assertEqual(program.stats["affine_nodes_fused"], 0)
            self.verify_decoders(program, image, baseline.reference(image))

    def test__given_cropped_or_duplicate_shifted_reads__when_rewriting__then_zero_borders_remain(self):
        cases = (
            lambda x: h264.weighted(h264.shift(h264.weighted(x, 0, 1), 1), 2, 3),
            lambda x: h264.bit_or(h264.shift(h264.shift(x, 1), -1), h264.shift(h264.shift(x, -1), 1)),
            lambda x: h264.bit_or(h264.shift(x, 1), h264.shift(x, 1)),
            lambda x: h264.bit_and(h264.shift(x, -1), h264.shift(x, -1)),
        )
        image = [[1]*16 for _ in range(16)]
        for kernel in cases:
            program = h264.compile(kernel, 16, 16)
            baseline = h264.compile(kernel, 16, 16, optimize=False)
            self.verify_decoders(program, image, baseline.reference(image))

    def test__given_named_affine_intermediate__when_fusing__then_both_outputs_keep_their_values(self):
        def kernel(x):
            inner = h264.weighted(x, 2, 3)
            return {"inner": inner, "output": h264.weighted(inner, 3, 4)}
        image = self.bits()
        program = h264.compile(kernel, 16, 16)
        baseline = h264.compile(kernel, 16, 16, optimize=False)
        self.assertEqual(program.stats["affine_nodes_fused"], 0)
        self.assertEqual(program.stats["lowered_instructions"], 2)
        self.verify_decoders(program, image, baseline.reference(image))


if __name__ == "__main__":
    unittest.main(verbosity=2)
