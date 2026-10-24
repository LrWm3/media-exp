"""Mathematical contracts and the unchanged H.264 realization."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import codec_compute as media


class SemanticTests(unittest.TestCase):
    def test__given_forged_range_proof__when_compiling__then_rejection_precedes_instruction_selection(self):
        with self.assertRaisesRegex(media.CompileError, "Invalid range proof"):
            media.compile(lambda x: media.threshold(media.Image("copy", (x,), (), 0, 0), 1),
                          width=16, height=16)
        graph = media.trace(lambda x: media.copy(x), width=16, height=16)
        forged = media.Image("copy", (graph.inputs[0],), (), 0, 0)
        with self.assertRaisesRegex(media.CompileError, "Invalid range proof"):
            media.compile_graph(replace(graph, nodes=(*graph.inputs, forged), outputs=(("output", forged),)))

    def test__given_positional_kernel__when_default_input_is_used__then_parameter_name_is_unrestricted(self):
        graph = media.trace(lambda x: media.copy(x), width=1, height=1)
        self.assertEqual(media.evaluate(graph, {"image": [[23]]}), {"output": [[23]]})

    def test__given_nonimage_operands__when_building_operations__then_errors_are_compilation_errors(self):
        for function in (media.copy, media.shift, lambda x: media.threshold(x, 1)):
            with self.subTest(function=function), self.assertRaises(media.CompileError):
                function(4)
        with self.assertRaises(media.CompileError):
            media.trace(lambda x: media.affine(x, weights=4), width=1, height=1)
        for operands in ((3,), None):
            with self.subTest(operands=operands), self.assertRaises(media.CompileError):
                media.trace(lambda x: media.Image("copy", operands, (), 0, 255), width=1, height=1)

    def test__given_full_range_average__when_interpreted__then_rounding_precedes_any_u8_storage(self):
        graph = media.trace(media.average, width=4, height=1, input_ranges={"a": (0, 255), "b": (0, 255)})
        self.assertEqual(media.evaluate(graph, {"a": [[255, 255, 0, 1]], "b": [[255, 254, 1, 2]]}),
                         {"output": [[255, 255, 1, 2]]})
        self.assertEqual(graph.manifest()["format"], "codec-compute-ir-2")

    def test__given_full_bytes__when_adding_exactly__then_overflow_requires_explicit_saturation(self):
        with self.assertRaisesRegex(media.CompileError, "exceeds u8"):
            media.trace(media.add, width=1, height=1, input_ranges={"a": (0, 255), "b": (0, 255)})
        graph = media.trace(media.add_sat, width=1, height=1, input_ranges={"a": (0, 255), "b": (0, 255)})
        self.assertEqual(media.evaluate(graph, {"a": [[200]], "b": [[100]]}), {"output": [[255]]})

    def test__given_a_clipped_intermediate__when_subtracting__then_clipping_remains_observable(self):
        graph = media.trace(lambda a, b: media.sub_sat(media.add_sat(a, b), b),
                            width=1, height=1, input_ranges={"a": (0, 255), "b": (0, 255)})
        self.assertEqual(media.evaluate(graph, {"a": [[200]], "b": [[100]]}), {"output": [[155]]})

    def test__given_shift_borders__when_interpreted__then_zero_and_clamp_are_distinct(self):
        graph = media.trace(lambda image: {"zero": media.shift(image, 1),
                                          "clamp": media.shift(image, 1, border="clamp")}, width=2, height=1)
        self.assertEqual(media.evaluate(graph, {"image": [[4, 9]]}), {"zero": [[9, 0]], "clamp": [[9, 9]]})

    def test__given_large_coefficients__when_tracing__then_target_limits_do_not_restrict_ir(self):
        graph = media.trace(lambda image: media.affine(image, weights=(1000,), saturating=True), width=3, height=1)
        self.assertEqual(media.evaluate(graph, {"image": [[0, 1, 255]]}), {"output": [[0, 255, 255]]})
        with self.assertRaises(media.UnsupportedOperation):
            media.compile_graph(graph)

    def test__given_unknown_targets_and_bad_inputs__when_compiling_or_encoding__then_errors_are_explicit(self):
        with self.assertRaisesRegex(media.CompileError, "Unknown codec"):
            media.compile(media.copy, target="unknown", input_ranges={"image": (0, 255)})
        for ranges in ({}, {"a": (-1, 255)}, {"a": (False, 1)}, {"bad name": (0, 1)}):
            with self.subTest(ranges=ranges), self.assertRaises(media.CompileError):
                media.trace(lambda **x: next(iter(x.values())), width=16, height=16, input_ranges=ranges)


class H264AdapterTests(unittest.TestCase):
    @staticmethod
    def inputs():
        return {"a": [[(x + y * 16) % 256 for x in range(16)] for y in range(16)],
                "b": [[(x * 37 + y * 19) % 256 for x in range(16)] for y in range(16)]}

    def test__given_composed_arithmetic__when_decoded__then_named_outputs_match_semantics(self):
        def kernel(a, b):
            total = media.add_sat(a, b)
            return {"sum": total, "difference": media.sub_sat(total, b), "original": a,
                    "threshold": media.threshold(total, 201)}
        program = media.compile(kernel, width=16, height=16, input_ranges={"a": (0, 255), "b": (0, 255)})
        program.verify(self.inputs())
        self.assertEqual(program.manifest()["target"], "h264")

    def test__given_changed_inputs__when_encoding__then_tail_stays_fixed_and_oracle_is_never_called(self):
        program = media.compile(media.add_sat, width=16, height=16,
                                input_ranges={"a": (0, 255), "b": (0, 255)})
        tail = program.tail
        with patch("codec_compute.ir.evaluate", side_effect=AssertionError("host inference")):
            first = program.encode(self.inputs())
            second_inputs = {"a": [[200] * 16 for _ in range(16)], "b": [[100] * 16 for _ in range(16)]}
            second = program.encode(second_inputs)
            self.assertEqual(program.decode(second_inputs)["output"], [[255] * 16 for _ in range(16)])
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(tail) and second.endswith(tail))

    def test__given_boolean_inputs__when_gates_and_shifts_run__then_literals_remain_zero_or_one(self):
        program = media.compile(lambda a, b: {"and": media.bit_and(a, b), "or": media.bit_or(a, b),
                                             "not": media.bit_not(a), "shift": media.shift(a, 1)},
                                width=16, height=16, input_ranges={"a": (0, 1), "b": (0, 1)})
        program.verify({"a": [[x % 2 for x in range(16)] for _ in range(16)],
                        "b": [[(x // 2) % 2 for x in range(16)] for _ in range(16)]})


if __name__ == "__main__":
    unittest.main()
