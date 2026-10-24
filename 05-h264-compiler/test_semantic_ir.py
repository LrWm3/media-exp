"""Boolean IR identities checked against independent tables and decoder pixels."""

from itertools import product
import random
import unittest
from unittest.mock import patch

import h264
from decoder_core.decode import available


def parity(a, b):
    return h264.bit_or(h264.bit_and(a, h264.bit_not(b)),
                       h264.bit_and(h264.bit_not(a), b))


def table_kernel(mask):
    def kernel(a, b):
        terms = [h264.bit_and(a if pair & 2 else h264.bit_not(a),
                              b if pair & 1 else h264.bit_not(b))
                 for pair in range(4) if mask >> pair & 1]
        if not terms:
            return h264.weighted(a, 0)
        result = terms[0]
        for term in terms[1:]:
            result = h264.bit_or(result, term)
        return result
    return kernel


class SemanticIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        if not cls.backends:
            raise AssertionError("Semantic IR tests require an actual decoder")

    def inputs(self, names=("a", "b")):
        return {name: [[((16*y+x) >> bit) & 1 for x in range(16)] for y in range(16)]
                for bit, name in enumerate(reversed(names))}

    def compile(self, kernel, names=("a", "b"), **kwargs):
        return h264.compile(kernel, 16, 16, input_ranges=dict.fromkeys(names, (0, 1)), **kwargs)

    def check(self, program, inputs, expected):
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(program.decode(inputs, backend), expected)

    def test__given_all_boolean_functions__when_canonicalizing__then_truth_tables_and_minimum_counts_hold(self):
        inputs = self.inputs()
        for mask in range(16):
            with self.subTest(mask=mask):
                program = self.compile(table_kernel(mask))
                expected = {"output": [[(mask >> (2*inputs["a"][y][x]+inputs["b"][y][x])) & 1
                                         for x in range(16)] for y in range(16)]}
                self.assertEqual(program.stats["lowered_instructions"],
                                 0 if mask in (10, 12) else 3 if mask in (6, 9) else 1)
                self.check(program, inputs, expected)

    def test__given_long_parity_recurrences__when_propagating_values__then_even_rounds_alias_the_input(self):
        inputs = self.inputs()
        for count in (2, 3, 40, 81):
            with self.subTest(count=count):
                kernel = lambda a, b: h264.repeat(count, lambda value: parity(value, b), a)
                program = self.compile(kernel)
                baseline = self.compile(kernel, semantic_ir=False)
                self.assertEqual(program.stats["lowered_instructions"], 3 if count % 2 else 0)
                self.assertLessEqual(program.pictures, baseline.pictures)
                if count == 40:
                    self.assertEqual(program.stats["semantic_ir_saved_instructions"], 151)
                expected = {"output": [[a ^ (b if count % 2 else 0) for a, b in zip(arow, brow)]
                                         for arow, brow in zip(inputs["a"], inputs["b"])]}
                self.check(program, inputs, expected)

    def test__given_absorption_and_equivalent_outputs__when_interning_semantics__then_frames_are_shared(self):
        def kernel(a, b):
            return {"input": a, "absorbed": h264.bit_or(a, h264.bit_and(a, b)),
                    "or": h264.bit_or(a, b),
                    "demorgan": h264.bit_not(h264.bit_and(h264.bit_not(a), h264.bit_not(b)))}
        inputs = self.inputs()
        program = self.compile(kernel)
        self.assertEqual(program.output_frames["input"], program.output_frames["absorbed"])
        self.assertEqual(program.output_frames["or"], program.output_frames["demorgan"])
        self.assertEqual(program.stats["lowered_instructions"], 1)
        values = [[a | b for a, b in zip(arow, brow)] for arow, brow in zip(inputs["a"], inputs["b"])]
        self.check(program, inputs, {"input": inputs["a"], "absorbed": inputs["a"],
                                    "or": values, "demorgan": values})

    def test__given_named_intermediates__when_eliminating_a_recurrence__then_every_named_value_survives(self):
        def kernel(a, b):
            first = parity(a, b)
            return {"first": first, "sum": h264.add(a, b), "output": parity(first, b)}
        inputs = self.inputs()
        program = self.compile(kernel)
        self.assertEqual(program.output_frames["output"], 0)
        self.check(program, inputs, {
            "first": [[a ^ b for a, b in zip(arow, brow)] for arow, brow in zip(inputs["a"], inputs["b"])],
            "sum": [[a + b for a, b in zip(arow, brow)] for arow, brow in zip(inputs["a"], inputs["b"])],
            "output": inputs["a"]})

    def test__given_constant_functions__when_folding__then_constants_share_one_prediction(self):
        def kernel(a, b):
            return {"zero": h264.bit_and(a, h264.bit_not(a)),
                    "also_zero": h264.bit_and(b, h264.bit_not(b)),
                    "one": h264.bit_or(a, h264.bit_not(a)),
                    "also_one": h264.bit_or(b, h264.bit_not(b))}
        program = self.compile(kernel)
        self.assertEqual(program.stats["lowered_instructions"], 2)
        self.assertEqual(program.output_frames["zero"], program.output_frames["also_zero"])
        self.assertEqual(program.output_frames["one"], program.output_frames["also_one"])
        self.check(program, self.inputs(), {name: [[int("one" in name)]*16 for _ in range(16)]
                                           for name in ("zero", "also_zero", "one", "also_one")})

    def test__given_movement_boundaries__when_propagating__then_crop_and_zero_padding_remain_observable(self):
        def kernel(a, b):
            one = h264.bit_or(a, h264.bit_not(a))
            shifted = h264.shift(one, 1)
            cropped = h264.shift(h264.shift(a, 1), -1)
            return {"shifted": shifted, "cropped": cropped,
                    "absorbed": h264.bit_or(cropped, h264.bit_and(cropped, b))}
        inputs = {"a": [[1]*16 for _ in range(16)], "b": [[1]*16 for _ in range(16)]}
        program = self.compile(kernel)
        cropped = [[0]+[1]*15 for _ in range(16)]
        self.check(program, inputs, {"shifted": [[1]*15+[0] for _ in range(16)],
                                    "cropped": cropped, "absorbed": cropped})

    def test__given_three_inputs_and_clipping__when_opening_frontiers__then_exhaustive_outputs_match(self):
        def kernel(a, b, c):
            hinge = h264.saturating_combine(a, b, 2, -1)
            return {"parity3": parity(parity(a, b), c), "hinge": hinge,
                    "selected": h264.bit_and(h264.threshold(hinge, 1), c)}
        inputs = self.inputs(("a", "b", "c"))
        expected = {name: [[0]*16 for _ in range(16)] for name in ("parity3", "hinge", "selected")}
        for y, x in product(range(16), repeat=2):
            a, b, c = (inputs[name][y][x] for name in ("a", "b", "c"))
            expected["parity3"][y][x] = a ^ b ^ c
            expected["hinge"][y][x] = max(0, 2*a-b)
            expected["selected"][y][x] = int(2*a-b >= 1) & c
        self.check(self.compile(kernel, ("a", "b", "c")), inputs, expected)

    def test__given_random_scalar_dags__when_rewriting__then_complete_three_input_domains_match_baseline(self):
        rng = random.Random(90909)
        for iteration in range(12):
            choices = [(rng.randrange(6), rng.randrange(index+3), rng.randrange(index+3))
                       for index in range(12)]
            def kernel(a, b, c):
                values = [a, b, c]
                for op, left, right in choices:
                    x, y = values[left], values[right]
                    values.append((lambda: h264.bit_not(x), lambda: h264.bit_or(x, y),
                                   lambda: h264.bit_and(x, y), lambda: parity(x, y),
                                   lambda: h264.threshold(h264.saturating_combine(x, y, 2, -1), 1),
                                   lambda: h264.equal(h264.add(x, y), 1))[op]())
                return {str(index): value for index, value in enumerate(values[-3:])}
            with self.subTest(iteration=iteration):
                program = self.compile(kernel, ("a", "b", "c"))
                baseline = self.compile(kernel, ("a", "b", "c"), semantic_ir=False)
                inputs = self.inputs(("a", "b", "c"))
                self.assertLessEqual(program.stats["lowered_instructions"], baseline.stats["lowered_instructions"])
                self.check(program, inputs, baseline.reference(inputs))

    def test__given_disabled_pass__when_compiling__then_baseline_and_manifest_controls_are_explicit(self):
        kernel = lambda a, b: h264.bit_or(a, h264.bit_and(a, b))
        enabled = self.compile(kernel, exact_search=False)
        disabled = self.compile(kernel, semantic_ir=False)
        plain = self.compile(kernel, optimize=False)
        self.assertEqual(enabled.stats["lowered_instructions"], 0)
        self.assertEqual(disabled.stats["lowered_instructions"], 1)
        self.assertFalse(disabled.stats["semantic_ir_enabled"])
        self.assertFalse(plain.stats["semantic_ir_enabled"])
        self.assertTrue(enabled.manifest()["stats"]["semantic_ir_proofs"])
        self.assertEqual(disabled.stats["semantic_ir_proofs"], [])

    def test__given_affine_fusion_exceeding_b_weight_sum__when_lowering__then_legal_stages_remain_separate(self):
        def kernel(a, b):
            return h264.weighted(h264.combine(a, b, 31, 31), 2)
        program = self.compile(kernel)
        self.assertEqual(program.stats["lowered_instructions"], 2)
        inputs = self.inputs()
        expected = {"output": [[62*(a+b) for a, b in zip(arow, brow)]
                                for arow, brow in zip(inputs["a"], inputs["b"])]}
        self.check(program, inputs, expected)
        with self.assertRaisesRegex(h264.CompileError, "weight sum"):
            self.compile(lambda a, b: h264.combine(a, b, 63, 63))

    def test__given_unrepresentable_dead_arithmetic__when_folding_constants__then_legal_output_still_compiles(self):
        for kernel in (lambda a, b: h264.weighted(h264.combine(a, b, 63, 63), 0),
                       lambda a, b: h264.threshold(h264.combine(a, b, 63, 63), 127),
                       lambda a, b: h264.weighted(h264.weighted(a, 200), 0)):
            with self.subTest(kernel=kernel):
                program = self.compile(kernel)
                self.assertEqual(program.stats["lowered_instructions"], 1)
                self.assertTrue(program.stats["semantic_ir_baseline_unavailable"])
                self.assertIsNone(program.stats["semantic_ir_saved_instructions"])
                with self.assertRaises(h264.CompileError):
                    self.compile(kernel, semantic_ir=False)
                self.check(program, self.inputs(), {"output": [[0]*16 for _ in range(16)]})

    def test__given_an_incorrect_ir_rewrite__when_verifying__then_original_graph_oracle_detects_it(self):
        from h264.ir import BooleanValues
        original = BooleanValues.rewrite
        def broken(values, node, identifier):
            result = original(values, node, identifier)
            if node.op == "threshold" and len(values.facts.get(node, ((),))[0]) == 2:
                return values.anchor
            return result
        kernel = lambda a, b: h264.bit_or(a, b)
        with patch.object(BooleanValues, "rewrite", broken):
            program = self.compile(kernel)
        expected = {"output": [[a | b for a, b in zip(arow, brow)]
                                for arow, brow in zip(self.inputs()["a"], self.inputs()["b"])]}
        for candidate in (program, h264.pack(program, tiles=2)):
            self.assertEqual(candidate.reference(self.inputs()), expected)
            with self.assertRaisesRegex(AssertionError, "differs from graph semantics"):
                candidate.verify(self.inputs(), self.backends[0])

    def test__given_exact_search_fallback__when_reporting_ir_savings__then_requested_baseline_is_used(self):
        def kernel(a, b):
            both = h264.bit_and(a, b)
            absorbed = h264.bit_or(both, h264.bit_and(both, b))
            return {"nand": h264.bit_not(both),
                    "zero": h264.bit_and(a, h264.equal(h264.add(absorbed, absorbed), 1))}
        for search in (False, True):
            with self.subTest(exact_search=search):
                program = self.compile(kernel, exact_search=search)
                previous = self.compile(kernel, exact_search=search, semantic_ir=False)
                self.assertEqual(program.stats["semantic_ir_saved_instructions"],
                                 previous.stats["lowered_instructions"]-program.stats["lowered_instructions"])
                self.check(program, self.inputs(), previous.reference(self.inputs()))


if __name__ == "__main__":
    unittest.main()
