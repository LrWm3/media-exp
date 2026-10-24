"""Absolute difference keeps honest IR ranges and runs through real decoders."""

from dataclasses import replace
from itertools import product
import shutil
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute import ir
from codec_compute.synthesis import library_for
from codec_compute.synthesis.lower import SynthesisError


BYTE_RANGES = {"a": (0, 255), "b": (0, 255)}


def difference(inputs):
    return [[abs(a-b) for a, b in zip(left, right)]
            for left, right in zip(inputs["a"], inputs["b"])]


class AbsDiffIRTests(unittest.TestCase):
    def test__given_disjoint_overlapping_and_singleton_ranges__when_tracing__then_bounds_match_all_reachable_pairs(self):
        cases = (
            ((0, 255), (0, 255), (0, 255)),
            ((0, 17), (90, 120), (73, 120)),
            ((90, 120), (0, 17), (73, 120)),
            ((25, 80), (60, 130), (0, 105)),
            ((40, 60), (40, 60), (0, 20)),
            ((5, 7), (0, 255), (0, 250)),
            ((9, 9), (37, 37), (28, 28)),
            ((37, 37), (37, 37), (0, 0)),
            ((0, 0), (255, 255), (255, 255)),
        )
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                graph = media.trace(media.absdiff, width=1, height=1, input_ranges={"a": left, "b": right})
                output = graph.outputs[0][1]
                values = [abs(a-b) for a, b in product(range(left[0], left[1]+1), range(right[0], right[1]+1))]
                self.assertEqual((min(values), max(values)), expected)
                self.assertEqual((output.lo, output.hi), expected)
                self.assertEqual(output.op, "absdiff")
                self.assertEqual(output.params, ())
                node = graph.manifest()["nodes"][-1]
                self.assertEqual((node["op"], node["range"]), ("absdiff", list(expected)))
                for a, b in product((left[0], left[1]), (right[0], right[1])):
                    self.assertEqual(media.evaluate(graph, {"a": [[a]], "b": [[b]]}), {"output": [[abs(a-b)]]})

    def test__given_same_symbolic_operand_twice__when_interpreting__then_zero_values_keep_conservative_interval_proof(self):
        graph = media.trace(lambda x: media.absdiff(x, x), width=16, height=16)
        output = graph.outputs[0][1]
        self.assertIs(output.args[0], output.args[1])
        self.assertEqual((output.lo, output.hi), (0, 255))
        inputs = {"image": [[16*y+x for x in range(16)] for y in range(16)]}
        self.assertEqual(media.evaluate(graph, inputs), {"output": [[0]*16 for _ in range(16)]})

    def test__given_forged_absdiff_range__when_compiling__then_validation_rejects_before_synthesis(self):
        graph = media.trace(media.absdiff, width=16, height=16, input_ranges=BYTE_RANGES)
        for lower, upper in ((0, 0), (0, 254), (1, 255)):
            original = graph.outputs[0][1]
            forged = replace(original, lo=lower, hi=upper)
            invalid = replace(graph, nodes=(*graph.inputs, forged), outputs=(("output", forged),))
            for target in ("h264", "hevc", "av1"):
                with self.subTest(lower=lower, upper=upper, target=target):
                    with patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("unvalidated search")) as search:
                        with self.assertRaisesRegex(media.CompileError, "Invalid range proof for absdiff"):
                            media.compile_graph(invalid, target=target)
                    search.assert_not_called()

    def test__given_invalid_operands_or_parameters__when_building_absdiff__then_malformed_semantics_are_rejected(self):
        for kernel in (lambda x: media.absdiff(x, 1), lambda x: media.absdiff(None, x),
                       lambda x: media.Image("absdiff", (x,), (), 0, 255),
                       lambda x: media.Image("absdiff", (x, x), ("extra",), 0, 255)):
            with self.subTest(kernel=kernel), self.assertRaises(media.CompileError):
                media.trace(kernel, width=1, height=1)

    def test__given_synthesis_disabled__when_compiling_absdiff__then_targets_report_missing_exact_lowering(self):
        for target in ("h264", "hevc", "av1"):
            with self.subTest(target=target):
                with patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("disabled search")) as search:
                    with self.assertRaises(media.UnsupportedOperation) as caught:
                        media.compile(media.absdiff, target=target, width=16, height=16,
                                      input_ranges=BYTE_RANGES, synthesis=False)
                self.assertEqual(caught.exception.operation, "absdiff")
                search.assert_not_called()

    def test__given_two_instruction_budget__when_searching_full_range_absdiff__then_failure_reports_only_that_bound(self):
        for target in ("h264", "hevc"):
            with self.subTest(target=target), self.assertRaises(SynthesisError) as caught:
                media.compile(media.absdiff, target=target, width=16, height=16,
                              input_ranges=BYTE_RANGES, synthesis_max_instructions=2)
            self.assertEqual(caught.exception.result.status, "instruction_bound")
            self.assertEqual(caught.exception.result.max_instructions, 2)
            self.assertIn("not a codec impossibility", str(caught.exception))

    def test__given_av1_average_only_grammar__when_searching_absdiff__then_bounded_miss_is_explicit(self):
        with self.assertRaises(SynthesisError) as caught:
            media.compile(media.absdiff, target="av1", width=16, height=16, input_ranges=BYTE_RANGES)
        error = caught.exception
        self.assertEqual((error.target, error.operation), ("av1", "absdiff"))
        self.assertEqual(error.result.status, "instruction_bound")
        self.assertEqual(error.result.max_instructions, 3)
        self.assertIsNone(error.result.plan)
        self.assertFalse(error.result.found)
        self.assertLess(error.result.candidates, error.result.max_candidates)
        self.assertIn("not a codec impossibility", str(error))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class AbsDiffDecoderTests(unittest.TestCase):
    def test__given_every_u8_pair__when_reusing_three_prediction_programs__then_h264_and_hevc_decode_all_65536_differences(self):
        graph = media.trace(media.absdiff, width=64, height=64, input_ranges=BYTE_RANGES)
        for target in ("h264", "hevc"):
            program = media.compile_graph(graph, target=target)
            manifest = program.manifest()
            self.assertEqual(manifest["costs"]["computation_pictures"], 3)
            self.assertEqual(manifest["costs"]["total_pictures"], 5)
            self.assertEqual(len(program.synthesis_records), 1)
            record = program.synthesis_records[0]
            self.assertEqual(record["operation"], "absdiff")
            verification = record["search"]["verification"]
            self.assertTrue(verification["verified"])
            self.assertEqual(verification["combinations_checked"], 65536)
            self.assertEqual(verification["expected_sha256"], verification["actual_sha256"])
            plan = record["search"]["plan"]["instructions"]
            self.assertEqual(len(plan), 3)
            self.assertTrue(all(step["primitive"] in library_for(target).by_name for step in plan))
            self.assertNotIn("absdiff", {node.op for node in program.lowered_graph.nodes})
            tail = program.tail
            seen = set()
            for chunk in range(16):
                inputs = {"a": [[(chunk*4096+y*64+x)//256 for x in range(64)] for y in range(64)],
                          "b": [[(y*64+x) % 256 for x in range(64)] for y in range(64)]}
                seen.update(zip((value for row in inputs["a"] for value in row),
                                (value for row in inputs["b"] for value in row)))
                with self.subTest(target=target, chunk=chunk):
                    self.assertEqual(program.decode(inputs, decoder="ffmpeg"), {"output": difference(inputs)})
                    self.assertEqual(program.tail, tail)
            self.assertEqual(len(seen), 65536)

    def test__given_composed_thresholds_and_aliases__when_decoding__then_all_named_outputs_preserve_semantics(self):
        def kernel(a, b):
            delta = media.absdiff(a, b)
            return {"difference": delta, "alias": delta, "a": a, "b": b,
                    "changed": media.threshold(delta, 1), "large": media.threshold(delta, 128)}
        inputs = {"a": [[16*y+x for x in range(16)] for y in range(16)],
                  "b": [[(37*x+19*y) % 256 for x in range(16)] for y in range(16)]}
        delta = difference(inputs)
        expected = {"difference": delta, "alias": delta, **inputs,
                    "changed": [[int(value >= 1) for value in row] for row in delta],
                    "large": [[int(value >= 128) for value in row] for row in delta]}
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile(kernel, target=target, width=16, height=16, input_ranges=BYTE_RANGES)
                ir.validate_graph(program.lowered_graph)
                self.assertEqual(program.verify(inputs), expected)
                outputs = program.manifest()["output_frames"]
                self.assertEqual(outputs["difference"], outputs["alias"])
                self.assertEqual(len(program.synthesis_records), 1)

    def test__given_correlated_byte_operands__when_decoding__then_repeated_references_produce_exact_zero(self):
        inputs = {"image": [[16*y+x for x in range(16)] for y in range(16)]}
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile(lambda x: media.absdiff(x, x), target=target, width=16, height=16)
                self.assertEqual((program.graph.outputs[0][1].lo, program.graph.outputs[0][1].hi), (0, 255))
                self.assertEqual(program.verify(inputs), {"output": [[0]*16 for _ in range(16)]})

    def test__given_changed_inputs_after_compilation__when_encoding_and_decoding__then_fixed_tail_uses_no_runtime_search_or_oracle(self):
        inputs = ({"a": [[0]*16 for _ in range(16)], "b": [[0]*16 for _ in range(16)]},
                  {"a": [[16*y+x for x in range(16)] for y in range(16)],
                   "b": [[255-16*y-x for x in range(16)] for y in range(16)]})
        expected = [{"output": difference(images)} for images in inputs]
        self.assertNotEqual(*expected)
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile(media.absdiff, target=target, width=16, height=16, input_ranges=BYTE_RANGES)
                tail, streams = program.tail, []
                with patch("codec_compute.ir.evaluate", side_effect=AssertionError("runtime oracle")), \
                     patch.object(program, "reference", side_effect=AssertionError("runtime reference")), \
                     patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("runtime synthesis")), \
                     patch("codec_compute.synthesis.search.synthesize", side_effect=AssertionError("runtime search")):
                    for images, output in zip(inputs, expected):
                        streams.append(program.encode(images))
                        self.assertEqual(program.decode(images), output)
                        self.assertTrue(streams[-1].endswith(tail))
                self.assertNotEqual(*streams)
                self.assertEqual(program.tail, tail)


if __name__ == "__main__":
    unittest.main()
