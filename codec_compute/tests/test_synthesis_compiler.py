"""Synthesis composition and actual software decoder execution are separate checks."""

import json
from operator import xor
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute import ir
from codec_compute.av1.bootstrap import bridge_path
from codec_compute.synthesis import Instruction, Plan, library_for, verify_record
from codec_compute.synthesis.lower import SynthesisError, lower_graph


BYTE_RANGES = {"a": (0, 255), "b": (0, 255)}
BOOL_RANGES = {"a": (0, 1), "b": (0, 1)}


def boolean_inputs():
    return {"a": [[x % 2 for x in range(16)] for _ in range(16)],
            "b": [[(x // 2) % 2 for x in range(16)] for _ in range(16)]}


class SynthesisCompilerTests(unittest.TestCase):
    def test__given_every_byte_pair__when_synthesized_extrema_decode__then_both_targets_match_specification(self):
        def kernel(a, b):
            return {"minimum": media.minimum(a, b), "maximum": media.maximum(a, b), "input": a}
        graph = media.trace(kernel, width=64, height=64, input_ranges=BYTE_RANGES)
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile_graph(graph, target=target)
                self.assertEqual(len(program.synthesis_records), 2)
                self.assertTrue(all(record["search"]["verification"]["combinations_checked"] == 65536
                                    for record in program.synthesis_records))
                for chunk in range(16):
                    inputs = {"a": [[(chunk * 4096 + y * 64 + x) // 256 for x in range(64)] for y in range(64)],
                              "b": [[(y * 64 + x) % 256 for x in range(64)] for y in range(64)]}
                    program.verify(inputs)

    def test__given_all_boolean_pairs__when_synthesized_gates_decode__then_every_requested_gate_is_exact(self):
        for target in ("h264", "hevc"):
            for operation in (media.bit_and, media.bit_or, media.bit_xor):
                with self.subTest(target=target, operation=operation.__name__):
                    program = media.compile(operation, target=target, width=16, height=16, input_ranges=BOOL_RANGES)
                    program.verify(boolean_inputs())
                    self.assertEqual(program.synthesis_records[0]["search"]["verification"]["combinations_checked"], 4)

    def test__given_correlated_boolean_intermediates__when_composed__then_conservative_lowered_ranges_remain_valid(self):
        def kernel(a, b):
            different = media.bit_xor(a, b)
            return {"not": media.bit_not(different), "and": media.bit_and(different, a),
                    "xor": different, "same": different}
        graph = media.trace(kernel, width=16, height=16, input_ranges=BOOL_RANGES)
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile_graph(graph, target=target)
                ir.validate_graph(program.lowered_graph)
                # Interval analysis cannot retain XOR's correlation. The
                # original proven range stays on the semantic graph only.
                self.assertEqual(dict(graph.outputs)["xor"].hi, 1)
                self.assertEqual(dict(program.lowered_graph.outputs)["xor"].hi, 2)
                program.verify(boolean_inputs())
                self.assertEqual(program.manifest()["output_frames"]["xor"],
                                 program.manifest()["output_frames"]["same"])

    def test__given_av1_average_only_library__when_searching_other_goals__then_failures_expose_bounds(self):
        for operation in (media.minimum, media.maximum, media.bit_and, media.bit_xor):
            ranges = BOOL_RANGES if operation in (media.bit_and, media.bit_xor) else BYTE_RANGES
            with self.subTest(operation=operation.__name__), self.assertRaises(SynthesisError) as caught:
                media.compile(operation, target="av1", width=16, height=16, input_ranges=ranges)
            self.assertEqual(caught.exception.result.status, "instruction_bound")
            self.assertIn("not a codec impossibility", str(caught.exception))

    def test__given_av1_or__when_synthesized_and_decoded__then_two_software_decoders_agree(self):
        try:
            bridge_path()
        except RuntimeError as error:
            self.skipTest(str(error))
        program = media.compile(media.bit_or, target="av1", width=16, height=16, input_ranges=BOOL_RANGES)
        self.assertEqual(program.synthesis_records[0]["search"]["plan"]["instructions"][0]["primitive"], "rounded_average")
        for decoder in ("libdav1d", "libaom-av1"):
            with self.subTest(decoder=decoder):
                program.verify(boolean_inputs(), decoder=decoder)

    def test__given_a_compiled_plan__when_inputs_change__then_encoding_uses_neither_search_nor_oracle(self):
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                program = media.compile(media.bit_xor, target=target, width=16, height=16, input_ranges=BOOL_RANGES)
                tail, streams = program.tail, []
                for swap in (False, True):
                    inputs = boolean_inputs()
                    if swap:
                        inputs["b"] = [[1 - value for value in row] for row in inputs["b"]]
                    expected = program.reference(inputs)
                    with patch("codec_compute.ir.evaluate", side_effect=AssertionError("host computation")), \
                         patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("runtime synthesis")):
                        streams.append(program.encode(inputs))
                        self.assertEqual(program.decode(inputs), expected)
                    self.assertTrue(streams[-1].endswith(tail))
                self.assertNotEqual(*streams)

    def test__given_serialized_program_evidence__when_replayed__then_plan_and_specification_remain_bound(self):
        program = media.compile(media.bit_xor, target="h265", width=16, height=16, input_ranges=BOOL_RANGES)
        manifest = json.loads(json.dumps(program.manifest()))
        record = manifest["synthesis"][0]
        encoded_plan = record["search"]["plan"]
        plan = Plan(encoded_plan["input_count"], tuple(Instruction(step["primitive"], step["operands"])
                    for step in encoded_plan["instructions"]), encoded_plan["output"])
        fresh = verify_record(plan, xor, ((0, 1), (0, 1)), library=library_for("hevc"),
                              record=record["search"]["verification"])
        self.assertTrue(fresh["verified"])
        self.assertFalse(fresh["bitstream_verified"])
        self.assertEqual(manifest["ir"]["nodes"][record["source_node"]]["op"], "xor")
        self.assertEqual(manifest["lowered_ir"]["outputs"]["output"], record["lowered_output_node"])
        record["search"]["verification"]["verified"] = False
        self.assertTrue(program.synthesis_records[0]["search"]["verification"]["verified"])

    def test__given_insufficient_search_budget__when_retried_with_more__then_failure_is_not_cached_as_impossible(self):
        graph = media.trace(media.minimum, width=16, height=16, input_ranges=BYTE_RANGES)
        with self.assertRaises(SynthesisError) as caught:
            media.compile_graph(graph, synthesis_max_candidates=0)
        self.assertEqual(caught.exception.result.status, "candidate_budget")
        with self.assertRaises(SynthesisError) as caught:
            media.compile_graph(graph, synthesis_max_instructions=1)
        self.assertEqual(caught.exception.result.status, "instruction_bound")
        program = media.compile_graph(graph, synthesis_max_instructions=2)
        self.assertTrue(program.synthesis_records[0]["search"]["verification"]["verified"])

    def test__given_forged_semantic_range__when_lowering__then_search_never_uses_untrusted_domains(self):
        graph = media.trace(media.bit_xor, width=16, height=16, input_ranges=BOOL_RANGES)
        original = graph.outputs[0][1]
        forged = ir.Image(original.op, original.args, original.params, 0, 0)
        graph = ir.Graph(16, 16, graph.inputs, (("output", forged),), (*graph.inputs, forged))
        with patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("unvalidated search")):
            with self.assertRaisesRegex(media.CompileError, "range proof"):
                lower_graph(graph, target="h264")

    def test__given_byte_operands__when_boolean_xor_is_requested__then_bitwise_bytes_are_not_silently_supported(self):
        with self.assertRaisesRegex(media.CompileError, "proven 0/1"):
            media.trace(media.bit_xor, width=16, height=16, input_ranges=BYTE_RANGES)

    def test__given_synthesis_disabled__when_compiling__then_existing_direct_gates_still_work(self):
        program = media.compile(media.bit_and, width=16, height=16, input_ranges=BOOL_RANGES, synthesis=False)
        self.assertEqual(program.synthesis_records, [])
        program.verify(boolean_inputs())
        with self.assertRaises(media.UnsupportedOperation):
            media.compile(media.minimum, width=16, height=16, input_ranges=BYTE_RANGES, synthesis=False)

    def test__given_safe_strict_affine_after_synthesis__when_lowered__then_offset_splitting_remains_available(self):
        def kernel(a, b):
            return media.affine(media.bit_or(a, b), weights=(-1,), bias=255)
        program = media.compile(kernel, target="h264", width=16, height=16, input_ranges=BOOL_RANGES)
        self.assertEqual(program.lowered_graph.outputs[0][1].op, "affine")
        program.verify(boolean_inputs())

    def test__given_scalar_proofs_but_too_many_live_values__when_scheduled__then_backend_limits_still_apply(self):
        def kernel(a, b):
            low, high = media.minimum(a, b), media.maximum(a, b)
            return media.add_sat(media.add_sat(low, high), media.average(a, b))
        graph = media.trace(kernel, width=16, height=16, input_ranges=BYTE_RANGES)
        _, records = lower_graph(graph, target="hevc")
        self.assertTrue(all(record["search"]["verification"]["verified"] for record in records))
        with self.assertRaisesRegex(media.UnsupportedOperation, "three simultaneously retained references"):
            media.compile_graph(graph, target="hevc")


if __name__ == "__main__":
    unittest.main()
