"""Real HEVC decode checks. Expected values exist only in tests."""
import random
import shutil
import unittest
from unittest.mock import patch

from codec_compute.hevc import HEVCError, Kernel, NativeProgram, NativeStep, compile_binary, compile_unary, decode_frames


def patterns(width=32, height=32):
    size = width * height
    rng = random.Random(227)
    return [bytes(size), bytes([255]) * size,
            bytes(i % 256 for i in range(size)),
            bytes(255 * ((i + i // width) % 2) for i in range(size)),
            bytes(rng.randrange(256) for _ in range(size))]


def clip(value):
    return max(0, min(255, value))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class HEVCNativeTests(unittest.TestCase):
    def test_given_u8_patterns_when_unary_predictions_decode_then_match_exact_arithmetic(self):
        for weight, offset, denominator in [(1, 0, 0), (2, -32, 0), (-1, 127, 0),
                                             (0, 45, 0), (3, -15, 2), (-7, 127, 3)]:
            kernel = compile_unary(32, 32, weight, offset, denominator)
            original_tail = kernel.tail
            for source in patterns():
                rounding = (1 << (denominator - 1)) if denominator else 0
                expected = bytes(clip(((weight * pixel + rounding) >> denominator) + offset)
                                 for pixel in source)
                with self.subTest(weight=weight, offset=offset, denominator=denominator, pattern=source[:8]):
                    self.assertEqual(kernel.decode(source), expected)
                    self.assertEqual(kernel.tail, original_tail)
                    self.assertTrue(kernel.assemble(source).endswith(original_tail))

    def test_given_independent_planes_when_binary_predictions_decode_then_match_add_subtract_average(self):
        for left_weight, right_weight, offset, denominator in [(2, 2, 0, 0), (2, -2, 0, 0),
                                                                (1, 1, 0, 0), (3, -5, 7, 2)]:
            kernel = compile_binary(32, 32, left_weight, right_weight, offset, denominator)
            original_tail = kernel.tail
            sources = patterns()
            for index, left in enumerate(sources):
                right = sources[(index + 2) % len(sources)]
                expected = bytes(clip(((left_weight * a + right_weight * b + (1 << denominator))
                                       >> (denominator + 1)) + offset)
                                 for a, b in zip(left, right))
                with self.subTest(weights=(left_weight, right_weight), pattern=index):
                    stream = kernel.assemble(left, right)
                    decoded = decode_frames(stream, 32, 32)
                    self.assertEqual(decoded, [left, expected, right])
                    self.assertTrue(stream.endswith(original_tail))

    def test_given_multiple_ctus_when_prediction_decodes_then_all_blocks_compute(self):
        for width, height in [(16, 16), (40, 24), (64, 48)]:
            left, right = patterns(width, height)[-2:]
            kernel = compile_binary(width, height, 2, -2)
            self.assertEqual(kernel.decode(left, right), bytes(clip(a-b) for a, b in zip(left, right)))

    def test_given_unsupported_parameters_when_compiling_then_reject_before_encoding(self):
        for arguments in [(8, 8), (32, 31), (32, 32, 1, 128), (32, 32, 1, 0, 8)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compile_unary(*arguments)

    def test_given_wrong_input_shape_when_assembling_then_reject(self):
        kernel = compile_unary(32, 32)
        with self.assertRaises(ValueError):
            kernel.assemble(bytes(1023))
        with self.assertRaises(ValueError):
            kernel.assemble([[1] * 32] * 31)

    def test_given_resident_dag_when_decoded_then_intermediates_remain_decoder_computed(self):
        from codec_compute.hevc.program import _encode
        program = NativeProgram(32, 32, ("a", "b"), (
            NativeStep("sum", ("a", "b"), (2, 2)),
            NativeStep("difference", ("sum", "b"), (2, -2)),
            NativeStep("result", ("difference", "a"), (1, 1)),
        ))
        self.assertEqual(program.max_live_references, 3)
        original_tail = program.tail
        inputs = patterns()
        for index, a in enumerate(inputs):
            b = inputs[(index + 2) % len(inputs)]
            summed = bytes(clip(x + y) for x, y in zip(a, b))
            difference = bytes(clip(x - y) for x, y in zip(summed, b))
            expected = bytes((x + y + 1) // 2 for x, y in zip(difference, a))
            with patch("codec_compute.hevc.program._encode", wraps=_encode) as encoder:
                actual = program.decode({"a": a, "b": b})
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(encoder.call_args.args[2], [a, a, b])
            self.assertEqual(actual, dict(a=a, b=b, sum=summed, difference=difference, result=expected))
            self.assertTrue(program.assemble({"a": a, "b": b}).endswith(original_tail))
        self.assertEqual(dict(program.output_frame_indices), dict(a=0, b=1, sum=2, difference=3, result=4))

    def test_given_long_unary_chain_when_decoded_then_dead_references_release(self):
        source = patterns()[-1]
        steps = tuple(NativeStep(f"stage{index}", ("a" if index == 0 else f"stage{index-1}",),
                                 (1,), 3 if index % 2 == 0 else -2)
                      for index in range(40))
        program = NativeProgram(32, 32, ("a",), steps)
        actual = program.decode({"a": source})
        expected = source
        for index in range(40):
            expected = bytes(clip(value + (3 if index % 2 == 0 else -2)) for value in expected)
            self.assertEqual(actual[f"stage{index}"], expected)
        self.assertEqual(program.max_live_references, 1)

    def test_given_repeated_operand_when_binary_prediction_decodes_then_both_weights_apply(self):
        source = patterns()[-1]
        program = NativeProgram(40, 24, ("a",), (NativeStep("result", ("a", "a"), (2, -2)),))
        self.assertEqual(program.decode({"a": source[:960]})["result"], bytes(960))

    def test_given_excess_reference_lifetime_when_compiling_then_reject(self):
        steps = (
            NativeStep("c", ("a",), (1,)), NativeStep("d", ("b",), (1,)),
            NativeStep("e", ("a", "b"), (2, 2)), NativeStep("f", ("c", "d"), (2, 2)),
        )
        with patch("codec_compute.hevc.program._template") as encoder:
            with self.assertRaisesRegex(ValueError, "three simultaneously retained"):
                NativeProgram(32, 32, ("a", "b"), steps)
        encoder.assert_not_called()

    def test_given_every_u8_pair_when_native_arithmetic_decodes_then_all_65536_pairs_match(self):
        a = bytes(index // 256 for index in range(65536))
        b = bytes(index % 256 for index in range(65536))
        operations = {"add": (2, 2), "subtract": (2, -2), "average": (1, 1)}
        expected = {name: bytes(clip((wa*x + wb*y + 1) // 2) for x, y in zip(a, b))
                    for name, (wa, wb) in operations.items()}
        for name, weights in operations.items():
            with self.subTest(name=name):
                self.assertEqual(compile_binary(256, 256, *weights).decode(a, b), expected[name])
        program = NativeProgram(256, 256, ("a", "b"),
                                tuple(NativeStep(name, ("a", "b"), weights)
                                      for name, weights in operations.items()))
        self.assertEqual(program.decode({"a": a, "b": b}), dict(a=a, b=b, **expected))

    def test_given_weight_syntax_boundaries_when_predictions_decode_then_exact_signed_rounding_holds(self):
        source = bytes(range(256))
        for denominator in range(8):
            for delta, offset in [(-128, 127), (127, -128)]:
                weight = (1 << denominator) + delta
                rounding = (1 << (denominator - 1)) if denominator else 0
                expected = bytes(clip(((weight * value + rounding) >> denominator) + offset)
                                 for value in source)
                with self.subTest(denominator=denominator, delta=delta, offset=offset):
                    self.assertEqual(compile_unary(16, 16, weight, offset, denominator).decode(source), expected)

    def test_given_every_u8_value_when_all_shared_threshold_cutoffs_decode_then_match_predicates(self):
        from codec_compute import compile, threshold
        source = [list(range(y * 16, (y + 1) * 16)) for y in range(16)]
        # Group outputs to exercise long resident schedules while keeping each
        # target program below its 128-instruction limit.
        for start in range(-1, 258, 40):
            cutoffs = tuple(range(start, min(start + 40, 258)))
            program = compile(lambda image: {f"cutoff_{c}": threshold(image, c) for c in cutoffs},
                              target="hevc", width=16, height=16)
            actual = program.decode({"image": source})
            for cutoff in cutoffs:
                with self.subTest(cutoff=cutoff):
                    self.assertEqual(actual[f"cutoff_{cutoff}"],
                                     [[int(value >= cutoff) for value in row] for row in source])

    def test_given_maximum_stage_count_when_old_reference_is_used_then_poc_and_liveness_remain_exact(self):
        a, b = bytes(range(256)), bytes(reversed(range(256)))
        steps = tuple(NativeStep(f"s{i}", ("a" if i == 0 else f"s{i-1}",), (1,),
                                 1 if i % 2 == 0 else -1) for i in range(127))
        steps += (NativeStep("result", ("s126", "a"), (2, -2)),)
        program = NativeProgram(16, 16, ("a", "b"), steps)
        result = program.decode({"a": a, "b": b})
        self.assertEqual(len(result), 130)
        self.assertEqual(result["result"], bytes([1] * 255 + [0]))
        self.assertEqual(program.picture_order_counts["result"], 130)
        self.assertEqual(program.output_frame_indices["result"], 129)
        self.assertEqual(program.max_live_references, 2)

    def test_given_boundary_shapes_when_resident_arithmetic_decodes_then_all_pixels_match(self):
        for width, height in [(512, 512), (16, 512), (512, 16)]:
            with self.subTest(shape=(width, height)):
                a = bytes((i * 17 + i // width * 9) % 256 for i in range(width * height))
                b = a[::-1]
                program = NativeProgram(width, height, ("a", "b"), (
                    NativeStep("sum", ("a", "b"), (2, 2)),
                    NativeStep("result", ("sum", "a"), (2, -2)),
                ))
                self.assertEqual(program.decode({"a": a, "b": b})["result"],
                                 bytes(min(255, x + y) - x for x, y in zip(a, b)))

    def test_given_invalid_names_or_resources_when_compiling_then_reject_before_encoding(self):
        valid = NativeStep("result", ("a",), (1,))
        cases = [
            (("a", "a"), (valid,)), (("a", "b", "c"), (valid,)), (("",), (valid,)),
            (("a",), ()), (("a",), (object(),)), (("a",), (valid,) * 129),
            (("a",), (valid, valid)), (("a",), (NativeStep("a", ("a",), (1,)),)),
            (("a",), (NativeStep("result", ("absent",), (1,)),)),
            (("a",), (NativeStep("first", ("later",), (1,)), NativeStep("later", ("a",), (1,)))),
        ]
        with patch("codec_compute.hevc.program._template") as encoder:
            for names, steps in cases:
                with self.subTest(names=names, steps=steps), self.assertRaises(ValueError):
                    NativeProgram(16, 16, names, steps)
        encoder.assert_not_called()
        for name, inputs in [("", ("a",)), ("result", ("",)), ("result", (42,)), ("result", ([],))]:
            with self.subTest(name=name, inputs=inputs), self.assertRaises(ValueError):
                NativeStep(name, inputs, (1,))

    def test_given_missing_computation_picture_when_decoding_then_frame_count_error_is_explicit(self):
        source = bytes(range(256))
        kernel = compile_unary(16, 16)
        prefix = kernel.assemble(source)[:-len(kernel.tail)]
        with patch.object(Kernel, "assemble", return_value=prefix):
            with self.assertRaisesRegex(HEVCError, "unexpected decoded frame count"):
                kernel.decode(source)
        program = NativeProgram(16, 16, ("a",), (NativeStep("result", ("a",), (1,)),))
        prefix = program.assemble({"a": source})[:-len(program.tail)]
        with patch.object(NativeProgram, "assemble", return_value=prefix):
            with self.assertRaisesRegex(HEVCError, "unexpected frame count"):
                program.decode({"a": source})

    def test_given_invalid_encoder_schedule_when_uploading_then_reject(self):
        source = bytes(range(256))
        program = NativeProgram(16, 16, ("a",), (NativeStep("result", ("a",), (1,)),))
        headers = b"".join(b"\x00\x00\x00\x01" + nal for nal in program.parameter_sets)
        with patch("codec_compute.hevc.program._encode", return_value=headers):
            with self.assertRaisesRegex(HEVCError, "unexpected HEVC.*schedule"):
                program.assemble({"a": source})
        with self.assertRaisesRegex(ValueError, "input names exactly"):
            program.assemble(["a"])

    def test_given_malformed_hevc_when_decoding_then_decoder_failure_propagates(self):
        with self.assertRaises(HEVCError):
            decode_frames(b"\x00\x00\x00\x01\x02\x01", 16, 16)


if __name__ == "__main__":
    unittest.main()
