"""Real HEVC decode checks. Expected values exist only in tests."""
import random
import shutil
import unittest
from unittest.mock import patch

from codec_compute.hevc import NativeProgram, NativeStep, compile_binary, compile_unary, decode_frames


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
        with self.assertRaisesRegex(ValueError, "three simultaneously retained"):
            NativeProgram(32, 32, ("a", "b"), steps)


if __name__ == "__main__":
    unittest.main()
