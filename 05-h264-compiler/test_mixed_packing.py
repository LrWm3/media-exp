"""Mixed packed pictures preserve signed, rounded P and B predictions exactly."""

from pathlib import Path
import sys
import tempfile
import unittest

import h264
from decoder_core.bitstream import Operation, Source
from decoder_core.decode import available, decode
from h264.packed import _promote_prediction


def prediction(operation, values):
    """Independent scalar H.264 weighted prediction, clipping only at the end."""
    denominator = operation.denominator
    total = sum(source.weight * value for source, value in zip(operation.sources, values))
    if len(operation.sources) == 1:
        rounded = ((total + (1 << (denominator - 1))) >> denominator) if denominator else total
        offset = operation.sources[0].offset
    else:
        rounded = (total + (1 << denominator)) >> (denominator + 1)
        offset = (sum(source.offset for source in operation.sources) + 1) >> 1
    return min(255, max(0, rounded + offset))


def chains(a, b):
    positive = h264.saturating_weighted(a, 2, -31)
    combined = h264.saturating_combine(a, b, 1, -1, 63)
    negative = h264.saturating_weighted(b, -1, 127)
    duplicate = h264.saturating_combine(a, a, 1, -1, 37)
    unary_chain = h264.saturating_weighted(combined, -1, 127)
    binary_chain = h264.saturating_combine(positive, negative, 1, -1, 59)
    return {"a": a, "b": b, "positive": positive, "combined": combined,
            "negative": negative, "duplicate": duplicate, "unary_chain": unary_chain,
            "binary_chain": binary_chain, "alias": unary_chain,
            "left": h264.shift(unary_chain, -16), "right": h264.shift(binary_chain, 16),
            "up": h264.shift(unary_chain, dy=-16), "down": h264.shift(binary_chain, dy=16),
            "corner": h264.shift(duplicate, -16, 16)}


class MixedPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))

    def require_decoder(self):
        if not self.backends:
            self.skipTest("No FFmpeg, OpenH264, or VAAPI decoder available")

    def test__given_all_u8_values_and_signed_weights__when_promoting_legal_p_sources__then_b_rounding_is_identical(self):
        # Every input/weight/denominator, with signed offset boundaries and
        # adjacent offset values exercising both clipping and odd averages.
        for denominator in range(8):
            for weight in range(-128, 128):
                for offset in (-128, -127, -1, 0, 1, 126, 127):
                    source = Source(0, weight, offset)
                    unary = Operation("P", (source,), denominator)
                    binary = _promote_prediction(unary)
                    with self.subTest(denominator=denominator, weight=weight, offset=offset):
                        if denominator == 0 and not -64 <= weight <= 64:
                            self.assertIsNone(binary)
                            continue
                        self.assertIsNotNone(binary)
                        self.assertTrue(-128 <= sum(ref.weight for ref in binary.sources) <=
                                        (127 if binary.denominator == 7 else 128))
                        self.assertEqual([prediction(unary, (value,)) for value in range(256)],
                                         [prediction(binary, (value, value)) for value in range(256)])

    def test__given_all_denominators_and_signed_extremes__when_decoding_promoted_predictions__then_all_u8_pixels_match(self):
        self.require_decoder()
        # The image DSL does not expose arbitrary codec denominators. Use a
        # Program's low-level operations to exercise their complete legal span.
        original = h264.compile(lambda x: x, 16, 16, input_range=(0, 255))
        original.operations = []
        image = [[16*y+x for x in range(16)] for y in range(16)]
        expected = {"input": image}
        original.output_frames = {"input": 0}
        original.graph_outputs = {"input": 0}
        for denominator in range(8):
            for index, weight in enumerate((-128, -1, 1, 127)):
                offset = (-128, 127, -1, 1)[index]
                unary = Operation(f"P d={denominator} w={weight}", (Source(0, weight, offset),), denominator)
                binary = Operation("duplicate B reference", (Source(0, 1), Source(0, -1, 127)), denominator)
                for operation in (unary, binary):
                    original.operations.append(operation)
                    name = str(len(original.operations))
                    original.output_frames[name] = len(original.operations)
                    original.graph_outputs[name] = 0
                    expected[name] = [[prediction(operation, (value,) * len(operation.sources))
                                       for value in row] for row in image]
        packed = h264.pack(original, tiles=2)
        self.assertGreater(packed.stats["p_to_b_promotions"], 0)
        self.assertGreater(packed.stats["mixed_prediction_pictures"], 0)
        for operation, group in zip(packed.operations, packed.groups):
            for tile, logical in zip(packed.tiles, group):
                unary = original.operations[logical-1]
                if len(unary.sources) != 1:
                    continue
                block = operation.block_operations[(tile["y"]//16)*(packed.canvas_width//16)+tile["x"]//16]
                if unary.denominator == 0 and abs(unary.sources[0].weight) > 64:
                    self.assertEqual(len(block.sources), 1)
                if len(block.sources) == 2:
                    expected_denominator = max(0, unary.denominator-1)
                    self.assertEqual(block.denominator, expected_denominator)
                    self.assertEqual(block.sources[0].offset, block.sources[1].offset)
                    self.assertEqual(block.sources[1].weight, 0 if unary.denominator else unary.sources[0].weight)
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(packed.decode(image, backend), expected)

    def test__given_mixed_chains_aliases_and_halo_reads__when_decoding__then_sources_outputs_and_guards_remain_exact(self):
        self.require_decoder()
        original = h264.compile(chains, 32, 32, input_ranges={"a": (0, 255), "b": (0, 255)},
                                optimize=False, reference_mode="retained")
        inputs = {"a": [[(y*31+x*7) % 256 for x in range(32)] for y in range(32)],
                  "b": [[(y*13+x*19) % 256 for x in range(32)] for y in range(32)]}
        expected = original.reference(inputs)
        packed = h264.pack(original, tiles=4)
        self.assertGreater(packed.stats["p_to_b_promotions"], 0)
        self.assertGreater(packed.stats["mixed_prediction_pictures"], 0)
        self.assertEqual(packed.output_regions["alias"], packed.output_regions["unary_chain"])
        for frame, (operation, group) in enumerate(zip(packed.operations, packed.groups), packed.input_count):
            for logical in group:
                for source in original.operations[logical-original.input_count].sources:
                    self.assertLess(packed.locations[source.frame][0], frame)
            self.assertTrue(all(source.frame < frame for child in operation.block_operations for source in child.sources))
            self.assertTrue(all(len(child.sources) == len(operation.sources) for child in operation.block_operations))
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(packed.verify(inputs, backend), expected)
                self.assertEqual(packed.decode(inputs, backend, readback="outputs"), expected)
        # Alter an actual mixed B picture, including its separator and chroma.
        frame = next(frame for frame, group in enumerate(packed.groups, packed.input_count)
                     if len({len(original.operations[value-original.input_count].sources) for value in group}) == 2)
        pixels = packed.writer.width*packed.writer.height
        start = frame*pixels*3//2
        offsets = (start, start+16*packed.writer.width+16+packed.width, start+pixels)
        with tempfile.TemporaryDirectory(prefix="mixed-packing-") as directory:
            path = Path(directory)/"mixed.h264"
            path.write_bytes(packed.encode(inputs))
            raw = decode(path, self.backends[0])
        for offset in offsets:
            damaged = bytearray(raw)
            damaged[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(AssertionError):
                packed.read(damaged)

    def test__given_only_p_predictions__when_packing__then_no_b_promotions_are_added(self):
        original = h264.compile(lambda x: {"positive": h264.saturating_weighted(x, 2, -31),
                                           "negative": h264.saturating_weighted(x, -1, 127)},
                                16, 16, input_range=(0, 255))
        packed = h264.pack(original, tiles=2)
        self.assertTrue(packed.stats["mixed_predictions"])
        self.assertEqual(packed.stats["p_to_b_promotions"], 0)
        self.assertEqual(packed.stats["mixed_prediction_pictures"], 0)
        self.assertTrue(all(len(op.sources) == 1 for op in packed.operations))

    def test__given_disabled_mixing__when_repacking__then_legacy_groups_and_bytes_are_deterministic(self):
        original = h264.compile(chains, 16, 16, input_ranges={"a": (0, 255), "b": (0, 255)},
                                optimize=False, reference_mode="retained")
        for tiles in (1, 2, 4, 8):
            with self.subTest(tiles=tiles):
                first = h264.pack(original, tiles, mixed_predictions=False, scheduling="stable")
                second = h264.PackedProgram(original, tiles, mixed_predictions=False, scheduling="stable")
                pending = dict(enumerate(original.operations, original.input_count))
                completed = set(range(original.input_count))
                expected = []
                while pending:
                    ready = [frame for frame, op in pending.items()
                             if all(source.frame in completed for source in op.sources)]
                    arity = len(pending[ready[0]].sources)
                    group = tuple(frame for frame in ready if len(pending[frame].sources) == arity)[:tiles]
                    expected.append(group)
                    completed.update(group)
                    for frame in group:
                        del pending[frame]
                self.assertEqual(first.groups, expected)
                self.assertEqual(first.operations, second.operations)
                self.assertEqual(first.tail, second.tail)
                self.assertEqual(first.output_regions, second.output_regions)
                self.assertEqual(first.reference_slots, second.reference_slots)
                self.assertFalse(first.stats["mixed_predictions"])
                self.assertEqual(first.stats["p_to_b_promotions"], 0)
                self.assertEqual(first.stats["mixed_prediction_pictures"], 0)
        single = h264.pack(original, tiles=1)
        self.assertEqual(single.tail, h264.pack(original, tiles=1, mixed_predictions=False).tail)

    def test__given_frozen_denoiser__when_mixing_four_or_eight_tiles__then_picture_counts_improve_or_stay_equal(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"03-denoising-decoder"/"compiled"))
        from denoiser import compile_model
        original = compile_model()
        for tiles, before, after in ((4, 20, 19), (8, 17, 17)):
            with self.subTest(tiles=tiles):
                separate = h264.pack(original, tiles, mixed_predictions=False, scheduling="stable")
                mixed = h264.pack(original, tiles, scheduling="stable")
                self.assertEqual(separate.pictures, before)
                self.assertEqual(mixed.pictures, after)
                self.assertLessEqual(mixed.pictures, separate.pictures)

    def test__given_nonboolean_mixing_flag__when_packing__then_configuration_is_rejected(self):
        original = h264.compile(lambda x: x, 16, 16)
        for flag in (None, 0, 1, "false"):
            with self.subTest(flag=flag), self.assertRaisesRegex(h264.CompileError, "mixed_predictions"):
                h264.pack(original, mixed_predictions=flag)


if __name__ == "__main__":
    unittest.main()
