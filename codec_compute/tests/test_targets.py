"""Cross-codec exactness on real decoders, including nonportable operations."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute.av1.bootstrap import bridge_path


def targets():
    try:
        bridge_path()
    except RuntimeError:
        return ("h264", "hevc")
    return ("h264", "hevc", "av1")


RANGES = {"a": (0, 255), "b": (0, 255)}


class CrossCodecTests(unittest.TestCase):
    def test__given_every_u8_pair__when_averaged__then_all_targets_match_exact_rounding(self):
        # Small chunks respect the existing H.264 writer's level-3.1 PCM
        # access-unit limit while still covering the complete Cartesian domain.
        graph = media.trace(media.average, width=64, height=64, input_ranges=RANGES)
        for target in targets():
            with self.subTest(target=target):
                program = media.compile_graph(graph, target=target)
                for chunk in range(16):
                    inputs = {"a": [[(chunk * 4096 + y * 64 + x) // 256 for x in range(64)] for y in range(64)],
                              "b": [[(y * 64 + x) % 256 for x in range(64)] for y in range(64)]}
                    program.verify(inputs)

    def test__given_clipped_chain__when_h264_and_hevc_execute__then_observed_intermediates_are_exact(self):
        def kernel(a, b):
            total = media.add_sat(a, b)
            remainder = media.sub_sat(total, b)
            return {"result": media.average(remainder, a), "sum": total, "remainder": remainder, "input": a}
        graph = media.trace(kernel, width=32, height=32, input_ranges=RANGES)
        inputs = {"a": [[(y * 37 + x * 19) % 256 for x in range(32)] for y in range(32)],
                  "b": [[(y * 13 + x * 73) % 256 for x in range(32)] for y in range(32)]}
        for target in ("h264", "hevc"):
            with self.subTest(target=target):
                media.compile_graph(graph, target=target).verify(inputs)

    def test__given_composed_means_and_changed_inputs__when_encoding__then_only_uploads_depend_on_pixels(self):
        def kernel(a, b):
            first = media.average(a, b)
            return {"output": media.average(first, a), "first": first, "input": b}
        graph = media.trace(kernel, width=16, height=16, input_ranges=RANGES)
        rng = random.Random(2031)
        for target in targets():
            with self.subTest(target=target):
                program = media.compile_graph(graph, target=target)
                tail = program.tail
                streams = []
                for _ in range(3):
                    inputs = {name: [[rng.randrange(256) for _ in range(16)] for _ in range(16)] for name in RANGES}
                    expected = program.reference(inputs)
                    with patch("codec_compute.ir.evaluate", side_effect=AssertionError("host evaluation during encoding")):
                        streams.append(program.encode(inputs))
                        self.assertEqual(program.decode(inputs), expected)
                    self.assertTrue(streams[-1].endswith(tail))
                self.assertEqual(len(set(streams)), 3)

    def test__given_binary_operands__when_or_is_lowered__then_av1_average_implements_the_same_gate(self):
        graph = media.trace(media.bit_or, width=16, height=16, input_ranges={"a": (0, 1), "b": (0, 1)})
        inputs = {"a": [[x % 2 for x in range(16)] for _ in range(16)],
                  "b": [[(x // 2) % 2 for x in range(16)] for _ in range(16)]}
        for target in targets():
            with self.subTest(target=target):
                media.compile_graph(graph, target=target).verify(inputs)

    def test__given_named_input_aliases__when_saved__then_manifests_and_decoded_inputs_are_correct(self):
        graph = media.trace(lambda a, b: {"one": a, "same": a, "two": b}, width=16, height=16, input_ranges=RANGES)
        inputs = {"a": [[0] * 16 for _ in range(16)], "b": [[255] * 16 for _ in range(16)]}
        for target in targets():
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                program = media.compile_graph(graph, target=target)
                program.verify(inputs)
                path = program.save(Path(directory) / ("program" + program.native.suffix), inputs)
                manifest = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(manifest["output_frames"]["one"], manifest["output_frames"]["same"])
                self.assertEqual(manifest["ir"]["format"], "codec-compute-ir-2")
                self.assertEqual(manifest["target"], target)

    def test__given_av1_nonportable_operations__when_lowering__then_supported_clamp_and_rejections_are_explicit(self):
        with self.assertRaisesRegex(media.UnsupportedOperation, "exact synthesis is unimplemented"):
            media.compile(media.add_sat, target="av1", input_ranges=RANGES)
        with self.assertRaisesRegex(media.UnsupportedOperation, "zero-border"):
            media.compile(lambda x: media.shift(x, 1), target="av1")
        with self.assertRaisesRegex(media.UnsupportedOperation, "clamp"):
            media.compile(lambda x: media.shift(x, 1, border="clamp"), target="h264")
        if "av1" in targets():
            program = media.compile(lambda x: media.shift(x, -3, 5, border="clamp"), target="av1", width=16, height=16)
            program.verify({"image": [[y * 16 + x for x in range(16)] for y in range(16)]})

    def test__given_hevc_specific_shape__when_compiling__then_ir_does_not_inherit_h264_geometry(self):
        graph = media.trace(lambda x: media.copy(x), width=24, height=16)
        media.compile_graph(graph, target="hevc").verify({"image": [[17] * 24 for _ in range(16)]})
        with self.assertRaises(media.UnsupportedOperation):
            media.compile_graph(graph, target="h264")

    def test__given_unsupported_coefficients__when_lowering__then_every_target_reports_its_limitation(self):
        graph = media.trace(lambda x: media.affine(x, weights=(1000,), saturating=True), width=16, height=16)
        for target in ("h264", "hevc", "av1"):
            with self.subTest(target=target), self.assertRaises(media.UnsupportedOperation) as error:
                media.compile_graph(graph, target=target)
            self.assertEqual(error.exception.target, target)


if __name__ == "__main__":
    unittest.main()
