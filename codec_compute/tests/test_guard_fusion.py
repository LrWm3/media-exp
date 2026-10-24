"""Shared H.264 compilation enables zero-guard row merging explicitly."""

from pathlib import Path
import tempfile
import unittest

import codec_compute as media
from decoder_core.bitstream import Writer
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture


class H264GuardFusionTests(unittest.TestCase):
    def test__given_nonboolean_guard_flag__when_using_either_entry_point__then_validation_rejects_it(self):
        graph = media.trace(media.copy, width=16, height=16)
        for value in (None, 0, 1, "false", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(media.CompileError, "h264_merge_zero_guards must be Boolean"):
                    media.compile(media.copy, width=16, height=16, h264_merge_zero_guards=value)
                with self.assertRaisesRegex(media.CompileError, "h264_merge_zero_guards must be Boolean"):
                    media.compile_graph(graph, h264_merge_zero_guards=value)

    def test__given_uniform_average_at_three_sizes__when_compiling__then_default_merges_and_false_preserves_old_writer_bytes(self):
        for size, level, old, new in ((128, 31, 26, 10), (256, 41, 50, 18), (512, 51, 98, 34)):
            graph = media.trace(media.average, width=size, height=size, input_ranges={"a": (0, 255), "b": (0, 255)})
            baseline = media.compile_graph(graph, h264_level=level, h264_merge_zero_guards=False)
            merged = media.compile_graph(graph, h264_level=level)
            native = baseline.native
            legacy_writer = Writer(size, size, slice_mode="row", reference_slots=native.reference_slots,
                                   level_idc=level, motion_bits=native.writer.motion_bits)
            self.assertEqual(baseline.tail, b"".join(legacy_writer.prediction(frame, op)
                             for frame, op in enumerate(native.operations, 2)))
            for program, enabled, count in ((baseline, False, old), (merged, True, new)):
                with self.subTest(size=size, enabled=enabled):
                    manifest = program.manifest()
                    self.assertIs(manifest["merge_zero_guards"], enabled)
                    self.assertEqual(manifest["optimizations"]["zero_guard_merge"],
                                     {"enabled": enabled, "eligible_pictures": 1, "merged_pictures": int(enabled)})
                    self.assertEqual(manifest["costs"]["prediction_slices"], count)
                    self.assertEqual(manifest["costs"]["total_pictures"], 3)
                    self.assertEqual(program.tail.count(b"\0\0\0\1"), count)
            self.assertLess(len(merged.tail), len(baseline.tail))

    def test__given_bias_and_already_zero_predictions__when_reporting__then_unchanged_pictures_are_not_claimed_as_merges(self):
        for kernel, eligible in ((lambda a: media.affine(a, weights=(1,), bias=1, saturating=True), 0),
                                 (lambda a: media.affine(a, weights=(0,)), 1)):
            baseline = media.compile(kernel, h264_merge_zero_guards=False)
            merged = media.compile(kernel)
            self.assertEqual(merged.tail, baseline.tail)
            self.assertEqual(merged.manifest()["optimizations"]["zero_guard_merge"],
                             {"enabled": True, "eligible_pictures": eligible, "merged_pictures": 0})

    def test__given_independent_fractional_fields_and_changing_inputs__when_decoding_both_guard_modes__then_full_i420_matches_ir(self):
        size = 64
        first = tuple((index % 4-4, index//4-4) for index in range(16))
        second = tuple((3-index % 4, 3-index//4) for index in range(16))

        def kernel(a, b):
            return {"blend": media.average(media.gather_h264(a, first), media.gather_h264(b, second))}

        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        programs = [media.compile(kernel, width=size, height=size, input_ranges={"a": (0, 255), "b": (0, 255)},
                                  h264_merge_zero_guards=enabled) for enabled in (False, True)]
        self.assertEqual([p.manifest()["costs"]["computation_pictures"] for p in programs], [1, 1])
        self.assertEqual([p.manifest()["costs"]["prediction_slices"] for p in programs], [14, 6])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"guard.h264"
            for fixture in range(2):
                inputs = {"a": [[(x*37+y*71+fixture*19) % 256 for x in range(size)] for y in range(size)],
                          "b": [[255*((x+3*y+fixture) % 7 < 3) for x in range(size)] for y in range(size)]}
                expected = programs[0].reference(inputs)
                raw = b"".join(i420_picture(board) for board in (inputs["a"], inputs["b"], expected["blend"]))
                for program in programs:
                    path.write_bytes(program.encode(inputs))
                    for backend in backends:
                        with self.subTest(fixture=fixture, merged=program.native.writer.merge_zero_guards, decoder=backend):
                            self.assertEqual(decode(path, backend), raw)


if __name__ == "__main__":
    unittest.main()
