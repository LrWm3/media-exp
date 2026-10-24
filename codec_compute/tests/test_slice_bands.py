"""Explicit H.264 row bands preserve shared IR semantics and cost reporting."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import codec_compute as media
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture


class H264SliceBandTests(unittest.TestCase):
    def test__given_invalid_or_non_h264_options__when_compiling__then_both_entry_points_reject_them(self):
        graph = media.trace(media.copy, width=16, height=16)
        for value in (None, True, False, -1, 1.0, "2", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(media.CompileError, "h264_prediction_slice_rows must be a nonnegative integer"):
                    media.compile(media.copy, width=16, height=16, h264_prediction_slice_rows=value)
                with self.assertRaisesRegex(media.CompileError, "h264_prediction_slice_rows must be a nonnegative integer"):
                    media.compile_graph(graph, h264_prediction_slice_rows=value)
        for target in ("hevc", "h265", "av1"):
            with self.subTest(target=target), self.assertRaisesRegex(media.CompileError, "applies only"):
                media.compile_graph(graph, target=target, h264_prediction_slice_rows=0)

    def test__given_uniform_programs_at_three_levels__when_selecting_bands__then_costs_and_manifest_match_emitted_slices(self):
        for size, level in ((128, 31), (256, 41), (512, 51)):
            graph = media.trace(media.average, width=size, height=size, input_ranges={"a": (0, 255), "b": (0, 255)})
            default = media.compile_graph(graph, h264_level=level)
            for rows in (1, 2, 4, 8, 0):
                program = media.compile_graph(graph, h264_level=level, h264_prediction_slice_rows=rows)
                manifest = program.manifest()
                coded_rows = size//16+2
                effective = rows or coded_rows
                expected = (coded_rows+effective-1)//effective
                with self.subTest(size=size, rows=rows):
                    self.assertEqual(manifest["prediction_slice_rows"], rows)
                    self.assertEqual(manifest["optimizations"]["prediction_slice_bands"],
                                     {"requested_rows": rows, "eligible_pictures": 1, "banded_pictures": int(rows != 1)})
                    self.assertEqual(manifest["costs"]["prediction_slices"], expected)
                    self.assertEqual(program.tail.count(b"\0\0\0\1"), expected)
                    self.assertEqual(manifest["costs"]["prediction_bytes"], len(program.tail))
                    self.assertEqual(manifest["costs"]["computation_pictures"], 1)
                    if rows == 1:
                        self.assertEqual(program.tail, default.tail)
                    else:
                        self.assertLess(len(program.tail), len(default.tail))

    def test__given_bias_or_disabled_guard_merging__when_requesting_full_picture__then_fallback_and_manifest_are_honest(self):
        for kernel, options in ((lambda a: media.affine(a, weights=(1,), bias=1, saturating=True), {}),
                                (lambda a: media.sample_h264(a, 1, 2), {"h264_merge_zero_guards": False})):
            default = media.compile(kernel, **options)
            banded = media.compile(kernel, h264_prediction_slice_rows=0, **options)
            self.assertEqual(default.tail, banded.tail)
            stats = banded.manifest()["optimizations"]["prediction_slice_bands"]
            self.assertEqual(stats, {"requested_rows": 0, "eligible_pictures": 0, "banded_pictures": 0})

    def test__given_two_independent_fields_and_changed_inputs__when_fusing_with_bands__then_decoder_outputs_match_fixed_ir(self):
        size = 64
        first = tuple((index % 4-4, index//4-4) for index in range(16))
        second = tuple((3-index % 4, 3-index//4) for index in range(16))

        def kernel(a, b):
            return media.average(media.gather_h264(a, first), media.gather_h264(b, second))

        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        programs = {rows: media.compile(kernel, width=size, height=size,
                                        input_ranges={"a": (0, 255), "b": (0, 255)},
                                        h264_prediction_slice_rows=rows) for rows in (1, 2, 4, 8, 0)}
        unfused = media.compile(kernel, width=size, height=size, input_ranges={"a": (0, 255), "b": (0, 255)},
                               h264_fuse_shifts=False, h264_prediction_slice_rows=0)
        self.assertEqual(unfused.manifest()["costs"]["computation_pictures"], 3)
        self.assertEqual(unfused.manifest()["costs"]["prediction_slices"], 3)
        tails = {rows: program.tail for rows, program in programs.items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bands.h264"
            for fixture in range(2):
                inputs = {"a": [[(x*37+y*71+fixture*19) % 256 for x in range(size)] for y in range(size)],
                          "b": [[255*((x+3*y+fixture) % 7 < 3) for x in range(size)] for y in range(size)]}
                expected = programs[1].reference(inputs)
                raw = b"".join(i420_picture(board) for board in (inputs["a"], inputs["b"], expected["output"]))
                with patch("codec_compute.ir.evaluate", side_effect=AssertionError("Runtime evaluated IR")):
                    for rows, program in programs.items():
                        self.assertEqual(program.tail, tails[rows])
                        self.assertEqual(program.manifest()["costs"]["computation_pictures"], 1)
                        path.write_bytes(program.encode(inputs))
                        for backend in backends:
                            with self.subTest(rows=rows, fixture=fixture, decoder=backend):
                                self.assertEqual(decode(path, backend), raw)
                    self.assertEqual(unfused.decode(inputs, decoder=backends[0]), expected)

    def test__given_512_byte_images__when_decoding_multirow_slices__then_complete_i420_matches(self):
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        size = 512
        a = [[(x*37+y*71) % 256 for x in range(size)] for y in range(size)]
        b = [[255*((x+3*y) % 7 < 3) for x in range(size)] for y in range(size)]
        expected = [[(x+y+1)//2 for x, y in zip(arow, brow)] for arow, brow in zip(a, b)]
        raw = b"".join(i420_picture(board) for board in (a, b, expected))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"large-bands.h264"
            for rows in (1, 8, 0):
                program = media.compile(media.average, width=size, height=size,
                                        input_ranges={"a": (0, 255), "b": (0, 255)},
                                        h264_level=51, h264_prediction_slice_rows=rows)
                path.write_bytes(program.encode({"a": a, "b": b}))
                for backend in backends:
                    with self.subTest(rows=rows, decoder=backend):
                        self.assertEqual(decode(path, backend), raw)


if __name__ == "__main__":
    unittest.main()
