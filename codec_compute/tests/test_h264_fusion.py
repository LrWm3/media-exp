"""Spatial read fusion preserves crops, named results, and scheduled costs."""

import shutil
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute.targets.h264 import H264Program
from codec_compute.workloads.kernels import WORKLOADS, fixtures, reference


TASKS = {workload.name: workload for workload in WORKLOADS}


def shifted(image, dx=0, dy=0):
    """Independent zero-border pixel oracle, including intermediate crops."""
    height, width = len(image), len(image[0])
    return [[image[y+dy][x+dx] if 0 <= y+dy < height and 0 <= x+dx < width else 0
             for x in range(width)] for y in range(height)]


def pointwise(function, *images):
    return [[function(*values) for values in zip(*rows)] for rows in zip(*images)]


def fusion(program):
    return program.manifest()["optimizations"]["shift_read_fusion"]


def pictures(program):
    return program.manifest()["costs"]["computation_pictures"]


class H264FusionSelectionTests(unittest.TestCase):
    def test__given_nonboolean_fusion_flag__when_using_either_compile_entry_point__then_validation_rejects_it(self):
        graph = media.trace(media.copy, width=16, height=16)
        for value in (None, 0, 1, "false", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(media.CompileError, "h264_fuse_shifts must be Boolean"):
                    media.compile(media.copy, width=16, height=16, h264_fuse_shifts=value)
                with self.assertRaisesRegex(media.CompileError, "h264_fuse_shifts must be Boolean"):
                    media.compile_graph(graph, h264_fuse_shifts=value)

    def test__given_two_copy_outputs_of_one_shift__when_fusion_duplicates_work_and_readback__then_original_plan_wins(self):
        def kernel(image):
            moved = media.shift(image, 1)
            return {"a": media.copy(moved), "b": media.copy(moved)}

        graph = media.trace(kernel, width=16, height=16)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        stats = fusion(selected)
        self.assertEqual(stats["eligible_nodes"], 1)
        self.assertEqual(stats["fused_nodes"], 0)
        self.assertEqual(stats["reason"], "scheduled cost regression")
        self.assertEqual(stats["baseline_cost"]["computation_pictures"], 1)
        self.assertEqual(stats["candidate_cost"]["computation_pictures"], 2)
        self.assertEqual(stats["baseline_cost"]["output_readback_pictures"], 1)
        self.assertEqual(stats["candidate_cost"]["output_readback_pictures"], 2)
        self.assertEqual(selected.tail, baseline.tail)
        self.assertEqual(selected.manifest()["output_frames"], baseline.manifest()["output_frames"])
        self.assertEqual(selected.manifest()["output_frames"]["a"], selected.manifest()["output_frames"]["b"])

    def test__given_candidate_schedule_failure__when_baseline_is_legal__then_baseline_remains_usable(self):
        graph = media.trace(lambda image: media.average(image, media.shift(image, 1)), width=16, height=16)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        build = H264Program._build.__func__

        def reject_candidate(cls, graph, fused):
            if fused:
                raise media.UnsupportedOperation("h264", "resources", "candidate schedule exhausted reference slots")
            return build(cls, graph, fused)

        with patch.object(H264Program, "_build", classmethod(reject_candidate)):
            selected = media.compile_graph(graph)
        self.assertEqual(selected.tail, baseline.tail)
        self.assertEqual(selected.manifest()["output_frames"], baseline.manifest()["output_frames"])
        self.assertEqual(fusion(selected)["fused_nodes"], 0)
        self.assertEqual(fusion(selected)["reason"], "candidate rejected")
        self.assertIn("reference slots", fusion(selected)["detail"])

    def test__given_baseline_schedule_failure__when_candidate_is_legal__then_candidate_can_still_compile(self):
        graph = media.trace(lambda image: media.average(image, media.shift(image, 1)), width=16, height=16)
        expected = media.compile_graph(graph)
        build = H264Program._build.__func__

        def reject_baseline(cls, graph, fused):
            if not fused:
                raise media.UnsupportedOperation("h264", "resources", "baseline schedule exhausted reference slots")
            return build(cls, graph, fused)

        with patch.object(H264Program, "_build", classmethod(reject_baseline)):
            selected = media.compile_graph(graph)
        self.assertEqual(selected.tail, expected.tail)
        self.assertEqual(fusion(selected)["reason"], "accepted")
        self.assertEqual(fusion(selected)["fused_nodes"], 1)
        self.assertIsNone(fusion(selected)["baseline_cost"])

    def test__given_clamp_border__when_compiling_with_either_flag__then_fusion_does_not_change_unsupported_semantics(self):
        graph = media.trace(lambda image: media.average(image, media.shift(image, 1, border="clamp")),
                            width=16, height=16)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), self.assertRaises(media.UnsupportedOperation) as caught:
                media.compile_graph(graph, h264_fuse_shifts=enabled)
            self.assertEqual(caught.exception.operation, "shift")
            self.assertIn("clamp", str(caught.exception))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class H264FusionDecoderTests(unittest.TestCase):
    def test__given_spatial_workloads__when_fusing_reads__then_fewer_pictures_decode_both_independent_fixtures_exactly(self):
        for name, counts in (("dilate-zero", (8, 4)), ("smooth-zero", (8, 4)), ("local-range", (21, 17))):
            workload = TASKS[name]
            graph = media.trace(workload.kernel, width=32, height=32, input_ranges=workload.input_ranges)
            baseline = media.compile_graph(graph, h264_fuse_shifts=False)
            selected = media.compile_graph(graph)
            with self.subTest(workload=name):
                self.assertEqual((pictures(baseline), pictures(selected)), counts)
                self.assertEqual(fusion(baseline)["reason"], "disabled")
                self.assertFalse(fusion(baseline)["enabled"])
                stats = fusion(selected)
                self.assertTrue(stats["enabled"])
                self.assertEqual(stats["reason"], "accepted")
                self.assertEqual((stats["eligible_nodes"], stats["fused_nodes"]), (4, 4))
                for metric, value in stats["candidate_cost"].items():
                    self.assertLessEqual(value, stats["baseline_cost"][metric], metric)
                self.assertEqual(graph.manifest(), selected.graph.manifest())
                tail = selected.tail
                for variant in (0, 1):
                    images = fixtures(workload, variant)
                    expected = reference(workload, images)
                    self.assertEqual(baseline.decode(images), expected)
                    self.assertEqual(selected.decode(images), expected)
                    self.assertEqual(selected.tail, tail)

    def test__given_two_moving_average_inputs_with_nonzero_minima__when_fusing__then_signed_motion_and_zero_borders_survive(self):
        graph = media.trace(lambda a, b: media.average(media.shift(a, -1, 1), media.shift(b, 1, -1)),
                            width=16, height=16, input_ranges={"a": (32, 255), "b": (16, 255)})
        images = {"a": [[32+(13*x+29*y) % 224 for x in range(16)] for y in range(16)],
                  "b": [[16+(31*x+11*y) % 240 for x in range(16)] for y in range(16)]}
        expected = {"output": pointwise(lambda a, b: (a+b+1)//2,
                                         shifted(images["a"], -1, 1), shifted(images["b"], 1, -1))}
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual((pictures(baseline), pictures(selected)), (3, 1))
        self.assertEqual(fusion(selected)["fused_nodes"], 2)
        self.assertEqual({(source.dx, source.dy) for source in selected.native.operations[0].sources},
                         {(-1, 1), (1, -1)})
        self.assertEqual(baseline.decode(images), expected)
        self.assertEqual(selected.decode(images), expected)

    def test__given_positive_input_range__when_fused_shift_feeds_threshold__then_padding_still_has_zero_range(self):
        images = {"image": [[80+(3*x+7*y) % 41 for x in range(16)] for y in range(16)]}
        expected = {"output": [[1]*15+[0] for _ in range(16)]}
        graph = media.trace(lambda image: media.threshold(media.shift(image, 1), 10), width=16, height=16,
                            input_ranges={"image": (80, 120)})
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                self.assertEqual(program.decode(images), expected)
                if enabled:
                    self.assertEqual(fusion(program)["fused_nodes"], 1)

    def test__given_named_shift_and_alias__when_compiling__then_intermediate_picture_and_border_bias_remain_observable(self):
        def kernel(image):
            moved = media.shift(image, 1)
            return {"moved": moved, "alias": moved,
                    "biased": media.affine(moved, bias=7, saturating=True)}

        images = {"image": [[80+(5*x+3*y) % 41 for x in range(16)] for y in range(16)]}
        moved = shifted(images["image"], 1)
        expected = {"moved": moved, "alias": moved, "biased": pointwise(lambda value: value+7, moved)}
        graph = media.trace(kernel, width=16, height=16, input_ranges={"image": (80, 120)})
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual((pictures(baseline), pictures(selected)), (2, 2))
        self.assertEqual((fusion(selected)["eligible_nodes"], fusion(selected)["fused_nodes"]), (0, 0))
        self.assertEqual(selected.tail, baseline.tail)
        frames = selected.manifest()["output_frames"]
        self.assertEqual(frames["moved"], frames["alias"])
        self.assertNotEqual(frames["moved"], frames["biased"])
        self.assertEqual(selected.decode(images), expected)

    def test__given_adjacent_opposite_shifts__when_outer_read_fuses_into_copy__then_intermediate_crop_is_not_cancelled(self):
        graph = media.trace(lambda image: media.copy(media.shift(media.shift(image, 1), -1)),
                            width=16, height=16, input_ranges={"image": (77, 77)})
        images = {"image": [[77]*16 for _ in range(16)]}
        expected = {"output": [[0]+[77]*15 for _ in range(16)]}
        self.assertNotEqual(expected["output"], images["image"])
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                self.assertEqual(pictures(program), 2)
                self.assertEqual(program.decode(images), expected)
                self.assertEqual(fusion(program)["eligible_nodes"], 1)
                self.assertEqual(fusion(program)["fused_nodes"], int(enabled))

    def test__given_shift_with_scalar_and_shift_consumers__when_compiling__then_shared_crop_stays_materialized(self):
        def kernel(image):
            moved = media.shift(image, 1)
            return {"roundtrip": media.shift(moved, -1),
                    "biased": media.affine(moved, bias=5, saturating=True)}

        images = {"image": [[16*y+x for x in range(16)] for y in range(16)]}
        moved = shifted(images["image"], 1)
        expected = {"roundtrip": shifted(moved, -1),
                    "biased": pointwise(lambda value: min(255, value+5), moved)}
        program = media.compile(kernel, width=16, height=16)
        self.assertEqual(fusion(program)["eligible_nodes"], 0)
        self.assertEqual(pictures(program), 3)
        self.assertEqual(program.decode(images), expected)

    def test__given_identity_copy_and_shifted_copy__when_decoding__then_only_stationary_copy_aliases_input(self):
        def kernel(image):
            return {"image": image, "identity": media.copy(image), "moved": media.copy(media.shift(image, 1))}

        images = {"image": [[16*y+x for x in range(16)] for y in range(16)]}
        expected = {"image": images["image"], "identity": images["image"], "moved": shifted(images["image"], 1)}
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                program = media.compile(kernel, width=16, height=16, h264_fuse_shifts=enabled)
                self.assertEqual(pictures(program), 1)
                frames = program.manifest()["output_frames"]
                self.assertEqual(frames["identity"], frames["image"])
                self.assertNotEqual(frames["moved"], frames["image"])
                self.assertEqual(program.decode(images), expected)

    def test__given_multistage_threshold__when_shift_read_fuses__then_motion_applies_only_at_first_scalar_stage(self):
        graph = media.trace(lambda image: media.threshold(media.shift(image, -1), 32), width=16, height=16)
        images = {"image": [[16*y+x for x in range(16)] for y in range(16)]}
        expected = {"output": pointwise(lambda value: int(value >= 32), shifted(images["image"], -1))}
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual((pictures(baseline), pictures(selected)), (3, 2))
        self.assertEqual(fusion(selected)["fused_nodes"], 1)
        moving = [operation for operation in selected.native.operations
                  if any(source.dx or source.dy for source in operation.sources)]
        self.assertEqual(len(moving), 1)
        self.assertIs(moving[0], selected.native.operations[0])
        self.assertEqual(baseline.decode(images), expected)
        self.assertEqual(selected.decode(images), expected)

    def test__given_synthesized_extrema_of_shifted_reads__when_fusing_after_synthesis__then_minimum_and_maximum_decode_exactly(self):
        def kernel(a, b):
            left, below = media.shift(a, -1), media.shift(b, dy=1)
            return {"low": media.minimum(left, below), "high": media.maximum(left, below)}

        images = {"a": [[16*y+x for x in range(16)] for y in range(16)],
                  "b": [[(37*x+19*y) % 256 for x in range(16)] for y in range(16)]}
        left, below = shifted(images["a"], -1), shifted(images["b"], dy=1)
        expected = {"low": pointwise(min, left, below), "high": pointwise(max, left, below)}
        graph = media.trace(kernel, width=16, height=16, input_ranges={"a": (0, 255), "b": (0, 255)})
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual((pictures(baseline), pictures(selected)), (6, 4))
        self.assertEqual(fusion(selected)["fused_nodes"], 2)
        self.assertEqual({record["operation"] for record in selected.synthesis_records}, {"minimum", "maximum"})
        self.assertEqual(baseline.decode(images), expected)
        self.assertEqual(selected.decode(images), expected)


if __name__ == "__main__":
    unittest.main()
