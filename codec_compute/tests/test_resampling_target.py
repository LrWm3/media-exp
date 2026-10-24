"""Spatial fields retain exact decoder semantics through H.264 instruction selection."""

from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute.targets.h264 import H264Program, _legacy
from decoder_core.decode import available, decode


BYTE_RANGES = {"a": (0, 255), "b": (0, 255)}
FIELD_A = tuple((fx-4, fy-4) for fy in range(4) for fx in range(4))
FIELD_B = tuple((3-fy, fx-1) for fy in range(4) for fx in range(4))


def pointwise(function, *images):
    return [[function(*values) for values in zip(*rows)] for rows in zip(*images)]


def integer_gather(image, offsets):
    """Direct pixel indexing, independent of graph evaluation and lowering."""
    height, width = len(image), len(image[0])
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            dx, dy = offsets[(y//16)*(width//16)+x//16]
            row.append(image[y+dy][x+dx] if 0 <= y+dy < height and 0 <= x+dx < width else 0)
        result.append(row)
    return result


def fractional_gather(image, offsets):
    """Use the previously decoder-validated oracle outside the shared IR."""
    height, width = len(image), len(image[0])
    plane = bytes(value for row in image for value in row)
    interpolate = _legacy().verify.interpolate
    return [[interpolate(plane, width, height,
                         4*x+offsets[(y//16)*(width//16)+x//16][0],
                         4*y+offsets[(y//16)*(width//16)+x//16][1])
             for x in range(width)] for y in range(height)]


def picture_count(program):
    return program.manifest()["costs"]["computation_pictures"]


class ResamplingTargetRejectionTests(unittest.TestCase):
    def test__given_codec_specific_or_unimplemented_block_reads__when_targeting_other_codecs__then_lowering_rejects_explicitly(self):
        kernels = (("gather", lambda x: media.gather(x, [(1, 0)])),
                   ("sample_h264", lambda x: media.sample_h264(x, 1, -3)),
                   ("gather_h264", lambda x: media.gather_h264(x, [(2, -1)])))
        for target in ("hevc", "av1"):
            for operation, kernel in kernels:
                with self.subTest(target=target, operation=operation), self.assertRaises(media.UnsupportedOperation) as caught:
                    media.compile(kernel, width=16, height=16, target=target)
                self.assertEqual(caught.exception.operation, operation)

    def test__given_unsupported_border_or_block_size__when_compiling_both_fusion_modes__then_backend_does_not_change_semantics(self):
        kernels = (("gather", lambda x: media.gather(x, [(1, 0)], border="clamp")),
                   ("gather", lambda x: media.gather(x, [(1, 0)]*4, block_size=8)),
                   ("gather_h264", lambda x: media.gather_h264(x, [(1, 0)]*4, block_size=8)))
        for operation, movement in kernels:
            graph = media.trace(lambda image: media.average(image, movement(image)), width=16, height=16)
            for enabled in (False, True):
                with self.subTest(operation=operation, enabled=enabled), self.assertRaises(media.UnsupportedOperation) as caught:
                    media.compile_graph(graph, h264_fuse_shifts=enabled)
                self.assertEqual(caught.exception.operation, operation)

    def test__given_motion_beyond_backend_limits__when_building_materialized_or_fused_reads__then_every_path_rejects(self):
        for qx, qy in ((-8193, 0), (8192, 0), (0, -2049), (0, 2048)):
            movements = (lambda x: media.sample_h264(x, qx, qy),
                         lambda x: media.gather_h264(x, [(qx, qy)]))
            for movement in movements:
                graph = media.trace(lambda image: media.average(image, movement(image)), width=16, height=16)
                moved = graph.outputs[0][1].args[1]
                for enabled in (False, True):
                    with self.subTest(operation=moved.op, offset=(qx, qy), enabled=enabled), self.assertRaises(media.UnsupportedOperation):
                        media.compile_graph(graph, h264_fuse_shifts=enabled)
                with self.assertRaises(media.UnsupportedOperation):
                    H264Program._build(graph, {moved})
        for dx, dy in ((-2049, 0), (2048, 0), (0, -513), (0, 512)):
            graph = media.trace(lambda image: media.average(image, media.gather(image, [(dx, dy)])),
                                width=16, height=16)
            for enabled in (False, True):
                with self.subTest(integer_offset=(dx, dy), enabled=enabled), self.assertRaises(media.UnsupportedOperation):
                    media.compile_graph(graph, h264_fuse_shifts=enabled)
            with self.assertRaises(media.UnsupportedOperation):
                H264Program._build(graph, {graph.outputs[0][1].args[1]})

    def test__given_exact_writer_motion_endpoints__when_compiling_either_level__then_materialized_and_fused_reads_emit(self):
        integer = ((-2048, 0), (2047, 0), (0, -512), (0, 511))
        fractional = ((-8192, 0), (8191, 0), (0, -2048), (0, 2047))
        movements = [(lambda x, d=pair: media.shift(x, *d)) for pair in integer]
        movements += [(lambda x, q=pair: media.sample_h264(x, *q)) for pair in fractional]
        movements += [lambda x: media.gather(x, integer), lambda x: media.gather_h264(x, fractional)]
        for level in (31, 41):
            for movement in movements:
                graph = media.trace(lambda image: media.average(image, movement(image)), width=32, height=32)
                for enabled in (False, True):
                    with self.subTest(level=level, operation=graph.outputs[0][1].args[1].op, enabled=enabled):
                        program = media.compile_graph(graph, h264_level=level, h264_fuse_shifts=enabled)
                        self.assertTrue(program.tail)
                        self.assertEqual(picture_count(program), 1 if enabled else 2)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class ResamplingTargetDecoderTests(unittest.TestCase):
    def assert_decoders(self, program, images, expected, *, all_backends=False):
        backends = ([name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
                    if all_backends else ["ffmpeg"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"program.h264"
            path.write_bytes(program.encode(images))
            first_raw = None
            for backend in backends:
                with self.subTest(decoder=backend):
                    raw = decode(path, backend)
                    # read() also verifies every picture's byte count, zero
                    # luma guard, neutral chroma, and output range proof.
                    self.assertEqual(_legacy().backend.read(raw, program.native), expected)
                    if first_raw is not None:
                        self.assertEqual(raw, first_raw)
                    first_raw = raw

    def test__given_extreme_signed_fields__when_decoding__then_zero_extension_matches_independent_oracle(self):
        integer = ((-2048, 0), (2047, 0), (0, -512), (0, 511))
        fractional = ((-8192, 0), (8191, 0), (0, -2048), (0, 2047))
        images = {"image": [[80+(13*x+29*y) % 176 for x in range(32)] for y in range(32)]}
        for movement, fields, oracle in ((media.gather, integer, integer_gather),
                                         (media.gather_h264, fractional, fractional_gather)):
            moved = oracle(images["image"], fields)
            self.assertEqual(moved, [[0]*32 for _ in range(32)])
            expected = {"output": pointwise(lambda a, b: (a+b+1)//2, images["image"], moved)}
            graph = media.trace(lambda image: media.average(image, movement(image, fields)),
                                width=32, height=32, input_ranges={"image": (80, 255)})
            for enabled in (False, True):
                with self.subTest(operation=movement.__name__, enabled=enabled):
                    program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                    self.assert_decoders(program, images, expected, all_backends=True)

    def test__given_fractional_reads_beyond_old_limit__when_crossing_large_active_image__then_both_signs_keep_filter_and_border(self):
        for width, height in ((256, 32), (32, 256)):
            horizontal = width > height
            offsets = ((509, 2), (-509, -1), (515, -5), (-515, 3))
            fields = tuple((x, y) if horizontal else (y, x) for x, y in offsets)*((width*height)//1024)
            images = {"image": [[(17*x+31*y+(x*y) % 29) % 256 for x in range(width)] for y in range(height)]}
            moved = fractional_gather(images["image"], fields)
            self.assertTrue(any(value for row in moved for value in row))
            self.assertTrue(any(value == 0 for row in moved for value in row))
            expected = {"output": pointwise(lambda a, b: (a+b+1)//2, images["image"], moved)}
            graph = media.trace(lambda image: media.average(image, media.gather_h264(image, fields)),
                                width=width, height=height)
            for enabled in (False, True):
                with self.subTest(shape=(width, height), enabled=enabled):
                    program = media.compile_graph(graph, h264_fuse_shifts=enabled, h264_level=41)
                    self.assert_decoders(program, images, expected, all_backends=True)

    def test__given_256_patch_atlas__when_routing_two_fixtures__then_roi_and_zero_filler_decode_exactly(self):
        from codec_compute.workloads.resampling.kernels import tasks, fixture
        task = next(task for task in tasks(256) if task.name == "patch-atlas")
        self.assertIn((255, 255), task.fields["integer"])
        graph = media.trace(task.kernel, width=256, height=256, input_ranges=task.input_ranges)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False, h264_level=41)
        selected = media.compile_graph(graph, h264_level=41)
        self.assertEqual(baseline.tail, selected.tail)
        self.assertEqual(picture_count(selected), 1)
        self.assertEqual((selected.native.writer.width, selected.native.writer.height), (288, 288))
        for index in (0, 1):
            images = fixture(task, index, 256)
            expected = {"output": integer_gather(images["image"], task.fields["integer"])}
            self.assertTrue(any(expected["output"][y][x] for y in range(128) for x in range(128)))
            self.assertTrue(all(expected["output"][y][x] == 0 for y in range(256) for x in range(256)
                                if x >= 128 or y >= 128))
            with self.subTest(fixture=index):
                self.assertEqual(selected.reference(images), expected)
                self.assert_decoders(selected, images, expected, all_backends=True)

    def test__given_two_independent_sixteen_phase_fields__when_averaging__then_one_b_picture_decodes_every_phase_and_edge(self):
        graph = media.trace(lambda a, b: media.average(media.gather_h264(a, FIELD_A),
                                                     media.gather_h264(b, FIELD_B)),
                            width=64, height=64, input_ranges=BYTE_RANGES)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual((picture_count(baseline), picture_count(selected)), (3, 1))
        operation, = selected.native.operations
        self.assertEqual(len(operation.block_operations), 16)
        self.assertEqual(len(operation.sources), 2)
        for child, left, right in zip(operation.block_operations, FIELD_A, FIELD_B):
            self.assertEqual(tuple((4*s.dx+s.qx, 4*s.dy+s.qy) for s in child.sources), (left, right))
            self.assertEqual(tuple(s.frame for s in child.sources), tuple(s.frame for s in operation.sources))
        rng = random.Random(132)
        images = {name: [[rng.randrange(256) for _ in range(64)] for _ in range(64)] for name in BYTE_RANGES}
        expected = {"output": pointwise(lambda a, b: (a+b+1)//2,
                                         fractional_gather(images["a"], FIELD_A),
                                         fractional_gather(images["b"], FIELD_B))}
        self.assert_decoders(baseline, images, expected)
        self.assert_decoders(selected, images, expected, all_backends=True)

    def test__given_derived_references_and_identity_aliases__when_scheduling_block_fields__then_children_use_remapped_parent_frames(self):
        left_field = ((-3, 2), (2, -1), (1, 3), (-2, -3))
        right_field = ((2, -3), (-1, 2), (3, 1), (-3, -2))

        def kernel(a, b):
            left = media.copy(media.affine(a, weights=(2,), bias=-31, saturating=True))
            right = media.copy(media.threshold(b, 128))
            return media.average(media.gather_h264(left, left_field), media.gather_h264(right, right_field))

        images = {"a": [[(13*x+29*y) % 256 for x in range(32)] for y in range(32)],
                  "b": [[(31*x+7*y) % 256 for x in range(32)] for y in range(32)]}
        left = pointwise(lambda value: min(255, max(0, 2*value-31)), images["a"])
        right = pointwise(lambda value: int(value >= 128), images["b"])
        expected = {"output": pointwise(lambda a, b: (a+b+1)//2,
                                         fractional_gather(left, left_field), fractional_gather(right, right_field))}
        graph = media.trace(kernel, width=32, height=32, input_ranges=BYTE_RANGES)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertLess(picture_count(selected), picture_count(baseline))
        final = selected.native.operations[-1]
        self.assertTrue(final.block_operations)
        self.assertTrue(all(source.frame >= 2 for source in final.sources))
        for child in final.block_operations:
            self.assertEqual(tuple(s.frame for s in child.sources), tuple(s.frame for s in final.sources))
        self.assert_decoders(baseline, images, expected)
        self.assert_decoders(selected, images, expected, all_backends=True)

    def test__given_nested_sampling_crops_and_named_aliases__when_optimizing__then_intermediate_filters_and_borders_remain_observable(self):
        def kernel(image):
            first = media.sample_h264(image, 1, -3)
            cropped = media.gather(image, [(1, 0)]*4)
            return {"first": first, "alias": first,
                    "nested": media.sample_h264(first, -1, 3),
                    "recropped": media.gather(cropped, [(-1, 0)]*4)}

        images = {"image": [[255 if (x+y) % 3 else 0 for x in range(32)] for y in range(32)]}
        first = fractional_gather(images["image"], [(1, -3)]*4)
        expected = {"first": first, "alias": first,
                    "nested": fractional_gather(first, [(-1, 3)]*4),
                    "recropped": integer_gather(integer_gather(images["image"], [(1, 0)]*4), [(-1, 0)]*4)}
        self.assertNotEqual(expected["nested"], images["image"])
        self.assertNotEqual(expected["recropped"], images["image"])
        graph = media.trace(kernel, width=32, height=32)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                frames = program.manifest()["output_frames"]
                self.assertEqual(frames["first"], frames["alias"])
                self.assertNotEqual(frames["first"], frames["nested"])
                self.assert_decoders(program, images, expected)
        # The filter boundary must also survive when the intermediate has no
        # name; the named-output rule alone cannot protect this case.
        nested = media.trace(lambda image: media.sample_h264(media.sample_h264(image, 1, -3), -1, 3),
                             width=32, height=32)
        for enabled in (False, True):
            with self.subTest(unnamed_intermediate=True, enabled=enabled):
                program = media.compile_graph(nested, h264_fuse_shifts=enabled)
                self.assertEqual(picture_count(program), 2)
                self.assert_decoders(program, images, {"output": expected["nested"]})

    def test__given_two_different_fields_reading_one_source__when_combining__then_same_reference_does_not_merge_distinct_reads(self):
        field_a, field_b = FIELD_A[:4], FIELD_B[:4]
        images = {"image": [[(31*x+17*y) % 256 for x in range(32)] for y in range(32)]}
        left = fractional_gather(images["image"], field_a)
        right = fractional_gather(images["image"], field_b)
        for combine, formula in ((media.average, lambda a, b: (a+b+1)//2),
                                 (media.sub_sat, lambda a, b: max(a-b, 0))):
            graph = media.trace(lambda image: combine(media.gather_h264(image, field_a),
                                                     media.gather_h264(image, field_b)), width=32, height=32)
            selected = media.compile_graph(graph)
            self.assertEqual(picture_count(selected), 1)
            for child in selected.native.operations[0].block_operations:
                self.assertEqual(child.sources[0].frame, child.sources[1].frame)
            self.assert_decoders(selected, images, {"output": pointwise(formula, left, right)})

    def test__given_two_named_copies_sharing_one_field__when_fusion_duplicates_work__then_cost_guard_preserves_shared_picture(self):
        field = ((1, 0), (-1, 0), (0, 1), (0, -1))

        def kernel(image):
            moved = media.gather(image, field)
            return {"first": media.copy(moved), "second": media.copy(moved)}

        graph = media.trace(kernel, width=32, height=32)
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertEqual(picture_count(selected), 1)
        self.assertEqual(selected.tail, baseline.tail)
        self.assertEqual(selected.native.output_frames["first"], selected.native.output_frames["second"])
        stats = selected.manifest()["optimizations"]["spatial_read_fusion"]
        self.assertEqual(stats["reason"], "scheduled cost regression")
        self.assertEqual(stats["candidate_cost"]["computation_pictures"], 2)
        images = {"image": [[(13*x+23*y) % 256 for x in range(32)] for y in range(32)]}
        moved = integer_gather(images["image"], field)
        self.assert_decoders(selected, images, {"first": moved, "second": moved})

    def test__given_moving_operands_in_multistage_arithmetic__when_selecting_instructions__then_temporaries_are_stationary(self):
        field = ((-3, 2), (2, -1), (1, 3), (-2, -3))
        integer_field = ((1, 0), (-1, 0), (0, 1), (0, -1))

        def kernel(image):
            moved = media.gather_h264(image, field)
            return {"threshold": media.threshold(moved, 24),
                    "weighted": media.affine(moved, weights=(2,), bias=-37, saturating=True),
                    "split_bias": media.affine(media.gather(image, integer_field), bias=150)}

        images = {"image": [[(13*x+17*y) % 101 for x in range(32)] for y in range(32)]}
        moved = fractional_gather(images["image"], field)
        expected = {"threshold": pointwise(lambda value: int(value >= 24), moved),
                    "weighted": pointwise(lambda value: min(255, max(0, 2*value-37)), moved),
                    "split_bias": pointwise(lambda value: value+150, integer_gather(images["image"], integer_field))}
        graph = media.trace(kernel, width=32, height=32, input_ranges={"image": (0, 100)})
        baseline = media.compile_graph(graph, h264_fuse_shifts=False)
        selected = media.compile_graph(graph)
        self.assertLess(picture_count(selected), picture_count(baseline))
        self.assert_decoders(baseline, images, expected)
        self.assert_decoders(selected, images, expected, all_backends=True)

    def test__given_identity_scalar_selection_on_moving_reads__when_selector_returns_an_alias__then_output_still_materializes_the_field(self):
        field = ((1, 0), (-1, 0), (0, 1), (0, -1))
        images = {"image": [[int((x+3*y) % 5 == 0) for x in range(32)] for y in range(32)]}
        expected = {"output": integer_gather(images["image"], field)}
        for consumer in (media.copy, lambda value: media.threshold(value, 1)):
            graph = media.trace(lambda image: consumer(media.gather(image, field)), width=32, height=32,
                                input_ranges={"image": (0, 1)})
            for enabled in (False, True):
                with self.subTest(consumer=consumer, enabled=enabled):
                    program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                    self.assertEqual(picture_count(program), 1)
                    self.assert_decoders(program, images, expected)

    def test__given_positive_input_minimum_and_boundary_motion__when_fusing__then_zero_extension_and_sps_motion_bounds_survive(self):
        fields = ((-508, 0), (508, 0), (0, -508), (0, 508))
        images = {"image": [[80+(x+3*y) % 41 for x in range(32)] for y in range(32)]}
        kernels = ((lambda image: media.average(image, media.gather_h264(image, fields)),
                    {"output": pointwise(lambda value: (value+1)//2, images["image"])}),
                   (lambda image: media.threshold(media.gather(image, [(1, 0)]*4), 10),
                    {"output": [[1]*31+[0] for _ in range(32)]}),
                   (lambda image: media.threshold(media.sample_h264(image, 4, 0), 10),
                    {"output": [[1]*31+[0] for _ in range(32)]}))
        for kernel, expected in kernels:
            graph = media.trace(kernel, width=32, height=32, input_ranges={"image": (80, 120)})
            for enabled in (False, True):
                with self.subTest(kernel=kernel, enabled=enabled):
                    program = media.compile_graph(graph, h264_fuse_shifts=enabled)
                    self.assert_decoders(program, images, expected)

    def test__given_changed_inputs__when_encoding_and_decoding__then_fixed_field_tail_calls_no_runtime_oracle_or_search(self):
        field_a, field_b = FIELD_A[:4], FIELD_B[:4]
        graph = media.trace(lambda a, b: media.average(media.gather_h264(a, field_a), media.gather_h264(b, field_b)),
                            width=32, height=32, input_ranges=BYTE_RANGES)
        program = media.compile_graph(graph)
        inputs = ({"a": [[0]*32 for _ in range(32)], "b": [[0]*32 for _ in range(32)]},
                  {"a": [[(13*x+17*y) % 256 for x in range(32)] for y in range(32)],
                   "b": [[(7*x+31*y) % 256 for x in range(32)] for y in range(32)]})
        expected = [{"output": pointwise(lambda a, b: (a+b+1)//2,
                                         fractional_gather(images["a"], field_a), fractional_gather(images["b"], field_b))}
                    for images in inputs]
        self.assertNotEqual(*expected)
        tail, streams = program.tail, []
        with patch("codec_compute.ir.evaluate", side_effect=AssertionError("runtime IR oracle")), \
             patch("codec_compute.sampling.interpolate_h264", side_effect=AssertionError("runtime interpolation")), \
             patch.object(_legacy().verify, "interpolate", side_effect=AssertionError("runtime legacy oracle")), \
             patch.object(program, "reference", side_effect=AssertionError("runtime reference")), \
             patch("codec_compute.synthesis.lower.synthesize", side_effect=AssertionError("runtime synthesis")):
            for images, output in zip(inputs, expected):
                streams.append(program.encode(images))
                self.assertEqual(program.decode(images), output)
                self.assertTrue(streams[-1].endswith(tail))
        self.assertNotEqual(*streams)
        self.assertEqual(program.tail, tail)


if __name__ == "__main__":
    unittest.main()
