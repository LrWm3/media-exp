"""Portable block reads and explicit H.264 interpolation have exact semantics."""

from dataclasses import replace
import json
import unittest

import codec_compute as media
from codec_compute import ir
from codec_compute.sampling import interpolate_h264
from codec_compute.synthesis.lower import lower_graph


FIELD = ((1, 0), (-1, 1), (10, 0), (0, -1), (0, 1), (-4, -2))
BOARD = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
ZERO_GATHER = [[2, 3, 7, 8, 0], [7, 8, 12, 13, 0], [6, 7, 0, 0, 1]]
CLAMP_GATHER = [[2, 3, 7, 8, 5], [7, 8, 12, 13, 10], [6, 7, 13, 14, 1]]


class ResamplingIRTests(unittest.TestCase):
    def test__given_partial_edge_blocks__when_gathering__then_offsets_follow_ceiling_grid_row_order(self):
        for border, expected in (("zero", ZERO_GATHER), ("clamp", CLAMP_GATHER)):
            with self.subTest(border=border):
                graph = media.trace(lambda image: media.gather(image, FIELD, block_size=2, border=border),
                                    width=5, height=3, input_ranges={"image": (1, 15)})
                self.assertEqual(media.evaluate(graph, {"image": BOARD}), {"output": expected})
                node = graph.outputs[0][1]
                self.assertEqual(node.params, (FIELD, 2, border))
                self.assertEqual((node.lo, node.hi), (0 if border == "zero" else 1, 15))
        native = media.trace(lambda image: media.gather_h264(image, [(4*x, 4*y) for x, y in FIELD], block_size=2),
                             width=5, height=3, input_ranges={"image": (1, 15)})
        self.assertEqual(media.evaluate(native, {"image": BOARD}), {"output": ZERO_GATHER})

    def test__given_mutable_offset_inputs__when_constructing__then_ir_holds_an_immutable_snapshot(self):
        for function in (media.gather, media.gather_h264):
            offsets = [[1, -2], [3, 4]]
            graph = media.trace(lambda image: function(image, offsets), width=17, height=1)
            offsets[0][0] = 99
            offsets.append([0, 0])
            with self.subTest(operation=function.__name__):
                self.assertEqual(graph.outputs[0][1].params[0], ((1, -2), (3, 4)))
                ir.validate_graph(graph)
                generated = media.trace(lambda image: function(image, (pair for pair in ((1, -2), (3, 4)))),
                                        width=17, height=1)
                self.assertEqual(generated.outputs[0][1].params, graph.outputs[0][1].params)

    def test__given_invalid_offsets_parameters_or_operands__when_constructing__then_contract_rejects_malformed_values(self):
        invalid_fields = (None, True, 3, (), [], [(0,)], [(0, 0, 0)], [1], [(True, 0)],
                          [(0, False)], [(0, 1.0)], [(0, "1")], "01", ["01"], [b"01"],
                          {(0, 0): 1}, [{0: 1, 1: 2}])
        for function in (media.gather, media.gather_h264):
            for offsets in invalid_fields:
                with self.subTest(operation=function.__name__, offsets=offsets), self.assertRaises(media.CompileError):
                    media.trace(lambda image: function(image, offsets), width=1, height=1)
            for size in (0, -1, True, False, 1.5, None, "16"):
                with self.subTest(operation=function.__name__, block_size=size), self.assertRaises(media.CompileError):
                    media.trace(lambda image: function(image, [(0, 0)], block_size=size), width=1, height=1)
            with self.assertRaises(media.CompileError):
                function(None, [(0, 0)])
        for offset in (True, False, None, "1", 1.5):
            for axis in ("qx", "qy"):
                with self.subTest(axis=axis, offset=offset), self.assertRaises(media.CompileError):
                    media.trace(lambda image: media.sample_h264(image, **{axis: offset}), width=1, height=1)
        with self.assertRaises(media.CompileError):
            media.sample_h264(None)
        for border in (None, True, "wrap", "reflect"):
            with self.subTest(border=border), self.assertRaises(media.CompileError):
                media.trace(lambda image: media.gather(image, [(0, 0)], border=border), width=1, height=1)

    def test__given_field_length_inconsistent_with_shape__when_validating_graph__then_partial_blocks_still_require_offsets(self):
        for function in (media.gather, media.gather_h264):
            for count in (1, 2, 5, 7):
                with self.subTest(operation=function.__name__, count=count), self.assertRaisesRegex(media.CompileError, "needs 6 offsets"):
                    media.trace(lambda image: function(image, [(0, 0)]*count, block_size=2), width=5, height=3)
            valid = media.trace(lambda image: function(image, [(0, 0)]*6, block_size=2), width=5, height=3)
            with self.assertRaisesRegex(media.CompileError, "needs 9 offsets"):
                ir.validate_graph(replace(valid, height=5))
            # One generic block can exceed the semantic image dimensions.
            one = media.trace(lambda image: function(image, [(0, 0)], block_size=100), width=5, height=3)
            self.assertEqual(media.evaluate(one, {"image": BOARD}), {"output": BOARD})

    def test__given_integer_and_fractional_offsets__when_deriving_ranges__then_fractional_overshoot_uses_full_u8(self):
        cases = (
            (lambda image: media.gather(image, [(0, 0)]), (0, 100)),
            (lambda image: media.gather(image, [(0, 0)], border="clamp"), (80, 100)),
            (lambda image: media.sample_h264(image), (0, 100)),
            (lambda image: media.sample_h264(image, -8, 12), (0, 100)),
            (lambda image: media.sample_h264(image, 1, 0), (0, 255)),
            (lambda image: media.sample_h264(image, 0, -1), (0, 255)),
            (lambda image: media.gather_h264(image, [(4, -4)]), (0, 100)),
            (lambda image: media.gather_h264(image, [(4, -3)]), (0, 255)),
        )
        for kernel, expected in cases:
            graph = media.trace(kernel, width=1, height=1, input_ranges={"image": (80, 100)})
            node = graph.outputs[0][1]
            with self.subTest(operation=node.op, params=node.params):
                self.assertEqual((node.lo, node.hi), expected)
        mixed = media.trace(lambda image: media.gather_h264(image, [(4, 0), (0, 1)]),
                            width=17, height=1, input_ranges={"image": (80, 100)})
        self.assertEqual((mixed.outputs[0][1].lo, mixed.outputs[0][1].hi), (0, 255))
        # Filtering may exceed the largest input even away from an image edge.
        overshoot = media.trace(lambda image: media.sample_h264(image, 2), width=6, height=1,
                                input_ranges={"image": (0, 100)})
        self.assertEqual(media.evaluate(overshoot, {"image": [[0, 0, 100, 100, 0, 0]]})["output"][0][2], 125)

    def test__given_forged_bounds_or_mutable_nested_fields__when_validating__then_untrusted_proofs_are_rejected(self):
        kernels = (lambda image: media.gather(image, [(0, 0)]),
                   lambda image: media.sample_h264(image, 1, 0),
                   lambda image: media.gather_h264(image, [(1, 0)]))
        for kernel in kernels:
            graph = media.trace(kernel, width=1, height=1, input_ranges={"image": (80, 100)})
            original = graph.outputs[0][1]
            for lo, hi in ((1, original.hi), (0, original.hi-1)):
                forged = replace(original, lo=lo, hi=hi)
                invalid = replace(graph, nodes=(*graph.inputs, forged), outputs=(("output", forged),))
                with self.subTest(operation=original.op, bounds=(lo, hi)), self.assertRaisesRegex(media.CompileError, "Invalid range proof"):
                    ir.validate_graph(invalid)
            if original.op != "sample_h264":
                for field in ([list(original.params[0][0])], (list(original.params[0][0]),)):
                    forged = replace(original, params=(field, *original.params[1:]))
                    invalid = replace(graph, nodes=(*graph.inputs, forged), outputs=(("output", forged),))
                    with self.subTest(operation=original.op, field=field), self.assertRaisesRegex(media.CompileError, "immutable"):
                        ir.validate_graph(invalid)

    def test__given_large_motion_and_partial_dimensions__when_interpreting__then_semantics_have_no_backend_limits(self):
        kernels = (lambda image: media.gather(image, [(10**20, -10**20)], block_size=3),
                   lambda image: media.sample_h264(image, -(10**20)-1, 10**20+3),
                   lambda image: media.gather_h264(image, [(10**20+1, -(10**20)-3)], block_size=3))
        for kernel in kernels:
            graph = media.trace(kernel, width=3, height=2)
            with self.subTest(operation=graph.outputs[0][1].op):
                self.assertEqual(media.evaluate(graph, {"image": [[255]*3 for _ in range(2)]}), {"output": [[0]*3 for _ in range(2)]})

    def test__given_sampling_extensions__when_serializing__then_revision_and_explicit_codec_names_are_preserved(self):
        def kernel(image):
            return {"portable": media.gather(image, [(1, 2)], border="clamp"),
                    "sample": media.sample_h264(image, -1, 3),
                    "field": media.gather_h264(image, [(2, -3)])}
        graph = media.trace(kernel, width=2, height=3)
        manifest = json.loads(json.dumps(graph.manifest()))
        self.assertEqual(manifest["format"], "codec-compute-ir-2")
        nodes = {node["op"]: node for node in manifest["nodes"]}
        self.assertEqual(nodes["gather"]["params"], [[[1, 2]], 16, "clamp"])
        self.assertEqual(nodes["sample_h264"]["params"], [-1, 3])
        self.assertEqual(nodes["gather_h264"]["params"], [[[2, -3]], 16])
        self.assertTrue({"gather", "sample_h264", "gather_h264"} <= set(media.__all__))

    def test__given_synthesized_source__when_rebuilding_sampling_consumers__then_parameters_and_pixels_survive(self):
        quarter_field = tuple((4*x, 4*y) for x, y in FIELD)
        kernels = (lambda source: media.gather(source, FIELD, block_size=2),
                   lambda source: media.sample_h264(source, 4, 0),
                   lambda source: media.gather_h264(source, quarter_field, block_size=2))
        images = {"a": BOARD, "b": [[15-value for value in row] for row in BOARD]}
        for sample in kernels:
            graph = media.trace(lambda a, b: sample(media.minimum(a, b)), width=5, height=3,
                                input_ranges={"a": (0, 15), "b": (0, 15)})
            with self.subTest(operation=graph.outputs[0][1].op):
                lowered, records = lower_graph(graph, target="h264")
                ir.validate_graph(lowered)
                self.assertEqual([record["operation"] for record in records], ["minimum"])
                self.assertNotIn("minimum", {node.op for node in lowered.nodes})
                self.assertIsNot(lowered.outputs[0][1], graph.outputs[0][1])
                self.assertEqual(lowered.outputs[0][1].params, graph.outputs[0][1].params)
                self.assertEqual(media.evaluate(lowered, images), media.evaluate(graph, images))


class H264SamplingOracleTests(unittest.TestCase):
    def test__given_six_tap_neighborhood__when_evaluating_all_sixteen_phases__then_normative_values_match(self):
        board = [[6, 240, 57, 201, 72, 22], [71, 57, 118, 71, 75, 16],
                 [31, 69, 118, 229, 211, 104], [47, 60, 10, 206, 174, 101],
                 [105, 169, 200, 186, 126, 107], [111, 213, 30, 26, 92, 182]]
        # H.264 8.4.2.2.1: G=118,H=229,M=10,b=177,h=33,m=239,s=103,j=132.
        expected = [[118, 148, 177, 203], [76, 105, 155, 208],
                    [33, 83, 132, 186], [22, 68, 118, 171]]
        graph = media.trace(lambda image: {f"{fx},{fy}": media.sample_h264(image, fx, fy)
                                           for fy in range(4) for fx in range(4)}, width=6, height=6)
        results = media.evaluate(graph, {"image": board})
        self.assertEqual([[results[f"{fx},{fy}"][2][2] for fx in range(4)] for fy in range(4)], expected)
        plane = [value for row in board for value in row]
        self.assertEqual([[interpolate_h264(plane, 6, 6, 8+fx, 8+fy) for fx in range(4)] for fy in range(4)], expected)

    def test__given_diagonal_filter__when_first_pass_rounding_or_clipping_would_change_result__then_precision_is_retained(self):
        plane = [0]*36
        plane[14] = 1
        self.assertEqual(interpolate_h264(plane, 6, 6, 10, 8), 1)
        self.assertEqual(interpolate_h264(plane, 6, 6, 10, 10), 0)
        plane[14] = plane[15] = 255
        self.assertEqual(interpolate_h264(plane, 6, 6, 10, 8), 255)
        self.assertEqual(interpolate_h264(plane, 6, 6, 10, 10), 199)

    def test__given_negative_fractional_offsets__when_sampling_single_impulse__then_floor_and_zero_extension_precede_filtering(self):
        positions = {(-1, 0): 207, (0, -1): 207, (-2, -2): 100, (-1, -1): 159,
                     (-3, -3): 0, (-6, 0): 0, (-1000, 0): 0, (0, 1000): 0}
        for (qx, qy), expected in positions.items():
            graph = media.trace(lambda image: media.sample_h264(image, qx, qy), width=1, height=1)
            with self.subTest(offset=(qx, qy)):
                self.assertEqual(media.evaluate(graph, {"image": [[255]]}), {"output": [[expected]]})
        # Relative quarter offsets in separate blocks can address one point.
        graph = media.trace(lambda image: media.gather_h264(image, [(1, 1), (-3, 1), (1, -3), (-3, -3)], block_size=1),
                            width=2, height=2)
        self.assertEqual(media.evaluate(graph, {"image": [[255, 0], [0, 0]]}), {"output": [[159, 159], [159, 159]]})

    def test__given_explicit_zero_padding__when_sampling_all_boundary_phases__then_extension_matches_padded_plane(self):
        width, height, padding = 3, 2, 4
        plane = [0, 255, 17, 201, 32, 99]
        stride, rows = width+2*padding, height+2*padding
        padded = [0]*(stride*rows)
        for y in range(height):
            padded[(y+padding)*stride+padding:(y+padding)*stride+padding+width] = plane[y*width:(y+1)*width]
        for qy in range(-7, 4*height+8):
            for qx in range(-7, 4*width+8):
                with self.subTest(position=(qx, qy)):
                    self.assertEqual(interpolate_h264(plane, width, height, qx, qy),
                                     interpolate_h264(padded, stride, rows, qx+4*padding, qy+4*padding))

    def test__given_two_sequential_samples__when_interpreting__then_each_intermediate_crop_remains_observable(self):
        graph = media.trace(lambda image: media.sample_h264(media.sample_h264(image, 4), -4), width=3, height=1)
        self.assertEqual(media.evaluate(graph, {"image": [[77, 88, 99]]}), {"output": [[0, 88, 99]]})


if __name__ == "__main__":
    unittest.main()
