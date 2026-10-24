"""Task mathematics and explicit borders checked without codec execution."""

from itertools import product
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute.workloads.kernels import WORKLOADS, fixtures, reference


TASKS = {workload.name: workload for workload in WORKLOADS}


def graph_for(name, width, height):
    workload = TASKS[name]
    return media.trace(workload.kernel, width=width, height=height, input_ranges=workload.input_ranges)


class WorkloadTests(unittest.TestCase):
    def test__given_all_task_fixtures__when_interpreting_graphs__then_independent_oracles_match_both_variants(self):
        self.assertEqual(len(TASKS), 9)
        for workload in WORKLOADS:
            graph = graph_for(workload.name, 32, 32)
            for variant in (0, 1):
                with self.subTest(workload=workload.name, variant=variant):
                    images = fixtures(workload, variant)
                    self.assertEqual(images, fixtures(workload, variant))
                    graph.validate(images)
                    # Task oracles must remain usable without the IR oracle.
                    with patch("codec_compute.ir.evaluate", side_effect=AssertionError("IR oracle reused")), \
                         patch("codec_compute.evaluate", side_effect=AssertionError("IR oracle reused")):
                        expected = reference(workload, images)
                    self.assertEqual(media.evaluate(graph, images), expected)

    def test__given_frame_changes_in_both_directions__when_measuring__then_positive_change_is_directional_and_absolute_change_is_symmetric(self):
        images = {"a": [[0, 255, 17, 23, 24, 25, 255]],
                  "b": [[255, 0, 20, 0, 0, 0, 255]]}
        reversed_images = {"a": images["b"], "b": images["a"]}
        positive = graph_for("positive-change", 7, 1)
        absolute = graph_for("absolute-change", 7, 1)
        forward = media.evaluate(positive, images)["output"]
        backward = media.evaluate(positive, reversed_images)["output"]
        self.assertEqual(forward, [[0, 255, 0, 23, 24, 25, 0]])
        self.assertEqual(backward, [[255, 0, 3, 0, 0, 0, 0]])
        result = media.evaluate(absolute, images)
        self.assertEqual(result, {"output": [[255, 255, 3, 23, 24, 25, 0]],
                                  "mask": [[1, 1, 0, 0, 1, 1, 0]]})
        self.assertEqual(result, media.evaluate(absolute, reversed_images))
        self.assertEqual(result["output"], [[a+b for a, b in zip(forward[0], backward[0])]])

    def test__given_extreme_and_nearby_byte_pairs__when_forming_temporal_envelope__then_range_equals_absolute_difference(self):
        samples = (0, 1, 23, 24, 127, 128, 254, 255)
        images = {"a": [[a]*len(samples) for a in samples], "b": [list(samples) for _ in samples]}
        result = media.evaluate(graph_for("temporal-envelope", len(samples), len(samples)), images)
        for y, a in enumerate(samples):
            for x, b in enumerate(samples):
                with self.subTest(a=a, b=b):
                    low, high, spread = (result[name][y][x] for name in ("low", "high", "output"))
                    self.assertEqual((low, high, spread), (min(a, b), max(a, b), abs(a-b)))
                    self.assertEqual(spread, high-low)
                    self.assertEqual(low+high, a+b)

    def test__given_odd_and_extreme_sums__when_temporally_averaging__then_rounding_is_half_up_without_clipping_sum(self):
        images = {"a": [[0, 1, 127, 254, 255]], "b": [[1, 2, 128, 255, 255]]}
        self.assertEqual(media.evaluate(graph_for("temporal-average", 5, 1), images),
                         {"output": [[1, 2, 128, 255, 255]]})

    def test__given_all_binary_cross_neighborhoods__when_dilating__then_only_four_neighbors_and_center_contribute(self):
        positions = ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))
        for name in ("dilate-zero", "dilate-clamp"):
            graph = graph_for(name, 3, 3)
            self.assertEqual((graph.inputs[0].lo, graph.inputs[0].hi), (0, 1))
            for bits in product((0, 1), repeat=5):
                # Nonzero diagonal corners must never reach the center in a
                # single four-connected step, even for the all-zero cross.
                image = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
                for (x, y), bit in zip(positions, bits):
                    image[y][x] = bit
                with self.subTest(workload=name, bits=bits):
                    actual = media.evaluate(graph, {"image": image})
                    self.assertEqual(actual, reference(TASKS[name], {"image": image}))
                    self.assertEqual(actual["output"][1][1], int(any(bits)))
                    self.assertTrue(all(value in (0, 1) for row in actual["output"] for value in row))
            with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
                graph.validate({"image": [[255]*3 for _ in range(3)]})
            with self.assertRaisesRegex(media.CompileError, "proven 0/1"):
                media.trace(TASKS[name].kernel, width=3, height=3, input_ranges={"image": (0, 255)})

    def test__given_three_level_cross_neighborhoods__when_computing_local_range__then_extrema_and_spread_match_direct_formulas(self):
        graph = graph_for("local-range", 3, 3)
        positions = ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))
        for values in product((0, 17, 255), repeat=5):
            image = [[0]*3 for _ in range(3)]
            for (x, y), value in zip(positions, values):
                image[y][x] = value
            with self.subTest(values=values):
                actual = media.evaluate(graph, {"image": image})
                self.assertEqual(actual, reference(TASKS["local-range"], {"image": image}))
                self.assertEqual((actual["low"][1][1], actual["high"][1][1], actual["output"][1][1]),
                                 (min(values), max(values), max(values)-min(values)))
        constant = media.evaluate(graph, {"image": [[64]*3 for _ in range(3)]})
        self.assertEqual(constant["output"], [[64, 64, 64], [64, 0, 64], [64, 64, 64]])

    def test__given_nonzero_constant_image__when_smoothing__then_zero_border_darkens_edges_and_clamp_preserves_constant(self):
        images = {"image": [[64]*3 for _ in range(3)]}
        zero = media.evaluate(graph_for("smooth-zero", 3, 3), images)
        clamp = media.evaluate(graph_for("smooth-clamp", 3, 3), images)
        self.assertEqual(zero["output"], [[36, 48, 36], [48, 64, 48], [36, 48, 36]])
        self.assertEqual(clamp["output"], images["image"])
        self.assertEqual(media.evaluate(graph_for("smooth-zero", 1, 1), {"image": [[64]]}), {"output": [[16]]})
        self.assertEqual(media.evaluate(graph_for("smooth-clamp", 1, 1), {"image": [[64]]}), {"output": [[64]]})

    def test__given_unit_center_impulse__when_nested_averages_round__then_blur_differs_from_one_rounded_outer_product(self):
        image = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        weights = (1, 2, 1)
        weighted_sum = sum(weights[y]*weights[x]*image[y][x] for y in range(3) for x in range(3))
        single_rounding = (weighted_sum+8)//16
        self.assertEqual(single_rounding, 0)
        for name in ("smooth-zero", "smooth-clamp"):
            with self.subTest(workload=name):
                actual = media.evaluate(graph_for(name, 3, 3), {"image": image})
                self.assertEqual(actual, reference(TASKS[name], {"image": image}))
                self.assertEqual(actual["output"][1][1], 1)
                self.assertNotEqual(actual["output"][1][1], single_rounding)

    def test__given_spatial_and_temporal_tasks__when_inspecting_semantic_graphs__then_border_choices_are_explicit(self):
        expected = {"dilate-zero": "zero", "dilate-clamp": "clamp", "smooth-zero": "zero",
                    "smooth-clamp": "clamp", "local-range": "zero"}
        for workload in WORKLOADS:
            with self.subTest(workload=workload.name):
                graph = graph_for(workload.name, 3, 3)
                shifts = [node for node in graph.manifest()["nodes"] if node["op"] == "shift"]
                if workload.name in expected:
                    self.assertEqual(workload.border, expected[workload.name])
                    self.assertTrue(shifts)
                    self.assertTrue(all(len(node["params"]) == 3 and node["params"][2] == workload.border
                                        for node in shifts))
                else:
                    self.assertIsNone(workload.border)
                    self.assertEqual(shifts, [])


if __name__ == "__main__":
    unittest.main()
