"""Bounded instruction search checked against actual decoded pixel values."""

import json
from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import h264
from decoder_core.decode import available


def rule(neighbors, current):
    return h264.bit_or(h264.equal(neighbors, 3),
                       h264.bit_and(current, h264.equal(neighbors, 2)))


def life_kernel(generations):
    def kernel(seed):
        states, current = [seed], seed
        for _ in range(generations):
            neighbors = [h264.shift(current, dx, dy)
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
            count = neighbors[0]
            for neighbor in neighbors[1:]:
                count = h264.add(count, neighbor)
            current = rule(count, current)
            states.append(current)
        return {"output": current,
                **{f"generation_{index}": state for index, state in enumerate(states)}}
    return kernel


def life_step(board):
    """Independent integer oracle with a dead border outside the image."""
    height, width = len(board), len(board[0])
    output = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            count = sum(board[ny][nx]
                        for ny in range(max(0, y - 1), min(height, y + 2))
                        for nx in range(max(0, x - 1), min(width, x + 2))
                        if (nx, ny) != (x, y))
            output[y][x] = int(count == 3 or (board[y][x] == 1 and count == 2))
    return output


def life_outputs(board, generations):
    expected, current = {"generation_0": board}, board
    for generation in range(1, generations + 1):
        current = life_step(current)
        expected[f"generation_{generation}"] = current
    return {"output": current, **expected}


def all_neighborhoods():
    board = [[0] * 128 for _ in range(64)]
    for pattern in range(512):
        x, y = (pattern % 32) * 4, (pattern // 32) * 4
        for bit in range(9):
            board[y + bit // 3][x + bit % 3] = (pattern >> bit) & 1
    return board


class ExactSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not available("ffmpeg"):
            raise AssertionError("Exact instruction integration tests require FFmpeg")
        cls.backends = ("ffmpeg",) + tuple(name for name in ("openh264", "vaapi") if available(name))

    def image(self, function, width=16, height=16):
        return [[function(y * width + x) for x in range(width)] for y in range(height)]

    def assert_decoders(self, program, inputs, expected):
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(program.decode(inputs, backend), expected)

    def truth_inputs(self):
        return {"neighbors": self.image(lambda index: index % 9),
                "current": self.image(lambda index: (index // 9) % 2)}

    def test__given_complete_rule_truth_table__when_searching__then_three_predictions_decode_exactly(self):
        inputs = self.truth_inputs()
        ranges = {"neighbors": (0, 8), "current": (0, 1)}
        program = h264.compile(rule, 16, 16, input_ranges=ranges)
        baseline = h264.compile(rule, 16, 16, input_ranges=ranges, exact_search=False)
        expected = {"output": self.image(lambda index: int(
            index % 9 == 3 or (index % 9 == 2 and (index // 9) % 2 == 1)))}
        self.assertEqual(program.stats["lowered_instructions"], 3)
        self.assertEqual(program.pictures, 5)
        self.assertEqual(baseline.stats["lowered_instructions"], 10)
        self.assertEqual(baseline.pictures, 12)
        self.assertEqual(program.stats["exact_search_rewrites"], 1)
        self.assertEqual(program.stats["exact_search_saved_instructions"], 7)
        self.assert_decoders(program, inputs, expected)
        self.assert_decoders(baseline, inputs, expected)

    def test__given_all_512_neighborhoods_and_borders__when_decoding_generations__then_independent_life_matches(self):
        board = all_neighborhoods()
        program = h264.compile(life_kernel(3), 128, 64)
        self.assertEqual(program.pictures, 32)
        self.assertEqual(program.stats["exact_search_rewrites"], 3)
        self.assert_decoders(program, board, life_outputs(board, 3))

    def test__given_random_board_and_forty_generations__when_decoding__then_402_pictures_replace_682(self):
        rng = random.Random(8128)
        board = self.image(lambda _: rng.randrange(2))
        # Exercise live cells touching every edge and all four corners.
        board[0] = [1] * 16
        board[-1] = [1] * 16
        for row in board:
            row[0] = row[-1] = 1
        kernel = life_kernel(40)
        program = h264.compile(kernel, 16, 16)
        baseline = h264.compile(kernel, 16, 16, exact_search=False)
        self.assertEqual(program.pictures, 402)
        self.assertEqual(baseline.pictures, 682)
        self.assertEqual(program.stats["exact_search_rewrites"], 40)
        self.assertEqual(program.stats["exact_search_saved_instructions"], 280)
        expected = life_outputs(board, 40)
        self.assert_decoders(program, board, expected)
        self.assert_decoders(baseline, board, expected)

    def test__given_unary_equalities_and_windows__when_searching__then_unrelated_predicates_also_shrink(self):
        image = self.image(lambda index: index % 9)
        cases = (
            ("equality", lambda x: h264.equal(x, 5), lambda value: value == 5),
            ("window", lambda x: h264.bit_and(h264.threshold(x, 2),
                                              h264.bit_not(h264.threshold(x, 6))),
             lambda value: 2 <= value < 6),
        )
        for name, kernel, predicate in cases:
            with self.subTest(predicate=name):
                program = h264.compile(kernel, 16, 16, input_range=(0, 8))
                baseline = h264.compile(kernel, 16, 16, input_range=(0, 8), exact_search=False)
                self.assertGreater(program.stats["exact_search_rewrites"], 0)
                self.assertLess(program.pictures, baseline.pictures)
                expected = {"output": [[int(predicate(value)) for value in row] for row in image]}
                self.assert_decoders(program, image, expected)

    def test__given_named_and_shared_arithmetic__when_searching__then_all_observable_values_remain(self):
        def kernel(a, b):
            total = h264.add(a, b)
            equals_three = h264.equal(total, 3)
            return {"sum": total, "double": h264.weighted(total, 2),
                    "predicate": equals_three,
                    "output": h264.bit_or(equals_three, h264.equal(total, 6))}
        inputs = {"a": self.image(lambda index: index % 5),
                  "b": self.image(lambda index: (index // 5) % 5)}
        program = h264.compile(kernel, 16, 16, input_ranges={"a": (0, 4), "b": (0, 4)})
        totals = [[a + b for a, b in zip(arow, brow)]
                  for arow, brow in zip(inputs["a"], inputs["b"])]
        expected = {"sum": totals,
                    "double": [[2 * value for value in row] for row in totals],
                    "predicate": [[int(value == 3) for value in row] for row in totals],
                    "output": [[int(value in (3, 6)) for value in row] for row in totals]}
        self.assertEqual(len(set(program.output_frames.values())), 4)
        self.assert_decoders(program, inputs, expected)

        def shared(a, b):
            total = h264.add(a, b)
            return {"predicate": h264.equal(total, 3), "double": h264.weighted(total, 2)}
        program = h264.compile(shared, 16, 16, input_ranges={"a": (0, 4), "b": (0, 4)})
        manifest = program.manifest()
        sums = [item for item in manifest["graph"]
                if item["op"] == "combine" and item["range"] == [0, 8]]
        self.assertEqual(len(sums), 1)
        self.assertIsNotNone(sums[0]["last_lowered_frame"])
        self.assert_decoders(program, inputs, {name: expected[name] for name in ("predicate", "double")})

    def test__given_shift_crop_and_fractional_sample_boundaries__when_searching__then_pixels_match_baseline(self):
        image = self.image(lambda index: (index * 7 + index // 16) % 9)
        cases = (
            lambda x: h264.equal(h264.shift(h264.weighted(x, 0, 3), 1), 3),
            lambda x: h264.equal(h264.shift(h264.shift(x, 1), -1), 3),
            lambda x: h264.bit_and(h264.threshold(h264.shift(x, 1, -1), 2),
                                   h264.bit_not(h264.threshold(h264.shift(x, 1, -1), 6))),
            lambda x: h264.equal(h264.sample(x, 1, 2), 3),
        )
        for index, kernel in enumerate(cases):
            with self.subTest(case=index):
                program = h264.compile(kernel, 16, 16, input_range=(0, 8))
                baseline = h264.compile(kernel, 16, 16, input_range=(0, 8), exact_search=False)
                expected = baseline.reference(image)
                if index == 0:
                    self.assertEqual(expected["output"], [[1] * 15 + [0] for _ in range(16)])
                self.assertLessEqual(program.pictures, baseline.pictures)
                self.assert_decoders(program, image, expected)

    def test__given_large_domains_or_three_independent_inputs__when_searching__then_fallback_stays_exact(self):
        def wide(a, b):
            return h264.threshold(h264.saturating_add(a, b), 117)
        def ternary(a, b, c):
            return h264.threshold(h264.add(h264.add(a, b), c), 2)
        cases = (
            (wide, {"a": (0, 255), "b": (0, 255)},
             {"a": self.image(lambda index: index), "b": self.image(lambda index: (index * 31 + 21) % 256)},
             lambda a, b: int(min(255, a + b) >= 117)),
            (ternary, dict.fromkeys(("a", "b", "c"), (0, 1)),
             {name: self.image(lambda index, bit=bit: (index >> bit) & 1)
              for bit, name in enumerate(("a", "b", "c"))},
             lambda a, b, c: int(a + b + c >= 2)),
        )
        for kernel, ranges, inputs, oracle in cases:
            with self.subTest(kernel=kernel.__name__):
                program = h264.compile(kernel, 16, 16, input_ranges=ranges)
                baseline = h264.compile(kernel, 16, 16, input_ranges=ranges, exact_search=False)
                expected = {"output": [[oracle(*(inputs[name][y][x] for name in inputs))
                                         for x in range(16)] for y in range(16)]}
                self.assertEqual({value for row in expected["output"] for value in row}, {0, 1})
                self.assertLessEqual(program.pictures, baseline.pictures)
                if kernel is wide:
                    self.assertEqual(program.stats["exact_search_rewrites"], 0)
                    self.assertEqual(program.operations, baseline.operations)
                for proof in program.stats["exact_search_proofs"]:
                    self.assertLessEqual(proof["checked_points"], 64)
                    self.assertLessEqual(len(proof["frontier"]), 2)
                self.assert_decoders(program, inputs, expected)

    def test__given_zero_or_one_instruction_kernels__when_searching__then_instruction_cost_never_increases(self):
        inputs = {"a": self.image(lambda index: index & 1),
                  "b": self.image(lambda index: (index >> 1) & 1)}
        cases = (lambda a, b: a, lambda a, b: h264.bit_not(a),
                 lambda a, b: h264.bit_or(a, b), lambda a, b: h264.bit_and(a, b))
        for kernel in cases:
            program = h264.compile(kernel, 16, 16, input_ranges=dict.fromkeys(inputs, (0, 1)))
            baseline = h264.compile(kernel, 16, 16, input_ranges=dict.fromkeys(inputs, (0, 1)), exact_search=False)
            self.assertEqual(program.stats["lowered_instructions"], baseline.stats["lowered_instructions"])
            self.assertEqual(program.stats["exact_search_rewrites"], 0)
            self.assert_decoders(program, inputs, baseline.reference(inputs))

    def test__given_disabled_search_or_optimization__when_compiling__then_previous_passes_keep_their_options(self):
        ranges = {"neighbors": (0, 8), "current": (0, 1)}
        old_optimized = h264.compile(rule, 16, 16, input_ranges=ranges, exact_search=False)
        plain = h264.compile(rule, 16, 16, input_ranges=ranges, optimize=False)
        plain_disabled = h264.compile(rule, 16, 16, input_ranges=ranges,
                                      optimize=False, exact_search=False)
        self.assertGreater(old_optimized.stats["direct_or_gates"], 0)
        self.assertEqual(plain.stats["direct_or_gates"], 0)
        self.assertEqual(plain.operations, plain_disabled.operations)
        self.assertEqual(plain.tail, plain_disabled.tail)
        for program in (old_optimized, plain, plain_disabled):
            self.assertEqual(program.stats["exact_search_rewrites"], 0)
            self.assertEqual(program.stats["exact_search_saved_instructions"], 0)
            self.assert_decoders(program, self.truth_inputs(), old_optimized.reference(self.truth_inputs()))

    def test__given_repeated_compilation_and_new_inputs__when_encoding__then_cached_predictions_stay_deterministic(self):
        ranges = {"neighbors": (0, 8), "current": (0, 1)}
        first = h264.compile(rule, 16, 16, input_ranges=ranges)
        second = h264.compile(rule, 16, 16, input_ranges=ranges)
        inputs = self.truth_inputs()
        changed = {"neighbors": [[3] * 16 for _ in range(16)],
                   "current": [[0] * 16 for _ in range(16)]}
        with patch("h264.backend.evaluate", side_effect=AssertionError("host inference")):
            first_stream = first.encode(inputs)
            changed_stream = first.encode(changed)
            self.assertEqual(first_stream, second.encode(inputs))
        self.assertIs(first.tail, first.tail)
        self.assertEqual(first.tail, second.tail)
        self.assertEqual(first_stream[len(first.encode_inputs(inputs)):], first.tail)
        self.assertEqual(changed_stream[len(first.encode_inputs(changed)):], first.tail)
        self.assertEqual(first.stats["exact_search_proofs"], second.stats["exact_search_proofs"])
        self.assert_decoders(first, changed, {"output": [[1] * 16 for _ in range(16)]})

    def test__given_saved_search_program__when_reading_manifest__then_proofs_describe_executable(self):
        program = h264.compile(rule, 16, 16,
                               input_ranges={"neighbors": (0, 8), "current": (0, 1)})
        with tempfile.TemporaryDirectory() as directory:
            path = program.save(Path(directory) / "exact.h264", self.truth_inputs())
            manifest = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(manifest["pictures"], 5)
        self.assertEqual(manifest["stats"]["exact_search_rewrites"], 1)
        proofs = manifest["stats"]["exact_search_proofs"]
        self.assertEqual(len(proofs), 1)
        self.assertEqual(proofs, json.loads(json.dumps(program.stats["exact_search_proofs"])))
        proof = proofs[0]
        self.assertEqual(proof["checked_points"], 18)
        self.assertEqual(proof["original_instructions"], 10)
        self.assertEqual(len(proof["replacement_instructions"]), 3)
        self.assertCountEqual([item["range"] for item in proof["frontier"]], [[0, 8], [0, 1]])
        names = [manifest["graph"][item["graph_id"]]["params"][0] for item in proof["frontier"]]
        domains = [range(item["range"][0], item["range"][1] + 1) for item in proof["frontier"]]
        truth_table, replayed = [], []
        for inputs in product(*domains):
            values = dict(zip(names, inputs))
            truth_table.append(int(values["neighbors"] == 3 or
                                   (values["neighbors"] == 2 and values["current"] == 1)))
            registers = list(inputs)
            for instruction in proof["replacement_instructions"]:
                refs, weights = instruction["inputs"], instruction["weights"]
                self.assertIn(len(refs), (1, 2))
                self.assertEqual(len(refs), len(weights))
                self.assertTrue(all(0 <= index < len(registers) for index in refs))
                self.assertTrue(all(-128 <= weight <= 127 for weight in weights))
                self.assertTrue(-128 <= instruction["bias"] <= 127)
                self.assertTrue(0 <= instruction["denominator"] <= 7)
                shift = instruction["denominator"] + int(len(refs) == 2)
                rounding = (1 << (shift - 1)) if shift else 0
                value = (sum(registers[index] * weight for index, weight in zip(refs, weights))
                         + rounding) >> shift
                registers.append(max(0, min(255, value + instruction["bias"])))
            replayed.append(registers[-1])
        self.assertEqual(proof["truth_table"], truth_table)
        self.assertEqual(replayed, truth_table)
        self.assertTrue(manifest["prediction_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
