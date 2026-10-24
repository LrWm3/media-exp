"""Packed scheduling preserves decoder semantics and every guarded static cost."""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import h264
from decoder_core.bitstream import Writer
from decoder_core.decode import available, decode


GUARDED_COSTS = ("total_pictures", "prediction_slices", "prediction_bytes", "output_readback_bytes")


def delayed_chain(x):
    side = {f"side{index}": h264.saturating_weighted(x, 1, index+2) for index in range(6)}
    chain = x
    for _ in range(5):
        chain = h264.saturating_weighted(chain, 1, 1)
    return {**side, "output": chain, "alias": chain, "input": x}


def mixed_predictions(x):
    pair = h264.saturating_combine(x, x, 1, -1, 63)
    safe = h264.saturating_weighted(x, -1, 127)
    extreme = h264.saturating_weighted(x, 127, -128)
    rounded = h264.threshold(x, 192)
    return {"pair": pair, "safe": safe, "extreme": extreme, "rounded": rounded,
            "binary_chain": h264.saturating_combine(pair, safe, 1, 1),
            "unary_chain": h264.saturating_weighted(rounded, 2)}


def random_kernel(seed):
    def kernel(a, b):
        rng = random.Random(seed)
        values = [a, b]
        for index in range(18):
            # Bound liveness while retaining branches, joins, repeated reads,
            # signed clipping and observable motion at both halo boundaries.
            choices = values[:2]+values[-6:]
            source = rng.choice(choices)
            kind = index % 4
            if kind == 0:
                value = h264.saturating_weighted(source, rng.choice((-128, -1, 1, 2, 127)),
                                                  rng.choice((-128, -1, 1, 127)))
            elif kind == 1:
                value = h264.saturating_combine(source, rng.choice(choices), 1, rng.choice((-1, 1)), 17)
            elif kind == 2:
                value = h264.shift(source, rng.choice((-16, -1, 1, 16)), rng.choice((-16, 0, 16)))
            else:
                value = h264.sample(source, qx=rng.choice((-64, -4, 4, 64)))
            values.append(value)
        return {**{str(index): value for index, value in enumerate(values)},
                "alias": values[-1], "input_alias": a}
    return kernel


class PackedScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"03-denoising-decoder"/"compiled"))
        from denoiser import compile_model
        cls.denoiser = compile_model()

    def compile(self, kernel, *, named=False):
        ranges = {"input_ranges": {"a": (0, 255), "b": (0, 255)}} if named else {"input_range": (0, 255)}
        return h264.compile(kernel, 16, 16, optimize=False, reference_mode="retained", **ranges)

    def assert_legal_schedule(self, packed):
        original = packed.original
        flattened = [logical for group in packed.groups for logical in group]
        self.assertEqual(sorted(flattened), list(range(original.input_count, original.pictures)))
        self.assertEqual(set(packed.reference_slots), set(range(packed.pictures)))
        self.assertTrue(all(type(slot) is int and 0 <= slot < Writer.MAX_REFS
                            for slot in packed.reference_slots.values()))
        occupants = {}
        for frame in range(packed.input_count):
            occupants[packed.reference_slots[frame]] = frame
        for frame, (group, operation) in enumerate(zip(packed.groups, packed.operations), packed.input_count):
            self.assertLessEqual(len(group), len(packed.tiles))
            for logical in group:
                self.assertEqual(packed.locations[logical][0], frame)
                for source in original.operations[logical-original.input_count].sources:
                    self.assertLess(packed.locations[source.frame][0], frame)
            for child in (operation,)+operation.block_operations:
                self.assertEqual(len(child.sources), len(operation.sources))
                for source in child.sources:
                    self.assertLess(source.frame, frame)
                    self.assertEqual(occupants[packed.reference_slots[source.frame]], source.frame)
                if len(child.sources) == 2:
                    maximum = 127 if child.denominator == 7 else 128
                    self.assertTrue(-128 <= sum(source.weight for source in child.sources) <= maximum)
            occupants[packed.reference_slots[frame]] = frame
        for name, logical in original.output_frames.items():
            frame, tile = packed.locations[logical]
            self.assertEqual(packed.output_regions[name], {"frame": frame, **packed.tiles[tile]})

    def assert_auto_selection(self, original, tiles, *, mixed=True):
        stable = h264.pack(original, tiles, mixed_predictions=mixed, scheduling="stable")
        critical = h264.pack(original, tiles, mixed_predictions=mixed, scheduling="critical")
        selected = h264.pack(original, tiles, mixed_predictions=mixed)
        improves = (all(critical.costs[key] <= stable.costs[key] for key in GUARDED_COSTS) and
                    any(critical.costs[key] < stable.costs[key] for key in GUARDED_COSTS))
        expected = critical if improves else stable
        self.assertEqual(selected.stats["scheduling"], "auto")
        self.assertEqual(selected.stats["selected_schedule"], "critical" if improves else "stable")
        self.assertEqual(selected.groups, expected.groups)
        self.assertEqual(selected.tail, expected.tail)
        self.assertEqual(selected.output_regions, expected.output_regions)
        self.assertEqual(selected.stats["saved_prediction_pictures"], stable.pictures-selected.pictures)
        candidates = {item["schedule"]: item for item in selected.stats["schedule_candidates"]}
        self.assertEqual(set(candidates), {"stable", "critical"})
        for strategy, program in (("stable", stable), ("critical", critical)):
            self.assertEqual(candidates[strategy]["status"], "valid")
            self.assertEqual(candidates[strategy]["costs"], program.costs)
            self.assertEqual(program.stats["scheduling"], strategy)
            self.assertEqual(program.stats["selected_schedule"], strategy)
        for key in GUARDED_COSTS:
            self.assertLessEqual(selected.costs[key], stable.costs[key], key)
        return stable, critical, selected

    def test__given_late_deep_chain__when_scheduling_critical_paths__then_ready_dependencies_start_before_short_branches(self):
        original = self.compile(delayed_chain)
        stable = h264.pack(original, 2, scheduling="stable")
        critical = h264.pack(original, 2, scheduling="critical")
        self.assertEqual(stable.groups, [(1, 2), (3, 4), (5, 6), (7,), (8,), (9,), (10,), (11,)])
        self.assertEqual(critical.groups, [(7, 1), (8, 2), (9, 3), (10, 4), (5, 6), (11,)])
        self.assertLess(critical.pictures, stable.pictures)
        self.assert_legal_schedule(stable)
        self.assert_legal_schedule(critical)

    def test__given_frozen_denoiser_and_varied_capacities__when_selecting_automatically__then_every_cost_is_nonincreasing(self):
        for tiles in (1, 2, 3, 4, 8):
            with self.subTest(tiles=tiles):
                stable, critical, selected = self.assert_auto_selection(self.denoiser, tiles)
                self.assert_legal_schedule(selected)
                if tiles == 4:
                    self.assertEqual((stable.pictures, critical.pictures, selected.pictures), (19, 17, 17))
                    self.assertEqual(selected.stats["selected_schedule"], "critical")
                    self.assertEqual(selected.stats["saved_prediction_pictures"], 2)

    def test__given_fewer_pictures_but_more_named_output_readback__when_selecting_automatically__then_stable_schedule_wins(self):
        stable, critical, selected = self.assert_auto_selection(self.denoiser, 3)
        self.assertEqual((stable.pictures, critical.pictures), (21, 20))
        self.assertEqual((stable.costs["output_readback_bytes"], critical.costs["output_readback_bytes"]), (76800, 122880))
        self.assertEqual(selected.stats["selected_schedule"], "stable")

    def test__given_critical_schedule_with_more_prediction_bytes__when_selecting_automatically__then_stable_schedule_wins(self):
        stable, critical, selected = self.assert_auto_selection(self.denoiser, 2)
        self.assertEqual((stable.costs["prediction_bytes"], critical.costs["prediction_bytes"]), (7414, 7415))
        self.assertEqual(selected.stats["selected_schedule"], "stable")

    def test__given_identical_candidates_or_no_predictions__when_selecting_automatically__then_ties_stay_stable(self):
        for kernel in (lambda x: x, lambda x: h264.saturating_weighted(h264.saturating_weighted(x, 1, 7), -1, 127)):
            with self.subTest(kernel=kernel):
                stable, critical, selected = self.assert_auto_selection(self.compile(kernel), 4)
                self.assertEqual(stable.costs, critical.costs)
                self.assertEqual(selected.stats["selected_schedule"], "stable")
                self.assertEqual(selected.stats["saved_prediction_pictures"], 0)

    def test__given_seeded_branching_graphs__when_repeating_schedules__then_costs_dependencies_and_retained_slots_are_deterministic(self):
        for seed in range(4):
            original = self.compile(random_kernel(seed), named=True)
            for tiles in (2, 4):
                with self.subTest(seed=seed, tiles=tiles):
                    stable, critical, selected = self.assert_auto_selection(original, tiles)
                    for program in (stable, critical, selected):
                        self.assert_legal_schedule(program)
                    repeated = h264.PackedProgram(original, tiles, scheduling="auto")
                    self.assertEqual(selected.groups, repeated.groups)
                    self.assertEqual(selected.reference_slots, repeated.reference_slots)
                    self.assertEqual(selected.manifest(), repeated.manifest())
                    self.assertEqual(selected.tail, repeated.tail)

    def test__given_mixed_and_unpromotable_predictions__when_reordering__then_arity_weight_limits_and_opt_out_remain_exact(self):
        original = self.compile(mixed_predictions)
        extreme = original.output_frames["extreme"]
        for mixing in (False, True):
            for strategy in ("stable", "critical", "auto"):
                with self.subTest(mixing=mixing, strategy=strategy):
                    packed = h264.pack(original, 2, mixed_predictions=mixing, scheduling=strategy)
                    self.assert_legal_schedule(packed)
                    frame, tile_index = packed.locations[extreme]
                    tile = packed.tiles[tile_index]
                    operation = packed.operations[frame-packed.input_count]
                    child = operation.block_operations[(tile["y"]//16)*(packed.canvas_width//16)+tile["x"]//16]
                    self.assertEqual(len(child.sources), 1)
                    self.assertEqual(child.sources[0].weight, 127)
                    if mixing:
                        self.assertGreater(packed.stats["p_to_b_promotions"], 0)
                    else:
                        self.assertEqual(packed.stats["p_to_b_promotions"], 0)
                        for group in packed.groups:
                            self.assertEqual(len({len(original.operations[frame-original.input_count].sources)
                                                  for frame in group}), 1)

    def test__given_motion_aliases_and_clipping__when_decoding_every_strategy__then_named_outputs_guards_and_chroma_match(self):
        if not self.backends:
            self.skipTest("No H.264 decoder available")
        original = self.compile(random_kernel(2), named=True)
        inputs = {"a": [[16*y+x for x in range(16)] for y in range(16)],
                  "b": [[(17*x+31*y) % 256 for x in range(16)] for y in range(16)]}
        expected = original.reference(inputs)
        for strategy in ("stable", "critical", "auto"):
            packed = h264.pack(original, 4, scheduling=strategy)
            self.assertEqual(packed.output_regions["alias"], packed.output_regions["19"])
            self.assertEqual(packed.output_regions["input_alias"], packed.output_regions["0"])
            for backend in self.backends:
                with self.subTest(strategy=strategy, backend=backend):
                    self.assertEqual(packed.verify(inputs, backend), expected)
                    self.assertEqual(packed.verify(inputs, backend, readback="outputs"), expected)
        # Corrupt an actual reordered prediction, including a tile separator.
        packed = h264.pack(original, 4, scheduling="critical")
        with tempfile.TemporaryDirectory(prefix="packed-schedule-") as directory:
            path = Path(directory)/"program.h264"
            path.write_bytes(packed.encode(inputs))
            raw = decode(path, self.backends[0])
        pixels = packed.writer.width*packed.writer.height
        start = packed.input_count*pixels*3//2
        for offset in (start, start+16*packed.writer.width+16+packed.width, start+pixels):
            damaged = bytearray(raw)
            damaged[offset] ^= 1
            with self.subTest(offset=offset), self.assertRaises(AssertionError):
                packed.read(damaged)

    def test__given_one_candidate_failing_during_encoding__when_selecting_automatically__then_valid_candidate_survives(self):
        original = self.compile(delayed_chain)
        emit = Writer.prediction
        # The first active tile uses offset 2 under stable and 1 under critical.
        # Failure occurs only while computing real encoded costs, after the
        # geometry, group formation and slice-count checks have all succeeded.
        for failing_strategy, first_offset in (("critical", 1), ("stable", 2)):
            def reject_candidate(writer, frame, operation):
                if frame == 1 and operation.block_operations[0].sources[0].offset == first_offset:
                    raise ValueError("injected encoded candidate failure")
                return emit(writer, frame, operation)
            with self.subTest(failing_strategy=failing_strategy), patch.object(Writer, "prediction", new=reject_candidate):
                selected = h264.pack(original, 2)
            records = {item["schedule"]: item for item in selected.stats["schedule_candidates"]}
            surviving = "stable" if failing_strategy == "critical" else "critical"
            self.assertEqual(selected.stats["selected_schedule"], surviving)
            self.assertEqual(records[failing_strategy]["status"], "invalid")
            self.assertIn("encoded candidate failure", records[failing_strategy]["error"])
            self.assertEqual(records[surviving]["status"], "valid")
            self.assertEqual(selected.stats["saved_prediction_pictures"], 0 if surviving == "stable" else None)
            self.assert_legal_schedule(selected)
        with patch.object(Writer, "prediction", side_effect=ValueError("both candidates fail")):
            with self.assertRaisesRegex(h264.CompileError, "No legal packed schedule.*stable:.*critical:"):
                h264.pack(original, 2)

    def test__given_unknown_scheduling_mode__when_packing__then_configuration_is_rejected(self):
        original = self.compile(lambda x: x)
        for strategy in (None, False, 1, "", "fast"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(h264.CompileError, "scheduling"):
                h264.pack(original, scheduling=strategy)


if __name__ == "__main__":
    unittest.main()
