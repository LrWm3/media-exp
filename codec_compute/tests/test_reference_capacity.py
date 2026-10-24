"""Tight H.264 DPB declarations preserve scheduled slots and image semantics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import codec_compute as media
from codec_compute.targets.h264 import _legacy
from decoder_core.bitstream import Writer
from decoder_core.decode import available, decode
from decoder_core.test_block_predictions import i420_picture


class ReferenceCapacityTests(unittest.TestCase):
    def test__given_invalid_or_non_h264_option__when_compiling__then_both_entry_points_reject_it(self):
        graph = media.trace(media.copy, width=16, height=16)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(media.CompileError, "h264_tight_dpb must be Boolean"):
                    media.compile(media.copy, h264_tight_dpb=value)
                with self.assertRaisesRegex(media.CompileError, "h264_tight_dpb must be Boolean"):
                    media.compile_graph(graph, h264_tight_dpb=value)
        for target in ("hevc", "h265", "av1"):
            with self.subTest(target=target), self.assertRaisesRegex(media.CompileError, "applies only"):
                media.compile_graph(graph, target=target, h264_tight_dpb=True)

    def test__given_default_or_disabled_option__when_compiling__then_existing_stream_bytes_and_slot_plans_match(self):
        for kernel, inputs in ((media.copy, {"image": [[37]*16 for _ in range(16)]}),
                               (media.average, {"a": [[17]*16 for _ in range(16)],
                                                "b": [[231]*16 for _ in range(16)]})):
            options = {"width": 16, "height": 16,
                       "input_ranges": {name: (0, 255) for name in inputs}}
            default = media.compile(kernel, **options)
            disabled = media.compile(kernel, **options, h264_tight_dpb=False)
            self.assertEqual(default.encode(inputs), disabled.encode(inputs))
            self.assertEqual(default.native.reference_slots, disabled.native.reference_slots)
            self.assertEqual(default.manifest()["reference_capacity"], 16)
            self.assertEqual(default.manifest()["optimizations"]["tight_dpb"],
                             {"enabled": False, "before_capacity": 16, "after_capacity": 16})

    def test__given_fusion_candidates_with_different_live_sets__when_tightening__then_each_uses_its_own_unchanged_plan(self):
        def kernel(a, b):
            return media.average(media.average(media.shift(a, 1, 0), b), a)

        options = {"width": 16, "height": 16, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
        with patch("codec_compute.targets.h264.Writer", wraps=Writer) as constructor:
            constructor.MAX_REFS = Writer.MAX_REFS
            tight = media.compile(kernel, **options, h264_tight_dpb=True)
        candidates = [call.kwargs for call in constructor.call_args_list if "reference_slots" in call.kwargs]
        self.assertEqual([call["reference_capacity"] for call in candidates], [3, 2])
        for call in candidates:
            self.assertEqual(call["reference_capacity"], max(call["reference_slots"].values())+1)
        baseline = media.compile(kernel, **options)
        self.assertEqual(tight.native.operations, baseline.native.operations)
        self.assertEqual(tight.native.reference_slots, baseline.native.reference_slots)
        self.assertEqual(tight.native.output_frames, baseline.native.output_frames)
        self.assertEqual(tight.manifest()["reference_capacity"], 2)
        self.assertEqual(tight.manifest()["optimizations"]["tight_dpb"],
                         {"enabled": True, "before_capacity": 16, "after_capacity": 2})
        self.assertEqual(tight.manifest()["optimizations"]["spatial_read_fusion"]["reason"], "accepted")

    def test__given_sparse_scheduled_indices__when_tightening__then_maximum_index_sets_capacity_without_remapping(self):
        schedule = _legacy().backend.schedule

        def sparse(*args, **kwargs):
            result = list(schedule(*args, **kwargs))
            result[-1] = {frame: 7 if slot == 1 else slot for frame, slot in result[-1].items()}
            return tuple(result)

        with patch.object(_legacy().backend, "schedule", side_effect=sparse):
            program = media.compile(media.average, h264_tight_dpb=True,
                                    input_ranges={"a": (0, 255), "b": (0, 255)})
        self.assertEqual(program.native.reference_slots, {0: 0, 1: 7, 2: 0})
        self.assertEqual(program.manifest()["reference_capacity"], 8)

    def test__given_thirty_two_named_outputs_and_retained_input__when_decoding_tight_and_default__then_every_full_picture_matches(self):
        def kernel(a, b):
            result, current = {"original": a}, b
            for step in range(32):
                current = (media.average(a, current) if step % 2 else
                           media.affine(current, weights=(1,), bias=1, saturating=True))
                result["step_"+str(step)] = current
            return result

        options = {"width": 16, "height": 16, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
        programs = {enabled: media.compile(kernel, **options, h264_tight_dpb=enabled) for enabled in (False, True)}
        self.assertEqual(programs[True].manifest()["reference_capacity"], 2)
        self.assertEqual(programs[True].native.reference_slots, programs[False].native.reference_slots)
        self.assertEqual(programs[True].manifest()["costs"]["computation_pictures"], 32)
        tails = {enabled: program.tail for enabled, program in programs.items()}
        backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
        if not backends:
            self.skipTest("No H.264 decoder available")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"capacity.h264"
            for fixture in range(2):
                inputs = {"a": [[(x*17+y*31+fixture*73) % 256 for x in range(16)] for y in range(16)],
                          "b": [[(x*71+y*37+fixture*19) % 256 for x in range(16)] for y in range(16)]}
                pictures = [inputs["a"], inputs["b"]]
                for step in range(32):
                    pictures.append([[(a+b+1)//2 if step % 2 else min(255, b+1)
                                      for a, b in zip(arow, brow)]
                                     for arow, brow in zip(inputs["a"], pictures[-1])])
                expected = {"original": pictures[0], **{"step_"+str(i): pictures[i+2] for i in range(32)}}
                raw = b"".join(i420_picture(board) for board in pictures)
                with patch("codec_compute.ir.evaluate", side_effect=AssertionError("Runtime evaluated IR")):
                    for enabled, program in programs.items():
                        self.assertEqual(program.tail, tails[enabled])
                        path.write_bytes(program.encode(inputs))
                        for backend in backends:
                            with self.subTest(tight=enabled, fixture=fixture, decoder=backend):
                                self.assertEqual(decode(path, backend), raw)
                        self.assertEqual(program.decode(inputs, decoder=backends[0]), expected)

    def test__given_one_input_chain__when_tightening__then_capacity_one_preserves_all_outputs(self):
        def kernel(image):
            current = image
            for _ in range(32):
                current = media.affine(current, weights=(1,), bias=1, saturating=True)
            return current

        program = media.compile(kernel, width=16, height=16, h264_tight_dpb=True,
                                input_ranges={"image": (0, 255)})
        self.assertEqual(program.manifest()["reference_capacity"], 1)
        self.assertEqual(set(program.native.reference_slots.values()), {0})
        self.assertEqual(program.manifest()["costs"]["computation_pictures"], 32)
        image = [[(x*17+y*31) % 256 for x in range(16)] for y in range(16)]
        expected = [[min(255, value+32) for value in row] for row in image]
        for backend in ("ffmpeg", "openh264", "vaapi"):
            if available(backend):
                with self.subTest(decoder=backend):
                    self.assertEqual(program.decode({"image": image}, decoder=backend), {"output": expected})


if __name__ == "__main__":
    unittest.main()
