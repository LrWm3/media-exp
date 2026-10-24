"""Application-level contracts for supplied-field resampling experiments."""

import json
from pathlib import Path
import tempfile
import unittest

import codec_compute as media
from codec_compute.workloads.resampling.kernels import (fixture, quality_errors, quality_fixture,
                                                       quality_kernel, reference, tasks)


class ResamplingWorkloadTests(unittest.TestCase):
    def test__given_six_tasks_and_controls__when_decoding_two_fixtures__then_both_plans_match_independent_formulas(self):
        for task in tasks(32):
            for fused in (False, True):
                program = media.compile(task.kernel, target="h264", width=32, height=32,
                                        input_ranges=task.input_ranges, h264_fuse_shifts=fused)
                tail = program.tail
                for index in range(2):
                    with self.subTest(task=task.name, fused=fused, fixture=index):
                        images = fixture(task, index, 32)
                        expected = reference(task, images)
                        self.assertEqual(program.reference(images), expected)
                        self.assertEqual(program.decode(images), expected)
                        self.assertEqual(program.tail, tail)

    def test__given_patch_atlas__when_routing__then_only_half_size_roi_remains_and_shape_stays_fixed(self):
        task = tasks(64)[0]
        inputs = fixture(task, size=64)
        output = reference(task, inputs)["output"]
        self.assertEqual((len(output), len(output[0])), (64, 64))
        self.assertTrue(all(output[y][x] == 0 for y in range(64) for x in range(64) if x >= 32 or y >= 32))
        for by in range(2):
            for bx in range(2):
                dx, dy = task.fields["integer"][by*4+bx]
                self.assertEqual(output[by*16][bx*16], inputs["image"][by*16+dy][bx*16+dx])

    def test__given_dense_polarity_maps__when_thresholding__then_positive_and_negative_events_never_overlap(self):
        task = next(t for t in tasks(32) if t.name == "bipolar-events")
        outputs = reference(task, fixture(task, 1, 32))
        positives = [v for row in outputs["positive"] for v in row]
        negatives = [v for row in outputs["negative"] for v in row]
        self.assertTrue(any(positives) and any(negatives))
        self.assertTrue(all(p+n <= 1 for p, n in zip(positives, negatives)))

    def test__given_nonlinear_known_motion__when_compensating__then_actual_decode_improves_midpoint_error(self):
        images, midpoint = quality_fixture(32)
        program = media.compile(quality_kernel, target="h264", width=32, height=32,
                                input_ranges={"a": (0, 255), "b": (0, 255)})
        actual = program.decode(images)
        self.assertEqual(actual, program.reference(images))
        errors = quality_errors(actual, midpoint)
        self.assertLess(errors["metrics"]["compensated"]["mae"], 1)
        self.assertGreater(errors["metrics"]["naive"]["mae"], 10)

    def test__given_256_square_pcm_inputs__when_uploading__then_current_writer_limit_is_explicit(self):
        program = media.compile(lambda image: image, target="h264", width=256, height=256,
                                input_ranges={"image": (0, 255)})
        with self.assertRaisesRegex(ValueError, "initial access-unit size limit"):
            program.encode({"image": [[0]*256 for _ in range(256)]})

    def test__given_capture_planes__when_binding_provenance__then_requires_decoder_evidence_and_actual_frame_pixels(self):
        from codec_compute.workloads.resampling.capture import bind_outputs, digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def record(name, data):
                (root / name).write_bytes(data)
                return {"path": name, "sha256": digest(data), "bytes": len(data)}
            manifest = {"ir": {"shape": [16, 16], "inputs": [{"name": "image"}]},
                        "coded_shape": [48, 48], "guard": 16, "costs": {"total_pictures": 1},
                        "output_frames": {"output": 0}}
            m = record("manifest.json", json.dumps(manifest).encode())
            raw = bytearray(48*48) + bytearray([128])*(48*48//2)
            raw[16*48+16] = 42
            expected = record("full.i420", raw)
            p = {**record("plane.u8", bytes([42])+bytes(255)), "width": 16, "height": 16, "range": [0, 255]}
            outputs, inputs = {"output": p}, {"image": p}
            evidence = [{"status": "pass", "i420_sha256": expected["sha256"]}]
            bind_outputs(root, m, expected, outputs, evidence, inputs)
            for records in ([], [{"status": "fail", "i420_sha256": expected["sha256"]}],
                            [{"status": "pass", "i420_sha256": "0"*64}]):
                with self.assertRaisesRegex(AssertionError, "decoder evidence"):
                    bind_outputs(root, m, expected, outputs, records, inputs)
            wrong = {**record("wrong.u8", bytes(256)), "width": 16, "height": 16, "range": [0, 255]}
            with self.assertRaisesRegex(AssertionError, "decoded I420 crop"):
                bind_outputs(root, m, expected, {"output": wrong}, evidence, inputs)
            with self.assertRaisesRegex(AssertionError, "decoded input"):
                bind_outputs(root, m, expected, outputs, evidence, {"image": wrong})


if __name__ == "__main__":
    unittest.main()
