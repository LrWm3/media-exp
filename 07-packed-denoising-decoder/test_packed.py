"""Exercise packed inference with real decoders and the unchanged frozen model."""

import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel import MODEL_PATH, FrozenNetwork, compile_baseline, compile_denoiser, h264, load
from reference import denoise
from decoder_core.decode import available, decode


def board(value=0, width=32, height=32):
    return [[value] * width for _ in range(height)]


def all_patch_images():
    """All 512 neighborhoods in eight small images, separated by zero pixels."""
    images = []
    for batch in range(8):
        image = board()
        for index in range(64):
            bits = batch * 64 + index
            left, top = index % 8 * 4, index // 8 * 4
            for bit in range(9):
                image[top + bit // 3][left + bit % 3] = (bits >> bit) & 1
        images.append(image)
    return images


def edge_images():
    corners = board()
    for x, y in ((0, 0), (31, 0), (0, 31), (31, 31)):
        corners[y][x] = 1
    perimeter = [[int(x in (0, 31) or y in (0, 31)) for x in range(32)] for y in range(32)]
    checkerboard = [[(x + y) % 2 for x in range(32)] for y in range(32)]
    rng = random.Random(70003)
    noise = [[rng.randrange(2) for _ in range(32)] for _ in range(32)]
    return [board(), board(1), corners, perimeter, checkerboard, noise]


class PackedDenoiserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backends = tuple(name for name in ("ffmpeg", "openh264", "vaapi") if available(name))
        if not cls.backends:
            raise unittest.SkipTest("No FFmpeg, OpenH264, or VAAPI decoder available")
        cls.model = FrozenNetwork.load(MODEL_PATH)
        cls.baseline = compile_baseline(cls.model)
        cls.programs = {tiles: h264.pack(cls.baseline, tiles=tiles) for tiles in (1, 2, 4, 8)}
        cls.program = cls.programs[8]

    def decoded(self, program, images, backend, stream=None):
        with tempfile.TemporaryDirectory(prefix="packed-denoiser-test-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(stream if stream is not None else b"".join(program.encode(image) for image in images))
            raw = decode(path, backend)
        stride = program.pictures * program.writer.width * program.writer.height * 3 // 2
        self.assertEqual(len(raw), len(images) * stride)
        return [program.read(raw[index * stride:(index + 1) * stride]) for index in range(len(images))]

    def assert_model(self, program, images, model=None):
        expected = [denoise(image, model or self.model) for image in images]
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(self.decoded(program, images, backend), expected)
        return expected

    def test__given_all_512_patches__when_decoding__then_every_feature_and_output_matches_frozen_network(self):
        images = all_patch_images()
        patterns = []
        for image in images:
            for index in range(64):
                left, top = index % 8 * 4, index // 8 * 4
                patterns.append(sum(image[top + bit // 3][left + bit % 3] << bit for bit in range(9)))
        self.assertEqual(patterns, list(range(512)))
        expected = self.assert_model(self.program, images)
        # Saved channels 1 and 2 alias exactly; channels 0 and 3 share weights
        # but use different thresholds. Packing must preserve both cases.
        self.assertEqual(self.model.kernels[1], self.model.kernels[2])
        self.assertEqual(self.model.thresholds[1], self.model.thresholds[2])
        self.assertEqual(self.program.output_frames["feature_1"], self.program.output_frames["feature_2"])
        self.assertTrue(any(item["feature_0"] != item["feature_3"] for item in expected))
        for item in expected:
            self.assertEqual(item["feature_1"], item["feature_2"])

    def test__given_zero_one_random_and_edge_inputs__when_decoding__then_padding_never_leaks_between_tiles(self):
        self.assert_model(self.program, edge_images())

    def test__given_one_two_four_or_eight_tiles__when_decoding__then_outputs_match_with_fewer_pictures(self):
        image = edge_images()[-1]
        self.assert_model(self.baseline, [image])
        for tiles, program in self.programs.items():
            with self.subTest(tiles=tiles):
                self.assertEqual((program.width, program.height), (32, 32))
                self.assertEqual(program.costs["total_pictures"], program.pictures)
                self.assertLessEqual(program.pictures, self.baseline.pictures)
                self.assert_model(program, [image])
        self.assertLess(self.program.pictures, self.programs[1].pictures)
        packed_samples = self.program.pictures * self.program.writer.width * self.program.writer.height
        baseline_samples = self.baseline.pictures * self.baseline.writer.width * self.baseline.writer.height
        self.assertGreater(packed_samples, baseline_samples)

    def test__given_other_legal_channel_weights__when_compiling__then_packing_does_not_assume_saved_layers(self):
        model = FrozenNetwork(
            ((1,) * 9, (-1,) * 9, (1, -1, 1, -1, 1, -1, 1, -1, 1),
             (-1, 1, -1, 1, -1, 1, -1, 1, -1)),
            (5, -4, 1, -1), (1, -1, 1, -1), 0)
        program = compile_denoiser(model, tiles=4)
        self.assert_model(program, edge_images()[1:], model)

    def test__given_disabled_host_inference__when_reusing_prediction_tail__then_decoders_compute_changed_results(self):
        first, second = board(), edge_images()[-1]
        expected = denoise(second, self.model)
        self.assertNotEqual(denoise(first, self.model)["output"], expected["output"])
        with patch.object(FrozenNetwork, "predict", side_effect=AssertionError("Host model inference invoked")), \
                patch("h264.backend.Program.reference", side_effect=AssertionError("Host program reference invoked")), \
                patch("h264.backend.evaluate", side_effect=AssertionError("Host graph evaluator invoked")), \
                patch("h264.verify.evaluate", side_effect=AssertionError("Host graph evaluator invoked")):
            program = compile_denoiser(self.model)
            first_prefix, second_prefix = program.encode_inputs(first), program.encode_inputs(second)
            original, changed = program.encode(first), program.encode(second)
            self.assertNotEqual(first_prefix, second_prefix)
            self.assertEqual(original[len(first_prefix):], program.tail)
            self.assertEqual(changed[len(second_prefix):], program.tail)
            for backend in self.backends:
                with self.subTest(backend=backend):
                    self.assertEqual(self.decoded(program, [second], backend,
                                                 second_prefix + original[len(first_prefix):]), [expected])

    def test__given_output_picture_selection__when_reading__then_named_features_and_aliases_remain_complete(self):
        image = edge_images()[-1]
        expected = denoise(image, self.model)
        self.assertEqual(self.program.output_picture_indices,
                         tuple(sorted(set(self.program.output_frames.values()))))
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(self.program.decode(image, backend, readback="outputs"), expected)

    def test__given_damaged_guards_or_unused_tiles__when_reading__then_corruption_is_rejected(self):
        program = self.program
        width, height = program.writer.width, program.writer.height
        first, unused = program.tiles[:2]
        self.assertEqual(program.manifest()["frame_tiles"]["0"], [0])
        offsets = {
            "outer_guard": 0,
            "separator": (first["y"] + 16) * width + first["x"] + first["width"] + 16,
            "unused_seed_tile": (unused["y"] + 16) * width + unused["x"] + 16,
            "chroma": width * height,
        }
        with tempfile.TemporaryDirectory(prefix="packed-denoiser-guards-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(program.encode(board(1)))
            for backend in self.backends:
                with self.subTest(backend=backend):
                    raw = decode(path, backend)
                    self.assertEqual(program.read(raw), denoise(board(1), self.model))
                    for name, offset in offsets.items():
                        changed = bytearray(raw)
                        changed[offset] = 1 if name != "chroma" else 127
                        with self.subTest(region=name), self.assertRaises(AssertionError):
                            program.read(changed)

    def test__given_saved_stream__when_loading_manifest__then_picture_counts_and_hashes_match(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix="packed-denoiser-save-") as directory:
            path = self.program.save(Path(directory) / "denoiser.h264", board())
            manifest = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(manifest["pictures"], self.program.pictures)
            self.assertEqual(manifest["output_frames"], self.program.output_frames)
            self.assertEqual(manifest["stream_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(manifest["prediction_sha256"], hashlib.sha256(self.program.tail).hexdigest())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment unavailable")
    def test__given_original_checkpoint__when_decoding__then_all_features_and_output_match_pytorch(self):
        import torch
        train = load("train")
        torch.set_num_threads(2)
        model = train.Denoiser()
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), weights_only=True))
        images = load("data").samples(4, seed=200000)[1]
        with torch.no_grad():
            tensor = train.tensor(images)
            scores = torch.nn.functional.conv2d(tensor, train.binary_weight(model.kernels), padding=1)
            features = (scores - model.thresholds.view(1, -1, 1, 1) >= 0).int().tolist()
            outputs = (model(tensor) >= 0).int().squeeze(1).tolist()
        expected = [{"output": output, **{f"feature_{index}": feature
                                         for index, feature in enumerate(feature_maps)}}
                    for output, feature_maps in zip(outputs, features)]
        for backend in self.backends:
            with self.subTest(backend=backend):
                self.assertEqual(self.decoded(self.program, images, backend), expected)


class PackedValidationTests(unittest.TestCase):
    def test__given_invalid_tile_counts__when_packing__then_compile_rejects_them(self):
        program = h264.compile(lambda image: h264.bit_not(image), 16, 16)
        for tiles in (0, -1, True, 1.5, "8"):
            with self.subTest(tiles=tiles), self.assertRaises(ValueError):
                h264.pack(program, tiles=tiles)

    def test__given_fractional_or_large_or_per_block_motion__when_packing__then_unsupported_reads_are_rejected(self):
        kernels = {
            "fractional": lambda image: h264.sample(image, qx=1),
            "large": lambda image: h264.shift(image, dx=17),
            "gather": lambda image: h264.gather(image, offsets=((4, 0),)),
        }
        for name, kernel in kernels.items():
            program = h264.compile(kernel, 16, 16, input_range=(0, 255))
            with self.subTest(motion=name), self.assertRaises(ValueError):
                h264.pack(program, tiles=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
