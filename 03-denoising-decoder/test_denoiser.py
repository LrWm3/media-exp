"""Real-decoder checks for learned inference, signed sums, and reference scheduling."""

import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from codec import available, decode, Writer
from compiler import compile_batch, compile_program
from data import error_rate, samples
from network import FrozenNetwork
from verify import extract, verify

MODEL_PATH = Path(__file__).parent / "model/weights.json"


def all_patches():
    image = [[0] * 128 for _ in range(64)]
    for bits in range(512):
        x, y = bits % 32 * 4, bits // 32 * 4
        for bit in range(9):
            image[y + bit // 3][x + bit % 3] = (bits >> bit) & 1
    return image


class DenoiserTests(unittest.TestCase):
    def infer(self, images, model, backend="ffmpeg", stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        if stream is None:
            stream, program = compile_batch(images, model)
        else:
            program = compile_program(model)
        with tempfile.TemporaryDirectory(prefix="denoiser-test-") as directory:
            path = Path(directory) / "test.h264"
            path.write_bytes(stream)
            outputs, features = extract(decode(path, backend), images, program)
        verify(outputs, features, images, model)
        return outputs, features

    def test__given_all_512_patches__when_running_learned_model_on_three_decoders__then_every_feature_and_output_bit_matches(self):
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                self.infer([all_patches()], FrozenNetwork.load(MODEL_PATH), backend)

    def test__given_extreme_thresholds_and_signed_kernels__when_decoding__then_clipping_never_corrupts_sums(self):
        rng = random.Random(31)
        for thresholds in (tuple(range(-9, 7)), tuple(range(7, 11))):
            count = len(thresholds)
            kernels = tuple(tuple(rng.choice((-1, 1)) for _ in range(9)) for _ in range(count))
            model = FrozenNetwork(kernels, thresholds, tuple(rng.choice((-1, 1)) for _ in range(count)), -count)
            with self.subTest(thresholds=thresholds):
                self.infer([all_patches()], model)

    def test__given_one_channel_and_extreme_mix_thresholds__when_decoding__then_constant_outputs_are_exact(self):
        _, images = samples(2, size=16)
        for threshold in (-1, 0, 1, 2):
            with self.subTest(threshold=threshold):
                self.infer(images, FrozenNetwork(((1,) * 9,), (5,), (-1,), threshold))

    def test__given_all_zero_all_one_and_random_edges__when_decoding__then_padding_matches_zero_padded_convolution(self):
        model = FrozenNetwork.load(MODEL_PATH)
        _, random_images = samples(2, size=16)
        self.infer([[[0] * 16 for _ in range(16)], [[1] * 16 for _ in range(16)], *random_images], model)

    def test__given_16_channels__when_scheduling__then_live_registers_are_refreshed_before_eviction(self):
        model = FrozenNetwork(((1,) * 9,) * 16, (5,) * 16, (1,) * 16, 8)
        program = compile_program(model)
        self.assertGreater(len(program.operations), Writer.MAX_REFS)
        self.assertGreater(len(program.refreshes), 0)
        for frame, op in enumerate(program.operations, 1):
            for source in op.sources:
                self.assertTrue(1 <= frame - source.frame <= Writer.MAX_REFS)
        self.infer(samples(1, size=16)[1], model)

    def test__given_new_input__when_reusing_identical_prediction_bytes__then_decoder_computes_new_inference(self):
        model = FrozenNetwork.load(MODEL_PATH)
        first, second = samples(2, size=16)[1]
        writer = Writer(16, 16)
        original, _ = compile_batch([first], model)
        tail = original[len(writer.seed(first)):]
        changed, _ = compile_batch([second], model)
        self.assertEqual(tail, changed[len(writer.seed(second)):])
        self.infer([second], model, stream=writer.seed(second) + tail)

    def test__given_disabled_reference_inference__when_compiling__then_generation_does_not_need_model_outputs(self):
        model = FrozenNetwork.load(MODEL_PATH)
        with patch.object(FrozenNetwork, "predict", side_effect=AssertionError("host inference called")):
            stream, program = compile_batch(samples(1, size=16)[1], model)
        self.assertGreater(len(stream), 0)
        self.assertGreater(program.output, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch training environment unavailable")
    def test__given_trained_checkpoint__when_decoding__then_output_equals_actual_pytorch_model(self):
        import torch
        from train import Denoiser, tensor
        torch.set_num_threads(2)
        model = Denoiser()
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), weights_only=True))
        images = samples(8, seed=200000)[1]
        actual, _ = self.infer(images, FrozenNetwork.load(MODEL_PATH))
        with torch.no_grad():
            expected = (model(tensor(images)) >= 0).int().squeeze(1).tolist()
        self.assertEqual(actual, expected)

    def test__given_training_report__when_assessing_held_out_quality__then_denoising_improves_over_corrupted_input(self):
        report = json.loads(MODEL_PATH.with_name("training.json").read_text())
        clean, noisy = samples(16, seed=200000)
        output, _ = self.infer(noisy, FrozenNetwork.load(MODEL_PATH))
        self.assertLess(error_rate(output, clean), error_rate(noisy, clean) / 2)
        self.assertLess(report["train_shape_seeds"][1], report["validation_shape_seeds"][0])
        self.assertLess(report["validation_shape_seeds"][1], report["test_shape_seeds"][0])

    def test__given_invalid_weights_thresholds_or_images__when_compiling__then_invalid_inputs_are_rejected(self):
        for kernel, threshold in (((0,) * 9, 0), ((1,) * 8, 0), ((1,) * 9, 11)):
            with self.subTest(kernel=kernel, threshold=threshold), self.assertRaises(ValueError):
                FrozenNetwork((kernel,), (threshold,), (1,), 0)
        with self.assertRaises(ValueError):
            compile_batch([], FrozenNetwork.load(MODEL_PATH))
        with self.assertRaises(ValueError):
            compile_batch([[[2] * 16 for _ in range(16)]], FrozenNetwork.load(MODEL_PATH))

    def test__given_truncated_inference_batch__when_verifying__then_missing_predictions_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "batch length"):
            verify([], [], samples(1, size=16)[1], FrozenNetwork.load(MODEL_PATH))


if __name__ == "__main__":
    unittest.main(verbosity=2)
