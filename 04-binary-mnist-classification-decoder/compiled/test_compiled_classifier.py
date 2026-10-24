"""Offline actual-decoder tests for the existing MNIST binary classifier."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import MODEL_PATH, CompiledClassifier, FrozenNetwork, fixtures
from classifier_original import load
from decoder_core.decode import available, decode


class CompiledClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FrozenNetwork.load(MODEL_PATH)
        cls.compiled = CompiledClassifier(cls.model)
        cls.digits = fixtures()

    def infer(self, images, *, compiled=None, backend="ffmpeg", stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        compiled = compiled or self.compiled
        with tempfile.TemporaryDirectory(prefix="compiled-mnist-test-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(compiled.encode(images) if stream is None else stream)
            records = compiled.read(decode(path, backend), len(images))
        load("verify").verify(records, images, compiled.model)
        return records

    def test__given_eight_offline_digits__when_available_decoders_classify__then_pooled_hidden_and_class_bits_match(self):
        images = [digit.binary() for digit in self.digits[:8]]
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                self.infer(images, backend=backend)

    def test__given_reordered_and_partial_tiles__when_reusing_tail__then_each_anchor_matches_its_digit(self):
        images = [digit.binary() for digit in self.digits[:8]]
        reversed_images = list(reversed(images))
        program = self.compiled.program
        original = self.compiled.encode(images)
        first_seed = program.writer.seed(self.compiled.board(images))
        tail = original[len(first_seed):]
        second_seed = program.writer.seed(self.compiled.board(reversed_images))
        self.assertEqual(tail, program.tail)
        self.assertEqual(second_seed + tail, self.compiled.encode(reversed_images))
        self.infer(reversed_images, stream=second_seed + tail)
        self.infer(images[:1])

    def test__given_zero_one_and_last_pixel_inputs__when_decoding__then_full_receptive_field_matches(self):
        zero = [[0] * 28 for _ in range(28)]
        last = [row[:] for row in zero]
        last[27][27] = 1
        self.infer([zero, [[1] * 28 for _ in range(28)], last])

    def test__given_disabled_host_inference__when_compiling_and_encoding__then_only_weights_and_inputs_are_used(self):
        with patch.object(FrozenNetwork, "predict", side_effect=AssertionError("host inference called")):
            compiled = CompiledClassifier(self.model, capacity=1)
            stream = compiled.encode([self.digits[0].binary()])
        self.assertTrue(stream.endswith(compiled.program.tail))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment unavailable")
    def test__given_original_checkpoint__when_decoding__then_actual_pytorch_hidden_and_output_bits_match(self):
        import torch
        train = load("train")
        torch.set_num_threads(2)
        model = train.Classifier()
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), weights_only=True))
        images = [digit.binary() for digit in self.digits[:8]]
        records = self.infer(images)
        with torch.no_grad():
            tensor, _ = train.tensors(self.digits[:8])
            pooled = (torch.nn.functional.avg_pool2d(tensor, 2) * 4 >= model.pool_threshold).float().flatten(1)
            hidden = train.binary_activation(torch.nn.functional.linear(pooled, train.binary_weight(model.weights))
                                             - model.thresholds).int().tolist()
            expected = (model(tensor) >= 0).int().tolist()
        self.assertEqual([record["hidden"] for record in records], hidden)
        self.assertEqual([record["bit"] for record in records], expected)

    def test__given_invalid_tiles__when_encoding__then_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compiled.encode([[[2] * 28 for _ in range(28)]])
        with self.assertRaises(ValueError):
            self.compiled.encode([])
        with self.assertRaises(ValueError):
            CompiledClassifier(self.model, capacity=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
