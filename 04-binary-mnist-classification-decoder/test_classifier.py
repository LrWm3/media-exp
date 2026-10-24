"""Real-decoder tests: arithmetic, input independence, tiling, and checkpoint parity."""

import base64
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from codec import Operation, Source, Writer, available, decode
from compiler import CompiledClassifier, compile_program
from data import Digit, pooled, resource, training_split
from network import FrozenNetwork
from verify import extract, verify

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model/weights.json"


def fixtures():
    obj = json.loads((ROOT / "assets/demo-inputs.json").read_text())
    return [Digit(row["index"], row["label"], base64.b64decode(row["pixels"])) for row in obj["samples"]]


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FrozenNetwork.load(MODEL_PATH)
        cls.compiled = CompiledClassifier(cls.model)
        cls.digits = fixtures()

    def infer(self, images, model=None, compiled=None, backend="ffmpeg", stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        model = model or self.model
        compiled = compiled or (self.compiled if model == self.model else CompiledClassifier(model))
        with tempfile.TemporaryDirectory(prefix="mnist-test-") as directory:
            path = Path(directory) / "test.h264"
            path.write_bytes(compiled.encode(images) if stream is None else stream)
            records = extract(decode(path, backend), images, compiled)
        verify(records, images, model)
        return records

    def test__given_mnist_digits__when_three_decoders_classify__then_pool_hidden_and_output_bits_match(self):
        images = [d.binary() for d in self.digits[:8]]
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                self.infer(images, backend=backend)

    def test__given_every_2x2_pattern__when_pooling__then_all_four_pool_thresholds_match_counts(self):
        image = [[0]*28 for _ in range(28)]
        for bits in range(16):
            x, y = (bits % 7)*2, (bits // 7)*2
            for bit in range(4):
                image[y+bit//2][x+bit%2] = (bits >> bit) & 1
        image[27][27] = 1
        for threshold in range(1, 5):
            with self.subTest(threshold=threshold):
                model = FrozenNetwork(((1,)*196,), (1,), (1,), 1, threshold)
                self.infer([image], model)

    def test__given_extreme_signed_neurons__when_decoding__then_bias_and_threshold_clipping_preserve_decisions(self):
        rng = random.Random(13)
        weights = ((1,)*196, (-1,)*196, (1,)*196, (-1,)*196,
                   tuple(rng.choice((-1, 1)) for _ in range(196)), (-1,)*195+(1,))
        model = FrozenNetwork(weights, (-196, -196, 197, 197, 0, 1), (1, -1, 1, -1, 1, 1), 1)
        blank = [[0]*28 for _ in range(28)]
        distant = [row[:] for row in blank]
        distant[27][27] = 1
        self.infer([blank, [[1]*28 for _ in range(28)], distant, self.digits[0].binary()], model)

    def test__given_one_neuron__when_output_threshold_is_extreme__then_constants_are_exact(self):
        images = [self.digits[0].binary()]
        for threshold in (-1, 0, 1, 2):
            with self.subTest(threshold=threshold):
                model = FrozenNetwork(((1,)*196,), (30,), (-1,), threshold)
                self.infer(images, model)

    def test__given_sixteen_neurons__when_scheduling__then_every_reference_remains_in_decoder_memory(self):
        model = FrozenNetwork(((1,)*196,)*16, (40,)*16, (1,)*16, 8)
        program = compile_program(model)
        self.assertGreater(len(program.refreshes), 100)
        self.assertTrue(all(1 <= frame-source.frame <= 16
                            for frame, op in enumerate(program.operations, 1) for source in op.sources))
        self.infer([self.digits[0].binary()], model)

    def test__given_swapped_and_partial_tiles__when_reusing_instruction_bytes__then_each_digit_is_independent(self):
        images = [d.binary() for d in self.digits[:8]]
        reversed_images = list(reversed(images))
        original = self.compiled.encode(images)
        seed = self.compiled.writer.seed(self.compiled.board(images))
        tail = original[len(seed):]
        replacement = self.compiled.writer.seed(self.compiled.board(reversed_images)) + tail
        self.assertEqual(replacement, self.compiled.encode(reversed_images))
        self.infer(reversed_images, stream=replacement)
        self.infer(images[:1])

    def test__given_disabled_host_inference__when_encoding__then_only_input_pixels_and_weights_are_needed(self):
        with patch.object(FrozenNetwork, "predict", side_effect=AssertionError("host inference called")):
            stream = self.compiled.encode([self.digits[0].binary()])
        self.assertTrue(stream.endswith(self.compiled.tail))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch training environment unavailable")
    def test__given_original_checkpoint__when_classifying__then_actual_pytorch_and_decoder_agree(self):
        import torch
        from train import Classifier, tensors
        torch.set_num_threads(2)
        model = Classifier()
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), weights_only=True))
        with torch.no_grad():
            expected = (model(tensors(self.digits[:8])[0]) >= 0).int().tolist()
        records = self.infer([d.binary() for d in self.digits[:8]])
        self.assertEqual([r["bit"] for r in records], expected)

    def test__given_wide_motion__when_writing__then_sps_bounds_are_enforced(self):
        operation = Operation("far tap", (Source(0, dx=26, dy=26),))
        with self.assertRaisesRegex(ValueError, "SPS bound"):
            Writer(32, 32).prediction(1, operation)
        self.assertTrue(Writer(32, 32, motion_bits=7).prediction(1, operation))
        with self.assertRaises(ValueError):
            Writer(32, 32, motion_bits=17)

    def test__given_validation_reservation__when_splitting_training_data__then_indices_do_not_overlap(self):
        digits = [Digit(i, 3 if i%2 else 8, b"") for i in range(12000)]
        train, validation = training_split(digits)
        self.assertEqual(len(validation), 1000)
        self.assertFalse({d.index for d in train} & {d.index for d in validation})
        self.assertEqual({d.index for d in train+validation}, {d.index for d in digits})

    def test__given_corrupt_dataset_or_inputs__when_loading__then_invalid_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "t10k-labels-idx1-ubyte.gz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resource("t10k-labels-idx1-ubyte.gz", directory)
        with self.assertRaises(ValueError):
            self.compiled.encode([[[2]*28 for _ in range(28)]])
        with self.assertRaises(ValueError):
            FrozenNetwork(((0,)*196,), (0,), (1,), 0)
        with self.assertRaisesRegex(AssertionError, "batch length"):
            verify([], [self.digits[0].binary()], self.model)
        with self.assertRaisesRegex(AssertionError, "picture count"):
            extract(b"", [self.digits[0].binary()], self.compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
