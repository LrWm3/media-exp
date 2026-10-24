"""Verify decoder inference against the unchanged model and checkpoint."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoiser import MODEL_PATH, FrozenNetwork, compile_model, extract
from denoiser_original import load
from decoder_core.decode import available, decode


def all_patches():
    image = [[0] * 128 for _ in range(64)]
    for bits in range(512):
        left, top = bits % 32 * 4, bits // 32 * 4
        for bit in range(9):
            image[top + bit // 3][left + bit % 3] = (bits >> bit) & 1
    return image


class CompiledDenoiserTests(unittest.TestCase):
    def infer(self, images, backend="ffmpeg", model=None, program=None, stream=None):
        if not available(backend):
            self.skipTest(f"{backend} unavailable")
        model = model or FrozenNetwork.load(MODEL_PATH)
        program = program or compile_model(model, width=len(images[0][0]), height=len(images[0]))
        with tempfile.TemporaryDirectory(prefix="compiled-denoiser-test-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(stream if stream is not None else b"".join(program.encode(image) for image in images))
            outputs, features = extract(decode(path, backend), images, program, len(model.kernels))
        load("verify").verify(outputs, features, images, model)
        return outputs, features

    def test__given_all_512_patches__when_decoded_on_available_backends__then_every_feature_and_output_matches(self):
        image = all_patches()
        program = compile_model(width=128, height=64)
        for backend in ("ffmpeg", "openh264", "vaapi"):
            with self.subTest(backend=backend):
                self.infer([image], backend, program=program)

    def test__given_zero_one_and_noisy_edges__when_decoding__then_zero_padded_model_matches(self):
        images = [[[0] * 16 for _ in range(16)], [[1] * 16 for _ in range(16)],
                  *load("data").samples(2, size=16)[1]]
        self.infer(images)

    def test__given_changed_input__when_reusing_prediction_tail__then_decoder_computes_new_result(self):
        first, second = load("data").samples(2, size=16)[1]
        program = compile_model(width=16, height=16)
        original = program.encode(first)
        changed = program.encode(second)
        first_seed = program.writer.seed(first)
        second_seed = program.writer.seed(second)
        self.assertEqual(original[len(first_seed):], changed[len(second_seed):])
        self.assertEqual(changed[len(second_seed):], program.tail)
        self.infer([second], program=program, stream=second_seed + original[len(first_seed):])

    def test__given_disabled_host_inference__when_compiling_and_encoding__then_model_predictions_are_unneeded(self):
        model = FrozenNetwork.load(MODEL_PATH)
        with patch.object(FrozenNetwork, "predict", side_effect=AssertionError("host inference called")):
            program = compile_model(model, width=16, height=16)
            stream = program.encode(load("data").samples(1, size=16)[1][0])
        self.assertTrue(stream.endswith(program.tail))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch environment unavailable")
    def test__given_original_checkpoint__when_decoding__then_actual_pytorch_output_matches(self):
        import torch
        train = load("train")
        torch.set_num_threads(2)
        model = train.Denoiser()
        model.load_state_dict(torch.load(MODEL_PATH.with_suffix(".pt"), weights_only=True))
        images = load("data").samples(8, seed=200000)[1]
        outputs, _ = self.infer(images)
        with torch.no_grad():
            expected = (model(train.tensor(images)) >= 0).int().squeeze(1).tolist()
        self.assertEqual(outputs, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
