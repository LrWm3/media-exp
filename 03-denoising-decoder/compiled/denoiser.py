"""Compile the existing frozen CNN through the shared image-expression DSL."""

from denoiser_original import ROOT, load
import h264

FrozenNetwork = load("network").FrozenNetwork
MODEL_PATH = ROOT / "model/weights.json"


def compile_model(model=None, *, width=32, height=32, optimize=True):
    model = model or FrozenNetwork.load(MODEL_PATH)

    def denoise(image):
        features = [h264.conv2d_binary(image, [weights[index:index + 3] for index in (0, 3, 6)], threshold)
                    for weights, threshold in zip(model.kernels, model.thresholds)]
        output = h264.binary_dot(features, model.mix_weights, model.mix_threshold)
        return {"output": output, **{f"feature_{index}": feature
                                      for index, feature in enumerate(features)}}

    return h264.compile(denoise, width=width, height=height,
                        input_range=(0, 1), optimize=optimize)


def extract(raw, images, program, channels):
    """Read actual decoder planes, retaining all learned feature activations."""
    if not images or len(raw) % len(images):
        raise ValueError("Decoded data must contain one complete program per image")
    stride = len(raw) // len(images)
    outputs, features = [], []
    for index in range(len(images)):
        values = program.read(raw[index * stride:(index + 1) * stride])
        outputs.append(values["output"])
        features.append([values[f"feature_{channel}"] for channel in range(channels)])
    return outputs, features
