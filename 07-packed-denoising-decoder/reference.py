"""Original frozen-network oracle, independent of compiler and codec formulas."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-denoising-decoder" / "compiled"))

from denoiser_original import ROOT, load

FrozenNetwork = load("network").FrozenNetwork
MODEL_PATH = ROOT / "model" / "weights.json"


def denoise(image, model=None):
    model = model or FrozenNetwork.load(MODEL_PATH)
    output, features = model.predict(image)
    return {"output": output, **{f"feature_{index}": feature
                                 for index, feature in enumerate(features)}}
