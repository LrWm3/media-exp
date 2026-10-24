"""Frozen binary MLP and an independent integer reference implementation."""

from dataclasses import dataclass
import json
from pathlib import Path

from data import pooled


@dataclass(frozen=True)
class FrozenNetwork:
    weights: tuple[tuple[int, ...], ...]
    thresholds: tuple[int, ...]
    mix_weights: tuple[int, ...]
    mix_threshold: int
    pool_threshold: int = 1

    def __post_init__(self):
        channels = len(self.weights)
        if not 1 <= channels <= 16 or len(self.thresholds) != channels or len(self.mix_weights) != channels:
            raise ValueError("Network needs 1..16 matching hidden neurons")
        if any(len(row) != 196 or any(type(w) is not int or w not in (-1, 1) for w in row) for row in self.weights):
            raise ValueError("Each neuron needs 196 signed binary weights")
        if any(type(t) is not int or not -196 <= t <= 197 for t in self.thresholds):
            raise ValueError("Hidden thresholds must be integers in [-196, 197]")
        if any(type(w) is not int or w not in (-1, 1) for w in self.mix_weights):
            raise ValueError("Output weights must be signed binary")
        if type(self.mix_threshold) is not int or not -channels <= self.mix_threshold <= channels + 1:
            raise ValueError("Output threshold outside representable range")
        if type(self.pool_threshold) is not int or not 1 <= self.pool_threshold <= 4:
            raise ValueError("Pool threshold must be an integer in [1, 4]")

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        if (obj.get("format"), obj.get("input_size"), obj.get("input_threshold"), obj.get("pool_size"), obj.get("classes")) != (
                "binary-mnist-3-vs-8-v1", 28, 128, 2, [3, 8]):
            raise ValueError("Unsupported model format or preprocessing")
        return cls(tuple(tuple(row) for row in obj["weights"]), tuple(obj["thresholds"]),
                   tuple(obj["mix_weights"]), obj["mix_threshold"], obj["pool_threshold"])

    def predict(self, image):
        cells = pooled(image, self.pool_threshold)
        flat = [value for row in cells for value in row]
        hidden = [int(sum(w*x for w, x in zip(weights, flat)) >= threshold)
                  for weights, threshold in zip(self.weights, self.thresholds)]
        bit = int(sum(w*x for w, x in zip(self.mix_weights, hidden)) >= self.mix_threshold)
        return bit, hidden, cells
