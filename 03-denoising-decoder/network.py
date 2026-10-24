"""Frozen binary network and independent integer inference (no PyTorch needed)."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class FrozenNetwork:
    kernels: tuple[tuple[int, ...], ...]
    thresholds: tuple[int, ...]
    mix_weights: tuple[int, ...]
    mix_threshold: int

    def __post_init__(self):
        count = len(self.kernels)
        if not 1 <= count <= 16 or len(self.thresholds) != count or len(self.mix_weights) != count:
            raise ValueError("Network requires 1..16 matching channels")
        if any(len(k) != 9 or any(type(w) is not int or w not in (-1, 1) for w in k) for k in self.kernels):
            raise ValueError("Every 3x3 kernel needs nine signed binary weights")
        if any(type(w) is not int or w not in (-1, 1) for w in self.mix_weights):
            raise ValueError("Mix weights must be signed binary weights")
        if any(type(t) is not int or not -9 <= t <= 10 for t in self.thresholds):
            raise ValueError("Convolution thresholds must be integers in [-9, 10]")
        if type(self.mix_threshold) is not int or not -count <= self.mix_threshold <= count + 1:
            raise ValueError("Mix threshold outside representable range")

    @classmethod
    def load(cls, path):
        model = json.loads(Path(path).read_text())
        if model.get("format") != "binary-denoiser-v1":
            raise ValueError("Unsupported frozen model format")
        return cls(tuple(tuple(k) for k in model["kernels"]), tuple(model["thresholds"]),
                   tuple(model["mix_weights"]), model["mix_threshold"])

    def predict(self, image):
        height, width = len(image), len(image[0])
        features = []
        for kernel, threshold in zip(self.kernels, self.thresholds):
            plane = []
            for y in range(height):
                row = []
                for x in range(width):
                    total = 0
                    for index, weight in enumerate(kernel):
                        nx, ny = x + index % 3 - 1, y + index // 3 - 1
                        if 0 <= nx < width and 0 <= ny < height:
                            total += weight * image[ny][nx]
                    row.append(int(total >= threshold))
                plane.append(row)
            features.append(plane)
        output = [[int(sum(weight * feature[y][x] for weight, feature in zip(self.mix_weights, features))
                       >= self.mix_threshold) for x in range(width)] for y in range(height)]
        return output, features
