"""Express the existing MNIST binary MLP as a shared H.264 compiler graph."""

import base64
import json
from math import ceil

from classifier_original import ROOT, load
import h264

FrozenNetwork = load("network").FrozenNetwork
MODEL_PATH = ROOT / "model/weights.json"


def fixtures():
    """Small original test fixture; offline tests never download MNIST."""
    data = load("data")
    obj = json.loads((ROOT / "assets/demo-inputs.json").read_text())
    return [data.Digit(row["index"], row["label"], base64.b64decode(row["pixels"]))
            for row in obj["samples"]]


class CompiledClassifier:
    """Tile up to eight 28×28 digits; all class bits are decoded at tile anchors."""

    def __init__(self, model=None, capacity=8, *, optimize=True):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError("Tile capacity must be 1..8")
        self.model = model or FrozenNetwork.load(MODEL_PATH)
        self.capacity = capacity
        self.columns = min(4, capacity)
        self.rows = ceil(capacity / self.columns)
        self.width, self.height = self.columns * 32, self.rows * 32

        def classify(image):
            horizontal = h264.add(image, h264.shift(image, 1, 0))
            pooled = h264.threshold(h264.add(horizontal, h264.shift(horizontal, 0, 1)),
                                    self.model.pool_threshold)
            terms = [h264.shift(pooled, x * 2, y * 2) for y in range(14) for x in range(14)]
            hidden = [h264.binary_dot(terms, weights, threshold)
                      for weights, threshold in zip(self.model.weights, self.model.thresholds)]
            output = h264.binary_dot(hidden, self.model.mix_weights, self.model.mix_threshold)
            return {"output": output,
                    **{f"hidden_{index}": node for index, node in enumerate(hidden)},
                    "pooled": pooled}

        self.program = h264.compile(classify, width=self.width, height=self.height,
                                    input_range=(0, 1), optimize=optimize)

    def anchor(self, index):
        return (index % self.columns) * 32, (index // self.columns) * 32

    def board(self, images):
        if not 1 <= len(images) <= self.capacity:
            raise ValueError("Batch must contain 1..capacity images")
        board = [[0] * self.width for _ in range(self.height)]
        for index, image in enumerate(images):
            if len(image) != 28 or any(len(row) != 28 or any(type(value) is not int or value not in (0, 1)
                                                            for value in row) for row in image):
                raise ValueError("Expected binary 28x28 images")
            left, top = self.anchor(index)
            for y, row in enumerate(image):
                board[top + y][left:left + 28] = row
        return board

    def encode(self, images):
        return self.program.encode(self.board(images))

    def read(self, raw, count):
        if type(count) is not int or not 1 <= count <= self.capacity:
            raise ValueError("Readback count must be 1..capacity")
        values = self.program.read(raw)
        records = []
        for index in range(count):
            left, top = self.anchor(index)
            record = {
                "bit": values["output"][top][left],
                "hidden": [values[f"hidden_{channel}"][top][left]
                           for channel in range(len(self.model.weights))],
                "pooled": [[values["pooled"][top + y * 2][left + x * 2]
                            for x in range(14)] for y in range(14)],
            }
            if any(value not in (0, 1) for value in [record["bit"], *record["hidden"],
                                                    *(v for row in record["pooled"] for v in row)]):
                raise AssertionError("Nonbinary decoded activation")
            records.append(record)
        return records
