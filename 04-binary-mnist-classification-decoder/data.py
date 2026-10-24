"""Checksum-verified original MNIST IDX files; no ML runtime needed to load them."""

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import random
import struct
from urllib.request import urlopen

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
# Original resource checksums, also published in torchvision.datasets.MNIST.
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
DEFAULT_CACHE = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Digit:
    index: int
    label: int
    pixels: bytes

    def binary(self):
        return [[int(p >= 128) for p in self.pixels[y * 28:(y + 1) * 28]] for y in range(28)]


def resource(name, cache):
    path = Path(cache) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}", flush=True)
        with urlopen(BASE + name, timeout=90) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != RESOURCES[name]:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    content = path.read_bytes()
    if hashlib.md5(content).hexdigest() != RESOURCES[name]:
        raise ValueError(f"MNIST checksum mismatch: {name}")
    return gzip.decompress(content)


def load(split="test", cache=DEFAULT_CACHE):
    if split not in ("train", "test"):
        raise ValueError("MNIST split must be train or test")
    prefix = "train" if split == "train" else "t10k"
    images = resource(prefix + "-images-idx3-ubyte.gz", cache)
    labels = resource(prefix + "-labels-idx1-ubyte.gz", cache)
    magic, count, rows, cols = struct.unpack(">IIII", images[:16])
    label_magic, label_count = struct.unpack(">II", labels[:8])
    if (magic, rows, cols, label_magic, label_count) != (2051, 28, 28, 2049, count):
        raise ValueError("Unexpected MNIST IDX headers")
    if len(images) != 16 + count * 784 or len(labels) != 8 + count:
        raise ValueError("Truncated MNIST IDX data")
    return [Digit(i, label, images[16 + i * 784:16 + (i + 1) * 784])
            for i, label in enumerate(labels[8:]) if label in (3, 8)]


def training_split(digits, seed=29):
    """Reserve 500 of each class from the original training split only."""
    rng = random.Random(seed)
    train, validation = [], []
    for label in (3, 8):
        group = [digit for digit in digits if digit.label == label]
        rng.shuffle(group)
        validation.extend(group[:500])
        train.extend(group[500:])
    return train, validation


def pooled(image, threshold=1):
    return [[int(sum(image[y * 2 + dy][x * 2 + dx] for dy in range(2) for dx in range(2)) >= threshold)
             for x in range(14)] for y in range(14)]
