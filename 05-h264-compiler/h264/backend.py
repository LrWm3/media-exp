"""A reusable prediction program specialized to a static board shape."""

from dataclasses import asdict
from functools import cached_property
import hashlib
import json
from pathlib import Path
import tempfile

from decoder_core.bitstream import Writer
from decoder_core.decode import decode
from .graph import CompileError, integer, trace
from .lower import lower
from .schedule import schedule
from .verify import evaluate, read


class Program:
    def __init__(self, function, width, height, input_range, optimize):
        integer(width, "width")
        integer(height, "height")
        self.width, self.height = width, height
        self.input_range = tuple(input_range)
        self.binary_input = self.input_range == (0, 1)
        source, outputs, traced_count = trace(function, input_range)
        instructions, virtual_outputs, self.graph, self.stats, self.graph_outputs = lower(source, outputs, optimize)
        self.operations, self.output_frames, self.refreshes, peak, self.graph_frames = schedule(instructions, virtual_outputs, self.binary_input)
        self.stats.update({"traced_nodes": traced_count, "dead_nodes": traced_count-self.stats["reachable_nodes_before_cse"],
                           "reference_copies": len(self.refreshes), "peak_live_values": peak, "optimized": optimize})
        maximum = max((abs(value)*4 for op in self.operations for ref in op.sources for value in (ref.dx, ref.dy)), default=0)
        self.writer = Writer(width, height, motion_bits=max(4, maximum.bit_length()))

    @property
    def pictures(self):
        return len(self.operations)+1

    @cached_property
    def tail(self):
        return b"".join(self.writer.prediction(frame, op) for frame, op in enumerate(self.operations, 1))

    def validate(self, image):
        if len(image) != self.height or any(len(row) != self.width for row in image):
            raise ValueError("Input dimensions differ from compiled static shape")
        if any(type(value) is not int or not self.input_range[0] <= value <= self.input_range[1] for row in image for value in row):
            raise ValueError("Input contains values outside its declared integer range")

    def encode(self, image):
        self.validate(image)
        seed = self.writer.seed(image) if self.binary_input else self.writer.seed_luma(image)
        return seed+self.tail

    def manifest(self):
        ids = {item: index for index, item in enumerate(self.graph)}
        return {"format": "h264-python-0.1", "width": self.width, "height": self.height,
                "input_range": list(self.input_range), "seed_encoding": "binary 0/255" if self.binary_input else "literal u8",
                "guard": 16, "pictures": self.pictures, "output_frames": self.output_frames,
                "graph_outputs": self.graph_outputs, "stats": self.stats,
                "graph": [{"id": ids[item], "op": item.op, "args": [ids[arg] for arg in item.args],
                           "params": list(item.params), "range": [item.lo, item.hi],
                           "last_lowered_frame": self.graph_frames.get(ids[item])} for item in self.graph],
                "instructions": [{"frame": frame, **asdict(op)} for frame, op in enumerate(self.operations, 1)],
                "refresh_frames": self.refreshes}

    def save(self, path, image):
        path = Path(path)
        manifest_path = path.with_suffix(".json")
        if path == manifest_path:
            raise ValueError("Stream path collides with JSON manifest; use a .h264 filename")
        stream = self.encode(image)
        manifest = self.manifest()
        manifest.update({"stream_sha256": hashlib.sha256(stream).hexdigest(),
                         "prediction_sha256": hashlib.sha256(self.tail).hexdigest()})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        return path

    def read(self, raw):
        return read(raw, self)

    def decode(self, image, backend="ffmpeg"):
        with tempfile.TemporaryDirectory(prefix="h264-kernel-") as directory:
            path = Path(directory)/"program.h264"
            path.write_bytes(self.encode(image))
            return self.read(decode(path, backend))

    def reference(self, image):
        self.validate(image)
        return evaluate(self.graph, self.graph_outputs, image)

    def verify(self, image, backend="ffmpeg"):
        actual = self.decode(image, backend)
        expected = self.reference(image)
        for name in actual:
            if actual[name] != expected[name]:
                raise AssertionError(f"Decoded output {name} differs from graph semantics")
        return actual


def compile(function, width=32, height=32, input_range=(0, 1), optimize=True):
    return Program(function, width, height, input_range, optimize)
