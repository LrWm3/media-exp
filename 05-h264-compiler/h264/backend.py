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
    def __init__(self, function, width, height, input_range, optimize, input_ranges=None,
                 reference_mode=None, slice_mode=None):
        integer(width, "width")
        integer(height, "height")
        self.width, self.height = width, height
        self.named_inputs = input_ranges is not None
        self.input_range = tuple(input_range) if not self.named_inputs else None
        self.binary_input = self.input_range == (0, 1)
        sources, outputs, traced_count = trace(function, input_range, input_ranges)
        self.input_ranges = {source.params[0]: (source.lo, source.hi) for source in sources}
        self.input_count = len(sources)
        self.reference_mode = reference_mode or ("retained" if optimize else "sliding")
        self.slice_mode = slice_mode or ("row" if optimize else "macroblock")
        if self.reference_mode not in ("retained", "sliding"):
            raise CompileError("reference_mode must be 'retained' or 'sliding'")
        if self.slice_mode not in ("row", "macroblock"):
            raise CompileError("slice_mode must be 'row' or 'macroblock'")
        instructions, virtual_outputs, self.graph, self.stats, self.graph_outputs = lower(sources, outputs, optimize)
        if any(item.op == "gather" and len(item.params[0]) != (width//16)*(height//16) for item in self.graph):
            raise CompileError("gather needs one offset per active 16x16 block")
        self.operations, self.output_frames, self.refreshes, peak, self.graph_frames, self.reference_slots = schedule(
            instructions, virtual_outputs, self.binary_input, self.input_count,
            retain=self.reference_mode == "retained")
        self.stats.update({"traced_nodes": traced_count, "dead_nodes": traced_count-self.stats["reachable_nodes_before_cse"],
                           "reference_copies": len(self.refreshes), "peak_live_values": peak, "optimized": optimize})
        maximum = max((abs(value) for op in self.operations for ref in op.sources
                       for value in (4*ref.dx+ref.qx, 4*ref.dy+ref.qy)), default=0)
        maximum += max((abs(value) for op in self.operations for pair in op.block_offsets for value in pair), default=0)
        self.writer = Writer(width, height, motion_bits=max(4, maximum.bit_length()),
                             reference_slots=self.reference_slots, slice_mode=self.slice_mode)

    @cached_property
    def costs(self):
        """Static costs; byte/slice totals describe the cached prediction tail."""
        return {"input_pictures": self.input_count,
                "normalization_pictures": int(self.binary_input),
                "arithmetic_pictures": self.stats["lowered_instructions"],
                "refresh_pictures": len(self.refreshes), "total_pictures": self.pictures,
                "prediction_slices": self.tail.count(b"\x00\x00\x00\x01"),
                "prediction_bytes": len(self.tail),
                "peak_live_values": self.stats["peak_live_values"],
                "reference_slots_used": len(set(self.reference_slots.values())) if self.reference_slots else None}

    @property
    def pictures(self):
        return len(self.operations)+self.input_count

    @cached_property
    def tail(self):
        return b"".join(self.writer.prediction(frame, op) for frame, op in enumerate(self.operations, self.input_count))

    def validate(self, image):
        images = image if self.named_inputs else {"image": image}
        if not isinstance(images, dict) or images.keys() != self.input_ranges.keys():
            raise ValueError("Supply exactly the named input images declared at compilation")
        for name, board in images.items():
            if len(board) != self.height or any(len(row) != self.width for row in board):
                raise ValueError(f"Input {name} dimensions differ from compiled static shape")
            lo, hi = self.input_ranges[name]
            if any(type(value) is not int or not lo <= value <= hi for row in board for value in row):
                raise ValueError(f"Input {name} contains values outside its declared integer range")
        return images

    def encode_inputs(self, image):
        """Load host images through I_PCM pictures; no reference-buffer injection."""
        images = self.validate(image)
        if self.binary_input:
            return self.writer.seed(images["image"])
        return b"".join(self.writer.input_luma(images[name], frame)
                        for frame, name in enumerate(self.input_ranges))

    def encode(self, image):
        return self.encode_inputs(image)+self.tail

    def manifest(self):
        ids = {item: index for index, item in enumerate(self.graph)}
        return {"format": "h264-python-0.2", "width": self.width, "height": self.height,
                "input_range": list(self.input_range) if self.input_range is not None else None,
                "seed_encoding": "binary 0/255" if self.binary_input else "literal u8",
                "inputs": [{"name": name, "range": list(value_range), "frame": frame,
                            "encoding": "binary 0/255" if self.binary_input else "literal u8"}
                           for frame, (name, value_range) in enumerate(self.input_ranges.items())],
                "guard": 16, "pictures": self.pictures, "output_frames": self.output_frames,
                "graph_outputs": self.graph_outputs, "stats": self.stats, "costs": self.costs,
                "reference_mode": self.reference_mode, "slice_mode": self.slice_mode,
                "reference_slots": self.reference_slots,
                "graph": [{"id": ids[item], "op": item.op, "args": [ids[arg] for arg in item.args],
                           "params": list(item.params), "range": [item.lo, item.hi],
                           "last_lowered_frame": self.graph_frames.get(ids[item])} for item in self.graph],
                "instructions": [{"frame": frame, **asdict(op)} for frame, op in enumerate(self.operations, self.input_count)],
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


def compile(function, width=32, height=32, input_range=(0, 1), optimize=True, input_ranges=None,
            reference_mode=None, slice_mode=None):
    return Program(function, width, height, input_range, optimize, input_ranges, reference_mode, slice_mode)
