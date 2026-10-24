"""Compile semantic graphs; input loading, decoding, and verification stay distinct."""

import hashlib
import json
from pathlib import Path

from . import ir


class Program:
    def __init__(self, graph, native):
        self.graph, self.native = graph, native
        self.target = native.target

    @property
    def tail(self):
        return self.native.tail

    def encode(self, inputs):
        self.graph.validate(inputs)
        return self.native.encode(inputs)

    def decode(self, inputs, *, decoder="ffmpeg"):
        self.graph.validate(inputs)
        result = self.native.decode(inputs, decoder)
        if set(result) != {name for name, _ in self.graph.outputs}:
            raise AssertionError("Decoder did not produce exactly the named outputs")
        for name, node in self.graph.outputs:
            board = result[name]
            if len(board) != self.graph.height or any(len(row) != self.graph.width for row in board):
                raise AssertionError(f"Decoded shape differs for {name}")
            if any(not node.lo <= value <= node.hi for row in board for value in row):
                raise AssertionError(f"Decoded output {name} violates its proven range")
        return result

    def reference(self, inputs):
        return ir.evaluate(self.graph, inputs)

    def verify(self, inputs, *, decoder="ffmpeg"):
        actual = self.decode(inputs, decoder=decoder)
        expected = self.reference(inputs)
        for name in expected:
            if actual[name] != expected[name]:
                for y, (got, want) in enumerate(zip(actual[name], expected[name])):
                    for x, (a, b) in enumerate(zip(got, want)):
                        if a != b:
                            raise AssertionError(f"{self.target} output {name}[{y}][{x}]: decoded {a}, expected {b}")
        return actual

    def manifest(self):
        return {"format": "codec-compute-program-1", "ir": self.graph.manifest(),
                **self.native.manifest(), "prediction_sha256": hashlib.sha256(self.tail).hexdigest()}

    def save(self, path, inputs):
        path = Path(path)
        manifest_path = path.with_suffix(".json")
        if path == manifest_path:
            raise ValueError("Stream and manifest paths must differ")
        stream = self.encode(inputs)
        manifest = self.manifest()
        manifest["stream_sha256"] = hashlib.sha256(stream).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path


def compile(function, *, target="h264", width=32, height=32, input_ranges=None):
    graph = ir.trace(function, width=width, height=height, input_ranges=input_ranges)
    return compile_graph(graph, target=target)


def compile_graph(graph, *, target="h264"):
    if not isinstance(graph, ir.Graph):
        raise TypeError("compile_graph requires a traced Graph")
    ir.validate_graph(graph)
    if target == "h264":
        from .targets.h264 import H264Program
        native = H264Program(graph)
    elif target in ("hevc", "h265"):
        from .targets.hevc import HEVCProgram
        native = HEVCProgram(graph)
    elif target == "av1":
        from .targets.av1 import AV1Program
        native = AV1Program(graph)
    else:
        raise ir.CompileError(f"Unknown codec target: {target}")
    return Program(graph, native)
