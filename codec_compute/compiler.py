"""Compile semantic graphs; input loading, decoding, and verification stay distinct."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from . import ir


class Program:
    def __init__(self, graph, native, *, lowered_graph=None, synthesis_records=()):
        self.graph, self.native = graph, native
        self.target = native.target
        self.lowered_graph = graph if lowered_graph is None else lowered_graph
        self._synthesis_records = deepcopy(list(synthesis_records))

    @property
    def synthesis_records(self):
        return deepcopy(self._synthesis_records)

    @property
    def tail(self):
        return self.native.tail

    def encode(self, inputs):
        self.graph.validate(inputs)
        return self.native.encode(inputs)

    def decode(self, inputs, *, decoder="ffmpeg"):
        self.graph.validate(inputs)
        result = self.native.decode(inputs, decoder)
        return self._validate_outputs(result)

    def _validate_outputs(self, result):
        if set(result) != {name for name, _ in self.graph.outputs}:
            raise AssertionError("Decoder did not produce exactly the named outputs")
        for name, node in self.graph.outputs:
            board = result[name]
            if len(board) != self.graph.height or any(len(row) != self.graph.width for row in board):
                raise AssertionError(f"Decoded shape differs for {name}")
            if any(not node.lo <= value <= node.hi for row in board for value in row):
                raise AssertionError(f"Decoded output {name} violates its proven range")
        return result

    def session(self, *, decoder="nvdec", helper=None, timeout=30):
        """Keep one decoder pipeline alive across sequential input submissions.

        Only H.264/NVDEC is currently implemented. Close the context to check
        final hardware diagnostics; each request independently uploads inputs
        and resets reference state with an IDR. No computation runs on the host.
        """
        from .runtime import H264Session
        if self.target != "h264" or decoder != "nvdec":
            raise ir.UnsupportedOperation(self.target, "session", "resident execution currently requires H.264/NVDEC")
        return H264Session(self, helper=helper, timeout=timeout)

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
                "lowered_ir": self.lowered_graph.manifest(), "synthesis": self.synthesis_records,
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


def compile(function, *, target="h264", width=32, height=32, input_ranges=None,
            synthesis=True, synthesis_max_instructions=3, synthesis_max_candidates=50000,
            h264_fuse_shifts=True, h264_level=31, h264_merge_zero_guards=True,
            h264_prediction_slice_rows=1, h264_tight_dpb=False):
    graph = ir.trace(function, width=width, height=height, input_ranges=input_ranges)
    return compile_graph(graph, target=target, synthesis=synthesis,
                         synthesis_max_instructions=synthesis_max_instructions,
                         synthesis_max_candidates=synthesis_max_candidates,
                         h264_fuse_shifts=h264_fuse_shifts, h264_level=h264_level,
                         h264_merge_zero_guards=h264_merge_zero_guards,
                         h264_prediction_slice_rows=h264_prediction_slice_rows,
                         h264_tight_dpb=h264_tight_dpb)


def compile_graph(graph, *, target="h264", synthesis=True,
                  synthesis_max_instructions=3, synthesis_max_candidates=50000,
                  h264_fuse_shifts=True, h264_level=31, h264_merge_zero_guards=True,
                  h264_prediction_slice_rows=1, h264_tight_dpb=False):
    if not isinstance(graph, ir.Graph):
        raise TypeError("compile_graph requires a traced Graph")
    ir.validate_graph(graph)
    if target not in ("h264", "hevc", "h265", "av1"):
        raise ir.CompileError(f"Unknown codec target: {target}")
    if type(synthesis) is not bool:
        raise ir.CompileError("synthesis must be Boolean")
    if type(h264_fuse_shifts) is not bool:
        raise ir.CompileError("h264_fuse_shifts must be Boolean")
    if type(h264_merge_zero_guards) is not bool:
        raise ir.CompileError("h264_merge_zero_guards must be Boolean")
    if type(h264_tight_dpb) is not bool:
        raise ir.CompileError("h264_tight_dpb must be Boolean")
    if target != "h264" and h264_tight_dpb:
        raise ir.CompileError("h264_tight_dpb applies only to the H.264 target")
    if type(h264_prediction_slice_rows) is not int or h264_prediction_slice_rows < 0:
        raise ir.CompileError("h264_prediction_slice_rows must be a nonnegative integer")
    if target != "h264" and h264_prediction_slice_rows != 1:
        raise ir.CompileError("h264_prediction_slice_rows applies only to the H.264 target")
    if type(h264_level) is not int or h264_level not in (31, 41, 51):
        raise ir.CompileError("h264_level must be 31, 41, or 51")
    if target != "h264" and h264_level != 31:
        raise ir.CompileError("h264_level applies only to the H.264 target")
    for name, value in (("synthesis_max_instructions", synthesis_max_instructions),
                        ("synthesis_max_candidates", synthesis_max_candidates)):
        if type(value) is not int or value < 0:
            raise ir.CompileError(f"{name} must be a nonnegative integer")
    lowered, records = graph, []
    if synthesis:
        from .synthesis.lower import lower_graph
        lowered, records = lower_graph(graph, target=target, max_instructions=synthesis_max_instructions,
                                      max_candidates=synthesis_max_candidates)
    if target == "h264":
        from .targets.h264 import H264Program
        native = H264Program(lowered, fuse_shifts=h264_fuse_shifts, level_idc=h264_level,
                             merge_zero_guards=h264_merge_zero_guards,
                             prediction_slice_rows=h264_prediction_slice_rows,
                             tight_dpb=h264_tight_dpb)
    elif target in ("hevc", "h265"):
        from .targets.hevc import HEVCProgram
        native = HEVCProgram(lowered)
    elif target == "av1":
        from .targets.av1 import AV1Program
        native = AV1Program(lowered)
    else:
        raise ir.CompileError(f"Unknown codec target: {target}")
    return Program(graph, native, lowered_graph=lowered, synthesis_records=records)
