"""Reuse the unchanged H.264 selector, reference scheduler, writer, and reader."""

from dataclasses import replace
from functools import cached_property, lru_cache
import importlib.util
from pathlib import Path
import sys
import tempfile

from decoder_core.bitstream import Writer
from decoder_core.decode import decode as decode_file
from ..ir import UnsupportedOperation


@lru_cache(maxsize=1)
def _legacy():
    path = Path(__file__).resolve().parents[2] / "05-h264-compiler" / "h264"
    # Keep private package identity separate from callers using `import h264`.
    name = "codec_compute._reference_h264"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class H264Program:
    target = "h264"
    suffix = ".h264"

    def __init__(self, graph):
        legacy = _legacy()
        if len(graph.inputs) > 15:
            raise UnsupportedOperation(self.target, "inputs", "this scheduler supports at most 15 input pictures")
        self.graph_spec = graph
        self.width, self.height = graph.width, graph.height
        self.graph = graph.nodes
        ids = {node: index for index, node in enumerate(graph.nodes)}
        self.graph_outputs = {name: ids[node] for name, node in graph.outputs}
        values = {node: index for index, node in enumerate(graph.inputs)}
        instructions, next_value = [], len(values)
        for graph_id, node in enumerate(graph.nodes):
            if node.op == "input":
                continue
            proxies = tuple(legacy.Input(f"x{i}", (arg.lo, arg.hi)) for i, arg in enumerate(node.args))
            try:
                expression = self._expression(legacy, node, proxies)
                selected, outputs, *_ = legacy.backend.lower(proxies, {"out": expression}, exact_search=False)
            except legacy.CompileError as error:
                raise UnsupportedOperation(self.target, node.op, str(error)) from error
            registers = {index: values[arg] for index, arg in enumerate(node.args)}
            for instruction in selected:
                refs = tuple(replace(source, frame=registers[source.frame]) for source in instruction.operation.sources)
                instructions.append(legacy.lower.Instruction(next_value,
                                    replace(instruction.operation, sources=refs), graph_id))
                registers[instruction.result] = next_value
                next_value += 1
            values[node] = registers[outputs["out"]]
        try:
            (self.operations, self.output_frames, self.refreshes, self.peak_live,
             _, self.reference_slots) = legacy.backend.schedule(
                instructions, {name: values[node] for name, node in graph.outputs}, False,
                input_count=len(graph.inputs), retain=True)
            maximum = max((abs(value) for op in self.operations for source in op.sources
                           for value in (4 * source.dx + source.qx, 4 * source.dy + source.qy)), default=0)
            self.writer = Writer(self.width, self.height, reference_slots=self.reference_slots,
                                 slice_mode="row", motion_bits=max(4, maximum.bit_length()))
        except (ValueError, legacy.CompileError) as error:
            raise UnsupportedOperation(self.target, "resources", str(error)) from error
        self.pictures = len(graph.inputs) + len(self.operations)

    @staticmethod
    def _expression(legacy, node, args):
        if node.op == "copy":
            return legacy.weighted(args[0])
        if node.op in ("affine", "affine_sat"):
            weights, bias = node.params[:-1], node.params[-1]
            if len(args) > 2:
                raise UnsupportedOperation("h264", node.op, "multi-source affine synthesis is not implemented")
            if len(args) == 1:
                fn = legacy.saturating_weighted if node.op == "affine_sat" else legacy.weighted
                return fn(args[0], weights[0], bias)
            fn = legacy.saturating_combine if node.op == "affine_sat" else legacy.combine
            return fn(*args, *weights, bias)
        if node.op == "threshold":
            return legacy.threshold(args[0], node.params[0])
        if node.op in ("and", "or"):
            return (legacy.bit_and if node.op == "and" else legacy.bit_or)(*args)
        if node.op == "shift":
            dx, dy, border = node.params
            if border != "zero":
                raise UnsupportedOperation("h264", "shift", "this writer implements zero borders; clamp synthesis is unimplemented")
            return legacy.shift(args[0], dx, dy)
        raise UnsupportedOperation("h264", node.op, "no exact lowering implemented")

    @cached_property
    def tail(self):
        return b"".join(self.writer.prediction(frame, operation)
                        for frame, operation in enumerate(self.operations, len(self.graph_spec.inputs)))

    def encode_inputs(self, images):
        return b"".join(self.writer.input_luma(images[node.params[0]], frame)
                        for frame, node in enumerate(self.graph_spec.inputs))

    def encode(self, images):
        return self.encode_inputs(images) + self.tail

    def decode(self, images, decoder="ffmpeg"):
        with tempfile.TemporaryDirectory(prefix="codec-compute-h264-") as directory:
            path = Path(directory) / "program.h264"
            path.write_bytes(self.encode(images))
            return _legacy().backend.read(decode_file(path, decoder), self)

    def manifest(self):
        return {"target": self.target, "coded_shape": [self.writer.height, self.writer.width],
                "output_frames": self.output_frames,
                "costs": {"input_pictures": len(self.graph_spec.inputs),
                          "computation_pictures": len(self.operations), "total_pictures": self.pictures,
                          "prediction_bytes": len(self.tail), "peak_live_references": self.peak_live},
                "reference_slots": self.reference_slots,
                "scheduling": "H.264 retained long-term references", "guard": 16}
