"""Fuse spatial reads, then reuse the H.264 selector, scheduler, and writer."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import cached_property, lru_cache
import importlib.util
from pathlib import Path
import sys
import tempfile

from decoder_core.bitstream import Writer, Operation, Source
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

    def __init__(self, graph, *, fuse_shifts=True):
        consumers = defaultdict(list)
        for node in graph.nodes:
            for arg in node.args:
                consumers[arg].append(node)
        named = {node for _, node in graph.outputs}
        scalar = {"copy", "affine", "affine_sat", "threshold", "and", "or", "average"}
        # Do not compose movements: each materialized crop is observable. A
        # named shift remains a real picture, as does any shift read by a shift.
        eligible = {node for node in graph.nodes if node.op == "shift"
                    and node.params[2] == "zero" and node not in named
                    and consumers[node] and all(user.op in scalar for user in consumers[node])}
        baseline_error, baseline_cost = None, None
        try:
            baseline = self._build(graph, set())
            if fuse_shifts and eligible:
                baseline_cost = baseline._cost()
        except UnsupportedOperation as error:
            if not fuse_shifts or not eligible or error.operation != "resources":
                raise
            baseline, baseline_error = None, error
        chosen = baseline
        stats = {"enabled": fuse_shifts, "eligible_nodes": len(eligible), "fused_nodes": 0,
                 "reason": "disabled" if not fuse_shifts else "no eligible shifts"}
        if fuse_shifts and eligible:
            try:
                candidate = self._build(graph, eligible)
                candidate_cost = candidate._cost()
            except UnsupportedOperation as error:
                if baseline is None:
                    raise baseline_error from error
                stats.update(reason="candidate rejected", detail=str(error))
            else:
                stats.update(baseline_cost=baseline_cost, candidate_cost=candidate_cost)
                # Fusion changes lifetimes and scheduling. Compare emitted
                # streams, not just the count of removed graph nodes.
                if baseline_cost is None or all(candidate_cost[key] <= baseline_cost[key]
                                                for key in candidate_cost):
                    chosen = candidate
                    stats.update(fused_nodes=len(eligible), reason="accepted")
                else:
                    stats.update(reason="scheduled cost regression")
        self.__dict__.update(chosen.__dict__)
        self.optimizations = {"shift_read_fusion": stats}

    def _cost(self):
        try:
            size = len(self.tail)
        except ValueError as error:
            raise UnsupportedOperation(self.target, "resources", str(error)) from error
        return {"computation_pictures": len(self.operations), "prediction_bytes": size,
                "output_readback_pictures": len(set(self.output_frames.values()))}

    @classmethod
    def _build(cls, graph, fused):
        self = cls.__new__(cls)
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
            if node.op == "input" or node in fused:
                continue
            bases = tuple(arg.args[0] if arg in fused else arg for arg in node.args)
            if node.op == "average":
                # Keep the complete sum in the decoder's weighted predictor;
                # materializing it as a u8 image would change full-range means.
                refs = tuple(Source(values[base], dx=arg.params[0], dy=arg.params[1])
                             if arg in fused else Source(values[base])
                             for arg, base in zip(node.args, bases))
                # Use the same motion limits as standalone shift selection.
                for arg in node.args:
                    if arg in fused:
                        try:
                            legacy.shift(legacy.Input(), *arg.params[:2])
                        except legacy.CompileError as error:
                            raise UnsupportedOperation(self.target, "shift", str(error)) from error
                operation = Operation("rounded average", refs)
                instructions.append(legacy.lower.Instruction(next_value, operation, graph_id))
                values[node] = next_value
                next_value += 1
                continue
            proxies = tuple(legacy.Input(f"x{i}", (arg.lo, arg.hi)) for i, arg in enumerate(bases))
            try:
                # Wrapping a proxy in a shift lets the existing selector fold
                # the moving read only at its first use in a scalar sequence.
                operands = tuple(legacy.shift(proxy, *arg.params[:2]) if arg in fused else proxy
                                 for proxy, arg in zip(proxies, node.args))
                expression = self._expression(legacy, node, operands)
                selected, outputs, *_ = legacy.backend.lower(proxies, {"out": expression}, exact_search=False)
            except legacy.CompileError as error:
                raise UnsupportedOperation(self.target, node.op, str(error)) from error
            registers = {index: values[arg] for index, arg in enumerate(bases)}
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
        return self

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
                "optimizations": deepcopy(self.optimizations),
                "scheduling": "H.264 retained long-term references", "guard": 16}
