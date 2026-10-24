"""Fuse exact spatial reads, then reuse the H.264 selector and scheduler."""

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


@lru_cache(maxsize=2)
def _motion_policy(level_idc):
    # Delegate component legality to the emitter's level policy. The generous
    # declared SPS bound here isolates that policy; the actual program declares
    # motion_bits from its selected vectors when its writer is constructed.
    return Writer(16, 16, motion_bits=16, level_idc=level_idc)._motion


class H264Program:
    target = "h264"
    suffix = ".h264"
    movements = {"shift", "gather", "sample_h264", "gather_h264"}

    def __init__(self, graph, *, fuse_shifts=True, level_idc=31, merge_zero_guards=True,
                 prediction_slice_rows=1):
        consumers = defaultdict(list)
        for node in graph.nodes:
            for arg in node.args:
                consumers[arg].append(node)
        named = {node for _, node in graph.outputs}
        scalar = {"copy", "affine", "affine_sat", "threshold", "and", "or", "average"}
        # Validate even reads that will disappear into consumers. Native
        # fractional filters have explicit IR semantics, not generic bilinear
        # semantics. Never compose movements across a crop/filter boundary.
        for node in graph.nodes:
            if node.op in self.movements:
                self._field(node, graph, level_idc=level_idc)
        eligible = {node for node in graph.nodes if node.op in self.movements and node not in named
                    and consumers[node] and all(user.op in scalar for user in consumers[node])}
        baseline_error, baseline_cost = None, None
        try:
            baseline = self._build(graph, set(), level_idc=level_idc,
                                   merge_zero_guards=merge_zero_guards,
                                   prediction_slice_rows=prediction_slice_rows)
            if fuse_shifts and eligible:
                baseline_cost = baseline._cost()
        except UnsupportedOperation as error:
            if not fuse_shifts or not eligible or error.operation != "resources":
                raise
            baseline, baseline_error = None, error
        chosen = baseline
        stats = {"enabled": fuse_shifts, "eligible_nodes": len(eligible), "fused_nodes": 0,
                 "eligible_kinds": sorted({node.op for node in eligible}),
                 "reason": "disabled" if not fuse_shifts else "no eligible shifts"}
        if fuse_shifts and eligible:
            try:
                candidate = self._build(graph, eligible, level_idc=level_idc,
                                        merge_zero_guards=merge_zero_guards,
                                        prediction_slice_rows=prediction_slice_rows)
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
        # Preserve the original manifest key/compile flag for existing clients.
        self.optimizations = {"spatial_read_fusion": stats, "shift_read_fusion": deepcopy(stats)}
        guard_eligible = sum(self.writer.zero_guard_merge_eligible(op) for op in self.operations)
        guard_reduced = sum(self.writer.zero_guard_merge_eligible(op)
                            and any(source.weight != 0 for source in op.sources) for op in self.operations)
        self.optimizations["zero_guard_merge"] = {
            "enabled": merge_zero_guards, "eligible_pictures": guard_eligible,
            "merged_pictures": guard_reduced if merge_zero_guards else 0}
        self.optimizations["prediction_slice_bands"] = {
            "requested_rows": prediction_slice_rows,
            "eligible_pictures": guard_eligible if merge_zero_guards else 0,
            "banded_pictures": sum(self.writer._prediction_band_rows(op) > 1 for op in self.operations)}

    @staticmethod
    def _field(node, graph, *, level_idc=31):
        """One static quarter-pixel displacement per active output macroblock."""
        if node.op == "shift":
            dx, dy, border = node.params
            if border != "zero":
                raise UnsupportedOperation("h264", node.op, "this writer implements zero borders; clamp synthesis is unimplemented")
            offsets = ((4 * dx, 4 * dy),)
        elif node.op == "sample_h264":
            offsets = (node.params,)
        else:
            offsets, block_size = node.params[:2]
            if block_size != 16:
                raise UnsupportedOperation("h264", node.op, "this adapter supports 16x16 output blocks")
            if node.op == "gather":
                if node.params[2] != "zero":
                    raise UnsupportedOperation("h264", node.op, "this writer implements zero borders; clamp synthesis is unimplemented")
                offsets = tuple((4 * dx, 4 * dy) for dx, dy in offsets)
        try:
            validate = _motion_policy(level_idc)
            for qx, qy in set(offsets):
                validate(Source(0, qx=qx, qy=qy))
        except ValueError as error:
            raise UnsupportedOperation("h264", node.op, str(error)) from error
        if node.op in ("shift", "sample_h264"):
            offsets *= ((graph.width + 15) // 16) * ((graph.height + 15) // 16)
        return offsets

    @staticmethod
    def _prediction(operation, fields):
        """Attach reads to scalar instructions without changing their rounding.

        Every child reads exactly its corresponding parent reference. This
        restricted form lets the existing scheduler see the full dependency
        set through the parent, before child frames are remapped below.
        """
        def moving(source, offset):
            dx, qx = divmod(4 * source.dx + source.qx + offset[0], 4)
            dy, qy = divmod(4 * source.dy + source.qy + offset[1], 4)
            return replace(source, dx=dx, dy=dy, qx=qx, qy=qy)

        if all(field is None or all(pair == field[0] for pair in field) for field in fields):
            return replace(operation, sources=tuple(moving(source, field[0]) if field else source
                           for source, field in zip(operation.sources, fields)))
        count = len(next(field for field in fields if field is not None))
        blocks = tuple(replace(operation, sources=tuple(
            moving(source, field[index]) if field else source
            for source, field in zip(operation.sources, fields))) for index in range(count))
        return replace(operation, block_operations=blocks)

    def _cost(self):
        try:
            size = len(self.tail)
        except ValueError as error:
            raise UnsupportedOperation(self.target, "resources", str(error)) from error
        return {"computation_pictures": len(self.operations), "prediction_bytes": size,
                "output_readback_pictures": len(set(self.output_frames.values()))}

    @classmethod
    def _build(cls, graph, fused, *, level_idc=31, merge_zero_guards=True,
               prediction_slice_rows=1):
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
            if node.op in self.movements:
                operation = self._prediction(Operation(node.op, (Source(values[node.args[0]]),)),
                                             (self._field(node, graph, level_idc=level_idc),))
                instructions.append(legacy.lower.Instruction(next_value, operation, graph_id))
                values[node] = next_value
                next_value += 1
                continue
            bases = tuple(arg.args[0] if arg in fused else arg for arg in node.args)
            fields = tuple(self._field(arg, graph, level_idc=level_idc) if arg in fused else None
                           for arg in node.args)
            if node.op == "average":
                # Keep the complete sum in the decoder's weighted predictor;
                # materializing it as a u8 image would change full-range means.
                refs = tuple(Source(values[base]) for base in bases)
                operation = self._prediction(Operation("rounded average", refs), fields)
                instructions.append(legacy.lower.Instruction(next_value, operation, graph_id))
                values[node] = next_value
                next_value += 1
                continue
            # Select pointwise math using the actual sampled ranges. Motion is
            # carried separately so arbitrary block fields can fold into the
            # same scalar selector as uniform translations.
            proxies = tuple(legacy.Input(f"x{i}", (arg.lo, arg.hi)) for i, arg in enumerate(node.args))
            try:
                expression = self._expression(legacy, node, proxies)
                selected, outputs, *_ = legacy.backend.lower(proxies, {"out": expression}, exact_search=False)
            except legacy.CompileError as error:
                raise UnsupportedOperation(self.target, node.op, str(error)) from error
            registers = {index: (values[arg], field) for index, (arg, field) in enumerate(zip(bases, fields))}
            for instruction in selected:
                refs = tuple(replace(source, frame=registers[source.frame][0]) for source in instruction.operation.sources)
                reads = tuple(registers[source.frame][1] for source in instruction.operation.sources)
                operation = self._prediction(replace(instruction.operation, sources=refs), reads)
                instructions.append(legacy.lower.Instruction(next_value,
                                    operation, graph_id))
                registers[instruction.result] = next_value, None
                next_value += 1
            result, field = registers[outputs["out"]]
            if field is not None and any(qx or qy for qx, qy in field):
                # An identity scalar may alias its operand, but a moving read
                # still needs a picture; otherwise it would return its base.
                operation = self._prediction(Operation("materialize sampled alias", (Source(result),)), (field,))
                instructions.append(legacy.lower.Instruction(next_value, operation, graph_id))
                result, next_value = next_value, next_value + 1
            values[node] = result
        try:
            (self.operations, self.output_frames, self.refreshes, self.peak_live,
             _, self.reference_slots) = legacy.backend.schedule(
                instructions, {name: values[node] for name, node in graph.outputs}, False,
                input_count=len(graph.inputs), retain=True)
            self.operations = [replace(op, block_operations=tuple(replace(block, sources=tuple(
                replace(source, frame=op.sources[index].frame) for index, source in enumerate(block.sources)))
                for block in op.block_operations)) if op.block_operations else op for op in self.operations]
            maximum = max((abs(value) for op in self.operations for block in (op, *op.block_operations)
                           for source in block.sources
                           for value in (4 * source.dx + source.qx, 4 * source.dy + source.qy)), default=0)
            self.writer = Writer(self.width, self.height, reference_slots=self.reference_slots,
                                 slice_mode="row", motion_bits=max(4, maximum.bit_length()), level_idc=level_idc,
                                 merge_zero_guards=merge_zero_guards,
                                 prediction_slice_rows=prediction_slice_rows)
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
    def prediction_units(self):
        return tuple(self.writer.prediction(frame, operation)
                     for frame, operation in enumerate(self.operations, len(self.graph_spec.inputs)))

    @cached_property
    def tail(self):
        return b"".join(self.prediction_units)

    def access_units(self, images):
        """Keep compiler-known picture boundaries for live decoder submission."""
        inputs = tuple(self.writer.input_luma(images[node.params[0]], frame)
                       for frame, node in enumerate(self.graph_spec.inputs))
        return inputs + self.prediction_units

    @cached_property
    def fence_unit(self):
        """One discarded IDR releases the preceding live computation sequence.

        The tested NVDEC path needs a subsequent IDR to release short sequences.
        A separate runtime fence preserves the ordinary saved stream and IR
        costs. It is always counted as additional work by resident execution.
        """
        from decoder_core.fence import neutral_idr
        return neutral_idr(self.writer)

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
                "level_idc": self.writer.level_idc,
                "merge_zero_guards": self.writer.merge_zero_guards,
                "prediction_slice_rows": self.writer.prediction_slice_rows,
                "output_frames": self.output_frames,
                "costs": {"input_pictures": len(self.graph_spec.inputs),
                          "computation_pictures": len(self.operations), "total_pictures": self.pictures,
                          "prediction_bytes": len(self.tail), "peak_live_references": self.peak_live,
                          "prediction_slices": sum(self.writer.slice_count(op) for op in self.operations)},
                "reference_slots": self.reference_slots,
                "optimizations": deepcopy(self.optimizations),
                "scheduling": "H.264 retained long-term references", "guard": 16}
