"""Lower exact image semantics to the independently implemented HEVC subset."""

from ..ir import UnsupportedOperation
from ..hevc import NativeProgram, NativeStep


class HEVCProgram:
    target = "hevc"
    suffix = ".hevc"

    def __init__(self, graph):
        self.graph = graph
        ids = {node: f"v{index}" for index, node in enumerate(graph.nodes)}
        self.input_names = {node.params[0]: ids[node] for node in graph.inputs}
        self.output_names = {name: ids[node] for name, node in graph.outputs}
        steps = []

        def emit(name, args, weights, bias=0, denominator=0):
            try:
                steps.append(NativeStep(name, tuple(args), tuple(weights), bias, denominator))
            except ValueError as error:
                raise UnsupportedOperation(self.target, "prediction", str(error)) from error

        for index, node in enumerate(graph.nodes):
            name, refs = ids[node], tuple(ids[arg] for arg in node.args)
            if node.op == "input":
                continue
            if node.op == "copy":
                emit(name, refs, (1,))
            elif node.op in ("affine", "affine_sat"):
                if len(refs) > 2:
                    raise UnsupportedOperation(self.target, node.op, "multi-source affine synthesis is not implemented")
                weights, bias = node.params[:-1], node.params[-1]
                if not -128 <= bias <= 127:
                    raise UnsupportedOperation(self.target, node.op, "offset splitting is not implemented; native offset must be [-128,127]")
                emit(name, refs, tuple(w * (2 if len(refs) == 2 else 1) for w in weights), bias)
            elif node.op in ("and", "or"):
                emit(name, refs, (2, 2) if node.op == "and" else (1, 1), -1 if node.op == "and" else 0)
            elif node.op == "average":
                emit(name, refs, (1, 1))
            elif node.op == "threshold":
                source, cutoff = refs[0], node.params[0]
                operand = node.args[0]
                if cutoff <= operand.lo or cutoff > operand.hi:
                    emit(name, (source,), (0,), int(cutoff <= operand.lo))
                else:
                    # Move the threshold to 192 through same-sign clipped
                    # offsets, then clip(((x+64)>>7)-1). All bytes are exact.
                    remaining, part_index = 192 - cutoff, 0
                    while remaining:
                        part = max(-128, min(127, remaining))
                        temporary = f"threshold_{index}_{part_index}"
                        emit(temporary, (source,), (1,), part)
                        source, remaining = temporary, remaining - part
                        part_index += 1
                    emit(name, (source,), (1,), -1, 7)
            else:
                raise UnsupportedOperation(self.target, node.op, "no exact lowering implemented by this HEVC subset")
        # NativeProgram deliberately requires a prediction: an input-only IR
        # still uploads once and executes a copy, whose value can be discarded.
        if not steps:
            first = next(iter(self.input_names.values()))
            emit("identity_observation", (first,), (1,))
        try:
            self.program = NativeProgram(graph.width, graph.height, tuple(self.input_names.values()), tuple(steps))
        except ValueError as error:
            raise UnsupportedOperation(self.target, "parameters/resources", str(error)) from error

    @property
    def tail(self):
        return self.program.tail

    def _inputs(self, images):
        return {native: images[name] for name, native in self.input_names.items()}

    def encode(self, images):
        return self.program.assemble(self._inputs(images))

    def decode(self, images, decoder="ffmpeg"):
        if decoder != "ffmpeg":
            raise ValueError("HEVC adapter currently validates only decoder='ffmpeg'")
        frames = self.program.decode(self._inputs(images))
        width, height = self.graph.width, self.graph.height
        return {name: [list(frames[key][y * width:(y + 1) * width]) for y in range(height)]
                for name, key in self.output_names.items()}

    def manifest(self):
        return {"target": self.target, "coded_shape": [self.graph.height, self.graph.width],
                "output_frames": {name: self.program.output_frame_indices[key] for name, key in self.output_names.items()},
                "picture_order_counts": dict(self.program.picture_order_counts),
                "costs": {"input_pictures": len(self.graph.inputs),
                          "computation_pictures": len(self.program.steps),
                          "total_pictures": len(self.graph.inputs) + len(self.program.steps),
                          "prediction_bytes": len(self.tail),
                          "peak_live_references": self.program.max_live_references},
                "scheduling": "HEVC short-term reference sets and explicit lists",
                "input_loading": "lossless libx265; computation uses fixed zero-residual CABAC templates"}
