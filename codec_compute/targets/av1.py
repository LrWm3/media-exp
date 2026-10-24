"""AV1 lowering preserves rounded averaging and explicit clamp borders."""

from ..ir import UnsupportedOperation
from ..av1 import AV1Step, compile_program


class AV1Program:
    target = "av1"
    suffix = ".ivf"

    def __init__(self, graph):
        self.graph = graph
        ids = {node: f"v{index}" for index, node in enumerate(graph.nodes)}
        self.input_names = {node.params[0]: ids[node] for node in graph.inputs}
        self.output_names = {name: ids[node] for name, node in graph.outputs}
        steps = []
        for node in graph.nodes:
            name, refs = ids[node], tuple(ids[arg] for arg in node.args)
            if node.op == "input":
                continue
            if node.op in ("copy", "average"):
                steps.append(AV1Step(name, node.op, refs))
            elif node.op == "or":
                # For proven 0/1 values rounded average is exactly Boolean OR.
                steps.append(AV1Step(name, "average", refs))
            elif node.op == "shift":
                dx, dy, border = node.params
                if border != "clamp" and (dx or dy):
                    raise UnsupportedOperation(self.target, "shift", "native translation clamps borders; zero-border synthesis is unimplemented")
                steps.append(AV1Step(name, "translate", refs, dx, dy))
            elif node.op in ("affine", "affine_sat") and len(refs) == 1 and node.params == (1, 0):
                steps.append(AV1Step(name, "copy", refs))
            else:
                raise UnsupportedOperation(self.target, node.op,
                    "exact synthesis is unimplemented for this operation; this is not a proof of AV1 impossibility")
        try:
            self.program = compile_program(graph.width, graph.height, tuple(self.input_names.values()), steps)
        except ValueError as error:
            raise UnsupportedOperation(self.target, "parameters/resources", str(error)) from error

    @property
    def tail(self):
        return self.program.compute_tail

    def encode(self, images):
        return self.program.encode({native: images[name] for name, native in self.input_names.items()})

    def decode(self, images, decoder="ffmpeg"):
        decoder = "libdav1d" if decoder == "ffmpeg" else decoder
        result = self.program.decode(self.encode(images), decoder=decoder)
        return {name: result.frames[key] for name, key in self.output_names.items()}

    def manifest(self):
        indices = {name: index for index, name in enumerate(self.program.frame_names)}
        return {"target": self.target, "coded_shape": [self.graph.height, self.graph.width],
                "output_frames": {name: indices[key] for name, key in self.output_names.items()},
                "costs": {"input_pictures": len(self.graph.inputs),
                          "computation_pictures": len(self.program.steps),
                          "total_pictures": len(self.program.frame_names),
                          "prediction_bytes": len(self.tail), "reference_capacity": 8},
                "scheduling": "AV1 reference-slot refresh with last-use allocation",
                "input_loading": "lossless intra blocks from pinned libaom serializer",
                "rounding": "average: floor((a+b+1)/2)", "translation_border": "clamp"}
