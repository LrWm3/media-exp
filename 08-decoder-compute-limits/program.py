"""Emit literal atlas witnesses without frontend rewriting or host inference.

Each pixel is a Boolean lane represented by literal luma 0 or 1. P and B
prediction instructions retain the atlas's signed weights and denominator.
Packing schedules independent instructions into two guarded image rectangles.
"""

from dataclasses import asdict, replace
from functools import cached_property
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decoder_core.bitstream import Operation, Source, Writer


RECURRENCE_NOTE = (
    "Rounds execute the single-evaluation witness repeatedly with original b retained. "
    "Composition can simplify algebraically; for example, an even number of XOR rounds "
    "is the identity on a. Executed instruction and picture counts are not minimum "
    "costs for the composed function."
)


def _slots(operations, input_count):
    """Replace a source only after its final read, including the current frame."""
    last = list(range(input_count + len(operations)))
    for frame, operation in enumerate(operations, input_count):
        for child in (operation,) + operation.block_operations:
            for source in child.sources:
                last[source.frame] = max(last[source.frame], frame)
    occupants, slots = {}, {}
    for frame in range(len(last)):
        free = [slot for slot in range(Writer.MAX_REFS)
                if slot not in occupants or last[occupants[slot]] <= frame]
        if not free:
            raise ValueError("Witness exceeds 16 simultaneously retained pictures")
        slot = min(free)
        slots[frame] = slot
        occupants[slot] = frame
    return slots


class Program:
    """An exact witness repeated as a dependent a <- f(a, original_b) kernel."""

    input_count = 2

    def __init__(self, case, width=32, height=32, packed=False, rounds=1):
        if any(type(value) is not int or value < 16 or value % 16
               for value in (width, height)):
            raise ValueError("Logical dimensions must be positive multiples of 16")
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if type(packed) is not bool:
            raise ValueError("packed must be Boolean")
        self.case, self.width, self.height = case, width, height
        self.packed, self.rounds = packed, rounds
        plan = case["plan"]
        if (len(case["truth"]) != 4 or any(value not in (0, 1) for value in case["truth"])
                or type(case["output_index"]) is not int
                or not 0 <= case["output_index"] < len(plan) + 2):
            raise ValueError("Invalid Boolean truth table or output index")
        for index, step in enumerate(plan, 2):
            if (len(step["inputs"]) not in (1, 2)
                    or len(step["inputs"]) != len(step["weights"])
                    or any(type(source) is not int or not 0 <= source < index
                           for source in step["inputs"])):
                raise ValueError("Witness inputs must precede their instruction")
            Writer._validate_operation(Operation("validate", tuple(
                Source(source, weight, step["bias"])
                for source, weight in zip(step["inputs"], step["weights"])), step["denominator"]))
        # Determine capacity from this witness; packing a one-step function
        # must not add empty tiles and inflate its coded geometry.
        capacity = 1
        if packed:
            known, pending = {0, 1}, dict(enumerate(plan, 2))
            while pending:
                ready = [index for index, step in pending.items()
                         if all(source in known for source in step["inputs"])]
                arity = len(pending[ready[0]]["inputs"])
                group = [index for index in ready if len(pending[index]["inputs"]) == arity][:2]
                capacity = max(capacity, len(group))
                known.update(group)
                for index in group:
                    del pending[index]
        self.tiles = tuple({"tile": tile, "x": tile * (width + 16), "y": 0,
                            "width": width, "height": height} for tile in range(capacity))
        self.canvas_width, self.canvas_height = capacity * width + (capacity - 1) * 16, height
        self.operations, self.steps = [], []
        self.frame_tiles = {0: (0,), 1: (0,)}
        current = (0, 0)
        for round_index in range(rounds):
            locations = {0: current, 1: (1, 0)}
            pending = dict(enumerate(plan, 2))
            while pending:
                ready = [index for index, step in pending.items()
                         if all(source in locations for source in step["inputs"])]
                arity = len(pending[ready[0]]["inputs"])
                group = [index for index in ready if len(pending[index]["inputs"]) == arity][:capacity]
                frame = self.input_count + len(self.operations)
                children = []
                for tile, index in enumerate(group):
                    step, target = pending[index], self.tiles[tile]
                    sources = []
                    for source_index, weight in zip(step["inputs"], step["weights"]):
                        source_frame, source_tile = locations[source_index]
                        origin = self.tiles[source_tile]
                        sources.append(Source(source_frame, weight, step["bias"],
                                              origin["x"] - target["x"], origin["y"] - target["y"]))
                    child = Operation("round %d step %d" % (round_index + 1, index - 1),
                                      tuple(sources), step["denominator"])
                    children.append(child)
                    self.steps.append({"round": round_index + 1, "index": index,
                                       "frame": frame, "tile": tile, **asdict(child)})
                if capacity == 1:
                    operation = children[0]
                else:
                    zero = Operation("zero padding", tuple(Source(source.frame, weight=0)
                                                           for source in children[0].sources))
                    blocks = [zero] * ((self.canvas_width // 16) * (height // 16))
                    for tile, child in enumerate(children):
                        left = self.tiles[tile]["x"] // 16
                        for y in range(height // 16):
                            for x in range(left, left + width // 16):
                                blocks[y * (self.canvas_width // 16) + x] = child
                    operation = replace(zero, label="; ".join(child.label for child in children),
                                        block_operations=tuple(blocks))
                self.operations.append(operation)
                self.frame_tiles[frame] = tuple(range(len(group)))
                # Publish only after all inputs in this picture were resolved.
                for tile, index in enumerate(group):
                    locations[index] = (frame, tile)
                    del pending[index]
            current = locations[case["output_index"]]
        self.operations = tuple(self.operations)
        self.output_regions = {"output": {"frame": current[0], **self.tiles[current[1]]}}
        self.output_frames = {"output": current[0]}
        self.output_picture_indices = (current[0],)
        self.reference_slots = _slots(self.operations, self.input_count)
        maximum = max((abs(value) for op in self.operations for child in (op,) + op.block_operations
                       for source in child.sources for value in (4 * source.dx, 4 * source.dy)), default=0)
        self.writer = Writer(self.canvas_width, self.canvas_height,
                             motion_bits=max(4, maximum.bit_length()),
                             reference_slots=self.reference_slots, slice_mode="row")
        if any(self.writer.slice_count(op) * self.writer.fps > 1800 for op in self.operations):
            raise ValueError("Witness exceeds level 3.1 slice rate")
        self.writer.seed_luma([[0] * self.canvas_width for _ in range(self.canvas_height)])

    @property
    def pictures(self):
        return self.input_count + len(self.operations)

    @property
    def single_evaluation_minima(self):
        return {"scalar_instructions": self.case["scalar_minimum"],
                "packed_pictures": self.case["packed_minimum"]}

    @cached_property
    def tail(self):
        return b"".join(self.writer.prediction(frame, operation)
                        for frame, operation in enumerate(self.operations, self.input_count))

    @cached_property
    def costs(self):
        pixels = self.writer.width * self.writer.height
        return {"input_pictures": self.input_count, "total_pictures": self.pictures,
                "prediction_pictures": len(self.operations), "logical_instructions": len(self.steps),
                "prediction_slices": sum(self.writer.slice_count(op) for op in self.operations),
                "prediction_bytes": len(self.tail), "coded_width": self.writer.width,
                "coded_height": self.writer.height, "coded_luma_samples": pixels * self.pictures,
                "coded_macroblocks": pixels // 256 * self.pictures,
                "full_readback_bytes": pixels * self.pictures * 3 // 2,
                "output_readback_bytes": pixels * 3 // 2,
                "tile_capacity": len(self.tiles), "reference_slots_used": len(set(self.reference_slots.values())),
                "computational_evaluations": self.rounds * self.width * self.height}

    def validate(self, inputs):
        if not isinstance(inputs, dict) or set(inputs) != {"a", "b"}:
            raise ValueError("Inputs must contain exactly a and b")
        for name, board in inputs.items():
            if (len(board) != self.height or any(len(row) != self.width for row in board)
                    or any(type(value) is not int or value not in (0, 1) for row in board for value in row)):
                raise ValueError("Input %s must match logical dimensions with literal Boolean pixels" % name)
        return inputs

    def encode_inputs(self, inputs):
        inputs = self.validate(inputs)
        pieces = []
        for frame, name in enumerate(("a", "b")):
            canvas = [[0] * self.canvas_width for _ in range(self.canvas_height)]
            for y, row in enumerate(inputs[name]):
                canvas[y][:self.width] = row
            pieces.append(self.writer.input_luma(canvas, frame))
        return b"".join(pieces)

    def encode(self, inputs):
        return self.encode_inputs(inputs) + self.tail

    def reference(self, inputs):
        """Independent oracle: direct truth-table lookup, never plan evaluation."""
        inputs = self.validate(inputs)
        current = [list(row) for row in inputs["a"]]
        truth, original_b = self.case["truth"], inputs["b"]
        for _ in range(self.rounds):
            current = [[truth[2 * a + b] for a, b in zip(row_a, row_b)]
                       for row_a, row_b in zip(current, original_b)]
        return {"output": current}

    def manifest(self):
        return {"format": "h264-compute-limits-program-0.1", "function": self.case["name"],
                "mask": self.case["id"], "truth": self.case["truth"],
                "mode": "packed" if self.packed else "scalar", "width": self.width, "height": self.height,
                "coded_width": self.writer.width, "coded_height": self.writer.height,
                "canvas_width": self.canvas_width, "canvas_height": self.canvas_height,
                "guard": 16, "tile_gap": 16, "tiles": list(self.tiles),
                "frame_tiles": {str(frame): list(tiles) for frame, tiles in self.frame_tiles.items()},
                "pictures": self.pictures, "input_count": self.input_count, "input_frames": {"a": 0, "b": 1},
                "rounds": self.rounds, "round_count": self.rounds, "compute_pictures": len(self.operations),
                "computational_evaluations": self.rounds * self.width * self.height,
                "single_evaluation_minima": self.single_evaluation_minima,
                "cost_scope": "executed witness counts across all rounds",
                "recurrence_note": RECURRENCE_NOTE,
                "output_frames": self.output_frames, "output_regions": self.output_regions,
                "reference_slots": self.reference_slots, "reference_mode": "retained", "slice_mode": "row",
                "prediction_slices": self.costs["prediction_slices"], "prediction_bytes": len(self.tail),
                "steps": self.steps, "costs": self.costs,
                "instructions": [{"frame": frame, **asdict(op)} for frame, op in
                                 enumerate(self.operations, self.input_count)]}

    def read(self, raw, *, selected_frames=None):
        selection = tuple(range(self.pictures)) if selected_frames is None else tuple(selected_frames)
        if (not selection or any(type(frame) is not int or not 0 <= frame < self.pictures for frame in selection)
                or any(a >= b for a, b in zip(selection, selection[1:]))):
            raise ValueError("selected_frames must be increasing unique integer picture indices")
        positions = {frame: index for index, frame in enumerate(selection)}
        if not set(self.output_frames.values()).issubset(positions):
            raise ValueError("selected_frames must cover every named output")
        width, height = self.writer.width, self.writer.height
        pixels, stride = width * height, width * height * 3 // 2
        if len(raw) != stride * len(selection):
            raise AssertionError("Decoded picture count does not match manifest")
        view, neutral = memoryview(raw), bytes([128]) * (pixels // 2)
        for position, frame in enumerate(selection):
            start = position * stride
            if view[start + pixels:start + stride] != neutral:
                raise AssertionError("Chroma changed at frame %d" % frame)
            active = [self.tiles[tile] for tile in self.frame_tiles[frame]]
            for y in range(height):
                spans = sorted((tile["x"] + 16, tile["x"] + 16 + self.width) for tile in active
                               if tile["y"] + 16 <= y < tile["y"] + 16 + self.height)
                end = 0
                for left, right in spans + [(width, width)]:
                    if any(view[start + y * width + end:start + y * width + left]):
                        raise AssertionError("Guard or unused tile changed at frame %d" % frame)
                    end = right
        outputs = {}
        for name, region in self.output_regions.items():
            start = positions[region["frame"]] * stride
            x, y = region["x"] + 16, region["y"] + 16
            board = [list(view[start + (y + row) * width + x:start + (y + row) * width + x + self.width])
                     for row in range(self.height)]
            if any(value not in (0, 1) for row in board for value in row):
                raise AssertionError("Output left the literal Boolean domain")
            outputs[name] = board
        return outputs
