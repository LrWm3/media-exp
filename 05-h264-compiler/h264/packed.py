"""Schedule independent integer predictions in separated rectangles of a picture.

Packing changes storage and instruction scheduling only. Input pictures contain
the user's pixels; every arithmetic value still comes from H.264 prediction.
"""

from dataclasses import asdict, replace
from functools import cached_property
from math import isqrt

from decoder_core.bitstream import Operation, Source, Writer
from .backend import Program
from .graph import CompileError


def _slots(operations, input_count):
    """Allocate complete pictures; a last-use picture may be replaced in place."""
    count = input_count + len(operations)
    last = list(range(count))
    for frame, operation in enumerate(operations, input_count):
        for child in (operation,) + operation.block_operations:
            for source in child.sources:
                last[source.frame] = max(last[source.frame], frame)
    occupants, slots = {}, {}
    peak = 0
    for frame in range(count):
        free = [slot for slot in range(Writer.MAX_REFS)
                if slot not in occupants or last[occupants[slot]] <= frame]
        if not free:
            raise CompileError("Packed schedule exceeds 16 reference pictures; use fewer tiles")
        slot = min(free)
        slots[frame] = slot
        occupants[slot] = frame
        peak = max(peak, sum(last[value] > frame for value in occupants.values()))
    return slots, peak


class PackedProgram(Program):
    """A Program with several independently computed logical images per picture.

    A tile has a 16-pixel zero halo. Version 0.1 supports uniform integer motion
    up to that distance, and rejects fractional reads and per-block gathers.
    Named outputs retain their logical dimensions and independent ranges.
    """

    def __init__(self, original, tiles=8):
        if not isinstance(original, Program) or isinstance(original, PackedProgram):
            raise CompileError("pack expects an unpacked compiled Program")
        if type(tiles) is not int or not 1 <= tiles <= 16:
            raise CompileError("tiles must be an integer from 1 through 16")
        for op in original.operations:
            if op.block_offsets or op.block_operations:
                raise CompileError("Packing does not yet support per-block motion or predictions")
            if any(value % 4 or abs(value) > 64 for ref in op.sources
                   for value in (4*ref.dx+ref.qx, 4*ref.dy+ref.qy)):
                raise CompileError("Packing requires integer motion within its 16-pixel halo")
        self.original = original
        for name in ("width", "height", "named_inputs", "input_range", "binary_input",
                     "input_ranges", "input_count", "graph", "graph_outputs"):
            setattr(self, name, getattr(original, name))
        self.reference_mode, self.slice_mode = "retained", "row"
        # Choose an exact factorization, avoiding extra unused rectangles. For
        # square logical images, eight tiles form four columns and two rows.
        rows = max(divisor for divisor in range(1, isqrt(tiles)+1) if tiles % divisor == 0)
        columns = tiles // rows
        self.tiles = tuple({"tile": index, "x": (index % columns)*(self.width+16),
                            "y": (index // columns)*(self.height+16),
                            "width": self.width, "height": self.height}
                           for index in range(tiles))
        self.canvas_width = columns*self.width + (columns-1)*16
        self.canvas_height = rows*self.height + (rows-1)*16
        self.frame_tiles = {frame: (0,) for frame in range(self.input_count)}
        locations = {frame: (frame, 0) for frame in range(self.input_count)}
        pending = dict(enumerate(original.operations, self.input_count))
        self.operations, self.groups = [], []
        while pending:
            ready = [frame for frame, op in pending.items()
                     if all(source.frame in locations for source in op.sources)]
            if not ready:
                raise CompileError("Prediction graph has an unresolved dependency")
            arity = len(pending[ready[0]].sources)
            group = [frame for frame in ready if len(pending[frame].sources) == arity][:tiles]
            physical = self.input_count + len(self.operations)
            children = []
            # Source locations are looked up before publishing any result in
            # this group: a picture cannot read its own unfinished pixels.
            for tile, frame in enumerate(group):
                op, target = pending[frame], self.tiles[tile]
                sources = []
                for source in op.sources:
                    source_frame, source_tile = locations[source.frame]
                    origin = self.tiles[source_tile]
                    sources.append(replace(source, frame=source_frame,
                                           dx=source.dx+source.qx//4+origin["x"]-target["x"],
                                           dy=source.dy+source.qy//4+origin["y"]-target["y"],
                                           qx=0, qy=0))
                children.append(replace(op, sources=tuple(sources)))
            zero = Operation("zero padding", tuple(Source(source.frame, weight=0)
                                                   for source in children[0].sources))
            blocks = [zero] * ((self.canvas_width//16)*(self.canvas_height//16))
            for tile, child in enumerate(children):
                rect = self.tiles[tile]
                for y in range(rect["y"]//16, (rect["y"]+self.height)//16):
                    for x in range(rect["x"]//16, (rect["x"]+self.width)//16):
                        blocks[y*(self.canvas_width//16)+x] = child
            operation = replace(zero, label="packed: "+", ".join(child.label for child in children),
                                block_operations=tuple(blocks))
            self.operations.append(operation)
            self.groups.append(tuple(group))
            self.frame_tiles[physical] = tuple(range(len(group)))
            for tile, frame in enumerate(group):
                locations[frame] = (physical, tile)
                del pending[frame]
        self.locations = locations
        self.output_regions = {}
        for name, old_frame in original.output_frames.items():
            frame, tile = locations[old_frame]
            self.output_regions[name] = {"frame": frame, **self.tiles[tile]}
        self.output_frames = {name: region["frame"] for name, region in self.output_regions.items()}
        self.reference_slots, peak = _slots(self.operations, self.input_count)
        maximum = max((abs(value) for op in self.operations for child in op.block_operations
                       for source in child.sources for value in (4*source.dx, 4*source.dy)), default=0)
        try:
            self.writer = Writer(self.canvas_width, self.canvas_height,
                                 motion_bits=max(4, maximum.bit_length()),
                                 reference_slots=self.reference_slots, slice_mode="row")
            # Check actual slice rates before returning a compiled program.
            if any(self.writer.slice_count(op)*self.writer.fps > 1800 for op in self.operations):
                raise ValueError("Packed slices exceed level 3.1 slice rate; use fewer tiles")
            # An all-zero luma canvas maximizes PCM emulation-prevention bytes
            # (each block has fixed header and neutral chroma). Conservatively
            # prove that every input fits the writer's initial access-unit bound.
            self.writer.seed_luma([[0]*self.canvas_width for _ in range(self.canvas_height)])
        except ValueError as error:
            raise CompileError(str(error)) from error
        self.stats = {"original_pictures": original.pictures, "tile_capacity": tiles,
                      "peak_live_values": peak, "scheduled_operations": len(original.operations),
                      "packed_prediction_pictures": len(self.operations)}

    @cached_property
    def costs(self):
        pixels = self.writer.width*self.writer.height
        return {"input_pictures": self.input_count, "total_pictures": self.pictures,
                "prediction_pictures": len(self.operations),
                "prediction_slices": self.tail.count(b"\x00\x00\x00\x01"),
                "prediction_bytes": len(self.tail), "coded_width": self.writer.width,
                "coded_height": self.writer.height, "coded_luma_samples": pixels*self.pictures,
                "coded_macroblocks": (pixels//256)*self.pictures,
                "full_readback_bytes": pixels*self.pictures*3//2,
                "output_readback_bytes": pixels*len(self.output_picture_indices)*3//2,
                "tile_capacity": len(self.tiles), "peak_live_values": self.stats["peak_live_values"],
                "reference_slots_used": len(set(self.reference_slots.values()))}

    def encode_inputs(self, image):
        images = self.validate(image)
        streams = []
        for frame, name in enumerate(self.input_ranges):
            canvas = [[0]*self.canvas_width for _ in range(self.canvas_height)]
            for y, row in enumerate(images[name]):
                canvas[y][:self.width] = row
            streams.append(self.writer.seed(canvas) if self.binary_input
                           else self.writer.input_luma(canvas, frame))
        return b"".join(streams)

    def manifest(self):
        return {"format": "h264-packed-python-0.1", "width": self.width, "height": self.height,
                "canvas_width": self.canvas_width, "canvas_height": self.canvas_height,
                "coded_width": self.writer.width, "coded_height": self.writer.height,
                "guard": 16, "tile_gap": 16, "tiles": list(self.tiles),
                "frame_tiles": {str(frame): list(tiles) for frame, tiles in self.frame_tiles.items()},
                "pictures": self.pictures, "output_frames": self.output_frames,
                "output_regions": self.output_regions,
                "inputs": [{"name": name, "range": list(bounds), "frame": frame, "tile": 0,
                            "encoding": "binary 0/255" if self.binary_input else "literal u8"}
                           for frame, (name, bounds) in enumerate(self.input_ranges.items())],
                "reference_mode": self.reference_mode, "slice_mode": self.slice_mode,
                "reference_slots": self.reference_slots, "stats": self.stats, "costs": self.costs,
                "original_frame_locations": {str(frame): {"frame": location[0], "tile": location[1]}
                                             for frame, location in self.locations.items()},
                "instructions": [{"frame": frame, "original_frames": list(group), **asdict(op)}
                                 for frame, group, op in zip(range(self.input_count, self.pictures),
                                                             self.groups, self.operations)],
                "original": self.original.manifest()}

    def read(self, raw, *, selected_frames=None):
        selection = tuple(range(self.pictures)) if selected_frames is None else tuple(selected_frames)
        if (not selection or any(type(frame) is not int or not 0 <= frame < self.pictures for frame in selection)
                or any(a >= b for a, b in zip(selection, selection[1:]))):
            raise ValueError("selected_frames must be increasing unique integer picture indices")
        positions = {frame: index for index, frame in enumerate(selection)}
        if not set(self.output_frames.values()).issubset(positions):
            raise ValueError("selected_frames must cover every named output")
        width, height = self.writer.width, self.writer.height
        pixels, stride = width*height, width*height*3//2
        if len(raw) != stride*len(selection):
            raise AssertionError("Decoded picture count does not match manifest")
        view = memoryview(raw)
        neutral = bytes([128])*(pixels//2)
        for index, frame in enumerate(selection):
            start = index*stride
            if view[start+pixels:start+stride] != neutral:
                raise AssertionError(f"Chroma changed at frame {frame}")
            active = [self.tiles[tile] for tile in self.frame_tiles[frame]]
            for y in range(height):
                spans = sorted((tile["x"]+16, tile["x"]+16+self.width) for tile in active
                               if tile["y"]+16 <= y < tile["y"]+16+self.height)
                end = 0
                for left, right in spans + [(width, width)]:
                    if any(view[start+y*width+end:start+y*width+left]):
                        raise AssertionError(f"Guard or unused tile changed at frame {frame}")
                    end = right
        outputs = {}
        for name, region in self.output_regions.items():
            start = positions[region["frame"]]*stride
            x, y = region["x"]+16, region["y"]+16
            board = [list(view[start+(y+row)*width+x:start+(y+row)*width+x+self.width])
                     for row in range(self.height)]
            node = self.graph[self.graph_outputs[name]]
            if any(not node.lo <= value <= node.hi for row in board for value in row):
                raise AssertionError(f"Decoded output {name} violates its proven range")
            outputs[name] = board
        return outputs


def pack(program, tiles=8):
    """Pack independent predictions; fewer pictures can still mean more work."""
    return PackedProgram(program, tiles)
