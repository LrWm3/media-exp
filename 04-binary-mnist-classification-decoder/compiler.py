"""Compile weights into prediction instructions; never compute classifications."""

from dataclasses import dataclass
from math import ceil

from codec import Operation, Source, Writer


@dataclass
class Program:
    operations: list[Operation]
    pooled: int
    hidden: list[int]
    output: int
    refreshes: list[int]

    @property
    def pictures(self):
        return len(self.operations) + 1


def compile_program(model):
    operations, hidden, refreshes = [], [], []
    registers = {"input": 0}

    def operand(name, weight=1, offset=0, dx=0, dy=0):
        return name, weight, offset, dx, dy

    def emit(target, label, *operands, denominator=0):
        while registers:
            oldest = min(registers.values())
            age = len(operations) + 1 - oldest
            if age < Writer.MAX_REFS:
                break
            if age > Writer.MAX_REFS:
                raise AssertionError("Scheduler lost a live reference")
            aliases = [name for name, frame in registers.items() if frame == oldest]
            operations.append(Operation("refresh " + ",".join(aliases), (Source(oldest),)))
            for name in aliases:
                registers[name] = len(operations)
            refreshes.append(len(operations))
        sources = tuple(Source(registers[name], weight, offset, dx, dy)
                        for name, weight, offset, dx, dy in operands)
        operations.append(Operation(label, sources, denominator))
        registers[target] = len(operations)
        return len(operations)

    emit("input", "normalize 0/255 input to 0/1", operand("input", 1, -1), denominator=7)
    # Two tree sums cover each 2x2 window with exact integer arithmetic.
    for step, (dx, dy) in enumerate(((1, 0), (0, 1))):
        offset = 64 - model.pool_threshold if step == 1 else 0
        emit("input", f"pool sum {step}", operand("input", 2, offset), operand("input", 2, offset, dx, dy))
    pooled_frame = emit("input", "pool binary activation", operand("input"), denominator=7)
    channels = len(model.weights)
    for channel, (weights, threshold, mix) in enumerate(zip(model.weights, model.thresholds, model.mix_weights)):
        negative = weights.count(-1)
        # N + signed_sum lies in [0, 196] for every possible binary input.
        # Split a large initial bias into two legal offsets before accumulation.
        bias = min(negative, 127)
        emit("hidden", f"neuron {channel}: tap 0 bias {bias}", operand("input", weights[0], bias))
        if negative > bias:
            emit("hidden", f"neuron {channel}: remaining bias", operand("hidden", 1, negative-bias))
        cutoff = max(0, min(197, negative + threshold))
        shift = 192 - cutoff
        for tap, weight in enumerate(weights[1:], 1):
            offset = min(127, shift) if tap == 195 else 0
            emit("hidden", f"neuron {channel}: tap {tap}", operand("hidden", 2, offset),
                 operand("input", weight * 2, offset, (tap % 14) * 2, (tap // 14) * 2))
        if shift > 127:
            emit("hidden", f"neuron {channel}: remaining threshold shift", operand("hidden", 1, shift-127))
        hidden.append(emit("hidden", f"neuron {channel}: binary activation", operand("hidden", 1, -1), denominator=7))
        if channel == 0:
            bias = 64 - model.mix_threshold if channels == 1 else channels
            emit("mix", "output: neuron 0", operand("hidden", mix, bias))
        else:
            offset = 64 - channels - model.mix_threshold if channel == channels - 1 else 0
            emit("mix", f"output: neuron {channel}", operand("mix", 2, offset), operand("hidden", 2 * mix, offset))
        del registers["hidden"]
    del registers["input"]
    emit("mix", "classification bit: 0=3 1=8", operand("mix"), denominator=7)
    emit("mix", "display x127", operand("mix", 127))
    output = emit("mix", "display x3", operand("mix", 3))
    return Program(operations, pooled_frame, hidden, output, refreshes)


class CompiledClassifier:
    """Reuse one instruction tail; classify up to eight digits spatially in parallel."""

    def __init__(self, model, capacity=8):
        if not 1 <= capacity <= 8:
            raise ValueError("Tile capacity must be 1..8")
        self.capacity = capacity
        self.columns = min(4, capacity)
        self.rows = ceil(capacity / self.columns)
        self.width, self.height = self.columns * 32, self.rows * 32
        self.writer = Writer(self.width, self.height, motion_bits=7)
        self.program = compile_program(model)
        self.tail = b"".join(self.writer.prediction(frame, op) for frame, op in enumerate(self.program.operations, 1))

    def anchor(self, index):
        return (index % self.columns) * 32, (index // self.columns) * 32

    def board(self, images):
        if not 1 <= len(images) <= self.capacity:
            raise ValueError("Batch must contain 1..capacity images")
        board = [[0] * self.width for _ in range(self.height)]
        for index, image in enumerate(images):
            if len(image) != 28 or any(len(row) != 28 or any(type(v) is not int or v not in (0, 1) for v in row) for row in image):
                raise ValueError("Expected binary 28x28 images")
            left, top = self.anchor(index)
            for y, row in enumerate(image):
                board[top+y][left:left+28] = row
        return board

    def encode(self, images):
        return self.writer.seed(self.board(images)) + self.tail
