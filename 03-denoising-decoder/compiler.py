"""Compile frozen weights, never model outputs, into decoder instructions."""

from dataclasses import dataclass

from codec import Operation, Source, Writer


@dataclass
class Program:
    operations: list[Operation]
    features: list[int]
    output: int
    refreshes: list[int]

    @property
    def pictures(self):
        return len(self.operations) + 1


def compile_program(model):
    operations, feature_frames, refreshes = [], [], []
    registers = {"input": 0}

    def emit(target, label, *operands, denominator=0):
        # Refresh before a still-live picture falls out of the sliding DPB.
        # At most input, mix, and the current feature accumulator are live.
        while registers:
            oldest = min(registers.values())
            if len(operations) + 1 - oldest < Writer.MAX_REFS:
                break
            if len(operations) + 1 - oldest > Writer.MAX_REFS:
                raise AssertionError("Scheduler lost a live reference")
            aliases = [key for key, frame in registers.items() if frame == oldest]
            operations.append(Operation("refresh " + ",".join(aliases), (Source(oldest),)))
            for key in aliases:
                registers[key] = len(operations)
            refreshes.append(len(operations))
        sources = tuple(Source(registers[name], weight, offset, dx, dy)
                        for name, weight, offset, dx, dy in operands)
        operations.append(Operation(label, sources, denominator))
        registers[target] = len(operations)
        return len(operations)

    def operand(name, weight=1, offset=0, dx=0, dy=0):
        return name, weight, offset, dx, dy

    # The visible seed is 0/255. All network arithmetic uses exact 0/1 bits.
    emit("input", "normalize noisy input", operand("input", 1, -1), denominator=7)
    channels = len(model.kernels)
    for channel, (kernel, threshold, mix_weight) in enumerate(zip(model.kernels, model.thresholds, model.mix_weights)):
        emit("feature", f"channel {channel}: tap 0, bias 9",
             operand("input", kernel[0], 9, -1, -1))
        for tap, weight in enumerate(kernel[1:], 1):
            offset = 55 - threshold if tap == 8 else 0
            emit("feature", f"channel {channel}: tap {tap}",
                 operand("feature", 2, offset), operand("input", 2 * weight, offset, tap % 3 - 1, tap // 3 - 1))
        feature_frames.append(emit("feature", f"channel {channel}: binary activation",
                                   operand("feature"), denominator=7))
        if channel == 0:
            bias = 64 - model.mix_threshold if channels == 1 else channels
            emit("mix", "1x1: channel 0", operand("feature", mix_weight, bias))
        else:
            offset = 64 - channels - model.mix_threshold if channel == channels - 1 else 0
            emit("mix", f"1x1: channel {channel}", operand("mix", 2, offset), operand("feature", 2 * mix_weight, offset))
        del registers["feature"]
    del registers["input"]
    emit("mix", "1x1: binary output", operand("mix"), denominator=7)
    emit("mix", "display scale x127", operand("mix", 127))
    output = emit("mix", "display scale x3", operand("mix", 3))
    return Program(operations, feature_frames, output, refreshes)


def compile_batch(images, model):
    if not images or not images[0] or not images[0][0]:
        raise ValueError("At least one nonempty image is required")
    height, width = len(images[0]), len(images[0][0])
    writer = Writer(width, height)
    program = compile_program(model)
    tail = b"".join(writer.prediction(frame, op) for frame, op in enumerate(program.operations, 1))
    # Every image receives exactly the same prediction bytes. Only the seed differs.
    stream = b"".join(writer.seed(image) + tail for image in images)
    return stream, program
