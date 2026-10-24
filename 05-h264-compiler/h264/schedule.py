"""Track last uses and refresh live pictures before the 16-picture DPB evicts them."""

from dataclasses import replace

from decoder_core.bitstream import Operation, Source, Writer
from .graph import CompileError


def schedule(instructions, outputs, binary_input, input_count=1):
    operations = [Operation("normalize binary seed", (Source(0, 1, -1),), 7)] if binary_input else []
    frame_offset = input_count-1
    registers = {index: index for index in range(input_count)}
    if binary_input:
        registers[0] = 1
    definitions = dict(registers)
    last_use = {}
    for index, instruction in enumerate(instructions):
        for ref in instruction.operation.sources:
            last_use[ref.frame] = index
    refreshes, peak_live, graph_frames = [], 0, {}
    for index, instruction in enumerate(instructions):
        registers = {value: frame for value, frame in registers.items() if last_use.get(value, -1) >= index}
        peak_live = max(peak_live, len(registers))
        if len(registers) >= Writer.MAX_REFS:
            raise CompileError("Graph exceeds 15 simultaneously live values; split the kernel or reduce shared intermediates")
        while registers and frame_offset+len(operations)+1-min(registers.values()) >= Writer.MAX_REFS:
            oldest = min(registers.values())
            if frame_offset+len(operations)+1-oldest > Writer.MAX_REFS:
                raise CompileError("Scheduler lost a live reference")
            operations.append(Operation("refresh live value", (Source(oldest),)))
            for value in registers:
                if registers[value] == oldest:
                    registers[value] = frame_offset+len(operations)
            refreshes.append(frame_offset+len(operations))
        op = instruction.operation
        refs = tuple(replace(s, frame=registers[s.frame]) for s in op.sources)
        operations.append(replace(op, sources=refs))
        definitions[instruction.result] = registers[instruction.result] = frame_offset+len(operations)
        graph_frames[instruction.graph_id] = frame_offset+len(operations)
    return operations, {name: definitions[value] for name, value in outputs.items()}, refreshes, peak_live, graph_frames
