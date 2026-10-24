"""AV1 decoder-compute: lossless inputs plus reusable zero-residual instructions.

Only standard-library Python is required. Build the pinned serializer explicitly
with ``python3 -m codec_compute.av1.bootstrap`` before compiling a program.
"""
from __future__ import annotations

from dataclasses import dataclass
import shutil
import struct
import subprocess
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AV1Step:
    output: str
    kind: str
    sources: tuple[str, ...]
    dx: int = 0
    dy: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True)
class AV1Execution:
    stream: bytes
    frames: dict[str, list[list[int]]]

    @property
    def bitstream(self) -> bytes:
        return self.stream


@dataclass(frozen=True)
class AV1Program:
    width: int
    height: int
    input_names: tuple[str, ...]
    steps: tuple[AV1Step, ...]
    frame_names: tuple[str, ...]
    compute_tail: bytes
    _header: bytes
    _instructions: tuple[tuple[int, int, int, int, int, int], ...]

    def encode_inputs(self, inputs: Mapping[str, object]) -> bytes:
        """Serialize only input planes losslessly; never evaluate any operation."""
        if set(inputs) != set(self.input_names):
            raise ValueError(f"Expected exactly these inputs: {self.input_names}")
        planes = tuple(_plane_bytes(inputs[name], self.width, self.height)
                       for name in self.input_names)
        return _serialize(self.width, self.height,
                          self._instructions[:len(self.input_names)], planes)

    def encode(self, inputs: Mapping[str, object]) -> bytes:
        """Append the compile-time instruction bytes to a lossless input prefix."""
        return self._header + self.encode_inputs(inputs)[32:] + self.compute_tail

    def decode(self, stream: bytes, decoder: str = "libdav1d") -> AV1Execution:
        """Extract displayed planes from a real FFmpeg AV1 decoder."""
        if decoder not in {"libdav1d", "libaom-av1"}:
            raise ValueError("Supported decoder choices: libdav1d, libaom-av1")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to decode AV1 programs")
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-xerror", "-threads", "1", "-c:v", decoder,
                 "-f", "ivf", "-i", "pipe:0", "-threads", "1", "-f", "rawvideo",
                 "-pix_fmt", "gray", "pipe:1"], input=stream, capture_output=True,
                timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("AV1 decoder exceeded its 60-second timeout") from exc
        if result.returncode or result.stderr:
            raise RuntimeError(f"AV1 decoder failed: {result.stderr.decode(errors='replace')}")
        area = self.width * self.height
        if len(result.stdout) != len(self.frame_names) * area:
            raise RuntimeError("AV1 decoder returned an unexpected frame count or geometry")
        frames = {}
        for index, name in enumerate(self.frame_names):
            plane = result.stdout[index * area:(index + 1) * area]
            frames[name] = [list(plane[y * self.width:(y + 1) * self.width])
                            for y in range(self.height)]
        return AV1Execution(stream, frames)

    def run(self, inputs: Mapping[str, object], decoder: str = "libdav1d") -> AV1Execution:
        return self.decode(self.encode(inputs), decoder=decoder)


def _plane_bytes(value: object, width: int, height: int) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        plane = bytes(value)
        if len(plane) != width * height:
            raise ValueError(f"Input plane must contain {width * height} bytes")
        return plane
    if not isinstance(value, (list, tuple)) or len(value) != height:
        raise ValueError(f"Input plane must have {height} rows")
    result = bytearray()
    for row in value:
        if not isinstance(row, (list, tuple, bytes, bytearray)) or len(row) != width:
            raise ValueError(f"Every input row must have {width} samples")
        for sample in row:
            if type(sample) is not int or not 0 <= sample <= 255:
                raise ValueError("AV1 inputs require integer samples in [0, 255]")
            result.append(sample)
    return bytes(result)


def _serialize(width: int, height: int,
               instructions: Sequence[tuple[int, int, int, int, int, int]],
               planes: Sequence[bytes]) -> bytes:
    request = bytearray(struct.pack("<III", width, height, len(instructions)))
    input_index = 0
    for instruction in instructions:
        request.extend(struct.pack("<IIIIii", *instruction))
        if instruction[0] == 0:
            request.extend(planes[input_index])
            input_index += 1
    from .bootstrap import bridge_path

    try:
        result = subprocess.run([str(bridge_path())], input=request, capture_output=True,
                                timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("AV1 serializer exceeded its 180-second timeout") from exc
    if result.returncode or result.stderr:
        raise RuntimeError(f"AV1 serializer failed: {result.stderr.decode(errors='replace')}")
    # Validate framing before preserving any native-generated bytes.
    packets = _packets(result.stdout)
    if len(packets) != len(instructions):
        raise RuntimeError("AV1 serializer returned an unexpected packet count")
    return result.stdout


def _packets(stream: bytes) -> tuple[bytes, ...]:
    if len(stream) < 32 or stream[:4] != b"DKIF" or stream[8:12] != b"AV01":
        raise RuntimeError("AV1 serializer returned invalid IVF framing")
    offset = 32
    packets = []
    while offset < len(stream):
        if offset + 12 > len(stream):
            raise RuntimeError("Truncated IVF packet header")
        size = struct.unpack_from("<I", stream, offset)[0]
        end = offset + 12 + size
        if end > len(stream):
            raise RuntimeError("Truncated IVF packet")
        packets.append(stream[offset:end])
        offset = end
    return tuple(packets)


def compile_program(width: int, height: int, input_names: Sequence[str],
                    steps: Sequence[AV1Step]) -> AV1Program:
    """Compile without real input samples, using fixed neutral placeholders.

    Translation reads src[clamp(y + dy), clamp(x + dx)]. Average rounds upward:
    (A + B + 1) // 2. Values are 8-bit monochrome planes. The compiler schedules
    reference lifetimes into eight AV1 slots; decoded outputs remain observable
    even after their reference slots are reused.
    """
    if any(type(n) is not int or n < 16 or n > 4096 or n % 16
           for n in (width, height)):
        raise ValueError("AV1 dimensions must be multiples of 16 in [16, 4096]")
    names, operations = tuple(input_names), tuple(steps)
    if not 1 <= len(names) <= 8:
        raise ValueError("AV1 requires between one and eight input planes")
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Input names must be unique nonempty strings")
    known = set(names)
    last_use: dict[str, int] = {}
    for index, step in enumerate(operations):
        if not isinstance(step, AV1Step):
            raise ValueError("AV1 steps must be AV1Step instances")
        if step.kind not in {"copy", "translate", "average"}:
            raise ValueError(f"Unsupported AV1 operation: {step.kind}")
        if not isinstance(step.output, str) or not step.output or step.output in known:
            raise ValueError("Every output must have a unique nonempty name")
        arity = 2 if step.kind == "average" else 1
        if len(step.sources) != arity or any(source not in known for source in step.sources):
            raise ValueError(f"{step.kind} needs {arity} previously defined source(s)")
        if any(type(n) is not int or not -16 <= n <= 16 for n in (step.dx, step.dy)):
            raise ValueError("AV1 integer translations must be in [-16, 16]")
        if step.kind != "translate" and (step.dx or step.dy):
            raise ValueError("Only translation accepts displacement")
        for source in step.sources:
            last_use[source] = index
        known.add(step.output)
    slots = {name: index for index, name in enumerate(names) if name in last_use}
    instructions = [(0, index, 0, 0, 0, 0) for index in range(len(names))]
    for index, step in enumerate(operations):
        source_slots = tuple(slots[name] for name in step.sources)
        occupied = {slot for name, slot in slots.items() if last_use[name] > index}
        free = set(range(8)) - occupied
        if not free:
            raise ValueError(f"AV1 reference capacity exceeded before {step.output}")
        target = min(free)
        compound = step.kind == "average" and step.sources[0] != step.sources[1]
        instructions.append((2 if compound else 1, target, source_slots[0],
                             source_slots[1] if compound else source_slots[0],
                             step.dx, step.dy))
        slots = {name: slot for name, slot in slots.items() if last_use[name] > index}
        if last_use.get(step.output, -1) > index:
            slots[step.output] = target
    instruction_tuple = tuple(instructions)
    template = _serialize(width, height, instruction_tuple,
                          tuple(bytes(width * height) for _ in names))
    tail = b"".join(_packets(template)[len(names):])
    return AV1Program(width, height, names, operations,
                      names + tuple(step.output for step in operations), tail,
                      template[:32], instruction_tuple)


__all__ = ["AV1Step", "AV1Program", "AV1Execution", "compile_program"]
