"""Named HEVC instructions with resident references and no host intermediates."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .native import (
    HEVCError, _Template, _bits, _dimensions, _encode, _from_bits, _nal_type,
    _nals, _plane, _rbsp, _template, _ue, _weight_bits, _weights, _wrap,
    _upload_nals, decode_frames,
)


@dataclass(frozen=True)
class NativeStep:
    """One unary or binary weighted prediction; semantics match ``Kernel``."""
    name: str
    inputs: tuple[str, ...]
    weights: tuple[int, ...]
    offset: int = 0
    denominator: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "weights", tuple(self.weights))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("HEVC instruction names must be nonempty strings")
        if len(self.inputs) not in (1, 2) or len(self.inputs) != len(self.weights):
            raise ValueError("HEVC instructions need one or two matching inputs and weights")
        if any(not isinstance(name, str) or not name for name in self.inputs):
            raise ValueError("HEVC instruction input names must be nonempty strings")
        _weights(self.weights, self.offset, self.denominator)


def _instruction(template: _Template, step: NativeStep, poc: int,
                 inputs: tuple[int, ...], retained: list[int]) -> bytes:
    # All reference pictures precede this one. Short-term RPS entries retain
    # live values, while used_by_curr_pic flags identify the active operands.
    retained = sorted(retained, reverse=True)
    active = [reference for reference in retained if reference in inputs]
    binary = len(inputs) == 2
    bits = "1" + _ue(0) + _ue(0 if binary else 1) + f"{poc:08b}"
    bits += "0" + _ue(len(retained)) + _ue(0)
    previous = poc
    for reference in retained:
        bits += _ue(previous - reference - 1) + str(int(reference in active))
        previous = reference
    bits += "0"  # num_ref_idx_active_override_flag: one active ref per list.
    if len(active) > 1:
        entry_bits = (len(active) - 1).bit_length()
        for reference in inputs:
            bits += "1" + format(active.index(reference), f"0{entry_bits}b")
    if binary:
        bits += "0"  # mvd_l1_zero_flag; every prediction is a zero-motion merge.
    bits += _weight_bits(step.weights, step.offset, step.denominator)
    bits += template.bits[template.weight_end:template.alignment] + "1"
    bits += "0" * (-len(bits) % 8)
    # TRAIL_R, temporal_id=0: subsequent computations may reference this frame.
    bits = _bits(b"\x02\x01") + bits + template.bits[template.cabac_start:]
    return _wrap(_from_bits(bits))


@dataclass(frozen=True)
class NativeProgram:
    """Compile a bounded DAG to resident HEVC P/B prediction instructions.

    One or two u8 luma inputs, at most 128 steps, and at most three retained
    references per step. Liveness analysis releases references after their last
    use. All inputs and results appear once in decoder display order. Decoder
    output includes unused intermediate results, which need not stay resident.
    ``tail`` is independent of input pixels and is never rewritten by assemble.
    """
    width: int
    height: int
    input_names: tuple[str, ...]
    steps: tuple[NativeStep, ...]
    tail: bytes = field(init=False, repr=False)
    parameter_sets: tuple[bytes, ...] = field(init=False, repr=False)
    output_frame_indices: Mapping[str, int] = field(init=False)
    picture_order_counts: Mapping[str, int] = field(init=False)
    max_live_references: int = field(init=False)
    _upload_parameter_sets: tuple[bytes, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _dimensions(self.width, self.height)
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "steps", tuple(self.steps))
        if len(self.input_names) not in (1, 2):
            raise ValueError("HEVC resident programs support one or two uploaded inputs")
        if any(not isinstance(name, str) or not name for name in self.input_names):
            raise ValueError("HEVC input names must be nonempty strings")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("HEVC input names must be unique")
        if not 1 <= len(self.steps) <= 128:
            raise ValueError("HEVC resident programs require 1 to 128 steps")
        known = set(self.input_names)
        last_use: dict[str, int] = {}
        for index, step in enumerate(self.steps):
            if not isinstance(step, NativeStep):
                raise ValueError("HEVC program steps must be NativeStep instances")
            if step.name in known or any(name not in known for name in step.inputs):
                raise ValueError("HEVC steps require unique names and previously defined inputs")
            for name in step.inputs:
                last_use[name] = index
            known.add(step.name)

        pocs = dict(zip(self.input_names, (0, 2)))
        frame_indices = {name: index for index, name in enumerate(self.input_names)}
        next_poc = 3 if len(self.input_names) == 2 else 1
        schedule = []
        max_live = 0
        for index, step in enumerate(self.steps):
            live = [poc for name, poc in pocs.items() if last_use.get(name, -1) >= index]
            max_live = max(max_live, len(live))
            if len(live) > 3:
                raise ValueError("HEVC program exceeds three simultaneously retained references")
            schedule.append((step, next_poc, tuple(pocs[name] for name in step.inputs), live))
            pocs[step.name] = next_poc
            frame_indices[step.name] = len(self.input_names) + index
            next_poc += 1

        binary_template = _template(self.width, self.height, True)
        unary_template = (_template(self.width, self.height, False)
                          if any(len(step.inputs) == 1 for step in self.steps) else None)
        if unary_template and unary_template.parameter_sets[-1] != binary_template.parameter_sets[-1]:
            raise HEVCError("libx265 P/B templates have incompatible picture parameter sets")
        headers = list(binary_template.parameter_sets)
        for index, nal in enumerate(headers):
            if _nal_type(nal) == 34:
                bits = _bits(_rbsp(nal))
                position = binary_template.list_modification_position
                bits = bits[:position] + "1" + bits[position + 1:]
                headers[index] = _nals(_wrap(_from_bits(bits)))[0]
        tail = bytearray()
        for step, poc, inputs, live in schedule:
            template = binary_template if len(step.inputs) == 2 else unary_template
            assert template is not None
            tail.extend(_instruction(template, step, poc, inputs, live))
        object.__setattr__(self, "tail", bytes(tail))
        object.__setattr__(self, "parameter_sets", tuple(headers))
        object.__setattr__(self, "_upload_parameter_sets", binary_template.parameter_sets)
        object.__setattr__(self, "output_frame_indices", MappingProxyType(frame_indices))
        object.__setattr__(self, "picture_order_counts", MappingProxyType(pocs))
        object.__setattr__(self, "max_live_references", max_live)

    def assemble(self, inputs: Mapping[str, bytes | Iterable[Iterable[int]]]) -> bytes:
        if not isinstance(inputs, Mapping) or set(inputs) != set(self.input_names):
            raise ValueError("HEVC inputs must match compiled input names exactly")
        planes = [_plane(inputs[name], self.width, self.height) for name in self.input_names]
        frames = planes if len(planes) == 1 else [planes[0], planes[0], planes[1]]
        encoded = _encode(self.width, self.height, frames, True)
        nals = _upload_nals(encoded, len(self.input_names), self._upload_parameter_sets)
        replacements = {_nal_type(n): n for n in self.parameter_sets}
        prefix = b"".join(b"\x00\x00\x00\x01" + replacements.get(_nal_type(nal), nal)
                          for nal in nals if _nal_type(nal) != 0)
        return prefix + self.tail

    def decode(self, inputs: Mapping[str, bytes | Iterable[Iterable[int]]]) -> dict[str, bytes]:
        frames = decode_frames(self.assemble(inputs), self.width, self.height)
        if len(frames) != len(self.output_frame_indices):
            raise HEVCError("unexpected frame count from HEVC resident program")
        return {name: frames[index] for name, index in self.output_frame_indices.items()}
