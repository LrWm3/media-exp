"""Constrained HEVC computation through weighted prediction, without residuals.

The compiler obtains CABAC skip syntax from an all-zero libx265 template. It
rewrites only slice prediction headers. Input uploading uses lossless encoding;
no requested arithmetic runs on the host. Actual execution is FFmpeg's HEVC
software decoder. The independent ``tail`` bytes never depend on input pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import shutil
import subprocess
from typing import Iterable


class HEVCError(RuntimeError):
    """An encoder/decoder capability or constrained-profile check failed."""


_PARAMS = (
    "lossless=1:b-adapt=0:b-pyramid=0:keyint=65535:min-keyint=65535:"
    "scenecut=0:ref=1:weightp=1:weightb=1:temporal-mvp=0:sao=0:"
    "no-deblock=1:wpp=0:pools=none:frame-threads=1:info=0:ctu=16:"
    "min-cu-size=8:repeat-headers=1:log-level=error"
)


def _run(arguments: list[str], data: bytes) -> bytes:
    if not shutil.which("ffmpeg"):
        raise HEVCError("ffmpeg with libx265 and HEVC decoding is required")
    result = subprocess.run(["ffmpeg", *arguments], input=data, capture_output=True)
    if result.returncode:
        raise HEVCError(result.stderr.decode(errors="replace"))
    return result.stdout


def _encode(width: int, height: int, frames: Iterable[bytes], binary: bool) -> bytes:
    chroma = bytes([128]) * (width * height // 2)
    data = b"".join(frame + chroma for frame in frames)
    return _run([
        "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
        "yuv420p", "-s", f"{width}x{height}", "-r", "1", "-i", "pipe:0",
        "-c:v", "libx265", "-preset", "medium", "-x265-params",
        _PARAMS + f":bframes={int(binary)}", "-f", "hevc", "pipe:1",
    ], data)


def _nals(data: bytes) -> list[bytes]:
    return [n for n in re.split(b"\x00\x00\x00?\x01", data) if n]


def _nal_type(nal: bytes) -> int:
    return (nal[0] >> 1) & 63


def _rbsp(nal: bytes) -> bytes:
    return nal.replace(b"\x00\x00\x03", b"\x00\x00")


def _wrap(raw: bytes) -> bytes:
    output = bytearray(b"\x00\x00\x00\x01")
    zeros = 0
    for value in raw:
        if zeros >= 2 and value <= 3:
            output.append(3)
            zeros = 0
        output.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(output)


def _ue(value: int) -> str:
    binary = f"{value + 1:b}"
    return "0" * (len(binary) - 1) + binary


def _se(value: int) -> str:
    return _ue(2 * abs(value) - int(value > 0))


def _bits(raw: bytes) -> str:
    return "".join(f"{value:08b}" for value in raw)


def _from_bits(bits: str) -> bytes:
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def _trace(data: bytes) -> str:
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", "pipe:0", "-c",
        "copy", "-bsf:v", "trace_headers", "-f", "null", "-",
    ], input=data, capture_output=True)
    if result.returncode:
        raise HEVCError(result.stderr.decode(errors="replace"))
    return result.stderr.decode()


def _position(trace: str, field: str) -> int:
    found = re.findall(r"\] (\d+)\s+" + re.escape(field) + r"\s", trace)
    if not found:
        raise HEVCError(f"libx265 template lacks expected HEVC field: {field}")
    return int(found[-1])


def _value(trace: str, field: str) -> int:
    found = re.findall(r"\]\s+\d+\s+" + re.escape(field) + r"\s+[01]+ = (-?\d+)", trace)
    if not found:
        raise HEVCError(f"libx265 template lacks expected HEVC field: {field}")
    return int(found[-1])


def _dimensions(width: int, height: int) -> None:
    if any(type(v) is not int or v < 16 or v > 512 or v % 8 for v in (width, height)):
        raise ValueError("HEVC dimensions must be multiples of 8 in [16, 512]")


def _weights(weights: tuple[int, ...], offset: int, denominator: int) -> None:
    if type(denominator) is not int or not 0 <= denominator <= 7:
        raise ValueError("HEVC denominator must be an integer in [0, 7]")
    if type(offset) is not int or not -128 <= offset <= 127:
        raise ValueError("HEVC offset must be an integer in [-128, 127]")
    if any(type(w) is not int or not -128 <= w - (1 << denominator) <= 127 for w in weights):
        raise ValueError("HEVC weight minus 2**denominator must be in [-128, 127]")


def _plane(value: bytes | Iterable[Iterable[int]], width: int, height: int) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        output = bytes(value)
    else:
        rows = [list(row) for row in value]
        if len(rows) != height or any(len(row) != width for row in rows):
            raise ValueError("input shape does not match compiled HEVC dimensions")
        if any(type(pixel) is not int or not 0 <= pixel <= 255 for row in rows for pixel in row):
            raise ValueError("HEVC input samples must be integers in [0, 255]")
        output = bytes(pixel for row in rows for pixel in row)
    if len(output) != width * height:
        raise ValueError("input length does not match compiled HEVC dimensions")
    return output


@dataclass(frozen=True)
class _Template:
    parameter_sets: tuple[bytes, ...]
    bits: str
    weight_start: int
    weight_end: int
    alignment: int
    cabac_start: int
    list_modification_position: int


@lru_cache(maxsize=32)
def _template(width: int, height: int, binary: bool) -> _Template:
    _dimensions(width, height)
    zero = bytes(width * height)
    encoded = _encode(width, height, [zero] * (3 if binary else 2), binary)
    nals = _nals(encoded)
    target_type = 0 if binary else 1
    targets = [n for n in nals if _nal_type(n) == target_type]
    if len(targets) != 1:
        raise HEVCError("libx265 template has unexpected prediction-frame structure")
    trace = _trace(encoded)
    for name, expected in {
        "weighted_pred_flag": 1, "weighted_bipred_flag": 1,
        "sps_temporal_mvp_enabled_flag": 0, "sample_adaptive_offset_enabled_flag": 0,
        "pps_deblocking_filter_disabled_flag": 1,
        "num_ref_idx_l0_default_active_minus1": 0,
        "num_ref_idx_l1_default_active_minus1": 0,
        "log2_max_pic_order_cnt_lsb_minus4": 4,
        "num_short_term_ref_pic_sets": 0,
        "long_term_ref_pics_present_flag": 0,
        "dependent_slice_segments_enabled_flag": 0,
        "output_flag_present_flag": 0,
        "num_extra_slice_header_bits": 0,
        "cabac_init_present_flag": 0,
        "lists_modification_present_flag": 0,
        "tiles_enabled_flag": 0,
        "entropy_coding_sync_enabled_flag": 0,
        "slice_segment_header_extension_present_flag": 0,
    }.items():
        if _value(trace, name) != expected:
            raise HEVCError(f"unsupported libx265 template setting: {name}")
    if binary and (_value(trace, "sps_max_dec_pic_buffering_minus1[0]") != 3
                   or _value(trace, "sps_max_num_reorder_pics[0]") != 1):
        raise HEVCError("unsupported libx265 decoded-picture-buffer configuration")
    alignment = _position(trace, "alignment_bit_equal_to_one")
    return _Template(
        tuple(n for n in nals if _nal_type(n) in (32, 33, 34)),
        _bits(_rbsp(targets[0])), _position(trace, "luma_log2_weight_denom"),
        _position(trace, "five_minus_max_num_merge_cand"), alignment,
        (alignment // 8 + 1) * 8, _position(trace, "lists_modification_present_flag"),
    )


def _weight_bits(weights: tuple[int, ...], offset: int, denominator: int) -> str:
    # Chroma remains identity; every list has one active reference.
    result = _ue(denominator) + _se(0)
    for weight in weights:
        result += "10" + _se(weight - (1 << denominator)) + _se(offset)
    return result


@dataclass(frozen=True)
class Kernel:
    """A reusable single HEVC weighted-prediction instruction.

    Unary: clip_u8(((weight*x + round) >> denominator) + offset),
    where round is 0 for denominator=0, else 2**(denominator-1).
    Binary: clip_u8(((left_weight*a + right_weight*b + 2**denominator)
                     >> (denominator+1)) + offset).

    ``decode`` returns only computed luma. ``assemble`` returns a standard
    Annex B HEVC stream containing uploads and the unchanged computation tail.
    """
    width: int
    height: int
    weights: tuple[int, ...]
    offset: int
    denominator: int
    tail: bytes
    parameter_sets: tuple[bytes, ...]

    @property
    def input_count(self) -> int:
        return len(self.weights)

    @property
    def output_frame_index(self) -> int:
        # Binary picture order: input A (POC 0), result (POC 1), input B (POC 2).
        return 1

    def assemble(self, *inputs: bytes | Iterable[Iterable[int]]) -> bytes:
        if len(inputs) != self.input_count:
            raise ValueError(f"expected {self.input_count} HEVC inputs")
        planes = [_plane(value, self.width, self.height) for value in inputs]
        frames = planes if self.input_count == 1 else [planes[0], planes[0], planes[1]]
        encoded = _encode(self.width, self.height, frames, self.input_count == 2)
        nals = _nals(encoded)
        headers = tuple(n for n in nals if _nal_type(n) in (32, 33, 34))
        if headers != self.parameter_sets:
            raise HEVCError("input upload parameter sets differ from compiled template")
        # The duplicated A is merely encoder lookahead input. Its B slice is
        # discarded; only the original A and B uploads remain in the prefix.
        prefix = [n for n in nals if _nal_type(n) != 0]
        return b"".join(b"\x00\x00\x00\x01" + n for n in prefix) + self.tail

    def decode(self, *inputs: bytes | Iterable[Iterable[int]]) -> bytes:
        frames = decode_frames(self.assemble(*inputs), self.width, self.height)
        if len(frames) != self.input_count + 1:
            raise HEVCError("unexpected decoded frame count")
        return frames[self.output_frame_index]


def decode_frames(bitstream: bytes, width: int, height: int) -> list[bytes]:
    """Decode Annex B HEVC and return its displayed luma planes in POC order."""
    raw = _run([
        "-hide_banner", "-loglevel", "error", "-xerror", "-threads", "1", "-i",
        "pipe:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-fps_mode",
        "passthrough", "pipe:1",
    ], bitstream)
    size = width * height
    frame_size = size * 3 // 2
    if len(raw) % frame_size:
        raise HEVCError("decoded frame size differs from compiled dimensions")
    return [raw[start:start + size] for start in range(0, len(raw), frame_size)]


def _compile(width: int, height: int, weights: tuple[int, ...], offset: int, denominator: int) -> Kernel:
    _dimensions(width, height)
    _weights(weights, offset, denominator)
    template = _template(width, height, len(weights) == 2)
    bits = (template.bits[:template.weight_start]
            + _weight_bits(weights, offset, denominator)
            + template.bits[template.weight_end:template.alignment] + "1")
    bits += "0" * (-len(bits) % 8)
    bits += template.bits[template.cabac_start:]
    return Kernel(width, height, weights, offset, denominator, _wrap(_from_bits(bits)), template.parameter_sets)


def compile_unary(width: int, height: int, weight: int = 1, offset: int = 0, denominator: int = 0) -> Kernel:
    return _compile(width, height, (weight,), offset, denominator)


def compile_binary(width: int, height: int, left_weight: int = 2, right_weight: int = 2,
                   offset: int = 0, denominator: int = 0) -> Kernel:
    return _compile(width, height, (left_weight, right_weight), offset, denominator)
