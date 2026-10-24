"""Decode externally and compare actual pixels against an independent Life oracle."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from life import step
from program import FRAMES_PER_GENERATION


def decode(path, backend="ffmpeg"):
    path = Path(path).resolve()
    if backend == "ffmpeg":
        command = ["ffmpeg", "-v", "error", "-xerror", "-threads", "1",
                   "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
                   "-pix_fmt", "yuvj420p", "-f", "rawvideo", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=True, timeout=180)
        if result.stderr:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result.stdout
    if backend not in ("vaapi", "openh264"):
        raise ValueError(f"Unknown decoder: {backend}")
    element = "vah264dec" if backend == "vaapi" else "openh264dec"
    with tempfile.TemporaryDirectory(prefix="decoder-life-") as directory:
        output = Path(directory) / "decoded.yuv"
        command = ["gst-launch-1.0", "-q", "filesrc", f"location={path}", "!",
                   "h264parse", "!", element, "!", "videoconvert", "!",
                   "video/x-raw,format=I420", "!", "filesink", f"location={output}"]
        result = subprocess.run(command, capture_output=True, check=True, timeout=180)
        if result.stderr:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return output.read_bytes()


def available(backend):
    if backend == "ffmpeg":
        return shutil.which("ffmpeg") is not None
    if shutil.which("gst-inspect-1.0") is None:
        return False
    element = "vah264dec" if backend == "vaapi" else "openh264dec"
    return subprocess.run(["gst-inspect-1.0", element], capture_output=True).returncode == 0


def check(raw, board, generations):
    height, width = len(board), len(board[0])
    coded_width, coded_height = width + 32, height + 32
    luma_size = coded_width * coded_height
    frame_size = luma_size * 3 // 2
    count = 1 + generations * FRAMES_PER_GENERATION
    if len(raw) != frame_size * count:
        raise AssertionError(f"Expected {count} frames ({frame_size * count} bytes), got {len(raw)} bytes")
    # Guard and neutral chroma are invariants for every temporary picture too.
    guard_indices = [y * coded_width + x for y in range(coded_height) for x in range(coded_width)
                     if x < 16 or x >= width + 16 or y < 16 or y >= height + 16]
    neutral_chroma = bytes([128]) * (luma_size // 2)
    for frame in range(count):
        start = frame * frame_size
        if raw[start + luma_size:start + frame_size] != neutral_chroma:
            raise AssertionError(f"Non-neutral chroma in picture {frame}")
        if any(raw[start + i] for i in guard_indices):
            raise AssertionError(f"Nonzero guard in picture {frame}")
    expected = board
    generation_luma = []
    for generation in range(generations + 1):
        start = generation * FRAMES_PER_GENERATION * frame_size
        luma = bytes(raw[start + (y + 16) * coded_width + x + 16]
                     for y in range(height) for x in range(width))
        expected_luma = bytes(value * 255 for row in expected for value in row)
        if luma != expected_luma:
            wrong = next(i for i, (a, b) in enumerate(zip(luma, expected_luma)) if a != b)
            raise AssertionError(f"Generation {generation}, cell ({wrong % width}, {wrong // width}): "
                                 f"decoder={luma[wrong]}, expected={expected_luma[wrong]}")
        generation_luma.append(luma)
        if generation < generations:
            expected = step(expected)
    return generation_luma
