"""Run unmodified H.264 decoders, preserving full-range sample values."""

from pathlib import Path
import shutil
import subprocess
import tempfile

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
