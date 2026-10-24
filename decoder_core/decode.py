"""Run unmodified H.264 decoders, preserving full-range sample values."""

from pathlib import Path
from dataclasses import dataclass
import os
import re
import shutil
import subprocess
import tempfile


_GSTREAMER_DECODERS = {"vaapi": "vah264dec", "openh264": "openh264dec", "nvdec": "nvv4l2decoder"}
_NVDEC_STARTUP = re.compile(
    r"(?:Opening in BLOCKING MODE|"
    r"NvMMLite(?:Open|BlockCreate)\s*:\s*Block\s*:\s*BlockType\s*=\s*\d+|"
    r"NVMMLITE_NVVIDEODEC,\s*<NvMMLiteNvVideoDecSetAttribute:\d+>\s*Error status reporting set to 1|"
    r"NvMMLiteNVMEDIADecSetAttribute::\s*Error status reporting set to 1)"
)
_NVDEC_FRAME = re.compile(
    r"ErrorType\s*=\s*(?P<error>0[xX][\da-fA-F]+|\d+)\s+"
    r"Decoded MBs\s*=\s*(?:0[xX][\da-fA-F]+|\d+)\s+"
    r"Concealed MBs\s*=\s*(?P<concealed>0[xX][\da-fA-F]+|\d+)\s+"
    r"FrameDecodeTime\s*(?:=\s*)?\d+"
)


def _validate_nvdec_diagnostics(stdout, stderr, expected_decoded=None,
                                 expected_copied=None, require_copied=False):
    """Accept known NVDEC status output, never errors or concealed macroblocks."""
    frame_counts = []
    copied_counts = []
    for output in (stdout, stderr):
        for raw_line in output.decode(errors="replace").splitlines():
            line = raw_line.strip()
            if not line or _NVDEC_STARTUP.fullmatch(line):
                continue
            count = re.fullmatch(r"NVDEC_DECODE_FRAMES=(\d+)", line)
            if count:
                frame_counts.append(int(count.group(1)))
                continue
            copied = re.fullmatch(r"NVDEC_COPIED_FRAMES=(\d+)", line)
            if copied:
                copied_counts.append(int(copied.group(1)))
                continue
            frame = _NVDEC_FRAME.fullmatch(line)
            if frame:
                values = [int(value, 16 if value.lower().startswith("0x") else 10)
                          for value in (frame.group("error"), frame.group("concealed"))]
                if not any(values):
                    continue
                raise RuntimeError("NVDEC decoder reported an error or concealed macroblocks: " + line)
            raise RuntimeError("Unexpected NVDEC decoder diagnostic: " + line)
    if len(frame_counts) != 1 or frame_counts[0] == 0:
        raise RuntimeError("NVDEC helper must report one successful nonempty picture count")
    if len(copied_counts) > 1 or (require_copied and len(copied_counts) != 1):
        raise RuntimeError("NVDEC helper must report one copied picture count")
    if expected_decoded is not None and frame_counts[0] != expected_decoded:
        raise RuntimeError("NVDEC decoded picture count mismatch: expected {}, got {}".format(
            expected_decoded, frame_counts[0]))
    if copied_counts:
        wanted = frame_counts[0] if expected_copied is None else expected_copied
        if copied_counts[0] != wanted or copied_counts[0] > frame_counts[0]:
            raise RuntimeError("NVDEC copied picture count mismatch: expected {}, got {}".format(
                wanted, copied_counts[0]))
    return frame_counts[0]


def _nvdec_helper():
    return Path(os.environ.get("H264_NVDEC_DECODER", str(Path(__file__).parent / ".build" / "nvdec_decode"))).resolve()


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
    if backend not in _GSTREAMER_DECODERS:
        raise ValueError(f"Unknown decoder: {backend}")
    with tempfile.TemporaryDirectory(prefix="decoder-life-") as directory:
        output = Path(directory) / "decoded.yuv"
        if backend == "nvdec":
            helper = _nvdec_helper()
            if not helper.is_file() or not os.access(helper, os.X_OK):
                raise RuntimeError("Build decoder_core/build_nvdec.sh or set H264_NVDEC_DECODER to the executable helper")
            command = [str(helper), str(path), str(output)]
        else:
            command = ["gst-launch-1.0", "-q", "filesrc", f"location={path}", "!",
                       "h264parse", "!", _GSTREAMER_DECODERS[backend], "!", "videoconvert", "!",
                       "video/x-raw,format=I420", "!", "filesink", f"location={output}"]
        result = subprocess.run(command, capture_output=True, check=True, timeout=180)
        if backend == "nvdec":
            _validate_nvdec_diagnostics(result.stdout, result.stderr)
        elif result.stderr:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        raw = output.read_bytes()
        if backend == "nvdec" and not raw:
            raise RuntimeError("NVDEC helper produced empty decoded output")
        return raw


@dataclass(frozen=True)
class BatchDecodeResult:
    """Packed I420 frames indexed by job, then position in selected_frames."""

    frames: tuple
    selected_frames: tuple
    width: int
    height: int
    pictures_per_job: int
    decoded_frames: int
    copied_frames: int

    @property
    def frame_bytes(self):
        return self.width * self.height * 3 // 2

    @property
    def job_count(self):
        return len(self.frames)


def decode_batch(paths, backend="nvdec", *, width, height, pictures_per_job,
                 selected_frames=None):
    """Decode a finite batch in one process and one decoder pipeline to EOS.

    Each immutable input must be an independent Annex-B H.264 stream, starting
    with SPS/PPS and an IDR, with the same even width/height and pictures_per_job
    displayed pictures. Inputs are concatenated once in argument order. This is
    a persistent finite batch, not an online submission service. Picture indices
    are zero-based display/output order, independently within each job.
    width/height describe complete decoded/coded pictures, including any compiler
    guard pixels (not the smaller active image dimensions).

    selected_frames is a nonempty, strictly increasing sequence of picture
    indices shared by every job; None copies all pictures. The result maps
    result.frames[job][i] to selected_frames[i], with exact packed I420 bytes.
    NVDEC releases each unselected GstSample without mapping/copying its image;
    the decoder retains any reference surfaces required by subsequent pictures.
    Only selected images undergo VIC layout copies and CPU readback. The ffmpeg,
    vaapi and openh264 compatibility paths decode/copy all images before selecting
    them in Python; copied_frames reports that actual readback count.

    NVDEC checks every picture's shape and the total decoded/copied counts.
    All backends require the exact expected raw byte count.
    Per-job picture counts are a caller contract: the concatenated stream does
    not expose job boundaries to the decoder. No input file is modified.
    """
    for name, value in (("width", width), ("height", height),
                        ("pictures_per_job", pictures_per_job)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + " must be a positive integer")
    if width % 2 or height % 2:
        raise ValueError("width and height must be even for packed I420")
    if backend != "ffmpeg" and backend not in _GSTREAMER_DECODERS:
        raise ValueError("Unknown decoder: " + str(backend))
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise ValueError("paths must be a nonempty sequence of input paths")
    paths = tuple(Path(path).resolve() for path in paths)
    if not paths:
        raise ValueError("paths must be a nonempty sequence of input paths")
    selection = tuple(range(pictures_per_job)) if selected_frames is None else tuple(selected_frames)
    if not selection:
        raise ValueError("selected_frames must be nonempty")
    if any(type(index) is not int for index in selection):
        raise ValueError("selected_frames must contain integer picture indices")
    if any(index < 0 or index >= pictures_per_job for index in selection):
        raise ValueError("selected_frames picture index out of range")
    if any(left >= right for left, right in zip(selection, selection[1:])):
        raise ValueError("selected_frames must be strictly increasing without duplicates")
    frame_bytes = width * height * 3 // 2
    expected_decoded = len(paths) * pictures_per_job
    global_selection = tuple(job * pictures_per_job + index
                             for job in range(len(paths)) for index in selection)
    with tempfile.TemporaryDirectory(prefix="decoder-batch-") as directory:
        directory = Path(directory)
        combined = directory / "batch.h264"
        with combined.open("wb") as destination:
            for path in paths:
                if path.stat().st_size == 0:
                    raise ValueError("Empty H.264 batch input: " + str(path))
                with path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        if backend == "nvdec":
            helper = _nvdec_helper()
            if not helper.is_file() or not os.access(helper, os.X_OK):
                raise RuntimeError("Build decoder_core/build_nvdec.sh or set H264_NVDEC_DECODER to the executable helper")
            output = directory / "decoded.i420"
            command = [str(helper), str(combined), str(output),
                       "--expected-frames", str(expected_decoded),
                       "--width", str(width), "--height", str(height)]
            if selected_frames is not None:
                indices = directory / "frames.txt"
                indices.write_text("".join("{}\n".format(index) for index in global_selection))
                command.extend(["--frames", str(indices)])
            result = subprocess.run(command, capture_output=True, check=True, timeout=180)
            copied_frames = len(global_selection)
            _validate_nvdec_diagnostics(result.stdout, result.stderr,
                                         expected_decoded=expected_decoded,
                                         expected_copied=copied_frames, require_copied=True)
            raw = output.read_bytes()
            expected_bytes = copied_frames * frame_bytes
            if len(raw) != expected_bytes:
                raise RuntimeError("Batch decoded byte count mismatch: expected {}, got {}".format(
                    expected_bytes, len(raw)))
        else:
            all_raw = decode(combined, backend)
            copied_frames = expected_decoded
            expected_bytes = expected_decoded * frame_bytes
            if len(all_raw) != expected_bytes:
                raise RuntimeError("Batch decoded byte count mismatch: expected {}, got {}".format(
                    expected_bytes, len(all_raw)))
            raw = b"".join(all_raw[index * frame_bytes:(index + 1) * frame_bytes]
                           for index in global_selection)
    frames = tuple(tuple(raw[(job * len(selection) + index) * frame_bytes:
                             (job * len(selection) + index + 1) * frame_bytes]
                         for index in range(len(selection))) for job in range(len(paths)))
    return BatchDecodeResult(frames, selection, width, height, pictures_per_job,
                             expected_decoded, copied_frames)


def available(backend):
    if backend == "ffmpeg":
        return shutil.which("ffmpeg") is not None
    if backend not in _GSTREAMER_DECODERS:
        return False
    if backend == "nvdec":
        helper = _nvdec_helper()
        if not helper.is_file() or not os.access(helper, os.X_OK):
            return False
    if shutil.which("gst-inspect-1.0") is None:
        return False
    return subprocess.run(["gst-inspect-1.0", _GSTREAMER_DECODERS[backend]], capture_output=True).returncode == 0
