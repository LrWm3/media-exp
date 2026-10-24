#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=10.1"]
# ///
"""Render verified NVDEC captures as PNG, GIF, and a 24-frame, 12 fps MP4."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
LABELS = ("output", "difference", "raw_mask", "eroded", "opened")
FPS = 12
FRAME_COUNT = 24
SOURCE_SIZE = (96, 64)
SCALE = 4
CANVAS_SIZE = (1280, 936)
COLORS = {
    "background": "#0c1421", "card": "#152234", "line": "#2c3e53",
    "white": "#edf4fa", "muted": "#a4b6c9", "cyan": "#55e5d0",
    "amber": "#ffc878", "blue": "#8abaff",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_pgm(path, expected_size):
    """Read literal P5 samples; never normalize binary samples to 255 here."""
    data = path.read_bytes()
    offset, tokens = 0, []
    while len(tokens) < 4:
        while offset < len(data):
            if data[offset] in b" \t\r\n\v\f":
                offset += 1
            elif data[offset] == ord("#"):
                end = data.find(b"\n", offset)
                if end < 0:
                    raise ValueError(f"Unterminated PGM comment: {path}")
                offset = end + 1
            else:
                break
        start = offset
        while offset < len(data) and data[offset] not in b" \t\r\n\v\f#":
            offset += 1
        if start == offset:
            raise ValueError(f"Incomplete PGM header: {path}")
        tokens.append(data[start:offset])
    if tokens[0] != b"P5" or tokens[3] != b"255":
        raise ValueError(f"Capture must be raw 8-bit P5 with maxval 255: {path}")
    size = (int(tokens[1]), int(tokens[2]))
    if size != expected_size:
        raise ValueError(f"PGM dimensions {size} != {expected_size}: {path}")
    if offset >= len(data) or data[offset] not in b" \t\r\n\v\f":
        raise ValueError(f"Missing PGM raster separator: {path}")
    # Consume the header delimiter only. A raster may itself start with whitespace.
    offset += 2 if data[offset:offset + 2] == b"\r\n" else 1
    samples = data[offset:]
    if len(samples) != size[0] * size[1]:
        raise ValueError(f"PGM raster length differs from dimensions: {path}")
    return Image.frombytes("L", size, samples)


def indexed_jobs(document, name):
    jobs = document.get("jobs", [])
    if not isinstance(jobs, list) or len(jobs) != FRAME_COUNT:
        raise ValueError(f"{name} must contain all {FRAME_COUNT} jobs")
    indices = [job.get("index") for job in jobs]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(FRAME_COUNT)):
        raise ValueError(f"{name} has duplicate, missing, or invalid job indices")
    return {job["index"]: job for job in jobs}


def load_verified(bundle_root, results_root):
    bundle_bytes = (bundle_root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    result_path = results_root / "results.json"
    if not result_path.is_file():
        raise ValueError(f"NVDEC results unavailable: {result_path}; run the remote benchmark first")
    results = json.loads(result_path.read_bytes())
    if results.get("status") != "pass":
        raise ValueError("NVDEC benchmark must finish with status 'pass' before rendering")
    if results.get("provenance", {}).get("backend") != "nvdec":
        raise ValueError("NVDEC artwork requires actual 'nvdec' capture provenance")
    if results.get("bundle_sha256") != digest(bundle_bytes):
        raise ValueError("NVDEC results do not match this bundle.json SHA-256")
    size = (bundle.get("width"), bundle.get("height"))
    if size != SOURCE_SIZE or bundle.get("job_count") != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} jobs with {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} inputs")
    references = indexed_jobs(bundle, "Bundle")
    captures = indexed_jobs(results, "NVDEC results")
    inputs = [read_pgm(bundle_root / "inputs" / f"{index:03d}.pgm", size)
              for index in range(FRAME_COUNT + 1)]
    verified = []
    for index in range(FRAME_COUNT):
        job, reference = captures[index], references[index]
        if job.get("status") != "pass" or job.get("raw_sha256") != reference.get("expected_sha256"):
            raise ValueError(f"Full NVDEC readback verification failed for job {index}")
        planes = {}
        for label in LABELS:
            record = job.get("named_outputs", {}).get(label, {})
            expected = reference.get("named_outputs", {}).get(label, {}).get("sha256")
            if record.get("status") != "pass" or not expected:
                raise ValueError(f"Missing passing capture record: job {index}/{label}")
            relative = record.get("capture")
            if not isinstance(relative, str):
                raise ValueError(f"Missing capture path: job {index}/{label}")
            path = (results_root / relative).resolve()
            if not path.is_relative_to(results_root):
                raise ValueError(f"Capture path escapes results directory: {relative}")
            plane = read_pgm(path, size)
            actual = digest(plane.tobytes())
            if actual != record.get("sha256") or actual != record.get("expected_sha256") or actual != expected:
                raise ValueError(f"Capture/reference SHA-256 mismatch: job {index}/{label}")
            if label != "difference" and not set(plane.tobytes()).issubset({0, 1}):
                raise ValueError(f"Mask contains samples other than 0/1: job {index}/{label}")
            planes[label] = plane
        verified.append({"current": inputs[index + 1], **planes})
    return bundle, results, verified


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [Path("/usr/share/fonts/truetype/dejavu") / name, Path(name)]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def display_mask(plane):
    """Display transformation only; verification always uses unscaled samples."""
    return plane.point([0, 255] + [0] * 254)


def render_frame(index, bundle, planes):
    canvas = Image.new("RGB", CANVAS_SIZE, COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    fonts = {size: font(size) for size in (14, 15, 16, 17)}
    heading, panel_heading = font(29, True), font(20, True)
    draw.text((24, 20), "NVDEC / synthetic video change mask", font=heading, fill=COLORS["white"])
    draw.text((25, 64), "Actual decoder readback  /  96 × 64 pixels  /  nearest-neighbor display  /  12 fps",
              font=fonts[16], fill=COLORS["muted"])
    draw.rounded_rectangle((1094, 22, 1255, 57), radius=8, fill="#183e3c")
    draw.text((1108, 29), "SHA-256 VERIFIED", font=font(13, True), fill=COLORS["cyan"])
    draw.line((24, 99, 1256, 99), fill=COLORS["line"], width=1)

    binary = {label: display_mask(planes[label]) for label in ("raw_mask", "opened", "output")}
    source = planes["current"].convert("RGB")
    tint = Image.blend(source, Image.new("RGB", SOURCE_SIZE, COLORS["cyan"]), 0.62)
    overlay = Image.composite(tint, source, binary["output"])
    panels = [
        ("01", "Current input", "Synthetic grayscale video · source frame %02d" % (index + 1),
         source, COLORS["blue"]),
        ("02", "Absolute difference", "|current − previous| · unscaled 8-bit values",
         planes["difference"], COLORS["amber"]),
        ("03", "Raw change mask", "Difference ≥ %d · decoded binary samples" % bundle["cutoff"],
         binary["raw_mask"], COLORS["amber"]),
        ("04", "Opened mask", "3 × 3 opening · isolated changes removed",
         binary["opened"], COLORS["cyan"]),
        ("05", "Cleaned mask", "3 × 3 closing · final decoded output",
         binary["output"], COLORS["cyan"]),
        ("06", "Mask overlay", "Visualization only · decoded mask on input",
         overlay, COLORS["cyan"]),
    ]
    for position, (number, title, subtitle, plane, accent) in enumerate(panels):
        left, top = 24 + (position % 3) * 416, 116 + (position // 3) * 354
        draw.rounded_rectangle((left, top, left + 400, top + 336), radius=12,
                               fill=COLORS["card"], outline=COLORS["line"])
        draw.text((left + 12, top + 13), number, font=font(14, True), fill=accent)
        draw.text((left + 43, top + 8), title, font=panel_heading, fill=COLORS["white"])
        draw.text((left + 12, top + 39), subtitle, font=fonts[14], fill=COLORS["muted"])
        enlarged = plane.convert("RGB").resize((SOURCE_SIZE[0] * SCALE, SOURCE_SIZE[1] * SCALE),
                                                Image.Resampling.NEAREST)
        canvas.paste(enlarged, (left + 8, top + 72))

    raw_pixels = sum(planes["raw_mask"].tobytes())
    clean_pixels = sum(planes["output"].tobytes())
    draw.text((25, 826), f"FRAME {index + 1:02d} / {FRAME_COUNT}   ·   {index / FPS:.2f} s",
              font=font(17, True), fill=COLORS["white"])
    draw.text((390, 827), f"Changed pixels: {raw_pixels} raw  →  {clean_pixels} cleaned",
              font=fonts[17], fill=COLORS["cyan"])
    draw.text((25, 860), "5 captured output planes / frame match NVDEC records and reference SHA-256 hashes.",
              font=fonts[15], fill=COLORS["muted"])
    draw.text((25, 884), "Binary 0/1 → 0/255 for display only. Difference samples unchanged. Overlay is visualization only.",
              font=fonts[15], fill=COLORS["muted"])
    draw.rounded_rectangle((24, 919, 1255, 923), radius=2, fill=COLORS["line"])
    draw.rounded_rectangle((24, 919, 24 + round(1231 * (index + 1) / FRAME_COUNT), 923),
                           radius=2, fill=COLORS["cyan"])
    return canvas


def run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{Path(command[0]).name} failed:\n{error.stderr.strip()}") from error


def capture(bundle_root, results_root, output):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise ValueError("FFmpeg is required to render GIF/MP4; install ffmpeg and put it on PATH")
    if not ffprobe:
        raise ValueError("ffprobe is required to verify MP4 frame count; install the FFmpeg tools")
    bundle, _results, verified = load_verified(bundle_root.resolve(), results_root.resolve())
    with tempfile.TemporaryDirectory(prefix="nvdec-motion-capture-") as temporary:
        staging = Path(temporary)
        for index, planes in enumerate(verified):
            render_frame(index, bundle, planes).save(staging / f"frame-{index:03d}.png")
        pattern = str(staging / "frame-%03d.png")
        video, animation = staging / "motion-capture.mp4", staging / "motion-capture.gif"
        common = [ffmpeg, "-v", "error", "-y", "-framerate", str(FPS), "-i", pattern]
        run(common + ["-frames:v", str(FRAME_COUNT), "-c:v", "libx264", "-crf", "18",
                      "-preset", "medium", "-pix_fmt", "yuv420p", "-threads", "2",
                      "-movflags", "+faststart", str(video)])
        run(common + ["-filter_complex_threads", "1", "-filter_complex",
                      "[0:v]split[a][b];[a]palettegen=max_colors=256[p];[b][p]paletteuse=dither=none",
                      "-frames:v", str(FRAME_COUNT), "-loop", "0", str(animation)])
        probe = json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                               "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
                               "-of", "json", str(video)]).stdout)["streams"][0]
        if int(probe["nb_read_frames"]) != FRAME_COUNT or probe["avg_frame_rate"] != f"{FPS}/1":
            raise ValueError("Rendered MP4 must contain all 24 frames at 12 fps")
        if (probe["width"], probe["height"]) != CANVAS_SIZE:
            raise ValueError("Rendered MP4 dimensions differ from capture canvas")
        with Image.open(animation) as gif:
            if gif.n_frames != FRAME_COUNT or gif.size != CANVAS_SIZE:
                raise ValueError("Rendered GIF must contain all 24 full-size frames")
        # Use the frame with most changed pixels for a useful static preview.
        still = max(range(FRAME_COUNT), key=lambda index: sum(verified[index]["output"].tobytes()))
        shutil.copyfile(staging / f"frame-{still:03d}.png", staging / "motion-capture.png")
        output.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "gif", "mp4"):
            name = f"motion-capture.{suffix}"
            pending = output / f".{name}.tmp"
            shutil.copyfile(staging / name, pending)
            pending.replace(output / name)
    print(f"PASS: 24 NVDEC frames, 120 verified planes, PNG/GIF/MP4 → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--results", type=Path, default=ROOT / "out" / "nvdec-results")
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    try:
        capture(args.bundle, args.results, args.output)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"capture: {error}\n")


if __name__ == "__main__":
    main()
