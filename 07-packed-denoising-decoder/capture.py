#!/usr/bin/env python3
"""Render verified NVDEC output planes; visualization performs no inference."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont
from benchmark import checked_file, digest


ROOT = Path(__file__).resolve().parent
BACKGROUND, FOREGROUND, MUTED, ACCENT = "#101821", "#edf5f7", "#aabcc7", "#65e0b4"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def decoded(results, report, variant, index, name):
    item = report["variants"][variant]["validation"]["jobs"][index]["named_outputs"][name]
    with Image.open(results / item["capture"]) as source:
        plane = source.convert("L")
    values = plane.tobytes()
    if any(value not in (0, 255) for value in values):
        raise ValueError("Capture contains a nonbinary output")
    if digest(bytes(value//255 for value in values)) != item["sha256"]:
        raise ValueError("Capture hash differs from validated decoder output")
    return plane


def capture(bundle, results, output):
    bundle, results, output = bundle.resolve(), results.resolve(), output.resolve()
    report = json.loads((results / "results.json").read_text())
    bundle_bytes = (bundle / "bundle.json").read_bytes()
    metadata = json.loads(bundle_bytes)
    if report["backend"] != "nvdec" or report["status"] != "pass":
        raise ValueError("Capture requires a passing NVDEC report")
    if digest(bundle_bytes) != report["bundle_sha256"]:
        raise ValueError("Result refers to another bundle")
    frames = []
    variant = report["variants"]["packed-8"]
    baseline_shape = report["variants"]["baseline"]["geometry"]
    shape = variant["geometry"]
    before, after = baseline_shape["pictures_per_job"], shape["pictures_per_job"]
    pixel_ratio = (after*shape["coded_width"]*shape["coded_height"] /
                   (before*baseline_shape["coded_width"]*baseline_shape["coded_height"]))
    for index, item in enumerate(metadata["inputs"]):
        noisy_bytes = checked_file(bundle, item["path"], item["sha256"])
        target_bytes = checked_file(bundle, item["clean"]["path"], item["clean"]["sha256"])
        noisy = Image.frombytes("L", (32, 32), bytes(value*255 for value in noisy_bytes))
        target = Image.frombytes("L", (32, 32), bytes(value*255 for value in target_bytes))
        baseline = decoded(results, report, "baseline", index, "output")
        packed = decoded(results, report, "packed-8", index, "output")
        if baseline.tobytes() != packed.tobytes():
            raise AssertionError("Packed and baseline captures differ")
        canvas = Image.new("RGB", (1100, 650), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 22), "07 / Packed denoising inside H.264", font=font(30), fill=FOREGROUND)
        draw.text((32, 67), "%d → %d pictures  |  unchanged frozen weights  |  actual NVDEC output" % (before, after),
                  font=font(18), fill=ACCENT)
        for column, (label, plane, detail) in enumerate([
                ("Noisy input", noisy, "PCM input pixels"),
                ("Packed decoder", packed, "8-tile schedule"),
                ("Baseline decoder", baseline, "Identical output pixels"),
                ("Clean target", target, "Verification only")]):
            left = 32+column*265
            draw.text((left, 116), label, font=font(20), fill=FOREGROUND)
            canvas.paste(plane.resize((160, 160), Image.Resampling.NEAREST).convert("RGB"), (left, 150))
            draw.text((left, 323), detail, font=font(15), fill=MUTED)
        draw.text((32, 370), "Decoded feature maps / picture and tile locations", font=font(20), fill=FOREGROUND)
        for channel in range(8):
            name, left = "feature_%d" % channel, 32+channel*132
            plane = decoded(results, report, "packed-8", index, name)
            region = variant["geometry"]["output_regions"][name]
            canvas.paste(plane.resize((96, 96), Image.Resampling.NEAREST).convert("RGB"), (left, 410))
            draw.text((left, 516), "Channel %d" % channel, font=font(14), fill=FOREGROUND)
            draw.text((left, 539), "P%d / tile %d" % (region["frame"], region["tile"]), font=font(13), fill=MUTED)
        draw.text((32, 586), "Coded picture: %d×%d → %d×%d. %.2f× decoded luma samples." %
                  (baseline_shape["coded_width"], baseline_shape["coded_height"],
                   shape["coded_width"], shape["coded_height"], pixel_ratio),
                  font=font(17), fill=ACCENT)
        draw.text((32, 619), "Sample %d/8 • Display enlarged; playback speed is illustrative." % (index+1),
                  font=font(14), fill=MUTED)
        frames.append(canvas)
    output.mkdir(parents=True, exist_ok=True)
    frames[0].save(output / "packed-capture.png")
    frames[0].save(output / "packed-capture.gif", save_all=True, append_images=frames[1:],
                   duration=800, loop=0, optimize=False)
    with tempfile.TemporaryDirectory(prefix="packed-capture-") as temporary:
        directory = Path(temporary)
        for index, frame in enumerate(frames):
            frame.save(directory / ("%03d.png" % index))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5/4", "-i",
                        str(directory / "%03d.png"), "-vf", "fps=25", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "packed-capture.mp4")], check=True)
    first = metadata["variants"]["packed-8"]["jobs"][0]
    stream = checked_file(bundle, first["stream"], first["stream_sha256"])
    (output / "packed.h264").write_bytes(stream)
    info = metadata["variants"]["packed-8"]
    manifest = json.loads(checked_file(bundle, info["manifest"], info["manifest_sha256"]))
    manifest.update(stream_sha256=digest(stream), prediction_sha256=info["tail_sha256"])
    (output / "program.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(results / "results.json", output / "nvdec-results.json")
    (output / "bundle.json").write_bytes(bundle_bytes)
    print("Saved verified NVDEC captures, executable, manifests, and measurements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--results", type=Path, default=ROOT / "out" / "nvdec-results")
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    capture(args.bundle, args.results, args.output)


if __name__ == "__main__":
    main()
