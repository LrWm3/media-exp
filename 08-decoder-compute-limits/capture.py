#!/usr/bin/env python3
"""Render the Boolean atlas from hash-verified decoder output planes."""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from benchmark import checked, digest


ROOT = Path(__file__).resolve().parent
BG, TEXT, SECONDARY, ACCENT = "#101923", "#edf4f7", "#b4c4d0", "#63e4b5"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def capture(bundle, results, output):
    bundle, results, output = bundle.resolve(), results.resolve(), output.resolve()
    bundle_bytes = (bundle/"bundle.json").read_bytes()
    metadata = json.loads(bundle_bytes)
    report = json.loads((results/"results.json").read_text())
    if report["status"] != "pass" or report["bundle_sha256"] != digest(bundle_bytes):
        raise ValueError("A passing report for this exact bundle is required")
    atlas_bytes = checked(bundle, metadata["atlas"], metadata["atlas_sha256"])
    atlas = json.loads(atlas_bytes)
    records = {case["mask"]: case for case in metadata["cases"]
               if case["purpose"] == "atlas" and case["mode"] == "packed"}
    canvas = Image.new("RGB", (1180, 1010), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), "08 / Boolean computation limits", font=font(32), fill=TEXT)
    draw.text((28, 77), "All 16 two-input functions. Exact minima within the stated P/B model.",
              font=font(18), fill=ACCENT)
    draw.text((28, 112), "Actual %s output pixels / 0 and 1 enlarged to black and white" % report["backend"].upper(),
              font=font(17), fill=SECONDARY)
    draw.text((28, 151), "Zero residuals. 8-bit intermediates. Input loading excluded from minima.",
              font=font(16), fill=SECONDARY)
    # Every case uses the same checked pair, so one legend suffices.
    first = records[0]
    for index, name in enumerate(("a", "b")):
        item = first["inputs"][name]
        raw = checked(bundle, item["path"], item["sha256"])
        plane = Image.frombytes("L", (first["width"], first["height"]), bytes(value*255 for value in raw))
        canvas.paste(plane.resize((80, 80), Image.Resampling.NEAREST).convert("RGB"), (934+index*120, 67))
        draw.text((934+index*120, 154), "Input "+name.upper(), font=font(15), fill=SECONDARY)
    for function in atlas["functions"]:
        mask = function["id"]
        case = records[mask]
        validation = report["cases"][case["id"]]["validation"]
        with Image.open(results/validation["capture"]) as source:
            plane = source.convert("L")
        raw = plane.tobytes()
        if any(value not in (0, 255) for value in raw):
            raise ValueError("Capture is not a binary output")
        native = bytes(value//255 for value in raw)
        if digest(native) != validation["output_sha256"] or digest(native) != case["expected_output_sha256"]:
            raise ValueError("Captured decoder pixels do not match their verified hashes")
        left, top = 28+(mask%4)*286, 210+(mask//4)*180
        draw.rounded_rectangle((left, top, left+270, top+163), radius=8, fill="#1b2a37")
        label = function["name"].replace("_", " ")
        draw.text((left+12, top+10), "%02d  %s" % (mask, label), font=font(17), fill=TEXT)
        canvas.paste(plane.resize((96, 96), Image.Resampling.NEAREST).convert("RGB"), (left+12, top+45))
        draw.text((left+123, top+49), "Instructions: %d" % function["scalar_minimum"], font=font(16), fill=SECONDARY)
        draw.text((left+123, top+78), "Pictures: %d" % function["packed_minimum"], font=font(16), fill=ACCENT)
        draw.text((left+123, top+117), "Truth: "+"".join(map(str, function["truth"])), font=font(14), fill=SECONDARY)
    draw.text((28, 945), "Truth order: 00, 01, 10, 11. XOR/XNOR require 3 instructions or 2 packed computation pictures.",
              font=font(17), fill=ACCENT)
    draw.text((28, 976), "Exhaustive single-prediction coverage + structural lower bounds + real decoder verification.",
              font=font(15), fill=SECONDARY)
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output/"boolean-atlas.png")
    (output/"atlas.json").write_bytes(atlas_bytes)
    (output/"bundle.json").write_bytes(bundle_bytes)
    shutil.copyfile(results/"results.json", output/(report["backend"]+"-results.json"))
    for mask in (6, 9):
        case = records[mask]
        label = case["function"].lower()
        (output/(label+".h264")).write_bytes(checked(bundle, case["stream"], case["stream_sha256"]))
        manifest = json.loads(checked(bundle, case["manifest"], case["manifest_sha256"]))
        manifest["stream_sha256"] = case["stream_sha256"]
        (output/(label+".json")).write_text(json.dumps(manifest, indent=2)+"\n")
    print("Saved verified Boolean atlas, representative streams, and reports.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT/"out"/"bundle")
    parser.add_argument("--results", type=Path, default=ROOT/"out"/"nvdec-results")
    parser.add_argument("--output", type=Path, default=ROOT/"assets")
    args = parser.parse_args()
    capture(args.bundle, args.results, args.output)


if __name__ == "__main__":
    main()
