#!/usr/bin/env python3
"""Capture actual named decoder outputs beside stable/automatic packing costs."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ir"))
from validate import checked, digest


def capture(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output)
    suite_bytes = (bundle / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    report_bytes = checked(bundle, "comparison.json", suite["comparison_sha256"])
    report = json.loads(report_bytes)
    groups = [name for name in ("denoise-tiles4", "chain-leaves-tiles4")
              if name in suite["representatives"]]
    if not groups:
        raise ValueError("Capture requires denoise or chain-leaves at four tiles")
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = lambda size: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    image = Image.new("RGB", (1100, 160+180*len(groups)), "#f3f5f7")
    draw = ImageDraw.Draw(image)
    draw.text((28, 22), "Packing dependent work earlier", font=font(29), fill="#172432")
    draw.text((28, 67), "Actual decoded outputs; same inputs, graph, tiles, and named results", font=font(17), fill="#455569")
    draw.text((654, 112), "Stable", font=font(17), fill="#455569")
    draw.text((872, 112), "Auto", font=font(17), fill="#455569")
    captures = []
    for index, name in enumerate(groups):
        top = 143+index*180
        draw.rounded_rectangle((18, top, 1082, top+167), radius=9, fill="white")
        record = next(item for item in report["comparisons"] if item["group"] == name)
        example = suite["representatives"][name]
        before, after = (example[variant]["costs"] for variant in ("baseline", "optimized"))
        draw.text((34, top+13), name, font=font(22), fill="#172432")
        draw.text((34, top+46), f"{before['pictures']} -> {after['pictures']} pictures", font=font(21), fill="#24674b")
        draw.text((34, top+79), f"Slices {before['prediction_slices']} -> {after['prediction_slices']} | "
                  f"Tail bytes {before['prediction_bytes']:,} -> {after['prediction_bytes']:,}",
                  font=font(16), fill="#455569")
        draw.text((34, top+106), f"Output readback {before['output_readback_bytes']:,} -> "
                  f"{after['output_readback_bytes']:,} bytes", font=font(16), fill="#455569")
        draw.text((34, top+134), f"All {record['named_outputs']} named outputs match; output shown", font=font(15), fill="#455569")
        variants = []
        for x, variant in ((650, "baseline"), (865, "optimized")):
            case = example[variant]
            raw = checked(bundle, case["expected"], case["expected_sha256"])
            info = case["named_outputs"]["output"]
            expected = checked(bundle, info["path"], info["sha256"])
            start = info["frame"]*case["frame_bytes"]
            pixels = b"".join(raw[start+(row+info["y"])*case["coded_width"]+info["x"]:
                                  start+(row+info["y"])*case["coded_width"]+info["x"]+case["width"]]
                              for row in range(case["height"]))
            if pixels != expected:
                raise AssertionError(name + "/" + variant + ": decoded capture mismatch")
            binary = max(pixels) <= 1
            visible = bytes(value*255 for value in pixels) if binary else pixels
            tile = Image.frombytes("L", (case["width"], case["height"]), visible)
            tile = tile.resize((128, 128), Image.Resampling.NEAREST)
            image.paste(tile, (x, top+19))
            variants.append({"variant": variant, "stream_sha256": case["stream_sha256"],
                "decoded_sha256": case["expected_sha256"], "output_sha256": digest(pixels),
                "display_mapping": "binary 0/1 to 0/255" if binary else "literal u8"})
        captures.append({"case": name, "output": "output", "variants": variants})
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    output.with_suffix(".capture.json").write_text(json.dumps({"suite_sha256": digest(suite_bytes),
        "comparison_sha256": digest(report_bytes), "capture_sha256": digest(output.read_bytes()),
        "capture_script_sha256": digest(Path(__file__).read_bytes()), "captures": captures,
        "timings": "not measured", "source": "Checked full FFmpeg I420 named output regions"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.bundle, args.output)
