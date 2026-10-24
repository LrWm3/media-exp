#!/usr/bin/env python3
"""Render checked decoder pixels and measured static picture counts from a bundle."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from validate import checked, digest


def capture(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output)
    suite_bytes = (bundle / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    report_bytes = checked(bundle, "comparison.json", suite["comparison_sha256"])
    report = json.loads(report_bytes)
    selected = [name for name in ("parity40", "absorption", "equivalent-or", "mixed-pb",
                                  "denoise-packed4", "denoise-packed8")
                if name in suite["representatives"]]
    if not selected:
        raise ValueError("No supported capture representatives in bundle")
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = lambda size: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    image = Image.new("RGB", (1120, 170+len(selected)*152), "#f3f5f7")
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), "Semantic IR and mixed prediction packing", font=font(29), fill="#172432")
    draw.text((30, 67), "Actual decoded named outputs; baseline reference checked on every output", font=font(18), fill="#455569")
    draw.text((30, 111), "Case / static picture count", font=font(17), fill="#455569")
    draw.text((602, 111), "Baseline", font=font(17), fill="#455569")
    draw.text((830, 111), "Optimized", font=font(17), fill="#455569")
    captures = []
    for index, name in enumerate(selected):
        top = 144+index*152
        draw.rounded_rectangle((20, top, 1100, top+140), radius=9, fill="white")
        example = suite["representatives"][name]
        comparison = next(item for item in report["comparisons"] if item["group"] == name)
        label = "output" if "output" in example["baseline"]["named_outputs"] else next(iter(example["baseline"]["named_outputs"]))
        before, after = (example[variant]["pictures"] for variant in ("baseline", "optimized"))
        draw.text((36, top+17), name, font=font(22), fill="#172432")
        draw.text((36, top+50), f"{before} -> {after} pictures", font=font(21), fill="#24674b")
        draw.text((36, top+82), f"All {comparison['named_outputs']} named outputs match", font=font(15), fill="#455569")
        draw.text((36, top+108), "Shown output: " + label, font=font(14), fill="#455569")
        outputs = []
        for x, variant in ((600, "baseline"), (828, "optimized")):
            case = example[variant]
            raw = checked(bundle, case["expected"], case["expected_sha256"])
            info = case["named_outputs"][label]
            expected = checked(bundle, info["path"], info["sha256"])
            start = info["frame"]*case["frame_bytes"]
            pixels = b"".join(raw[start+(row+info["y"])*case["coded_width"]+info["x"]:
                                  start+(row+info["y"])*case["coded_width"]+info["x"]+case["width"]]
                              for row in range(case["height"]))
            if pixels != expected:
                raise AssertionError(name + "/" + variant + ": capture differs from reference")
            binary = max(pixels) <= 1
            visible = bytes(value*255 for value in pixels) if binary else pixels
            tile = Image.frombytes("L", (case["width"], case["height"]), visible)
            tile.thumbnail((120, 120), Image.Resampling.NEAREST)
            if max(tile.size) < 120:
                scale = 120 // max(tile.size)
                tile = tile.resize((tile.width*scale, tile.height*scale), Image.Resampling.NEAREST)
            image.paste(tile, (x, top+10))
            outputs.append({"variant": variant, "stream_sha256": case["stream_sha256"],
                            "decoded_sha256": case["expected_sha256"], "output_sha256": digest(pixels),
                            "display_mapping": "binary 0/1 to 0/255" if binary else "literal u8"})
        captures.append({"case": name, "output": label, "variants": outputs})
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    output.with_suffix(".json").write_text(json.dumps({"suite_sha256": digest(suite_bytes),
        "comparison_sha256": digest(report_bytes), "capture_sha256": digest(output.read_bytes()),
        "capture_script_sha256": digest(Path(__file__).read_bytes()),
        "source": "Checked full FFmpeg I420 bytes; named output regions cross-checked against old graph reference",
        "timings": "not measured", "captures": captures}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.bundle, args.output)
