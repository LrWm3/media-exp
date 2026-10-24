"""Render verified workload inputs and actual decoded output planes.

Run: python3 -m codec_compute.workloads.capture BUNDLE --output capture.png
The optional --report-sha256 pins the expected validation report. Every displayed
plane and its stream is hash-checked, and the report digest is always recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROWS = (
    ("absolute-change", "h264", "Absolute change", "Magnitude and change mask, threshold >=24",
     (("inputs", "a", "Current"), ("inputs", "b", "Reference"),
      ("outputs", "output", "Magnitude"), ("outputs", "mask", "Mask >=24"))),
    ("temporal-envelope", "hevc", "Temporal envelope", "Two-frame minimum, maximum, and observed range",
     (("inputs", "a", "Current"), ("inputs", "b", "Reference"),
      ("outputs", "low", "Minimum"), ("outputs", "high", "Maximum"),
      ("outputs", "output", "Range"))),
    ("dilate-clamp", "av1", "Binary dilation", "One four-connected expansion; clamp borders",
     (("inputs", "image", "Input mask"), ("outputs", "output", "Expanded mask"))),
    ("smooth-clamp", "av1", "Spatial smoothing", "Separable three-tap averages; nested rounding, clamp borders",
     (("inputs", "image", "Input"), ("outputs", "output", "Smoothed"))),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(bundle, relative, checksum):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact path must be a nonempty relative string")
    path = (bundle / relative).resolve()
    if Path(relative).is_absolute() or bundle not in path.parents:
        raise ValueError("Artifact path escapes bundle: " + relative)
    data = path.read_bytes()
    if digest(data) != checksum:
        raise AssertionError("Artifact hash mismatch: " + relative)
    return data


def plane(bundle, record):
    width, height = record["width"], record["height"]
    if any(type(value) is not int or value < 1 for value in (width, height)):
        raise ValueError("Plane dimensions must be positive integers")
    lo, hi = record["range"]
    if any(type(value) is not int for value in (lo, hi)) or not 0 <= lo <= hi <= 255:
        raise ValueError("Plane range must lie inside u8")
    values = checked(bundle, record["path"], record["sha256"])
    if len(values) != width*height or record.get("bytes", len(values)) != len(values):
        raise AssertionError("Plane size differs from its declared dimensions")
    if any(not lo <= value <= hi for value in values):
        raise AssertionError("Plane pixels violate their declared range")
    return values


def capture(bundle, output, *, report_sha256=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("Capture output must use a .png filename")
    report_path = bundle / "validation.json"
    report_bytes = report_path.read_bytes()
    report_digest = digest(report_bytes)
    if report_sha256 is not None and report_digest != report_sha256:
        raise AssertionError("Validation report hash mismatch")
    report = json.loads(report_bytes)
    if report.get("status") != "pass":
        raise AssertionError("Capture requires a passing validation report")
    prepared = []
    for task, target, title, description, columns in ROWS:
        matches = [case for case in report["cases"] if (case["task"], case["target"]) == (task, target)]
        if len(matches) != 1 or matches[0].get("status") != "pass":
            raise AssertionError(f"Missing unique passing case: {task}/{target}")
        case = matches[0]
        variants = [variant for variant in case["variants"] if type(variant["index"]) is int and variant["index"] == 0]
        if len(variants) != 1:
            raise AssertionError(f"Missing unique fixture variant zero: {task}/{target}")
        variant = variants[0]
        stream = checked(bundle, variant["stream"], variant["stream_sha256"])
        if variant.get("stream_bytes", len(stream)) != len(stream):
            raise AssertionError("Stream size differs from report")
        for key in ("manifest", "expected"):
            if key in variant:
                checked(bundle, variant[key], variant[key + "_sha256"])
        if any(item.get("status") != "pass" for item in variant.get("decoders", [])):
            raise AssertionError("A selected stream failed decoder validation")
        boards = {(kind, name): plane(bundle, record)
                  for kind in ("inputs", "outputs") for name, record in variant[kind].items()}
        costs = case["costs"]
        if any(type(costs[key]) is not int or costs[key] < 0 for key in
               ("input_pictures", "computation_pictures", "total_pictures", "prediction_bytes")):
            raise ValueError("Stream costs must be nonnegative integers")
        if costs["input_pictures"] + costs["computation_pictures"] != costs["total_pictures"]:
            raise AssertionError("Picture costs do not add up")
        prepared.append((task, target, title, description, columns, variant, boards, costs))

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    canvas = Image.new("RGB", (1320, 1210), "#f1f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Shared IR, actual decoder outputs", font=font(31), fill="#142634")
    draw.text((28, 70), "H.264 change detection  |  HEVC temporal envelope  |  AV1 dilation and smoothing",
              font=font(17), fill="#4c5f6e")
    draw.text((28, 99), "Each row uses saved input pixels and verified named outputs from fixture zero.",
              font=font(16), fill="#4c5f6e")
    provenance = []
    for index, (task, target, title, description, columns, variant, boards, costs) in enumerate(prepared):
        top = 142+index*246
        draw.rounded_rectangle((18, top, 1302, top+232), radius=10, fill="white")
        draw.text((34, top+17), title, font=font(22), fill="#142634")
        draw.text((34, top+49), {"h264": "H.264", "hevc": "HEVC", "av1": "AV1"}[target],
                  font=font(18), fill="#206747")
        draw.text((34, top+82), f"{costs['total_pictures']} pictures / {costs['computation_pictures']} compute",
                  font=font(16), fill="#4c5f6e")
        draw.text((34, top+107), f"{costs['prediction_bytes']:,} prediction bytes", font=font(16), fill="#4c5f6e")
        for line_index, line in enumerate(textwrap.wrap(description, width=34)):
            draw.text((34, top+147+line_index*22), line, font=font(15), fill="#4c5f6e")
        planes = []
        for column, (kind, name, label) in enumerate(columns):
            record = variant[kind][name]
            values = boards[(kind, name)]
            width, height = record["width"], record["height"]
            zoom = 160 // max(width, height)
            if zoom < 1:
                raise ValueError("Capture requires planes no larger than 160 pixels per side")
            binary = record["range"][1] <= 1
            visible = bytes(value*255 for value in values) if binary else values
            tile = Image.frombytes("L", (width, height), visible)
            tile = tile.resize((width*zoom, height*zoom), Image.Resampling.NEAREST)
            left = 360+column*184+(160-tile.width)//2
            y = top+50+(160-tile.height)//2
            draw.text((360+column*184+80, top+16), label, font=font(16), fill="#314b5c", anchor="mt")
            draw.rectangle((left-1, y-1, left+tile.width, y+tile.height), outline="#c5cfd6")
            canvas.paste(tile, (left, y))
            planes.append({"kind": kind, "name": name, **record, "integer_zoom": zoom,
                           "display_mapping": "binary 0/1 to 0/255" if binary else "literal u8"})
        provenance.append({"task": task, "target": target, "variant": 0, "costs": costs,
                           "stream": variant["stream"], "stream_sha256": variant["stream_sha256"],
                           "manifest_sha256": variant.get("manifest_sha256"),
                           "decoders": variant.get("decoders", []), "planes": planes})
    draw.text((28, 1140), "Binary planes: 0/1 shown as black/white. Byte planes retain original intensity. Integer nearest-neighbor enlargement.",
              font=font(16), fill="#4c5f6e")
    draw.text((28, 1170), "Picture and byte counts are static stream costs. Decoder latency was not measured.",
              font=font(16), fill="#4c5f6e")
    if report_path.read_bytes() != report_bytes:
        raise AssertionError("Validation report changed during capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    metadata = {"format": "codec-compute-workload-capture-1", "report": "validation.json",
                "report_sha256": report_digest, "report_hash_pinned": report_sha256 is not None,
                "repository_commit": report.get("repository_commit"),
                "capture_script_sha256": digest(Path(__file__).read_bytes()),
                "capture_sha256": digest(output.read_bytes()), "size": list(canvas.size),
                "source": "Hash-checked actual fixture planes and decoded named output planes",
                "timings": "not measured", "rows": provenance}
    output.with_suffix(".capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-sha256", help="Expected SHA256 of validation.json")
    args = parser.parse_args()
    capture(args.bundle, args.output, report_sha256=args.report_sha256)
