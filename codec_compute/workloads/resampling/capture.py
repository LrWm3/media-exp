"""Render actual decoded resampling outputs, with hash-checked provenance."""

import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from ..capture import checked, digest, plane
from ..temporal_benchmark import extract, geometry


LABELS = {"image": "Input", "a": "Input A", "b": "Input B", "output": "Decoded output",
          "positive": "Positive >=20", "negative": "Negative >=20", "cost_mhalf": "Cost -0.5 px",
          "cost_phalf": "Cost +0.5 px", "cost_pthreehalf": "Cost +1.5 px",
          "compensated": "Compensated", "naive": "Naive average", "midpoint": "Analytic midpoint"}


def bind_outputs(root, manifest_record, expected_record, outputs, decoders, inputs):
    """Require decoder evidence and bind every output plane to actual I420."""
    if not decoders or any(record.get("status") != "pass" or record.get("i420_sha256") != expected_record["sha256"]
                           for record in decoders):
        raise AssertionError("Expected nonempty passing decoder evidence for the full I420 hash")
    manifest = json.loads(checked(root, manifest_record["path"], manifest_record["sha256"]))
    raw = checked(root, expected_record["path"], expected_record["sha256"])
    shape = geometry(manifest)
    if len(raw) != shape["frame_bytes"]*shape["pictures_per_job"] or set(outputs) != set(shape["output_frames"]):
        raise AssertionError("Full I420 or named output metadata differs from manifest")
    for name, record in outputs.items():
        index = shape["output_frames"][name]
        frame = raw[index*shape["frame_bytes"]:(index+1)*shape["frame_bytes"]]
        if (record["width"], record["height"]) != (shape["width"], shape["height"]) or plane(root, record) != extract(frame, shape):
            raise AssertionError("Displayed output plane differs from its actual decoded I420 crop")
    names = [item["name"] for item in manifest["ir"]["inputs"]]
    if set(names) != set(inputs):
        raise AssertionError("Input names differ from manifest")
    for index, name in enumerate(names):
        record = inputs[name]
        frame = raw[index*shape["frame_bytes"]:(index+1)*shape["frame_bytes"]]
        if (record["width"], record["height"]) != (shape["width"], shape["height"]) or plane(root, record) != extract(frame, shape):
            raise AssertionError("Displayed input differs from actual decoded input picture")


def capture(bundle, output):
    root, output = Path(bundle).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("Capture output must be PNG")
    report_bytes = (root / "bundle.json").read_bytes()
    report = json.loads(report_bytes)
    if report.get("format") != "codec-compute-resampling-1" or report.get("status") != "pass":
        raise AssertionError("Expected passing resampling bundle")
    rows = []
    for name in ("patch-atlas", "motion-interpolation", "rolling-shutter", "motion-residual", "bipolar-events", "stereo-costs"):
        case = report["cases"][name + "/fused"]
        job = case["jobs"][0]
        if job["index"] != 0:
            raise AssertionError("Expected passing fixture zero")
        bind_outputs(root, case["manifest"], job["expected"], job["outputs"], job["decoders"], job["inputs"])
        for record in (case["manifest"], case["fields"], job["stream"], job["expected"]):
            checked(root, record["path"], record["sha256"])
        columns = [(LABELS[key], value) for key, value in job["inputs"].items()]
        columns += [(LABELS[key], value) for key, value in job["outputs"].items()]
        before = report["cases"][name + "/baseline"]["costs"]["computation_pictures"]
        after = case["costs"]["computation_pictures"]
        note = "%d -> %d compute pictures" % (before, after)
        if name == "patch-atlas":
            note += "; same coded size"
        rows.append((name, case["title"], case["description"], note, columns, job["stream"], case["costs"]))
    quality = report["quality"]
    bind_outputs(root, quality["manifest"], quality["expected"], quality["outputs"], quality["decoders"], quality["inputs"])
    for record in (quality["stream"], quality["manifest"], quality["expected"]):
        checked(root, record["path"], record["sha256"])
    columns = [(LABELS[key], record) for key, record in quality["inputs"].items()]
    columns += [(LABELS[key], quality["outputs"][key]) for key in ("naive", "compensated")]
    columns.append((LABELS["midpoint"], quality["midpoint"]))
    metrics = quality["metrics"]
    note = "Interior MAE %.2f -> %.2f" % (metrics["naive"]["mae"], metrics["compensated"]["mae"])
    rows.insert(2, ("known-motion", "Known-motion witness", "Analytic texture, +/-2.5 px translation; supplied half-pel correction.",
                    note, columns, quality["stream"], quality["costs"]))
    width, height = 1330, 128+len(rows)*205+66
    canvas = Image.new("RGB", (width, height), "#f1f4f6")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    def font(size):
        return ImageFont.truetype(font_path, size)
    draw.text((25, 20), "Static motion fields, actual decoder outputs", font=font(30), fill="#142634")
    draw.text((25, 62), "H.264  |  supplied fields  |  exact byte arithmetic  |  all panels use saved fixture pixels",
              font=font(17), fill="#496171")
    draw.text((25, 91), "Known-motion midpoint is analytic ground truth; every other output panel comes from the decoder.",
              font=font(15), fill="#496171")
    provenance = []
    for index, (key, title, description, note, columns, stream, costs) in enumerate(rows):
        top = 128+index*205
        draw.rounded_rectangle((16, top, width-16, top+193), radius=10, fill="white")
        draw.text((30, top+15), title, font=font(21), fill="#142634")
        for line, text in enumerate(textwrap.wrap(description, 33)):
            draw.text((30, top+49+line*21), text, font=font(14), fill="#496171")
        for line, text in enumerate(textwrap.wrap(note, 35)):
            draw.text((30, top+137+line*19), text, font=font(14), fill="#206747")
        records = []
        for column, (label, record) in enumerate(columns):
            values = plane(root, record)
            zoom = 144//max(record["width"], record["height"])
            if zoom < 1:
                raise ValueError("Capture requires inputs at most 144 pixels per side")
            binary = record["range"][1] <= 1
            pixels = bytes(v*255 for v in values) if binary else values
            tile = Image.frombytes("L", (record["width"], record["height"]), pixels)
            tile = tile.resize((tile.width*zoom, tile.height*zoom), Image.Resampling.NEAREST)
            center = 363+column*183+72
            left, y = center-tile.width//2, top+39+(144-tile.height)//2
            draw.text((center, top+12), label, font=font(14), fill="#314b5c", anchor="mt")
            draw.rectangle((left-1, y-1, left+tile.width, y+tile.height), outline="#c5cfd6")
            canvas.paste(tile, (left, y))
            records.append({"label": label, **record, "integer_zoom": zoom,
                            "display": "binary0/1-to-0/255" if binary else "literal u8"})
        provenance.append({"case": key, "stream": stream, "costs": costs, "planes": records})
    draw.text((25, height-54), "Binary maps use black/white. Byte planes retain literal intensity. Integer nearest-neighbor display.",
              font=font(15), fill="#496171")
    draw.text((25, height-29), "Fields are compile-time constants; no estimated flow, occlusion solve, asynchronous sensor, or smaller coded surface.",
              font=font(15), fill="#496171")
    if (root / "bundle.json").read_bytes() != report_bytes:
        raise AssertionError("Bundle changed during capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    metadata = {"format": "codec-compute-resampling-capture-1", "bundle_sha256": digest(report_bytes),
                "repository_commit": report["repository_commit"], "script_sha256": digest(Path(__file__).read_bytes()),
                "capture_sha256": digest(output.read_bytes()), "size": [width, height], "rows": provenance}
    output.with_suffix(".capture.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.bundle, args.output)
