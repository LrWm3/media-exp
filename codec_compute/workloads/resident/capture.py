"""Render actual frozen decoder outputs, bound to complete I420 references."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .benchmark import load


def capture(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    bundle_bytes, bundle, workloads = load(root)
    sizes = sorted(bundle["sizes"])
    image = Image.new("RGB", (1150, 95+335*len(sizes)), "#111827")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("DejaVuSans.ttf", 24)
    text = ImageFont.truetype("DejaVuSans.ttf", 16)
    draw.text((24, 14), "Larger inputs, identical fused interpolation", fill="white", font=title)
    draw.text((24, 49), "Actual decoded planes • supplied static motion • runtime IDR fence is discarded", fill="#aab7ca", font=text)
    evidence = {"bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(), "rows": []}
    for row, size in enumerate(sizes):
        base = workloads[f"{size}/motion-interpolation/baseline"]
        fused = workloads[f"{size}/motion-interpolation/fused"]
        bjob, fjob = base["jobs"][0], fused["jobs"][0]
        shape = fused["shape"]
        def plane(frame):
            w = shape["coded_width"]
            return b"".join(frame[(y+16)*w+16:(y+16)*w+16+size] for y in range(size))
        baseline = plane(bjob["frames"][base["case"]["output_frames"]["output"]])
        optimized = plane(fjob["frames"][fused["case"]["output_frames"]["output"]])
        if baseline != optimized:
            raise AssertionError("Baseline and fused decoder outputs differ")
        columns = (("Input A", plane(fjob["frames"][0])), ("Input B", plane(fjob["frames"][1])),
                   ("Unfused: 3 compute pictures", baseline), ("Fused: 1 compute picture", optimized))
        y = 95+row*335
        draw.text((24, y), f"{size}×{size} active / {shape['coded_width']}×{shape['coded_height']} coded · High {fused['case']['level_idc']/10:.1f}",
                  fill="#83d9bb", font=text)
        hashes = {}
        for col, (label, data) in enumerate(columns):
            x = 24+col*282
            draw.text((x, y+28), label, fill="white", font=text)
            panel = Image.frombytes("L", (size, size), data).resize((256, 256), Image.Resampling.NEAREST)
            image.paste(panel.convert("RGB"), (x, y+55))
            hashes[label] = hashlib.sha256(data).hexdigest()
        evidence["rows"].append({"size": size, "fixture": 0, "displayed_plane_sha256": hashes,
                                 "baseline_raw": base["case"]["jobs"][0]["expected"],
                                 "fused_raw": fused["case"]["jobs"][0]["expected"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    evidence["capture_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps(evidence, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.bundle, args.output)
