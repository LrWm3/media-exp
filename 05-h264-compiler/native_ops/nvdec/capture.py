#!/usr/bin/env python3
"""Render verified NVDEC PGM readbacks; retain the original measurement reports."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import contact_sheet
from demos import cases


def capture(results, suite, output):
    report = json.loads((results / "results.json").read_text())
    if report["summary"]["passed"] != report["summary"]["cases"]:
        raise ValueError("Cannot publish a successful capture from a failed suite")
    if hashlib.sha256(suite.read_bytes()).hexdigest() != report["suite_sha256"]:
        raise ValueError("Suite does not match the measured report")
    measured = {case["name"]: case for case in report["cases"]}
    for name, title, _, _, inputs in cases():
        case = measured["demo-" + name]
        boards = inputs if isinstance(inputs, dict) else {"input": inputs}
        panels = [("HOST / " + label, board) for label, board in boards.items()]
        for label, info in case["named_outputs"].items():
            with Image.open(results / info["capture"]) as image:
                if image.mode != "L" or image.size != (case["width"], case["height"]):
                    raise ValueError("Unexpected capture dimensions or pixel format")
                pixels = image.tobytes()
            checksum = hashlib.sha256(pixels).hexdigest()
            if checksum != info["sha256"] or checksum != info["expected_sha256"]:
                raise ValueError("Capture pixels differ from verified report: " + label)
            width = case["width"]
            board = [list(pixels[y*width:(y+1)*width]) for y in range(case["height"])]
            panels.append(("NVDEC / " + label, board))
        contact_sheet(title,
                      "NVDEC | %d pictures | exact FFmpeg + integer-oracle match"
                      % case["pictures"], panels, output / "assets" / (name + ".png"))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(results / "results.json", output / "results.json")
    shutil.copyfile(suite, output / "suite.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    capture(args.results, args.suite, args.output)


if __name__ == "__main__":
    main()
