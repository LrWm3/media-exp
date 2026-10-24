#!/usr/bin/env python3
"""Validate every prepared IR comparison stream, including packed output regions.

Uses Python stdlib plus decoder_core/decode.py, without compiler or demo imports.
Example: python3 validate.py out/bundle --backend nvdec --helper /path/nvdec_decode
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


for parent in Path(__file__).resolve().parents:
    if (parent / "decoder_core" / "decode.py").exists():
        sys.path.insert(0, str(parent))
        break
from decoder_core.decode import available, decode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(root, relative, checksum):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Bundle path escapes directory")
    data = path.read_bytes()
    if digest(data) != checksum:
        raise AssertionError("Bundle hash mismatch: " + relative)
    return data


def validate(root, backend):
    root = Path(root).resolve()
    suite_bytes = (root / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    checked(root, "comparison.json", suite["comparison_sha256"])
    if not available(backend):
        raise RuntimeError("Requested decoder unavailable: " + backend)
    results = []
    for case in suite["cases"]:
        for key in ("stream", "manifest", "expected"):
            checked(root, case[key], case[key + "_sha256"])
        actual = decode(root / case["stream"], backend)
        if len(actual) != case["frame_bytes"]*case["pictures"] or digest(actual) != case["expected_sha256"]:
            raise AssertionError(case["name"] + ": full I420 mismatch")
        for label, info in case["named_outputs"].items():
            expected = checked(root, info["path"], info["sha256"])
            start = info["frame"]*case["frame_bytes"]
            pixels = b"".join(actual[start+(row+info["y"])*case["coded_width"]+info["x"]:
                                     start+(row+info["y"])*case["coded_width"]+info["x"]+case["width"]]
                              for row in range(case["height"]))
            if pixels != expected:
                raise AssertionError(case["name"] + "/" + label + ": named output mismatch")
        results.append({"name": case["name"], "status": "pass", "pictures": case["pictures"],
                        "named_outputs": len(case["named_outputs"]), "actual_sha256": digest(actual)})
        print("PASS " + case["name"], flush=True)
    return {"status": "pass", "backend": backend, "suite_sha256": digest(suite_bytes),
            "validator_sha256": digest(Path(__file__).read_bytes()),
            "decode_module_sha256": digest(Path(sys.modules[decode.__module__].__file__).read_bytes()),
            "stream_count": len(results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--backend", choices=("ffmpeg", "openh264", "vaapi", "nvdec"), default="nvdec")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.helper:
        os.environ["H264_NVDEC_DECODER"] = str(args.helper.resolve())
    report = validate(args.bundle, args.backend)
    if args.helper:
        report["helper_sha256"] = digest(args.helper.read_bytes())
    output = args.output or args.bundle / (args.backend + "-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
