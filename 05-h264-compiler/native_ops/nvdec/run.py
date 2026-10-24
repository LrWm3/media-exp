#!/usr/bin/env python3
"""Validate a prepared bundle on NVDEC using only Python 3.8's standard library."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (parent / "decoder_core" / "decode.py").exists():
        sys.path.insert(0, str(parent))
        break
from decoder_core.decode import available, decode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(bundle, relative, expected_hash):
    path = (bundle / relative).resolve()
    if bundle not in path.parents:
        raise ValueError("Bundle path escapes its directory")
    data = path.read_bytes()
    if digest(data) != expected_hash:
        raise AssertionError("Bundle file hash differs: " + relative)
    return data


def command_output(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=10, universal_newlines=True)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def provenance(helper):
    return {"decoder": "NVDEC", "python": platform.python_version(),
            "readback": "Same-color-format surface copy; CPU plane copy/reorder",
            "helper_sha256": digest(helper.read_bytes()),
            "decode_module_sha256": digest(Path(sys.modules[decode.__module__].__file__).read_bytes()),
            "power_or_clock_settings_changed": False}


def plane(raw, case, frame):
    start = frame*case["frame_bytes"]
    width, height, stride, guard = case["width"], case["height"], case["coded_width"], case["guard"]
    return b"".join(raw[start+(y+guard)*stride+guard:start+(y+guard)*stride+guard+width]
                    for y in range(height))


def validate_case(bundle, case, output):
    checked_file(bundle, case["stream"], case["stream_sha256"])
    expected = checked_file(bundle, case["expected"], case["expected_sha256"])
    started = time.perf_counter()
    actual = decode(bundle / case["stream"], "nvdec")
    elapsed = (time.perf_counter()-started)*1000
    result = {key: case[key] for key in ("name", "kind", "width", "height", "coded_width", "coded_height",
                                        "pictures", "stream_sha256", "expected_bytes", "expected_sha256")}
    result.update({"pipeline_ms": elapsed, "actual_bytes": len(actual), "actual_sha256": digest(actual),
                   "status": "pass" if actual == expected else "mismatch", "named_outputs": {}})
    if actual != expected:
        result["different_bytes"] = sum(a != b for a, b in zip(actual, expected)) + abs(len(actual)-len(expected))
        result["first_differences"] = []
        for index, (a, b) in enumerate(zip(actual, expected)):
            if a != b:
                result["first_differences"].append({"byte": index, "actual": a, "expected": b})
                if len(result["first_differences"]) == 10:
                    break
    for index, (label, info) in enumerate(case["named_outputs"].items()):
        independent = checked_file(bundle, info["path"], info["sha256"])
        values = plane(actual, case, info["frame"])
        if values != independent:
            result["status"] = "mismatch"
        path = output / "captures" / case["name"] / ("%03d.pgm" % index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(values) == case["width"]*case["height"]:
            path.write_bytes(("P5\n%d %d\n255\n" % (case["width"], case["height"])).encode() + values)
        result["named_outputs"][label] = {"frame": info["frame"], "sha256": digest(values),
                                           "expected_sha256": info["sha256"], "pixels": len(values),
                                           "status": "pass" if values == independent else "mismatch",
                                           "capture": str(path.relative_to(output))}
    return result


def benchmark(bundle, cases, repeats):
    selected = [case for case in cases if case["kind"] == "benchmark"]
    expected = {case["name"]: checked_file(bundle, case["expected"], case["expected_sha256"])
                for case in selected}
    samples = {case["name"]: [] for case in selected}
    # One untimed warmup each, followed by alternating order to limit order bias.
    for iteration in range(repeats+1):
        for case in (selected if iteration % 2 == 0 else list(reversed(selected))):
            started = time.perf_counter()
            actual = decode(bundle / case["stream"], "nvdec")
            if actual != expected[case["name"]]:
                raise AssertionError("Benchmark failed exact comparison: " + case["name"])
            elapsed = (time.perf_counter()-started)*1000
            if iteration:
                samples[case["name"]].append(elapsed)
    return {"repeats": repeats, "warmup_per_case": 1,
            "timing": "On-device process startup, parsing, NVDEC, VIC readback, file I/O, diagnostics, byte comparison",
            "excluded": "SSH/network transfer, graph compilation, input packing, reference file loading",
            "cases": {case["name"]: {"samples_ms": samples[case["name"]],
                                       "median_ms": statistics.median(samples[case["name"]]),
                                       "min_ms": min(samples[case["name"]]),
                                       "max_ms": max(samples[case["name"]]),
                                       "pictures": case["pictures"], "logical_additions": 32,
                                       "actual_sha256": digest(expected[case["name"]])}
                      for case in selected}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    bundle, output, helper = args.bundle.resolve(), args.output.resolve(), args.helper.resolve()
    os.environ["H264_NVDEC_DECODER"] = str(helper)
    if not available("nvdec"):
        raise RuntimeError("NVDEC helper or NVIDIA GStreamer decoder unavailable")
    output.mkdir(parents=True, exist_ok=True)
    suite_bytes = (bundle / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "suite_sha256": digest(suite_bytes),
              "source_commit": suite["repository_commit"], "provenance": provenance(helper), "cases": []}
    for case in suite["cases"]:
        try:
            result = validate_case(bundle, case, output)
        except Exception as error:
            result = {"name": case["name"], "kind": case["kind"], "status": "error",
                      "error": str(error)}
            if isinstance(error, subprocess.CalledProcessError):
                result["stderr"] = (error.stderr or b"").decode(errors="replace")
                result["stdout"] = (error.stdout or b"").decode(errors="replace")
        report["cases"].append(result)
        print("%s %s" % (result["status"].upper(), case["name"]), flush=True)
        (output / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    report["summary"] = {"cases": len(report["cases"]),
                         "passed": sum(case["status"] == "pass" for case in report["cases"]),
                         "verified_pictures": sum(case.get("pictures", 0) for case in report["cases"] if case["status"] == "pass"),
                         "verified_raw_bytes": sum(case.get("actual_bytes", 0) for case in report["cases"] if case["status"] == "pass")}
    if report["summary"]["passed"] == len(report["cases"]):
        report["benchmark"] = benchmark(bundle, suite["cases"], args.repeats)
    (output / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"]), flush=True)
    return int(report["summary"]["passed"] != len(report["cases"]))


if __name__ == "__main__":
    sys.exit(main())
