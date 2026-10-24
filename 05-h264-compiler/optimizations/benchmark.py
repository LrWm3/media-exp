#!/usr/bin/env python3
"""Compare saved and optimized streams using the same decoder and input pixels."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

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
        raise ValueError("Bundle hash mismatch: " + relative)
    return path, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvdec", "ffmpeg", "vaapi", "openh264"), default="nvdec")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.helper:
        os.environ["H264_NVDEC_DECODER"] = str(args.helper.resolve())
    if not available(args.backend):
        raise RuntimeError("Requested decoder unavailable: " + args.backend)
    root = args.bundle.resolve()
    suite_bytes = (root / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    representatives = suite["representatives"]
    if not representatives:
        raise ValueError("No representative streams")
    cases = {}
    for name, case in representatives.items():
        cases[name] = {}
        for variant in ("baseline", "optimized"):
            item = case[variant]
            stream, _ = checked(root, item["stream"], item["stream_sha256"])
            _, expected = checked(root, item["expected"], item["expected_sha256"])
            if len(expected) != item["pictures"] * case["frame_bytes"]:
                raise ValueError("Expected picture count mismatch")
            cases[name][variant] = stream, expected
    report = {"format": "h264-optimization-timings-0.1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
              "python": platform.python_version(), "suite_sha256": digest(suite_bytes),
              "runner_sha256": digest(Path(__file__).read_bytes()),
              "helper_sha256": digest(args.helper.read_bytes()) if args.helper else None,
              "repeats": args.repeats, "warmups_per_variant": 1,
              "timed": "process startup, parsing, decoding, full readback, file I/O, diagnostics and exact byte comparison",
              "excluded": "network transfer, compilation, input packing, loading reference files",
              "order": "rotating group order; alternating baseline/optimized within every group",
              "cases": {name: {variant: {"samples_ms": [], **representatives[name][variant]}
                                for variant in ("baseline", "optimized")} for name in cases}}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, variant):
        stream, expected = cases[name][variant]
        started = time.perf_counter()
        actual = decode(stream, args.backend)
        if actual != expected:
            raise AssertionError("Decoded pixels differ: " + name + "/" + variant)
        return (time.perf_counter() - started) * 1000

    save()
    for name in cases:
        for variant in ("baseline", "optimized"):
            run(name, variant)
    names = list(cases)
    for iteration in range(args.repeats):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            order = ("baseline", "optimized") if iteration % 2 == 0 else ("optimized", "baseline")
            for variant in order:
                report["cases"][name][variant]["samples_ms"].append(run(name, variant))
        save()
        print("PASS measured repetition %d/%d" % (iteration + 1, args.repeats), flush=True)
    for case in report["cases"].values():
        for variant in ("baseline", "optimized"):
            values = case[variant]["samples_ms"]
            case[variant].update(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
        case["speedup"] = case["baseline"]["median_ms"] / case["optimized"]["median_ms"]
    report["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
