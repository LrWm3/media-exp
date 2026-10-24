#!/usr/bin/env python3
"""Verify and time four execution/readback modes, including on Python 3.8 NVDEC."""

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

for parent in Path(__file__).resolve().parents:
    if (parent/"decoder_core"/"decode.py").exists():
        sys.path.insert(0, str(parent))
        break
from decoder_core.decode import _nvdec_helper, available, decode_batch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(root, relative, checksum):
    path = (root/relative).resolve()
    if root not in path.parents:
        raise ValueError("Bundle path escapes directory")
    data = path.read_bytes()
    if digest(data) != checksum:
        raise ValueError("Input hash mismatch: " + relative)
    return data


def read_text(path):
    try:
        return Path(path).read_text().strip("\x00\n")
    except OSError:
        return "unavailable"


def provenance(backend, helper):
    return {"python": platform.python_version(), "backend": backend,
            "power_or_clock_settings_changed": False,
            "helper_sha256": digest(helper.read_bytes()) if helper else None,
            "decode_module_sha256": digest(Path(sys.modules[decode_batch.__module__].__file__).read_bytes())}


def active_plane(raw_frame, bundle):
    width, height, stride, guard = (bundle[key] for key in ("width", "height", "coded_width", "guard"))
    return b"".join(raw_frame[(y+guard)*stride+guard:(y+guard)*stride+guard+width] for y in range(height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--backend", choices=("nvdec", "ffmpeg", "vaapi", "openh264"), default="nvdec")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.helper or args.backend == "nvdec":
        args.helper = args.helper.resolve() if args.helper else _nvdec_helper()
        os.environ["H264_NVDEC_DECODER"] = str(args.helper)
    if not available(args.backend):
        raise RuntimeError("Requested backend is unavailable: " + args.backend)
    root, output = args.bundle.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_bytes = (root/"bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    jobs = bundle["jobs"]
    if not jobs or len(jobs) != bundle["job_count"]:
        raise ValueError("Invalid bundle job count")
    paths, expected = [], []
    for job in jobs:
        checked_file(root, job["stream"], job["stream_sha256"])
        raw = checked_file(root, job["expected"], job["expected_sha256"])
        if len(raw) != bundle["pictures_per_job"]*bundle["frame_bytes"]:
            raise ValueError("Reference picture count mismatch")
        paths.append(root/job["stream"])
        expected.append(raw)
    options = {"width": bundle["coded_width"], "height": bundle["coded_height"],
               "pictures_per_job": bundle["pictures_per_job"]}
    final_frame = bundle["output_frames"]["output"]
    frame_bytes = bundle["frame_bytes"]
    expected_final = [raw[final_frame*frame_bytes:(final_frame+1)*frame_bytes] for raw in expected]
    report = {"format": "h264-motion-results-0.1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "bundle_sha256": digest(bundle_bytes),
              "provenance": provenance(args.backend, args.helper), "jobs": [],
              "benchmark": {"repeats": args.repeats, "warmups_per_mode": 1, "modes": {},
                            "timed": "batch assembly/file I/O, process/pipeline startup, parsing, decode, selected surface copies, readback, diagnostics and exact byte comparisons",
                            "excluded": "SSH/network, compilation, PCM input packing, loading reference files and capture rendering",
                            "order": "rotating modes each repetition; one untimed warmup per mode"}}
    def save():
        (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    # One full readback proves every intermediate and supplies actual captures.
    captured = decode_batch(paths, args.backend, **options)
    for index, (job, frames, reference) in enumerate(zip(jobs, captured.frames, expected)):
        raw = b"".join(frames)
        if raw != reference:
            raise AssertionError("Full capture differs for job %d" % index)
        entry = {"index": index, "status": "pass", "raw_sha256": digest(raw), "named_outputs": {}}
        for name, info in job["named_outputs"].items():
            values = active_plane(frames[info["frame"]], bundle)
            if digest(values) != info["sha256"]:
                raise AssertionError("Named output differs: %d/%s" % (index, name))
            path = output/"captures"/("%03d" % index)/(name+".pgm")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("P5\n%d %d\n255\n" % (bundle["width"], bundle["height"])).encode()+values)
            entry["named_outputs"][name] = {"sha256": digest(values), "expected_sha256": info["sha256"],
                                             "capture": str(path.relative_to(output)), "status": "pass",
                                             "nonzero_pixels": sum(value != 0 for value in values)}
        report["jobs"].append(entry)
    report["validation"] = {"jobs": len(jobs), "pictures": captured.decoded_frames,
                             "raw_bytes": sum(map(len, expected)), "named_outputs": sum(len(j["named_outputs"]) for j in jobs)}
    save()
    print("PASS full capture: %d jobs, %d pictures" % (len(jobs), captured.decoded_frames), flush=True)

    def execute(persistent, selective):
        selection = (final_frame,) if selective else None
        decoded, copied, returned = 0, 0, 0
        groups = [(paths, expected, expected_final)] if persistent else [([path], [raw], [final]) for path, raw, final in zip(paths, expected, expected_final)]
        for batch_paths, raw_references, final_references in groups:
            result = decode_batch(batch_paths, args.backend, selected_frames=selection, **options)
            decoded += result.decoded_frames
            copied += result.copied_frames
            for frames, full, final in zip(result.frames, raw_references, final_references):
                actual = b"".join(frames)
                if actual != (final if selective else full):
                    raise AssertionError("Timed workload differs from reference")
                returned += len(actual)
        return {"decoder_sessions": len(groups), "decoded_pictures": decoded,
                "copied_pictures": copied, "copied_bytes": copied*frame_bytes, "returned_bytes": returned}

    modes = [("restart-all", False, False), ("restart-final", False, True),
             ("batch-all", True, False), ("batch-final", True, True)]
    samples = {name: [] for name, _, _ in modes}
    for iteration in range(args.repeats+1):
        ordered = modes[iteration % len(modes):]+modes[:iteration % len(modes)]
        for name, persistent, selective in ordered:
            started = time.perf_counter()
            counters = execute(persistent, selective)
            elapsed = (time.perf_counter()-started)*1000
            if iteration:
                samples[name].append(elapsed)
                report["benchmark"]["modes"][name] = {**counters, "samples_ms": samples[name],
                    "median_ms": statistics.median(samples[name]), "min_ms": min(samples[name]),
                    "max_ms": max(samples[name]), "status": "pass"}
                save()
            print("PASS %s %s: %.2f ms" % (name, "warmup" if not iteration else "run %d" % iteration, elapsed), flush=True)
    report["status"] = "pass"
    save()
    print(json.dumps(report["validation"]), flush=True)


if __name__ == "__main__":
    main()
