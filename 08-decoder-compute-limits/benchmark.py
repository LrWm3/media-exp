#!/usr/bin/env python3
"""Portable decoder validation and timed Boolean recurrence microbenchmarks.

Python 3.8+, standard library, and decoder_core.decode suffice. There is no
compiler, search, or host implementation of the Boolean computation here.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from decoder_core.decode import available, decode_batch, _nvdec_helper


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def checked(root, relative, sha):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Bundle path escapes its directory")
    data = path.read_bytes()
    if digest(data) != sha:
        raise ValueError("Bundle checksum mismatch: "+relative)
    return data


def extract(raw, region, width):
    left, top = region["x"]+16, region["y"]+16
    return b"".join(raw[(top+y)*width+left:(top+y)*width+left+region["width"]]
                    for y in range(region["height"]))


def load_bundle(root):
    bundle_bytes = (root/"bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle["format"] != "h264-compute-limits-bundle-0.1":
        raise ValueError("Unsupported bundle format")
    for name in ("atlas", "model", "provenance", "proofs"):
        checked(root, bundle[name], bundle[name+"_sha256"])
    workloads = {}
    for record in bundle["cases"]:
        key = record["id"]
        if key in workloads:
            raise ValueError("Duplicate workload identifier")
        stream = checked(root, record["stream"], record["stream_sha256"])
        raw = checked(root, record["expected"], record["expected_sha256"])
        output = checked(root, record["expected_output"], record["expected_output_sha256"])
        manifest = json.loads(checked(root, record["manifest"], record["manifest_sha256"]))
        tail = record["prediction"]
        prediction = checked(root, tail["path"], tail["sha256"])
        if stream[record["input_bytes"]:] != prediction:
            raise ValueError("Frozen prediction tail differs from stream")
        for name, info in record["inputs"].items():
            values = checked(root, info["path"], info["sha256"])
            region = {"x": 0, "y": 0, "width": record["width"], "height": record["height"]}
            start = (0 if name == "a" else 1)*record["frame_bytes"]
            if extract(raw[start:start+record["frame_bytes"]], region, record["coded_width"]) != values:
                raise ValueError("Reference input pixels differ from supplied inputs")
        expected_size = record["pictures"]*record["frame_bytes"]
        if len(raw) != expected_size or len(output) != record["width"]*record["height"]:
            raise ValueError("Reference dimensions or picture count differ")
        if record["frame_bytes"] != record["coded_width"]*record["coded_height"]*3//2:
            raise ValueError("Invalid I420 frame stride")
        if manifest["output_regions"] != record["output_regions"] or manifest["pictures"] != record["pictures"]:
            raise ValueError("Case geometry differs from instruction manifest")
        workloads[key] = {"record": record, "manifest": manifest, "expected": raw,
                          "output": output, "path": root/record["stream"]}
    return bundle, bundle_bytes, workloads


def validate(result, workload):
    record = workload["record"]
    if result.job_count != 1 or result.decoded_frames != record["pictures"]:
        raise AssertionError("Decoder picture count differs from manifest")
    expected, size = workload["expected"], record["frame_bytes"]
    if tuple(result.frames[0]) != tuple(expected[frame*size:(frame+1)*size]
                                        for frame in result.selected_frames):
        raise AssertionError("Decoded I420 differs: "+record["id"])
    region = record["output_regions"]["output"]
    position = result.selected_frames.index(region["frame"])
    native = extract(result.frames[0][position], region, record["coded_width"])
    if native != workload["output"]:
        raise AssertionError("Decoded output differs from truth-table oracle")
    return native


def run(workload, backend, mode):
    record = workload["record"]
    selection = tuple(sorted(set(record["output_frames"].values()))) if mode == "outputs" else None
    started = time.perf_counter()
    result = decode_batch([workload["path"]], backend,
                          width=record["coded_width"], height=record["coded_height"],
                          pictures_per_job=record["pictures"], selected_frames=selection)
    elapsed = (time.perf_counter()-started)*1000
    # Full reference loading and every equality check are outside the timer.
    native = validate(result, workload)
    return elapsed, result, native


def counters(result):
    return {"decoded_pictures": result.decoded_frames, "copied_pictures": result.copied_frames,
            "returned_pictures": len(result.selected_frames),
            "decoded_i420_bytes": result.decoded_frames*result.frame_bytes,
            "copied_i420_bytes": result.copied_frames*result.frame_bytes,
            "returned_i420_bytes": len(result.selected_frames)*result.frame_bytes}


def benchmark(bundle_path, output, backend="nvdec", repeats=5, helper=None, modes=("outputs", "all")):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not modes or len(set(modes)) != len(modes) or any(mode not in ("outputs", "all") for mode in modes):
        raise ValueError("modes must contain unique outputs/all selections")
    if helper and backend != "nvdec":
        raise ValueError("A helper is supported only for NVDEC")
    if backend == "nvdec":
        helper = Path(helper).resolve() if helper else _nvdec_helper()
        os.environ["H264_NVDEC_DECODER"] = str(helper)
    if not available(backend):
        raise RuntimeError("Requested decoder unavailable: "+backend)
    root, output = Path(bundle_path).resolve(), Path(output).resolve()
    bundle, bundle_bytes, workloads = load_bundle(root)
    report = {"format": "h264-compute-limits-results-0.1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "backend": backend,
              "bundle_sha256": digest(bundle_bytes), "atlas_sha256": bundle["atlas_sha256"],
              "helper_sha256": digest(helper.read_bytes()) if helper else None,
              "benchmark_sha256": digest(Path(__file__).read_bytes()),
              "decode_module_sha256": digest(Path(sys.modules[decode_batch.__module__].__file__).read_bytes()),
              "timing": {"repeats": repeats, "warmups_per_case_mode": 1, "modes": list(modes),
                         "order": "rotate case/mode order between repetitions",
                         "included": "stream assembly/file I/O, process and decoder startup, parsing, decoding, surface copies/readback, diagnostic and byte-count validation, frame slicing",
                         "excluded": "compilation, input packing, reference loading, pixel comparison, network transfer, capture writing",
                         "counter_meaning": "recurrent Boolean update evaluations; repeated iterations can be algebraically redundant, not independent useful tasks or equivalent GPU operations",
                         "steady_state": "not isolated; compare fixed-geometry 32/128-round runs to estimate incremental pipeline time"},
              "cases": {}, "incremental_estimates": []}

    def save():
        write_json(output/"results.json", report)

    save()
    for key, workload in workloads.items():
        record = workload["record"]
        _, result, native = run(workload, backend, "all")
        capture = Path("captures")/(key+".pgm")
        (output/capture).parent.mkdir(parents=True, exist_ok=True)
        (output/capture).write_bytes(("P5\n%d %d\n255\n" % (record["width"], record["height"])).encode()
                                    + bytes(value*255 for value in native))
        report["cases"][key] = {"function": record["function"], "mask": record["mask"],
                               "mode": record["mode"], "purpose": record["purpose"],
                               "width": record["width"], "height": record["height"],
                               "coded_width": record["coded_width"], "coded_height": record["coded_height"],
                               "pictures": record["pictures"], "rounds": record["rounds"],
                               "evaluations": record["evaluations"], "costs": record["costs"],
                               "validation": {"status": "pass", **counters(result),
                                              "i420_sha256": digest(b"".join(result.frames[0])),
                                              "output_sha256": digest(native), "capture": str(capture)},
                               "timings": {}}
        print("PASS %s: %d complete pictures, exact truth output" % (key, result.decoded_frames), flush=True)
        save()

    runs = [(key, mode) for key, value in workloads.items()
            if value["record"]["purpose"] == "throughput" for mode in modes]
    samples = {run: [] for run in runs}
    for iteration in range(repeats+1):
        order = runs[iteration % len(runs):]+runs[:iteration % len(runs)] if runs else []
        for key, mode in order:
            elapsed, result, _ = run(workloads[key], backend, mode)
            label = "warmup" if not iteration else "run %d" % iteration
            print("PASS %s %s %s: %.2f ms" % (key, mode, label, elapsed), flush=True)
            if not iteration:
                continue
            samples[(key, mode)].append(elapsed)
            values = samples[(key, mode)]
            median = statistics.median(values)
            record = workloads[key]["record"]
            area = record["coded_width"]*record["coded_height"]
            report["cases"][key]["timings"][mode] = {
                "status": "pass", **counters(result), "samples_ms": list(values),
                "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                "recurrent_evaluations_per_second": record["evaluations"]*1000/median,
                "decoded_luma_samples_per_second": result.decoded_frames*area*1000/median,
                "compute_luma_samples_per_evaluation": (record["pictures"]-2)*area/record["evaluations"],
                "all_luma_samples_per_evaluation": record["pictures"]*area/record["evaluations"]}
            save()

    # A slope is an estimate, not an isolated hardware timer. Keep negative or
    # noisy measurements visible rather than inventing a positive throughput.
    for key, workload in workloads.items():
        short = workload["record"]
        if short["purpose"] != "throughput" or short["rounds"] != 32:
            continue
        for other, candidate in workloads.items():
            long = candidate["record"]
            if (long["rounds"] != 128 or any(short[field] != long[field]
                    for field in ("mask", "mode", "width", "height"))):
                continue
            for mode in modes:
                delta = report["cases"][other]["timings"][mode]["median_ms"]-report["cases"][key]["timings"][mode]["median_ms"]
                evaluations = long["evaluations"]-short["evaluations"]
                report["incremental_estimates"].append({"short_case": key, "long_case": other,
                    "readback": mode, "additional_rounds": 96, "delta_median_ms": delta,
                    "incremental_evaluations_per_second": evaluations*1000/delta if delta > 0 else None,
                    "qualification": "difference of warmed process/pipeline medians; startup is approximately canceled, not measured in isolation"})
    report["status"] = "pass"
    report["totals"] = {"cases": len(workloads),
                        "validated_pictures": sum(entry["validation"]["decoded_pictures"] for entry in report["cases"].values()),
                        "validated_i420_bytes": sum(entry["validation"]["decoded_i420_bytes"] for entry in report["cases"].values())}
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT/"out"/"bundle")
    parser.add_argument("--output", type=Path, default=ROOT/"out"/"nvdec-results")
    parser.add_argument("--backend", choices=("ffmpeg", "openh264", "vaapi", "nvdec"), default="nvdec")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--modes", choices=("outputs", "all"), nargs="+", default=["outputs", "all"])
    args = parser.parse_args()
    report = benchmark(args.bundle, args.output, args.backend, args.repeats, args.helper, tuple(args.modes))
    print(json.dumps({"status": report["status"], **report["totals"]}), flush=True)


if __name__ == "__main__":
    main()
