#!/usr/bin/env python3
"""Portable exact resampling benchmark, Python 3.8 and the existing decoder only.

Reuses geometry, hash checks, I420 extraction, guard checks, copy counters and
the postponed-annotation Writer loader from the prior temporal benchmark.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

COMMON = Path(__file__).resolve().parents[1] / "temporal_benchmark.py"
spec = importlib.util.spec_from_file_location("_resampling_common", COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
decoder = common.decoder


def load(root):
    data = (root / "bundle.json").read_bytes()
    bundle = json.loads(data)
    if bundle.get("format") != "codec-compute-resampling-1" or bundle.get("status") != "pass":
        raise ValueError("Expected passing resampling bundle")
    workloads = {}
    for key, case in bundle["cases"].items():
        manifest = json.loads(common.checked(root, case["manifest"]))
        shape = common.geometry(manifest)
        tail = common.checked(root, case["tail"])
        common.checked(root, case["fields"])
        if case["geometry"] != shape or len(tail) != case["costs"]["prediction_bytes"]:
            raise AssertionError("Geometry or prediction cost mismatch")
        if len(case["input_names"]) != case["costs"]["input_pictures"]:
            raise AssertionError("Input picture count mismatch")
        jobs = []
        for index, job in enumerate(case["jobs"]):
            if job["index"] != index:
                raise ValueError("Job indices must be consecutive")
            prefix, stream = (common.checked(root, job[name]) for name in ("input_prefix", "stream"))
            if stream != prefix + tail:
                raise AssertionError("Stream differs from frozen prefix plus tail")
            inputs = {name: common.checked(root, record) for name, record in job["inputs"].items()}
            if set(inputs) != set(case["input_names"]) or any(len(p) != shape["width"]*shape["height"] for p in inputs.values()):
                raise AssertionError("Input plane names or dimensions differ")
            outputs = {name: common.checked(root, record) for name, record in job["outputs"].items()}
            if set(outputs) != set(shape["output_frames"]):
                raise AssertionError("Named outputs differ from manifest")
            raw = common.checked(root, job["expected"])
            if len(raw) != shape["frame_bytes"]*shape["pictures_per_job"]:
                raise AssertionError("Full I420 reference size differs")
            frames = tuple(raw[i:i+shape["frame_bytes"]] for i in range(0, len(raw), shape["frame_bytes"]))
            common.validate_guards(frames, shape)
            for frame, name in enumerate(case["input_names"]):
                if common.extract(frames[frame], shape) != inputs[name]:
                    raise AssertionError("Full decoded input reference differs from saved input")
            for name, frame in shape["output_frames"].items():
                if common.extract(frames[frame], shape) != outputs[name]:
                    raise AssertionError("Full decoded reference differs from saved named output")
            boards = {name: [list(value[y*shape["width"]:(y+1)*shape["width"]])
                             for y in range(shape["height"])] for name, value in inputs.items()}
            jobs.append({"path": root / job["stream"]["path"], "prefix": prefix,
                         "frames": frames, "outputs": outputs, "boards": boards})
        if len(jobs) != bundle["job_count"] or len({j["input_prefix"]["sha256"] for j in case["jobs"]}) != len(jobs):
            raise AssertionError("Expected complete distinct input fixtures")
        workloads[key] = {"record": case, "shape": shape, "tail": tail, "jobs": jobs}
    return data, bundle, workloads


def run(bundle_path, output, *, backend="ffmpeg", repeats=5, batch_sizes=(1, 8), helper=None, cases=None):
    if type(repeats) is not int or repeats < 1 or not batch_sizes or any(type(n) is not int or n < 1 for n in batch_sizes):
        raise ValueError("Positive repeats and batch sizes required")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Batch sizes must be unique")
    if helper is not None:
        if backend != "nvdec":
            raise ValueError("--helper applies only to NVDEC")
        os.environ["H264_NVDEC_DECODER"] = str(Path(helper).resolve())
    if not decoder.available(backend):
        raise RuntimeError("Decoder unavailable: " + backend)
    root, output = Path(bundle_path).resolve(), Path(output).resolve()
    bundle_bytes, bundle, workloads = load(root)
    selected = list(cases) if cases else [key for key in workloads if key.endswith("/fused") or key == "motion-interpolation/baseline"]
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(workloads):
        raise ValueError("Select unique known cases")
    if max(batch_sizes) > bundle["job_count"]:
        raise ValueError("Batch exceeds distinct prepared jobs")
    writer_path = Path(decoder.__file__).with_name("bitstream.py")
    files = {"runner": Path(__file__), "common": COMMON, "decoder": Path(decoder.__file__), "writer": writer_path}
    hashes = {name: common.digest(path.read_bytes()) for name, path in files.items()}
    if hashes != bundle["runtime_source_sha256"]:
        raise AssertionError("Runtime sources differ from bundle")
    Writer = common.writer_class(writer_path)
    for workload in workloads.values():
        params = dict(workload["record"]["writer"])
        params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
        workload["writer"] = Writer(**params)
    report = {"format": "codec-compute-resampling-benchmark-1", "status": "running", "backend": backend,
              "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              "repository_commit": bundle["repository_commit"], "bundle_sha256": common.digest(bundle_bytes),
              "runtime_source_sha256": hashes, "repeats": repeats, "warmups_per_mode": 1,
              "batch_sizes": list(batch_sizes), "timed_cases": selected, "cases": {},
              "protocol": {"input-serialization": "Validate and serialize already loaded input boards; exact input-prefix comparison; no disk I/O, compilation or decoding",
                           "preencoded-all": "Batch assembly and file I/O, one process/pipeline startup, decode/copy all pictures, diagnostics, exact returned-I420 and named-output comparisons",
                           "preencoded-outputs": "Same scope, selecting named-output pictures; NVDEC skips unselected surface copies, FFmpeg still copies all pictures",
                           "host-to-outputs": "Measured together: serialize/check inputs, append cached tail, write streams, then selected-output decode and exact comparisons",
                           "excluded": "Compilation, motion fields, fixtures, independent formula/IR checks, reference loading, source/hash auditing, initial guard/chroma checks, network and report writing",
                           "order": "Rotate case/batch/mode order after one warmup per mode",
                           "caveats": "Finite batches to EOS include startup; per-job figures are amortized throughput, not individual latency. Fields are cached compile-time constants. No flow estimation, resident video service, GPU baseline or concurrent GPU workload is measured.",
                           "writer": "Unmodified Writer source with postponed annotations on Python3.8; every serialized prefix checked byte for byte"}}
    if backend == "nvdec":
        report["decoder_helper_sha256"] = common.digest(decoder._nvdec_helper().read_bytes())
    else:
        report["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], universal_newlines=True).splitlines()[0]

    def save():
        common.write_json(output, report)

    def encode(workload, jobs):
        prefixes = []
        for job in jobs:
            prefix = b"".join(workload["writer"].input_luma(job["boards"][name], frame)
                              for frame, name in enumerate(workload["record"]["input_names"]))
            if prefix != job["prefix"]:
                raise AssertionError("Input serialization differs from frozen prefix")
            prefixes.append(prefix)
        return prefixes

    def decode(workload, jobs, paths, selection):
        shape = workload["shape"]
        result = decoder.decode_batch(paths, backend, width=shape["coded_width"], height=shape["coded_height"],
                                      pictures_per_job=shape["pictures_per_job"], selected_frames=selection)
        if len(result.frames) != len(jobs):
            raise AssertionError("Decoded job count differs")
        positions = {frame: i for i, frame in enumerate(result.selected_frames)}
        for frames, job in zip(result.frames, jobs):
            if frames != tuple(job["frames"][frame] for frame in result.selected_frames):
                raise AssertionError("Actual decoded I420 differs")
            for name, frame in shape["output_frames"].items():
                if common.extract(frames[positions[frame]], shape) != job["outputs"][name]:
                    raise AssertionError("Named output differs from frozen independent oracle")
        return result

    # Validate every saved baseline and fused stream, including untimed cases.
    for key, workload in workloads.items():
        jobs, shape = workload["jobs"], workload["shape"]
        result = decode(workload, jobs, [job["path"] for job in jobs], None)
        guards = {"guard_bytes": 0, "chroma_bytes": 0}
        for frames in result.frames:
            for name, value in common.validate_guards(frames, shape).items():
                guards[name] += value
        report["cases"][key] = {"costs": workload["record"]["costs"], "geometry": shape,
                               "validation": {"status": "pass", **common.counters(result), **guards,
                                              "named_outputs": len(jobs)*len(shape["output_frames"])}, "batches": {}}
        print("PASS full validation " + key, flush=True)
    quality = bundle["quality"]
    shape = common.geometry(json.loads(common.checked(root, quality["manifest"])))
    common.checked(root, quality["stream"])
    result = decoder.decode_batch([root / quality["stream"]["path"]], backend, width=shape["coded_width"],
                                  height=shape["coded_height"], pictures_per_job=shape["pictures_per_job"])
    raw = b"".join(result.frames[0])
    if raw != common.checked(root, quality["expected"]):
        raise AssertionError("Quality witness full I420 differs")
    for frame, name in enumerate(("a", "b")):
        if common.extract(result.frames[0][frame], shape) != common.checked(root, quality["inputs"][name]):
            raise AssertionError("Quality witness input differs")
    midpoint = common.checked(root, quality["midpoint"])
    if len(midpoint) != shape["width"]*shape["height"]:
        raise AssertionError("Quality midpoint dimensions differ")
    crop = quality["crop"]
    if any(type(crop[k]) is not int for k in ("x", "y", "width", "height")) or min(crop["x"], crop["y"]) < 0 or min(crop["width"], crop["height"]) < 1 or crop["x"]+crop["width"] > shape["width"] or crop["y"]+crop["height"] > shape["height"]:
        raise AssertionError("Quality crop is invalid")
    metrics = {}
    for name, frame in shape["output_frames"].items():
        values = common.extract(result.frames[0][frame], shape)
        if values != common.checked(root, quality["outputs"][name]):
            raise AssertionError("Quality witness named output differs")
        errors = [abs(values[y*shape["width"]+x]-midpoint[y*shape["width"]+x])
                  for y in range(crop["y"], crop["y"]+crop["height"])
                  for x in range(crop["x"], crop["x"]+crop["width"])]
        metrics[name] = {"mae": sum(errors)/len(errors), "max_absolute_error": max(errors),
                         "squared_error_sum": sum(error*error for error in errors)}
    if metrics != quality["metrics"]:
        raise AssertionError("Actual quality metrics differ from prepared report")
    report["quality_validation"] = {"status": "pass", **common.counters(result),
                                     **common.validate_guards(result.frames[0], shape), "i420_sha256": common.digest(raw),
                                     "crop": crop, "metrics": metrics}
    save()

    def execute(key, count, mode):
        workload = workloads[key]
        jobs = workload["jobs"][:count]
        encoded = {"encoded_input_pictures": count*len(workload["record"]["input_names"]),
                   "encoded_input_bytes": sum(len(j["prefix"]) for j in jobs)}
        if mode == "input-serialization":
            encode(workload, jobs)
            return {**encoded, "decoder_sessions": 0}
        selection = None if mode == "preencoded-all" else workload["shape"]["output_picture_indices"]
        if mode == "host-to-outputs":
            prefixes = encode(workload, jobs)
            with tempfile.TemporaryDirectory(prefix="resampling-inputs-") as directory:
                paths = [Path(directory) / ("%03d.h264" % index) for index in range(count)]
                for path, prefix in zip(paths, prefixes):
                    path.write_bytes(prefix + workload["tail"])
                result = decode(workload, jobs, paths, selection)
            return {**encoded, **common.counters(result)}
        return common.counters(decode(workload, jobs, [j["path"] for j in jobs], selection))

    runs = [(key, count, mode) for key in selected for count in batch_sizes
            for mode in ("input-serialization", "preencoded-all", "preencoded-outputs", "host-to-outputs")]
    samples = {key: [] for key in runs}
    for iteration in range(repeats+1):
        offset = iteration % len(runs)
        for key, count, mode in runs[offset:] + runs[:offset]:
            started = time.perf_counter()
            counts = execute(key, count, mode)
            elapsed = (time.perf_counter()-started)*1000
            if iteration:
                values = samples[key, count, mode]
                values.append(elapsed)
                median = statistics.median(values)
                report["cases"][key]["batches"].setdefault(str(count), {})[mode] = {
                    "status": "pass", **counts, "jobs": count, "samples_ms": list(values),
                    "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                    "amortized_ms_per_job": median/count, "jobs_per_second": 1000*count/median}
                save()
            print("PASS %s %d %s %s %.3f ms" %
                  (key, count, mode, "warmup" if not iteration else "run%d" % iteration, elapsed), flush=True)
    if (root / "bundle.json").read_bytes() != bundle_bytes or hashes != {name: common.digest(path.read_bytes()) for name, path in files.items()}:
        raise AssertionError("Bundle or runtime changed during measurements")
    report["status"] = "pass"
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("ffmpeg", "nvdec"), default="ffmpeg")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    run(args.bundle, args.output, backend=args.backend, repeats=args.repeats,
        batch_sizes=args.batch_sizes, helper=args.helper, cases=args.cases)
